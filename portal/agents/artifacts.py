"""
Artifact Storage and Retrieval Layer for SXI Explanation Engine.

Supports:
1. FileSystemArtifactStore: Local filesystem storage for development and testing.
2. DatabaseArtifactStore: Durable production storage in Neon PostgreSQL (agent_explanation_artifacts)
   using gzip compression, size caps, and retention policies.

Guarantees:
- Pure native serialization for SXI predictors (XGBoost JSON) — ZERO pickle anywhere.
- Immutability per run with SHA-256 integrity checks.
- Major version compatibility audit on load.
- Strict path and identifier sanitization.
"""

import gzip
import hashlib
import json
import os
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from portal.agents.db import get_neon_connection

MAX_ARTIFACT_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB safety cap


class ArtifactIntegrityError(Exception):
    """Raised when an SXI artifact SHA-256 checksum does not match its manifest."""
    pass


class SXICompatibilityError(Exception):
    """Raised when SXI runtime versions are incompatible."""
    pass


def sanitize_identifier(value: str) -> str:
    """
    Sanitizes customer_id, session_id, run_id, and solution_code.
    Only allows alphanumeric, hyphens, and underscores.
    """
    if not value or not re.match(r"^[A-Za-z0-9_-]+$", str(value)):
        raise ValueError(f"Invalid identifier '{value}'. Must match '^[A-Za-z0-9_-]+$'.")
    return str(value)


def compute_sha256(content: bytes) -> str:
    """Computes hexadecimal SHA-256 checksum for given bytes."""
    return hashlib.sha256(content).hexdigest()


@dataclass
class ArtifactRef:
    artifact_id: str
    run_id: str
    solution: str
    kind: str
    sha256: str
    size_bytes: int
    uri: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "run_id": self.run_id,
            "solution": self.solution,
            "kind": self.kind,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "uri": self.uri,
            "created_at": self.created_at,
        }


class ArtifactStore(ABC):
    """Abstract interface for storing and retrieving SXI explanation artifacts."""

    @abstractmethod
    def save_artifact(
        self,
        run_id: str,
        solution: str,
        kind: str,
        content: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRef:
        """Persists artifact bytes immutably. Returns reference metadata."""
        pass

    @abstractmethod
    def load_artifact(
        self,
        run_id: str,
        solution: str,
        kind: str,
        expected_sha256: Optional[str] = None,
    ) -> bytes:
        """Loads artifact bytes, verifying SHA-256 integrity."""
        pass

    @abstractmethod
    def artifact_exists(self, run_id: str, solution: str, kind: str) -> bool:
        """Checks if artifact already exists."""
        pass


class FileSystemArtifactStore(ArtifactStore):
    """Local filesystem backend for development, testing, and offline runs."""

    def __init__(self, base_dir: str = ""):
        if not base_dir:
            try:
                from django.conf import settings
                if settings.configured:
                    base_dir = getattr(settings, "AGENT_ARTIFACT_ROOT", "")
            except Exception:
                base_dir = ""
        if not base_dir:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads", "artifacts")
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_path(self, run_id: str, solution: str, kind: str) -> str:
        s_run = sanitize_identifier(run_id)
        s_sol = sanitize_identifier(solution)
        s_kind = sanitize_identifier(kind)
        target_dir = os.path.join(self.base_dir, s_run, s_sol)
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, f"{s_kind}.bin")

    def save_artifact(
        self,
        run_id: str,
        solution: str,
        kind: str,
        content: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRef:
        if len(content) > MAX_ARTIFACT_SIZE_BYTES:
            raise ValueError(f"Artifact size {len(content)} exceeds cap of {MAX_ARTIFACT_SIZE_BYTES} bytes.")

        file_path = self._get_path(run_id, solution, kind)
        sha = compute_sha256(content)

        # Immutability check
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                existing_bytes = f.read()
            if compute_sha256(existing_bytes) != sha:
                raise RuntimeError(f"Artifact immutable: cannot overwrite {run_id}/{solution}/{kind}.")
            created_at = datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc).isoformat()
            return ArtifactRef(
                artifact_id=f"fs_{run_id}_{solution}_{kind}",
                run_id=run_id,
                solution=solution,
                kind=kind,
                sha256=sha,
                size_bytes=len(content),
                uri=f"file://{file_path.replace(os.sep, '/')}",
                created_at=created_at,
            )

        with open(file_path, "wb") as f:
            f.write(content)

        created_at = datetime.now(timezone.utc).isoformat()
        return ArtifactRef(
            artifact_id=f"fs_{run_id}_{solution}_{kind}",
            run_id=run_id,
            solution=solution,
            kind=kind,
            sha256=sha,
            size_bytes=len(content),
            uri=f"file://{file_path.replace(os.sep, '/')}",
            created_at=created_at,
        )

    def load_artifact(
        self,
        run_id: str,
        solution: str,
        kind: str,
        expected_sha256: Optional[str] = None,
    ) -> bytes:
        file_path = self._get_path(run_id, solution, kind)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Artifact not found: {file_path}")

        with open(file_path, "rb") as f:
            content = f.read()

        actual_sha = compute_sha256(content)
        if expected_sha256 and actual_sha != expected_sha256:
            raise ArtifactIntegrityError(
                f"sxi_integrity_failed: expected {expected_sha256}, got {actual_sha}"
            )

        return content

    def artifact_exists(self, run_id: str, solution: str, kind: str) -> bool:
        file_path = self._get_path(run_id, solution, kind)
        return os.path.exists(file_path)


class DatabaseArtifactStore(ArtifactStore):
    """
    Production storage backend using PostgreSQL table agent_explanation_artifacts.
    Compresses content with gzip, enforces size caps, and provides retention purging.
    """

    def save_artifact(
        self,
        run_id: str,
        solution: str,
        kind: str,
        content: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRef:
        s_run = sanitize_identifier(run_id)
        s_sol = sanitize_identifier(solution)
        s_kind = sanitize_identifier(kind)

        if len(content) > MAX_ARTIFACT_SIZE_BYTES:
            raise ValueError(f"Artifact size {len(content)} exceeds cap of {MAX_ARTIFACT_SIZE_BYTES} bytes.")

        sha = compute_sha256(content)
        compressed = gzip.compress(content, compresslevel=6)

        query_check = """
            SELECT id, sha256, size_bytes, created_at
            FROM agent_explanation_artifacts
            WHERE run_id = %s AND solution = %s AND kind = %s;
        """
        insert_query = """
            INSERT INTO agent_explanation_artifacts
                (run_id, solution, kind, sha256, size_bytes, content, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id, created_at;
        """

        with get_neon_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query_check, (s_run, s_sol, s_kind))
                row = cur.fetchone()
                if row:
                    existing_id, existing_sha, existing_size, existing_created = row
                    if existing_sha != sha:
                        raise RuntimeError(f"Artifact immutable: cannot overwrite {s_run}/{s_sol}/{s_kind}.")
                    return ArtifactRef(
                        artifact_id=str(existing_id),
                        run_id=s_run,
                        solution=s_sol,
                        kind=s_kind,
                        sha256=existing_sha,
                        size_bytes=existing_size,
                        uri=f"db://agent_explanation_artifacts/{existing_id}",
                        created_at=existing_created.isoformat() if hasattr(existing_created, "isoformat") else str(existing_created),
                    )

                cur.execute(
                    insert_query,
                    (
                        s_run,
                        s_sol,
                        s_kind,
                        sha,
                        len(content),
                        compressed,
                        json.dumps(metadata or {}),
                    ),
                )
                new_id, created_at = cur.fetchone()
                conn.commit()

        return ArtifactRef(
            artifact_id=str(new_id),
            run_id=s_run,
            solution=s_sol,
            kind=s_kind,
            sha256=sha,
            size_bytes=len(content),
            uri=f"db://agent_explanation_artifacts/{new_id}",
            created_at=created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
        )

    def load_artifact(
        self,
        run_id: str,
        solution: str,
        kind: str,
        expected_sha256: Optional[str] = None,
    ) -> bytes:
        s_run = sanitize_identifier(run_id)
        s_sol = sanitize_identifier(solution)
        s_kind = sanitize_identifier(kind)

        query = """
            SELECT content, sha256
            FROM agent_explanation_artifacts
            WHERE run_id = %s AND solution = %s AND kind = %s;
        """
        with get_neon_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (s_run, s_sol, s_kind))
                row = cur.fetchone()
                if not row:
                    raise FileNotFoundError(f"Artifact not found in DB: {s_run}/{s_sol}/{s_kind}")
                compressed_content, db_sha = row

        content = gzip.decompress(compressed_content)
        actual_sha = compute_sha256(content)

        if actual_sha != db_sha:
            raise ArtifactIntegrityError(f"sxi_integrity_failed: DB stored checksum mismatch.")
        if expected_sha256 and actual_sha != expected_sha256:
            raise ArtifactIntegrityError(
                f"sxi_integrity_failed: expected {expected_sha256}, got {actual_sha}"
            )

        return content

    def artifact_exists(self, run_id: str, solution: str, kind: str) -> bool:
        s_run = sanitize_identifier(run_id)
        s_sol = sanitize_identifier(solution)
        s_kind = sanitize_identifier(kind)
        query = """
            SELECT 1 FROM agent_explanation_artifacts
            WHERE run_id = %s AND solution = %s AND kind = %s;
        """
        with get_neon_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (s_run, s_sol, s_kind))
                return cur.fetchone() is not None

    def purge_old_artifacts(self, retention_days: int = 30) -> int:
        """Purges artifacts older than retention_days. Returns count of deleted rows."""
        purge_query = """
            DELETE FROM agent_explanation_artifacts
            WHERE created_at < NOW() - INTERVAL '%s days';
        """
        with get_neon_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(purge_query, (int(retention_days),))
                deleted = cur.rowcount
                conn.commit()
                return deleted


def get_artifact_store(backend: Optional[str] = None) -> ArtifactStore:
    """Factory helper to obtain the configured ArtifactStore instance."""
    selected = backend
    if not selected:
        try:
            from django.conf import settings
            if settings.configured:
                selected = getattr(settings, "AGENT_ARTIFACT_BACKEND", "filesystem")
        except Exception:
            selected = "filesystem"
    selected = selected or "filesystem"
    if selected == "database":
        return DatabaseArtifactStore()
    return FileSystemArtifactStore()


@dataclass
class ArtifactBundle:
    """Typed container for loaded SXI explanation artifacts."""
    run_id: str
    solution: str
    manifest: Dict[str, Any]
    sxi_predictor: Any
    preprocessing_spec: Dict[str, Any]


def load_sxi_artifacts(
    run_id: str,
    solution: str,
    store: Optional[ArtifactStore] = None,
) -> Tuple[Optional[ArtifactBundle], Optional[str]]:
    """
    Safely loads and verifies all SXI artifacts for a solution run.
    Performs:
    1. Path sanitization
    2. Manifest loading
    3. Major version compatibility check
    4. SHA-256 integrity verification for each artifact
    5. Native non-pickle SXI deserialization

    Returns (bundle, None) on success, or (None, skip_reason) on failure.
    """
    try:
        s_run = sanitize_identifier(run_id)
        s_sol = sanitize_identifier(solution)
    except ValueError:
        return None, "invalid_identifier"

    if store is None:
        store = get_artifact_store()

    # 1. Load Manifest
    try:
        manifest_bytes = store.load_artifact(s_run, s_sol, "manifest")
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except FileNotFoundError:
        return None, "no_sxi_artifact"
    except Exception as e:
        return None, f"manifest_load_error: {str(e)}"

    # 2. Check SXI Major Version Compatibility
    recorded_xgb_ver = manifest.get("xgboost_version", "")
    if recorded_xgb_ver and xgb is not None:
        rec_major = recorded_xgb_ver.split(".")[0]
        cur_major = xgb.__version__.split(".")[0]
        if rec_major != cur_major:
            return None, f"incompatible_sxi_version: recorded {recorded_xgb_ver} vs runtime {xgb.__version__}"

    artifact_hashes = manifest.get("artifact_hashes", {})

    # 3. Load Preprocessing Spec
    spec_sha = artifact_hashes.get("preprocessing_spec")
    try:
        spec_bytes = store.load_artifact(s_run, s_sol, "preprocessing_spec", expected_sha256=spec_sha)
        preprocessing_spec = json.loads(spec_bytes.decode("utf-8"))
    except ArtifactIntegrityError:
        return None, "sxi_integrity_failed"
    except FileNotFoundError:
        return None, "no_sxi_artifact"
    except Exception as e:
        return None, f"preprocessing_spec_error: {str(e)}"

    # 4. Load SXI Predictor
    pred_sha = artifact_hashes.get("sxi_predictor")
    try:
        pred_bytes = store.load_artifact(s_run, s_sol, "sxi_predictor", expected_sha256=pred_sha)
    except ArtifactIntegrityError:
        return None, "sxi_integrity_failed"
    except FileNotFoundError:
        return None, "no_sxi_artifact"
    except Exception as e:
        return None, f"sxi_predictor_error: {str(e)}"

    # Zero-pickle check on predictor bytes
    if pred_bytes.startswith(b"\x80") or b"cos\nsystem" in pred_bytes:
        return None, "sxi_integrity_failed: pickle_detected"

    task_type = manifest.get("task_type", "regression")
    if xgb is None:
        return None, "sxi_runtime_missing"

    try:
        if task_type == "classification":
            predictor = xgb.XGBClassifier()
        else:
            predictor = xgb.XGBRegressor()
        predictor.load_model(bytearray(pred_bytes))
    except Exception as e:
        return None, f"sxi_deserialization_failed: {str(e)}"

    return ArtifactBundle(
        run_id=s_run,
        solution=s_sol,
        manifest=manifest,
        sxi_predictor=predictor,
        preprocessing_spec=preprocessing_spec,
    ), None


def persist_sxi_run_artifacts(
    fitted_sxi: Any,
    task_type: str,
    target_name: str,
    feature_names: List[str],
    df: Any,
    X_numeric: Any,
    X_train: Any,
    X_test: Any,
    y: Any,
    y_test: Any,
    raw_r2: Optional[float],
    acc: Optional[float],
    raw_rmse: Optional[float],
    raw_mae: Optional[float],
    imputation_values: Dict[str, Any],
    categorical_encodings: Dict[str, Any],
    run_id: str,
    code: str,
    session_id: str = "",
    output_dir: str = "",
    store: Optional[ArtifactStore] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Persists native SXI predictor and associated explanation artifacts:
    1. sxi_predictor: Native XGBoost JSON (ZERO pickle).
    2. preprocessing_spec: Imputation values, categorical classes, scaler params.
    3. manifest: Row ID lists (train & holdout), direction hints, dataset hash, SHA-256 hashes.

    Emits skip 'primary_sxi_not_serializable' if non-XGBoost fallback was used.
    """
    import numpy as np
    import pandas as pd

    # 1. Fallback / Non-XGBoost check -> emit skip reason
    if xgb is None or not isinstance(fitted_sxi, (xgb.XGBClassifier, xgb.XGBRegressor)):
        return None, "primary_sxi_not_serializable"

    try:
        s_run = sanitize_identifier(run_id)
        s_sol = sanitize_identifier(code)
    except ValueError:
        return None, "invalid_identifier"

    if store is None:
        if output_dir:
            store = FileSystemArtifactStore(base_dir=output_dir)
        else:
            store = get_artifact_store()

    # 2. Extract Native Predictor Bytes (pure JSON, zero pickle)
    try:
        predictor_bytes = bytes(fitted_sxi.get_booster().save_raw(raw_format="json"))
    except Exception as e:
        return None, f"primary_sxi_not_serializable: {str(e)}"

    if predictor_bytes.startswith(b"\x80") or b"cos\nsystem" in predictor_bytes:
        return None, "primary_sxi_not_serializable: pickle_detected"

    # 3. Stable Row ID Mapping (train and holdout raw_row_id lists go in manifest only)
    if "raw_row_id" in df.columns:
        row_id_series = df["raw_row_id"]
    elif "master_id" in df.columns:
        row_id_series = df["master_id"]
    else:
        row_id_series = pd.Series(range(1, len(df) + 1), index=df.index)

    train_row_ids = [int(x) for x in row_id_series.loc[X_train.index].tolist()]
    holdout_row_ids = [int(x) for x in row_id_series.loc[X_test.index].tolist()]

    # 4. Direction Hint: Lasso sign + Spearman correlation with target
    direction_hint = {}
    try:
        from scipy.stats import spearmanr
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
        lasso = Lasso(alpha=0.01, random_state=42, max_iter=2000)
        y_numeric = pd.to_numeric(y, errors="coerce").fillna(0)
        lasso.fit(X_scaled, y_numeric)
        lasso_coefs = dict(zip(feature_names, lasso.coef_))
    except Exception:
        lasso_coefs = {}

    for col in feature_names:
        try:
            sp_val, _ = spearmanr(X_numeric[col], y)
            sp_corr = float(sp_val) if np.isfinite(sp_val) else 0.0
        except Exception:
            sp_corr = 0.0
        c_coef = lasso_coefs.get(col, 0.0)
        sign = 1 if c_coef > 1e-5 else (-1 if c_coef < -1e-5 else 0)
        direction_hint[col] = {
            "lasso_sign": sign,
            "spearman_corr": round(sp_corr, 4),
        }

    # 5. Extract master encoding metadata & scaler params if available
    meta_path = ""
    if output_dir and session_id:
        p1 = os.path.join(output_dir, "master", f"master_{session_id}_encoding.json")
        p2 = os.path.join(output_dir, f"master_{session_id}_encoding.json")
        if os.path.exists(p1):
            meta_path = p1
        elif os.path.exists(p2):
            meta_path = p2

    enc_meta = {}
    scaler_params = {}
    feature_aliases = {}
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                mdata = json.load(mf)
                enc_meta = mdata.get("enc_meta", {})
                scaler_params = mdata.get("scaler_params", {})
                feature_aliases = mdata.get("feature_aliases", {})
        except Exception:
            pass

    # Verify if sxi_engine categorical encoding matches enc_meta; persist re-encoding mapping
    reencoding_mapping = {}
    for col, enc_info in categorical_encodings.items():
        prior = enc_meta.get(col) or enc_meta.get("mappings", {}).get(col)
        if prior and isinstance(prior, dict):
            if prior != enc_info.get("mapping", {}):
                reencoding_mapping[col] = {
                    "enc_meta": prior,
                    "training": enc_info.get("mapping", {}),
                }
            else:
                reencoding_mapping[col] = "identity"
        else:
            reencoding_mapping[col] = "identity"

    # 6. Build Preprocessing Spec
    preprocessing_spec = {
        "imputation_values": imputation_values,
        "categorical_encodings": categorical_encodings,
        "scaler_params": scaler_params,
        "reencoding_mapping": reencoding_mapping,
        "feature_aliases": feature_aliases,
    }
    spec_bytes = json.dumps(preprocessing_spec, indent=2).encode("utf-8")

    # 7. Build Manifest
    dataset_hash = compute_sha256(df.to_csv(index=False).encode("utf-8"))
    pred_sha = compute_sha256(predictor_bytes)
    spec_sha = compute_sha256(spec_bytes)

    manifest = {
        "run_id": s_run,
        "solution": s_sol,
        "sxi_version": "2.1a",
        "xgboost_version": xgb.__version__ if xgb is not None else "",
        "dataset_hash": dataset_hash,
        "task_type": task_type,
        "target_column": target_name,
        "final_feature_order": feature_names,
        "dropped_features": [
            c for c in df.columns
            if c not in feature_names and c not in ("raw_row_id", "master_id", target_name)
        ],
        "feature_aliases": feature_aliases,
        "train_row_ids": train_row_ids,
        "holdout_row_ids": holdout_row_ids,
        "split_params": {"test_size": 0.20, "random_state": 42},
        "direction_hint": direction_hint,
        "holdout_metrics": {
            "r2_score": round(float(raw_r2), 4) if raw_r2 is not None else None,
            "accuracy": round(float(acc), 4) if acc is not None else None,
            "rmse": round(float(raw_rmse), 4) if raw_rmse is not None else None,
            "mae": round(float(raw_mae), 4) if raw_mae is not None else None,
        },
        "artifact_hashes": {
            "sxi_predictor": pred_sha,
            "preprocessing_spec": spec_sha,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    manifest_sha = compute_sha256(manifest_bytes)

    # 8. Persist All Artifacts Immutably
    store.save_artifact(s_run, s_sol, "sxi_predictor", predictor_bytes)
    store.save_artifact(s_run, s_sol, "preprocessing_spec", spec_bytes)
    store.save_artifact(s_run, s_sol, "manifest", manifest_bytes)

    # 9. Return Pointers Only
    sxi_ptr = {
        "artifact_id": f"{s_run}_{s_sol}",
        "manifest_sha256": manifest_sha,
        "artifact_ref": f"{s_run}/{s_sol}/manifest",
        "dataset_hash": dataset_hash,
        "xgboost_version": xgb.__version__ if xgb is not None else "",
    }
    return sxi_ptr, None

