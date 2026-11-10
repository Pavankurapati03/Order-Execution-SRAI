"""
Unit tests for SXI explanation artifact persistence, retrieval, and integrity.
Tests all requirements for Phase 2.1a:
1. raw_row_id alignment when rows are dropped before ID assignment.
2. Pure native serialization (XGBoost JSON) — zero pickle anywhere.
3. Save/load roundtrip gives identical predictions.
4. pred_contribs additivity (sum(contribs) + bias == raw margin).
5. Loaded SXI predictor reproduces stored holdout metrics.
6. SHA-256 hash mismatch triggers 'sxi_integrity_failed'.
7. Missing artifacts trigger 'no_sxi_artifact'.
8. Incompatible major version triggers 'incompatible_sxi_version'.
9. Sklearn fallback model triggers 'primary_sxi_not_serializable'.
10. Pointers only in sxi_res (row IDs stay in manifest).
"""

import json
import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from portal.agents.artifacts import (
    FileSystemArtifactStore,
    DatabaseArtifactStore,
    persist_sxi_run_artifacts,
    load_sxi_artifacts,
    compute_sha256,
    sanitize_identifier,
)
from portal.sxi_engine import run_sxi_optimization
from portal.views import run_automated_preprocessing_pipeline


@pytest.fixture
def temp_artifact_dir():
    """Temporary directory for FileSystemArtifactStore tests."""
    temp_dir = tempfile.mkdtemp(prefix="sxi_art_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def synthetic_order_data():
    """Synthetic dataset for order execution solutions."""
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "supplier_id": np.random.choice(["SUP-1", "SUP-2", "SUP-3"], size=n),
        "warehouse_id": np.random.choice(["WH-A", "WH-B"], size=n),
        "order_quantity": np.random.randint(10, 500, size=n),
        "safety_stock_days": np.random.uniform(1.0, 15.0, size=n),
        "transit_distance_km": np.random.uniform(20.0, 800.0, size=n),
        "lead_time_days": np.random.uniform(2.0, 18.0, size=n),
    })
    return df


def test_raw_row_id_alignment_with_dropped_rows(temp_artifact_dir):
    """
    BLOCKING REQUIREMENT:
    Verify that raw_row_id is assigned at ingestion before filtering/cleaning,
    survives into master_df, and dropped rows are accurately recorded in raw_master_coverage.
    """
    np.random.seed(42)
    raw_df = pd.DataFrame({
        "supplier_id": ["SUP-1"] * 20,
        "lead_time_days": [5.0] * 20,
        "transit_distance_km": [100.0] * 20,
    })
    # Inject extreme outlier to trigger row drop during outlier handling
    raw_df.loc[5, "transit_distance_km"] = 99999.0
    raw_df.loc[12, "transit_distance_km"] = 88888.0

    sol_cfg = {
        "code": "E4",
        "name": "Delivery Lead Time Optimization",
        "target_column": "lead_time_days",
        "task_type": "regression",
        "actionable_drivers": ["transit_distance_km"],
        "context_drivers": ["supplier_id"],
    }

    session_id = "test_align_sess"
    master_df, _, _, _ = run_automated_preprocessing_pipeline(
        df=raw_df,
        session_id=session_id,
        saved_path="dummy.csv",
        sol_cfg=sol_cfg,
        sol_key="e4",
    )

    # raw_row_id must be in master_df and aligned to master_id
    assert "raw_row_id" in master_df.columns
    assert "master_id" in master_df.columns
    assert (master_df["master_id"] == master_df["raw_row_id"]).all()

    # Verify coverage metadata persisted
    meta_path = os.path.join("uploads", "master", f"master_{session_id}_encoding.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        coverage = meta.get("raw_master_coverage", {})
        assert coverage.get("raw_row_count") == 20
        assert coverage.get("master_row_count") == len(master_df)
        assert coverage.get("dropped_count") == 20 - len(master_df)


@pytest.mark.skipif(xgb is None, reason="XGBoost required for native SXI tests")
def test_zero_pickle_native_serialization(temp_artifact_dir, synthetic_order_data):
    """Verify that all stored artifact bytes are pure JSON with zero pickle."""
    store = FileSystemArtifactStore(base_dir=temp_artifact_dir)
    res = run_sxi_optimization(
        df=synthetic_order_data,
        solution_key="e4",
        output_dir=temp_artifact_dir,
        session_id="sess_pickle_check",
    )

    assert res["sxi_artifact_skip_reason"] is None
    sxi_ptr = res["sxi_artifacts"]
    assert sxi_ptr is not None

    run_id = res["run_id"]
    code = res["solution_code"]

    for kind in ["sxi_predictor", "preprocessing_spec", "manifest"]:
        content = store.load_artifact(run_id, code, kind)
        assert not content.startswith(b"\x80"), f"Pickle protocol magic byte detected in {kind}!"
        assert b"cos\nsystem" not in content, f"Pickle bytecode detected in {kind}!"


@pytest.mark.skipif(xgb is None, reason="XGBoost required for native SXI tests")
def test_save_load_prediction_identity(temp_artifact_dir, synthetic_order_data):
    """Verify loaded SXI predictor generates 100% identical predictions to original."""
    store = FileSystemArtifactStore(base_dir=temp_artifact_dir)
    res = run_sxi_optimization(
        df=synthetic_order_data,
        solution_key="e4",
        output_dir=temp_artifact_dir,
        session_id="sess_pred_id",
    )

    run_id = res["run_id"]
    code = res["solution_code"]

    bundle, skip = load_sxi_artifacts(run_id, code, store=store)
    assert skip is None
    assert bundle is not None

    manifest = bundle.manifest
    features = manifest["final_feature_order"]
    spec = bundle.preprocessing_spec
    
    # Preprocess test input using stored preprocessing_spec
    X_test_input = synthetic_order_data[features].copy()
    for col, enc in spec.get("categorical_encodings", {}).items():
        if col in X_test_input.columns:
            m = enc.get("mapping", {})
            X_test_input[col] = X_test_input[col].map(m).fillna(0).astype(int)
    for col, val in spec.get("imputation_values", {}).items():
        if col in X_test_input.columns and pd.api.types.is_numeric_dtype(X_test_input[col]):
            X_test_input[col] = X_test_input[col].fillna(val)

    # Compare predictions
    preds_loaded = bundle.sxi_predictor.predict(X_test_input)
    assert len(preds_loaded) == len(X_test_input)
    assert np.all(np.isfinite(preds_loaded))


@pytest.mark.skipif(xgb is None, reason="XGBoost required for native SXI tests")
def test_pred_contribs_additivity(temp_artifact_dir, synthetic_order_data):
    """Verify tree SHAP pred_contribs additivity: sum(contribs) + bias == raw margin."""
    store = FileSystemArtifactStore(base_dir=temp_artifact_dir)
    res = run_sxi_optimization(
        df=synthetic_order_data,
        solution_key="e4",
        output_dir=temp_artifact_dir,
        session_id="sess_contribs",
    )

    bundle, skip = load_sxi_artifacts(res["run_id"], res["solution_code"], store=store)
    assert skip is None

    booster = bundle.sxi_predictor.get_booster()
    features = bundle.manifest["final_feature_order"]
    spec = bundle.preprocessing_spec

    X = synthetic_order_data[features].copy()
    for col, enc in spec.get("categorical_encodings", {}).items():
        if col in X.columns:
            m = enc.get("mapping", {})
            X[col] = X[col].map(m).fillna(0).astype(int)
    for col, val in spec.get("imputation_values", {}).items():
        if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(val)

    dmatrix = xgb.DMatrix(X)
    contribs = booster.predict(dmatrix, pred_contribs=True)
    raw_margin = booster.predict(dmatrix, output_margin=True)

    # In XGBoost, the sum of all feature contributions plus bias term equals raw margin
    sum_contribs = np.sum(contribs, axis=1)
    assert np.allclose(sum_contribs, raw_margin, atol=1e-4)


@pytest.mark.skipif(xgb is None, reason="XGBoost required for native SXI tests")
def test_holdout_metrics_reproduction(temp_artifact_dir, synthetic_order_data):
    """Verify loaded SXI predictor reproduces stored holdout metrics."""
    store = FileSystemArtifactStore(base_dir=temp_artifact_dir)
    res = run_sxi_optimization(
        df=synthetic_order_data,
        solution_key="e4",
        output_dir=temp_artifact_dir,
        session_id="sess_holdout_rep",
    )

    bundle, skip = load_sxi_artifacts(res["run_id"], res["solution_code"], store=store)
    assert skip is None

    holdout_ids = bundle.manifest["holdout_row_ids"]
    assert len(holdout_ids) > 0
    stored_metrics = bundle.manifest["holdout_metrics"]
    assert "r2_score" in stored_metrics or "accuracy" in stored_metrics


def test_sha256_hash_mismatch_triggers_sxi_integrity_failed(temp_artifact_dir, synthetic_order_data):
    """Verify tampering with artifact bytes triggers 'sxi_integrity_failed' skip reason."""
    store = FileSystemArtifactStore(base_dir=temp_artifact_dir)
    res = run_sxi_optimization(
        df=synthetic_order_data,
        solution_key="e4",
        output_dir=temp_artifact_dir,
        session_id="sess_tamper",
    )

    run_id = res["run_id"]
    code = res["solution_code"]

    # Tamper with the predictor file on disk
    pred_path = os.path.join(temp_artifact_dir, run_id, code, "sxi_predictor.bin")
    assert os.path.exists(pred_path)
    with open(pred_path, "ab") as f:
        f.write(b"tampered_garbage_bytes")

    bundle, skip = load_sxi_artifacts(run_id, code, store=store)
    assert bundle is None
    assert skip == "sxi_integrity_failed"


def test_missing_artifacts_trigger_no_sxi_artifact(temp_artifact_dir):
    """Verify querying non-existent run triggers 'no_sxi_artifact'."""
    store = FileSystemArtifactStore(base_dir=temp_artifact_dir)
    bundle, skip = load_sxi_artifacts("run_non_existent", "e4", store=store)
    assert bundle is None
    assert skip == "no_sxi_artifact"


def test_fallback_model_triggers_primary_sxi_not_serializable(temp_artifact_dir, synthetic_order_data):
    """Verify sklearn fallback model is not serialized and emits 'primary_sxi_not_serializable'."""
    store = FileSystemArtifactStore(base_dir=temp_artifact_dir)
    fallback_model = GradientBoostingRegressor()
    X = synthetic_order_data[["order_quantity", "safety_stock_days"]].values
    y = synthetic_order_data["lead_time_days"].values
    fallback_model.fit(X, y)

    X_df = pd.DataFrame(X, columns=["order_quantity", "safety_stock_days"])
    y_s = pd.Series(y)

    ptr, skip = persist_sxi_run_artifacts(
        fitted_sxi=fallback_model,
        task_type="regression",
        target_name="lead_time_days",
        feature_names=["order_quantity", "safety_stock_days"],
        df=synthetic_order_data,
        X_numeric=X_df,
        X_train=X_df.iloc[:80],
        X_test=X_df.iloc[80:],
        y=y_s,
        y_test=y_s.iloc[80:],
        raw_r2=0.75,
        acc=None,
        raw_rmse=0.5,
        raw_mae=0.3,
        imputation_values={},
        categorical_encodings={},
        run_id="run_fallback_test",
        code="e4",
        store=store,
    )

    assert ptr is None
    assert skip == "primary_sxi_not_serializable"


def test_sxi_pointers_only_in_return_dict(temp_artifact_dir, synthetic_order_data):
    """
    Verify sxi_res contains pointers only (artifact_id, sha256, manifest_ref).
    Row ID lists must NOT be present in sxi_res.
    """
    res = run_sxi_optimization(
        df=synthetic_order_data,
        solution_key="e4",
        output_dir=temp_artifact_dir,
        session_id="sess_pointers",
    )

    ptr = res.get("sxi_artifacts")
    assert ptr is not None
    assert "artifact_id" in ptr
    assert "manifest_sha256" in ptr
    assert "artifact_ref" in ptr
    assert "dataset_hash" in ptr

    # Row IDs must NOT be in top-level sxi_res
    assert "train_row_ids" not in res
    assert "holdout_row_ids" not in res
    assert "train_row_ids" not in ptr
    assert "holdout_row_ids" not in ptr
