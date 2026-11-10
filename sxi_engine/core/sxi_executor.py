# agent2/utils/sxi_executor.py
import os
import base64
import json
import random
import re
import html
import time
import sys
from datetime import datetime
from urllib import request
from xmlrpc import client
import pandas as pd
import numpy as np
from pathlib import Path
from django.conf import settings

# Import your SXI classes
from agent2.sxi_exe import (
    model_execution,
    SxiProcess,
    sxirl_engine,
    format_display_value,
    normalize_metric_score,
)
from agent2.utils.json_utils import make_json_safe
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, PageBreak, Image
)
from reportlab.platypus import ListFlowable, ListItem, Paragraph
from pipeline.models import UploadedFile
from agent2.utils.path_utils import resolve_media_path
from agent2.utils.llm_provider import (
    DEFAULT_LLM_PROVIDER,
    LLMProviderError,
    build_credit_exhausted_message,
    finalize_llm_usage_tracking,
    print_llm_usage_summary,
    is_credit_exhausted_error,
    request_llm,
    reset_llm_usage_tracking,
    resolve_llm_provider,
)


def _ensure_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_ensure_utf8_streams()

class SXIExecutor:
    """
    Wrapper class to execute SXI analysis pipeline
    """
    LLM_PROVIDER = DEFAULT_LLM_PROVIDER  # Shared default: "anthropic", "gemini", "groq", or "openai"
    TARGET_MISSING_ERROR_THRESHOLD = 0.5
    MIN_VALID_TARGET_ROWS = 10
    
    def __init__(self, request, dataframe_path, target_info):
        """
        Args:
            request: Django request object
            dataframe_path: Path to uploaded CSV/Excel file
            target_info: Dict with target configuration
        """
        self.request = request
        self.dataframe_path = dataframe_path
        self.target_info = target_info
        self.buyerid = request.session.get('run_id') or request.session.get('buyerid') or getattr(request.user, "id", None) or 1
        
        # Create output directory
        self.output_dir = Path(settings.MEDIA_ROOT) / 'files' / 'chatbot' / str(self.buyerid)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._last_llm_credit_exhausted = False
        self.encoded_feature_mapping = {}
        self.grouped_encoded_features = {}
        self.master_encoding_validation = {}
        self._encoded_schema_columns = []

    def check_cancelled(self):
        if not self.request or not hasattr(self.request, 'session') or not self.request.session:
            return
        session_key = self.request.session.session_key
        if not session_key or len(session_key) < 20:
            return
        try:
            # Use the configured session backend (file/cached/db). Never assume
            # django_session DB rows — local mode uses file sessions, so a Neon
            # Session.objects lookup would false-cancel every SXI run.
            from importlib import import_module
            from django.conf import settings

            engine = import_module(settings.SESSION_ENGINE)
            store = engine.SessionStore(session_key)
            if not store.exists(session_key):
                print(
                    f"[SXI CANCEL] Session {session_key} no longer exists "
                    f"({settings.SESSION_ENGINE}). Terminating execution."
                )
                raise Exception("SXI execution cancelled: session reset")
        except Exception as e:
            if str(e) == "SXI execution cancelled: session reset":
                raise e
            pass

    def _reset_llm_status(self):
        self._last_llm_credit_exhausted = False

    def _is_credit_exhausted_error(self, error_or_text) -> bool:
        return is_credit_exhausted_error(error_or_text)

    def _record_llm_failure(self, error_or_text):
        if self._is_credit_exhausted_error(error_or_text):
            self._last_llm_credit_exhausted = True

    def _llm_out_of_credits(self) -> bool:
        return bool(getattr(self, "_last_llm_credit_exhausted", False))

    def _out_of_credits_text(self) -> str:
        provider = resolve_llm_provider(request=self.request, default=self.LLM_PROVIDER)
        return build_credit_exhausted_message(provider)

    def _update_report_progress(self, **updates):
        self.check_cancelled()
        report_status = dict(self.request.session.get("agent2_report_status", {}) or {})
        report_status.setdefault("status", "idle")
        report_status.setdefault("full_report_path", "")
        report_status.setdefault("summary_report_path", "")
        report_status.setdefault("error", "")

        for key, value in updates.items():
            if value is not None:
                report_status[key] = value

        report_status["updated_at"] = datetime.utcnow().isoformat()
        self.request.session["agent2_report_status"] = report_status
        self.request.session.modified = True
        try:
            from pipeline.session_utils import safe_session_save

            if not safe_session_save(self.request.session, label="sxi_report_progress"):
                # File/DB blip — do not cancel a long SXI run for a progress write.
                print("[SXI] report progress session save soft-failed; continuing.")
        except Exception as e:
            err_str = str(e)
            if "Forced update did not affect any rows" in err_str or "UpdateError" in err_str or "SessionInterrupted" in type(e).__name__:
                print("[SXI] Session save race during progress update; continuing.")
            pass
    
    def load_dataframe(self):
        """Load the uploaded dataset"""
        df = self._read_tabular_file(self.dataframe_path)
        self._normalize_target_info_from_dataframe(df)
        df = self._sanitize_numeric_target_column(df)
        df = self._sanitize_categorical_target_column(df)
        self._derive_optimization_target_class()
        self._hydrate_master_encoding_metadata()
        return df

    def _target_column_name(self):
        if not isinstance(self.target_info, dict):
            return None
        return self.target_info.get("Target Outcome")

    def _dataset_quality_error_message(self, df: pd.DataFrame):
        target_col = self._target_column_name()
        if not target_col:
            return "Dataset error: Target column is missing from the configuration. Please select a target column and run again."

        if target_col not in df.columns:
            return (
                f"Dataset error: Target column '{target_col}' was not found in the dataset. "
                "Please choose an existing column as the target and run again."
            )

        total_rows = len(df)
        if total_rows == 0:
            return "Dataset error: The dataset has no rows after preprocessing. Please upload a dataset with valid records."

        target_type = str(self.target_info.get("Target Outcome Type", "")).strip().lower()
        target_series = df[target_col]
        if target_type in {"numeric", "continuous"}:
            target_series = self._coerce_numeric_like_series(target_series)

        missing_count = int(target_series.isna().sum())
        valid_count = int(target_series.notna().sum())
        missing_pct = (missing_count / total_rows) if total_rows else 1

        if valid_count == 0:
            return (
                f"Dataset error: Target column '{target_col}' has no usable values. "
                "Please clean or select a target column with enough non-missing values."
            )

        if valid_count < self.MIN_VALID_TARGET_ROWS:
            return (
                f"Dataset error: Target column '{target_col}' has only {valid_count} usable rows. "
                f"DXI needs at least {self.MIN_VALID_TARGET_ROWS} valid target values to run reliably."
            )

        if missing_pct >= self.TARGET_MISSING_ERROR_THRESHOLD:
            return (
                f"Dataset error: Target column '{target_col}' has too many missing values "
                f"({missing_count} of {total_rows} rows, {missing_pct:.0%}). "
                "Please clean the dataset, fill the missing target values, or choose a better target column."
            )

        target_type = str(self.target_info.get("Target Outcome Type", "")).strip().lower()
        task_type = str(self.target_info.get("Task Type", "")).strip().lower()
        if task_type == "classification" or target_type == "categorical":
            class_counts = target_series.dropna().value_counts()
            if len(class_counts) < 2:
                return (
                    f"Dataset error: Target column '{target_col}' has only one usable class after preprocessing. "
                    "SXI classification needs at least two target classes."
                )

        return None

    def _problem_contract_quality_warning(self, df: pd.DataFrame) -> str | None:
        """Disclose P2/P4 proxy targets. Does not change SXI train/test split."""
        if not isinstance(self.target_info, dict):
            return None
        try:
            pid = int(self.target_info.get("Problem Id") or 0)
        except (TypeError, ValueError):
            pid = 0
        if pid not in (2, 4):
            return None
        source = str(self.target_info.get("Target Source") or "").strip().lower()
        quality = self.target_info.get("Data Quality") if isinstance(self.target_info.get("Data Quality"), dict) else {}
        if not quality and self.request and hasattr(self.request, "session"):
            quality = self.request.session.get("data_quality") or {}
            source = source or str(self.request.session.get("target_source") or "")
        has_history = bool(quality.get("has_repeat_history"))
        if source == "longitudinal" and has_history:
            return None
        if pid == 2:
            return (
                "Problem 2 target is a disclosed proxy (no true repeat-purchase history). "
                "is_repeat_buyer is not a longitudinal second-order label on this upload."
            )
        return (
            "Problem 4 target is a disclosed proxy (no uncensored next-purchase dates). "
            "days_until_next_purchase is not a true inter-purchase interval on this upload."
        )

    def _looks_like_dataset_quality_error(self, error_or_text) -> bool:
        text = str(error_or_text or "").lower()
        markers = [
            "dataset error",
            "missing",
            "nan",
            "inf",
            "infinity",
            "0 sample",
            "0 row",
            "empty",
            "at least one sample",
            "contains nan",
            "input y",
            "could not convert",
        ]
        return any(marker in text for marker in markers)

    def _friendly_dataset_error_message(self, fallback_error=None, df=None):
        if isinstance(df, pd.DataFrame):
            message = self._dataset_quality_error_message(df)
            if message:
                return message

        target_col = self._target_column_name() or "the selected target column"
        return (
            f"Dataset error: SXI could not run because the dataset or target column '{target_col}' "
            "contains too many missing or unusable values. Please clean the dataset, fill missing values, "
            "or choose a target column with enough valid rows."
        )

    def _hydrate_master_encoding_metadata(self):
        if self.encoded_feature_mapping or self.grouped_encoded_features:
            return

        csv_master_info = self.request.session.get("csv_master_info") or {}
        summary = csv_master_info.get("summary") or {}

        self.encoded_feature_mapping = (
            csv_master_info.get("encoded_feature_mapping")
            or summary.get("encoded_feature_mapping")
            or {}
        )
        self.grouped_encoded_features = (
            csv_master_info.get("grouped_encoded_features")
            or summary.get("grouped_encoded_features")
            or {}
        )
        self.master_encoding_validation = (
            csv_master_info.get("validation")
            or summary.get("validation")
            or {}
        )
        if self.encoded_feature_mapping or self.grouped_encoded_features:
            print(
                "Loaded master encoding metadata from session:",
                len(self.encoded_feature_mapping),
                "encoded feature mappings",
            )
            return

        metadata_url = csv_master_info.get("encoding_metadata_json")
        if not metadata_url:
            return

        try:
            metadata_path = resolve_media_path(metadata_url)
            if not os.path.exists(metadata_path):
                return

            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)

            self.encoded_feature_mapping = metadata.get("encoded_feature_mapping") or {}
            self.grouped_encoded_features = metadata.get("grouped_encoded_features") or {}
            self.master_encoding_validation = metadata.get("validation") or {}
            print(
                "Loaded master encoding metadata:",
                len(self.encoded_feature_mapping),
                "encoded feature mappings",
            )
        except Exception as exc:
            print(f"Failed to load master encoding metadata: {exc}")

    def _read_tabular_file(self, file_path: str) -> pd.DataFrame:
        extension = os.path.splitext(str(file_path))[1].lower()

        if extension == ".csv":
            encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
            for encoding in encodings_to_try:
                try:
                    return pd.read_csv(file_path, low_memory=False, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            return pd.read_csv(file_path, low_memory=False, encoding_errors="ignore")

        if extension in (".xlsx", ".xls"):
            return pd.read_excel(file_path)

        if extension == ".json":
            return pd.read_json(file_path)

        raise ValueError(f"Unsupported file format: {file_path}")

    def _coerce_numeric_like_series(self, series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")

        cleaned = (
            series.astype(str)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA})
            .str.replace(",", "", regex=False)
            .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
        )
        return pd.to_numeric(cleaned, errors="coerce")

    def _normalize_target_info_from_dataframe(self, df: pd.DataFrame):
        if not isinstance(self.target_info, dict) or not isinstance(df, pd.DataFrame):
            return

        target_col = self.target_info.get("Target Outcome")
        if not target_col or target_col not in df.columns:
            return

        self.target_info = dict(self.target_info)
        if self.target_info.get("Good Outcome Label") and not self.target_info.get("Good Outcome"):
            self.target_info["Good Outcome"] = self.target_info["Good Outcome Label"]
        if self.target_info.get("Bad Outcome Label") and not self.target_info.get("Bad Outcome"):
            self.target_info["Bad Outcome"] = self.target_info["Bad Outcome Label"]

        existing_type = str(self.target_info.get("Target Outcome Type") or "").strip()
        if existing_type:
            return

        target_series = df[target_col].dropna()
        unique_count = target_series.nunique()
        is_timeseries = self.target_info.get("is_timeseries", False)
        if isinstance(is_timeseries, str):
            is_timeseries = is_timeseries.strip().lower() == "true"

        if is_timeseries:
            return

        task_type = str(self.target_info.get("Task Type") or "").strip().lower()
        selected_outcome_text = str(self.target_info.get("Selected Outcome") or "").strip().lower()
        good_value_text = str(self.target_info.get("Good Outcome Value") or "").strip().lower()
        bad_value_text = str(self.target_info.get("Bad Outcome Value") or "").strip().lower()
        has_regression_range_config = any(
            value in {"above_mean", "above mean", "below_mean", "below mean"}
            for value in (selected_outcome_text, good_value_text, bad_value_text)
        )
        if (
            not is_timeseries
            and (task_type == "regression" or (has_regression_range_config and pd.api.types.is_numeric_dtype(target_series) and unique_count > 2))
        ):
            self.target_info["Target Outcome Type"] = "Numeric"
            self.target_info["Task Type"] = "regression"
            print(f"Normalized missing Target Outcome Type to Numeric for range target '{target_col}'")
            return

        has_classification_config = any(
            self.target_info.get(key) not in (None, "")
            for key in (
                "Good Outcome",
                "Bad Outcome",
                "Good Outcome Value",
                "Bad Outcome Value",
                "Selected Outcome",
            )
        )
        if task_type == "classification" or (task_type != "regression" and (has_classification_config or unique_count <= 2)):
            self.target_info["Target Outcome Type"] = "Categorical"
            self.target_info["Task Type"] = "classification"
            print(f"Normalized missing Target Outcome Type to Categorical for '{target_col}'")
            return

        if pd.api.types.is_numeric_dtype(target_series) and (unique_count > 2 or task_type == "regression"):
            self.target_info["Target Outcome Type"] = "Numeric"
            self.target_info["Task Type"] = "regression"
            print(f"Normalized missing Target Outcome Type to Numeric for '{target_col}'")

    def _sanitize_numeric_target_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(self.target_info, dict):
            return df

        target_type = str(self.target_info.get("Target Outcome Type", "")).strip().lower()
        if target_type not in {"numeric", "continuous"}:
            return df

        target_col = self.target_info.get("Target Outcome")
        if not target_col or target_col not in df.columns:
            return df

        coerced = self._coerce_numeric_like_series(df[target_col])
        non_null_original = df[target_col].notna().sum()
        valid_ratio = (coerced.notna().sum() / non_null_original) if non_null_original else 0
        if valid_ratio < 0.8:
            return df

        df = df.copy()
        df[target_col] = coerced
        print(f"Sanitized numeric-like target '{target_col}' for SXI execution (valid ratio {valid_ratio:.0%})")
        return df

    def _normalize_binary_code(self, value):
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (int, float)) and not pd.isna(value):
            return int(float(value))

        text = str(value).strip()
        if text == "":
            return None

        try:
            return int(float(text))
        except (TypeError, ValueError):
            return None

    def _normalize_target_value(self, value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        try:
            numeric = float(value)
            if numeric.is_integer():
                return int(numeric)
            return numeric
        except (TypeError, ValueError):
            return str(value).strip()

    def _same_target_value(self, left, right):
        if left is None or right is None:
            return False
        left_norm = self._normalize_target_value(left)
        right_norm = self._normalize_target_value(right)
        if left_norm == right_norm:
            return True
        return str(left).strip().lower() == str(right).strip().lower()

    def _target_change_from_info(self):
        explicit = str(
            self.target_info.get("Target Outcome change")
            or self.target_info.get("Target Outcome Change")
            or ""
        ).strip().lower()
        if explicit in {"decrease", "decreasing", "reduce", "reduction", "reduced"}:
            return "decreasing"
        if explicit in {"increase", "increasing", "raise", "increased"}:
            return "increasing"
        try:
            return "decreasing" if float(self.target_info.get("Target Outcome Improvement", 0)) < 0 else "increasing"
        except (TypeError, ValueError):
            return "increasing"

    def _derive_optimization_target_class(self):
        if not isinstance(self.target_info, dict):
            return None

        self.target_info = dict(self.target_info)
        selected_outcome = self.target_info.get("Selected Outcome")
        selected_meaning = str(self.target_info.get("Selected Outcome Meaning") or "").strip().lower()
        good_value = self.target_info.get("Good Outcome Value")
        bad_value = self.target_info.get("Bad Outcome Value")
        good_label = self.target_info.get("Good Outcome Label") or self.target_info.get("Good Outcome")
        bad_label = self.target_info.get("Bad Outcome Label") or self.target_info.get("Bad Outcome")

        selected_is_bad = selected_meaning in {"no", "bad", "negative", "fraud", "risk", "failure", "defect"}
        selected_is_good = selected_meaning in {"yes", "good", "positive", "success"}

        if self._same_target_value(selected_outcome, bad_value) or self._same_target_value(selected_outcome, bad_label):
            optimization_target_class = bad_value
            selected_is_bad = True
        elif self._same_target_value(selected_outcome, good_value) or self._same_target_value(selected_outcome, good_label):
            optimization_target_class = good_value
            selected_is_good = True
        else:
            normalized_selected = self._normalize_target_value(selected_outcome)
            if normalized_selected is not None:
                optimization_target_class = normalized_selected
            elif selected_is_bad:
                optimization_target_class = bad_value
            else:
                optimization_target_class = good_value

        target_polarity = "negative_outcome" if selected_is_bad else "positive_outcome" if selected_is_good else ""
        self.target_info["Optimization Target Class"] = optimization_target_class
        self.target_info["optimization_target_class"] = optimization_target_class
        self.target_info["Selected Outcome Is Bad"] = selected_is_bad
        self.target_info["Selected Outcome Is Good"] = selected_is_good
        if target_polarity:
            self.target_info["target_polarity"] = target_polarity

        target_change = self._target_change_from_info()
        print({
            "selected_outcome_ui": selected_outcome,
            "selected_outcome_backend": optimization_target_class,
            "selected_outcome_meaning": selected_meaning,
            "good_outcome": good_value,
            "bad_outcome": bad_value,
            "target_change": target_change,
            "optimization_target_class": optimization_target_class,
            "slope": None,
            "current_sxi": None,
            "target_sxi": None,
            "correlation_direction": None,
            "iteration_selected": self.target_info.get("iteration_selected") or self.request.session.get("selected_smote_candidate"),
        })
        print({
            "regression_target_column": self.target_info.get("Target Outcome"),
            "target_mean": None,
            "above_mean_meaning": (
                "good" if self._same_target_value(good_value, "Above_mean") else
                "bad" if self._same_target_value(bad_value, "Above_mean") else ""
            ),
            "below_mean_meaning": (
                "good" if self._same_target_value(good_value, "Below_mean") else
                "bad" if self._same_target_value(bad_value, "Below_mean") else ""
            ),
            "optimization_target_class": optimization_target_class,
            "optimization_direction": target_change,
            "slope": None,
            "current_sxi": None,
            "target_sxi": None,
        })
        return optimization_target_class

    def _sanitize_categorical_target_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(self.target_info, dict):
            return df

        target_type = str(self.target_info.get("Target Outcome Type", "")).strip().lower()
        if target_type != "categorical":
            return df

        target_col = self.target_info.get("Target Outcome")
        if not target_col or target_col not in df.columns:
            return df

        good_raw = self.target_info.get("Good Outcome Value")
        bad_raw = self.target_info.get("Bad Outcome Value")
        good_code = self._normalize_binary_code(good_raw)
        bad_code = self._normalize_binary_code(bad_raw)
        good_label = (
            self.target_info.get("Good Outcome Label")
            or self.target_info.get("Good Outcome")
        )
        bad_label = (
            self.target_info.get("Bad Outcome Label")
            or self.target_info.get("Bad Outcome")
        )

        numeric_series = df[target_col].dropna().map(self._normalize_binary_code)
        numeric_ratio = numeric_series.notna().mean() if len(numeric_series) else 0

        if good_code is not None and bad_code is not None and numeric_ratio >= 0.9:
            self.target_info = dict(self.target_info)
            self.target_info["Good Outcome Value"] = good_code
            self.target_info["Bad Outcome Value"] = bad_code
            return df

        def normalize_label(value):
            if value is None or pd.isna(value):
                return None
            return str(value).strip()

        good_label = normalize_label(
            self.target_info.get("Good Outcome Label", self.target_info.get("Good Outcome", good_raw))
        )
        bad_label = normalize_label(
            self.target_info.get("Bad Outcome Label", self.target_info.get("Bad Outcome", bad_raw))
        )
        if not good_label or not bad_label or good_label == bad_label:
            return df

        good_code = good_code if good_code is not None else 1
        bad_code = bad_code if bad_code is not None else 0
        raw_to_code = {
            good_label: good_code,
            bad_label: bad_code,
            normalize_label(good_raw): good_code,
            normalize_label(bad_raw): bad_code,
            str(good_code): good_code,
            str(bad_code): bad_code,
            str(float(good_code)) if isinstance(good_code, (int, float)) else str(good_code): good_code,
            str(float(bad_code)) if isinstance(bad_code, (int, float)) else str(bad_code): bad_code,
        }

        encoded = df[target_col].map(lambda value: raw_to_code.get(normalize_label(value), np.nan))
        unmapped_mask = df[target_col].notna() & encoded.isna()
        if unmapped_mask.any():
            unresolved = sorted({str(value) for value in df.loc[unmapped_mask, target_col].dropna().unique()})
            raise ValueError(
                f"Could not map categorical target values in '{target_col}' to SXI classes. "
                f"Expected {good_raw!r} and {bad_raw!r}, but found unmapped values: {unresolved[:5]}"
            )

        df = df.copy()
        df[target_col] = encoded.astype("Int64")
        self.target_info = dict(self.target_info)
        self.target_info["Good Outcome Raw Value"] = good_raw
        self.target_info["Bad Outcome Raw Value"] = bad_raw
        self.target_info["Good Outcome Value"] = good_code
        self.target_info["Bad Outcome Value"] = bad_code
        print(
            f"Sanitized categorical target '{target_col}' for SXI execution "
            f"with mapping: {good_code} -> {good_raw!r}, {bad_code} -> {bad_raw!r}"
        )
        return df

    def _resolve_local_path(self, file_path):
        if not file_path:
            return None

        candidate_paths = [str(file_path)]
        if not os.path.isabs(str(file_path)):
            base_dir = getattr(settings, "BASE_DIR", None)
            if base_dir:
                candidate_paths.append(os.path.join(str(base_dir), str(file_path)))
            candidate_paths.append(os.path.join(os.getcwd(), str(file_path)))

        for candidate in candidate_paths:
            normalized = os.path.normpath(candidate)
            if os.path.exists(normalized):
                return os.path.abspath(normalized)

        return None

    def _report_assets_dir(self) -> str:
        base_dir = getattr(settings, "BASE_DIR", None)
        if base_dir:
            return os.path.join(str(base_dir), "static", "images")
        return os.path.join(os.getcwd(), "static", "images")

    def _ensure_report_assets(self) -> str:
        """Create default PDF branding assets when missing from the repo."""
        asset_dir = self._report_assets_dir()
        os.makedirs(asset_dir, exist_ok=True)

        blue_path = os.path.join(asset_dir, "blue.png")
        logo_path = os.path.join(asset_dir, "Sriya_new_logo.png")

        try:
            from PIL import Image, ImageDraw

            if not os.path.exists(blue_path):
                Image.new("RGB", (1200, 200), color=(31, 115, 183)).save(blue_path)

            if not os.path.exists(logo_path):
                logo = Image.new("RGBA", (300, 80), color=(255, 255, 255, 0))
                draw = ImageDraw.Draw(logo)
                draw.text((8, 22), "Sriya.AI", fill=(255, 255, 255, 255))
                logo.save(logo_path)
        except Exception as exc:
            print(f"Warning: could not create default report assets: {exc}")

        return asset_dir

    def _get_report_asset(self, filename: str):
        self._ensure_report_assets()
        resolved = self._resolve_local_path(os.path.join("static", "images", filename))
        if resolved:
            return resolved
        fallback = os.path.join(self._report_assets_dir(), filename)
        return fallback if os.path.exists(fallback) else None

    def _draw_pdf_header_banner(self, canvas, path, x, y, width, height):
        if path and os.path.exists(str(path)):
            canvas.drawImage(str(path), x, y, width=width, height=height, mask='auto')
            return
        canvas.setFillColorRGB(0.12, 0.45, 0.72)
        canvas.rect(x, y, width, height, stroke=0, fill=1)

    def _draw_pdf_logo(self, canvas, path, x, y, width, height):
        if path and os.path.exists(str(path)):
            canvas.drawImage(
                str(path),
                x,
                y,
                width=width,
                height=height,
                preserveAspectRatio=True,
                mask='auto',
            )
            return
        canvas.setFillColorRGB(1, 1, 1)
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(x + 4, y + (height / 2) - 4, "Sriya.AI")

    def _encode_image_for_llm(self, image_path):
        resolved_path = self._resolve_local_path(image_path)
        if not resolved_path:
            return None

        extension = os.path.splitext(resolved_path)[1].lower()
        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(extension)

        if not mime_type:
            return None

        try:
            with open(resolved_path, "rb") as image_file:
                return {
                    "mime_type": mime_type,
                    "data": base64.b64encode(image_file.read()).decode("utf-8"),
                    "path": resolved_path,
                }
        except Exception as exc:
            print(f"Warning: failed to encode image for LLM: {exc}")
            return None

    def _claude_request(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024, image_path=None) -> str:
        """
        Priority fallback LLM caller:
        1. Anthropic Claude  — uses ANTHROPIC_API_KEY from settings.py
        2. Google Gemini     — uses GOOGLE_API_KEY or GEMINI_API_KEY from settings.py
        3. Groq fallback     — uses llm_engine._groq_request
        Always returns a string — never raises, so PDF generation never breaks.
        """
        provider = resolve_llm_provider(request=self.request, default=self.LLM_PROVIDER)
        self._reset_llm_status()
        return self._llm_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            image_path=image_path,
        )
        """
        try:
            response_text = request_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                request=self.request,
                provider=provider,
                max_tokens=max_tokens,
                image_path=image_path,
            )
            print(f"LLM provider: {provider}")
            return response_text
        except LLMProviderError as exc:
            self._record_llm_failure(exc)
            print(f"{provider} failed: {exc}")
            return ""
        full_prompt = f"{system_prompt}\n\n{user_prompt}".strip()
        encoded_image = self._encode_image_for_llm(image_path)

        # ── Priority 1: Anthropic Claude ─────────────────────────────────────
        try:
            _ant_key = getattr(settings, "ANTHROPIC_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
            if _ant_key:
                import anthropic
                _client = anthropic.Anthropic(
                    api_key=_ant_key,
                    timeout=20.0,
                    max_retries=1,
                )
                user_content = []
                if encoded_image:
                    user_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": encoded_image["mime_type"],
                            "data": encoded_image["data"],
                        },
                    })
                user_content.append({
                    "type": "text",
                    "text": user_prompt,
                })

                _resp = _client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                    timeout=20.0,
                )
                print("✅ LLM: Anthropic Claude used")
                response_text = _resp.content[0].text.strip() # type: ignore
                if self._is_credit_exhausted_error(response_text):
                    self._last_llm_credit_exhausted = True
                    return ""
                return response_text
        except Exception as _e:
            self._record_llm_failure(_e)
            print(f"⚠️  Anthropic failed: {_e}")

        # ── Priority 2: Google Gemini ─────────────────────────────────────────
        try:
            _gem_key = (getattr(settings, "GOOGLE_API_KEY",  "") or
                        getattr(settings, "GEMINI_API_KEY",  "") or
                        os.environ.get("GOOGLE_API_KEY",  "") or
                        os.environ.get("GEMINI_API_KEY",  ""))
            if _gem_key:
                import google.generativeai as genai
                genai.configure(api_key=_gem_key) # type: ignore
                _model = genai.GenerativeModel("gemini-1.5-flash") # type: ignore
                prompt_parts = [full_prompt]
                if encoded_image:
                    prompt_parts.append({
                        "mime_type": encoded_image["mime_type"],
                        "data": encoded_image["data"],
                    })
                _resp  = _model.generate_content(prompt_parts)
                print("✅ LLM: Google Gemini used")
                response_text = _resp.text.strip()
                if self._is_credit_exhausted_error(response_text):
                    self._last_llm_credit_exhausted = True
                    return ""
                return response_text
        except Exception as _e:
            self._record_llm_failure(_e)
            print(f"⚠️  Gemini failed: {_e}")

        # ── Priority 3: Groq via llm_engine ──────────────────────────────────
        try:
            from agent2.utils.llm_engine import _groq_request
            _resp = _groq_request(full_prompt, max_tokens=max_tokens)
            print("✅ LLM: Groq used")
            response_text = _resp.strip() if isinstance(_resp, str) else str(_resp)
            if self._is_credit_exhausted_error(response_text):
                self._last_llm_credit_exhausted = True
                return ""
            return response_text
        except Exception as _e:
            self._record_llm_failure(_e)
            print(f"⚠️  Groq failed: {_e}")

        # ── All failed — return empty string so PDF still generates ──────────
        print("❌ All LLM providers failed — returning empty string")
        return ""
        """

    def _llm_request(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024, image_path=None) -> str:
        """
        Single-provider LLM caller controlled by SXIExecutor.LLM_PROVIDER.
        """
        self.check_cancelled()
        self._reset_llm_status()
        provider = resolve_llm_provider(
            request=self.request,
            provider=getattr(self, "LLM_PROVIDER", None),
            default=self.LLM_PROVIDER,
        )

        try:
            response_text = request_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                request=self.request,
                provider=provider,
                max_tokens=max_tokens,
                image_path=image_path,
            )
            print(f"LLM provider: {provider}")
            return response_text
        except LLMProviderError as exc:
            self._record_llm_failure(exc)
            print(f"{provider.title()} failed: {exc}")
            return ""

    def _save_text_artifact(self, filename: str, content: str) -> str:
        file_path = self.output_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content or "")
        return str(file_path).replace("\\", "/")

    def _sanitize_reportlab_markup(self, text) -> str:
        """
        Sanitize LLM text for reportlab.platypus.Paragraph.
        Keeps a small safe tag set and guarantees balanced tags.
        """
        if text is None:
            return ""

        raw = str(text).replace("\r\n", "\n")
        parts = re.split(r"(<[^>]+>)", raw)
        out = []
        stack = []

        def close_to(tag_name: str):
            while stack and stack[-1] != tag_name:
                out.append(f"</{stack.pop()}>")
            if stack and stack[-1] == tag_name:
                out.append(f"</{stack.pop()}>")

        for part in parts:
            if not part:
                continue

            if not (part.startswith("<") and part.endswith(">")):
                out.append(html.escape(part, quote=False))
                continue

            token = part[1:-1].strip()
            token_lower = token.lower()

            if token_lower in {"br", "br/", "br /"}:
                out.append("<br/>")
                continue

            if token_lower in {"para", "/para", "p", "/p", "ul", "/ul", "li", "/li", "div", "/div", "span", "/span"}:
                continue

            if token_lower in {"b", "i", "u"}:
                out.append(f"<{token_lower}>")
                stack.append(token_lower)
                continue

            if token_lower in {"/b", "/i", "/u"}:
                close_to(token_lower[1:])
                continue

            if token_lower.startswith("font"):
                match = re.search(r'color\s*=\s*["\']?([^"\'>\s]+)', token, flags=re.IGNORECASE)
                color = match.group(1) if match else "#000000"
                if not re.match(r"^#[0-9A-Fa-f]{3,8}$|^[A-Za-z]+$", color):
                    color = "#000000"
                out.append(f'<font color="{color}">')
                stack.append("font")
                continue

            if token_lower.startswith("/font"):
                close_to("font")
                continue

        while stack:
            out.append(f"</{stack.pop()}>")

        return "".join(out).strip()

    def _split_report_lines(self, text):
        """
        Split multiline report text into individual PDF-safe lines.
        """
        if text is None:
            return []

        if isinstance(text, (list, tuple, set)):
            raw_lines = []
            for item in text:
                raw_lines.extend(str(item).splitlines())
        else:
            raw_lines = str(text).splitlines()

        cleaned_lines = []
        for line in raw_lines:
            clean_line = self._sanitize_reportlab_markup(line).strip()
            if clean_line:
                cleaned_lines.append(clean_line)

        return cleaned_lines

    def _append_eda_content(self, elements, eda_text, styles):
        lines = self._split_report_lines(eda_text)
        header_style = ParagraphStyle(
            name="EdaTableHeader",
            parent=styles["SmallBulletStyle"],
            textColor=colors.whitesmoke,
            fontName="Helvetica-Bold",
        )
        index = 0
        while index < len(lines):
            line = lines[index]
            if line.startswith("|") and line.endswith("|"):
                table_lines = []
                while index < len(lines) and lines[index].startswith("|") and lines[index].endswith("|"):
                    table_lines.append(lines[index])
                    index += 1

                rows = []
                for row_index, table_line in enumerate(table_lines):
                    cells = [cell.strip() for cell in table_line.strip("|").split("|")]
                    if cells and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
                        continue
                    cell_style = header_style if not rows else styles["SmallBulletStyle"]
                    rows.append([Paragraph(cell, cell_style) for cell in cells])

                if rows:
                    table = Table(rows, colWidths=[125, 185, 185], repeatRows=1)
                    table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.midnightblue),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ]))
                    elements.append(table)
                    elements.append(Spacer(1, 6))
                continue

            elements.append(Paragraph(line, styles["SmallBulletStyle"]))
            elements.append(Spacer(1, 3))
            index += 1

    def _strip_plain_text(self, value) -> str:
        text = re.sub(r"<[^>]+>", " ", str(value or ""))
        text = html.unescape(text)
        text = re.sub(r"(?m)^\s*[•\-]\s*$", "", text)
        text = re.sub(r"(?:â€¢|Ã¢â‚¬Â¢|•)\s*(?:â€¢|Ã¢â‚¬Â¢|•)+", "-", text)
        return re.sub(r"\s+", " ", text).strip()

    def _decision_logic_report_rows(self, text):
        """
        Convert deterministic decision-logic HTML or legacy bullet text into
        simple rows for ReportLab rendering.
        """
        rows = []
        raw_text = str(text or "")
        if re.search(r"</?(?:div|h4|ul|li)\b", raw_text, flags=re.IGNORECASE):
            token_pattern = re.compile(
                r"<h4[^>]*>(.*?)</h4>|<div[^>]*>(.*?)</div>|<li[^>]*>(.*?)</li>",
                flags=re.IGNORECASE | re.DOTALL,
            )
            for match in token_pattern.finditer(raw_text):
                heading_html, div_html, li_html = match.groups()
                if heading_html is not None:
                    heading = self._strip_plain_text(heading_html)
                    if heading:
                        rows.append(("heading", heading))
                    continue

                if div_html is not None:
                    div_text = html.unescape(re.sub(r"<[^>]+>", "", div_html))
                    div_text = div_text.replace("&bull;", "-")
                    for raw_line in div_text.splitlines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        clean = self._strip_tree_bullet_prefix(line)
                        if line.lower().startswith("best path for class"):
                            rows.append(("path_heading", clean))
                        elif clean:
                            rows.append(("path_condition", clean))
                    continue

                if li_html is not None:
                    item = self._strip_plain_text(li_html)
                    if item:
                        rows.append(("bullet", item))

            return rows

        plain_lines = [line.strip() for line in raw_text.splitlines()]
        nonempty_lines = [line for line in plain_lines if line]

        for line_index, raw_line in enumerate(plain_lines):
            line = raw_line.strip()
            if not line:
                continue
            if set(line) <= {"=", "-", " "}:
                continue
            if line.endswith(":") or line.startswith("TOP 3 ") or line in {
                "Best Business Paths:",
                "Worst Business Paths:",
                "Top Drivers:",
                "Business Recommendation:",
                "Conditions:",
                "Business Interpretation:",
            }:
                rows.append(("heading", line))
                continue
            if line.lower().startswith(("low ", "high ")):
                rows.append(("heading", line))
                continue
            is_bullet_line = bool(re.match(r"^(?:â€¢|Ã¢â‚¬Â¢|•|\*|-)\s+", line))
            next_nonempty = next((candidate for candidate in plain_lines[line_index + 1:] if candidate), "")
            if not is_bullet_line and next_nonempty and re.match(r"^(?:â€¢|Ã¢â‚¬Â¢|•|\*|-)\s+", next_nonempty):
                rows.append(("heading", line))
                continue
            clean = self._strip_tree_bullet_prefix(line)
            if line.lower().startswith("best path for class"):
                rows.append(("path_heading", clean))
            elif clean:
                rows.append(("bullet", clean))
        return rows

    def _extract_json_object(self, response_text):
        if not response_text:
            return None

        cleaned = str(response_text).strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        json_candidate = fenced_match.group(1) if fenced_match else None

        if json_candidate is None:
            raw_match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
            json_candidate = raw_match.group(1) if raw_match else None

        if not json_candidate:
            return None

        try:
            return json.loads(json_candidate)
        except Exception as exc:
            print(f"Improvement text JSON parse failed: {exc}")
            return None

    def _coerce_summary_float(self, value):
        if value in (None, "", "None", "nan"):
            return None
        try:
            cleaned = value.replace(",", "").replace("%", "").strip() if isinstance(value, str) else value
            return float(cleaned)
        except (TypeError, ValueError):
            return None

    def _normalize_classification_percentage(self, value):
        numeric_value = self._coerce_summary_float(value)
        if numeric_value is None:
            return 0.0
        return numeric_value * 100 if 0 <= numeric_value <= 1 else numeric_value

    def _metric(self, value, reduction_percent=5.0):
        numeric_value = self._coerce_summary_float(value)
        if numeric_value is None:
            return 0.0
        reduction_factor = max(0.0, 1.0 - (float(reduction_percent) / 100.0))
        return max(0.0, numeric_value * reduction_factor)

    def _format_optional_float(self, value, decimals=2, default="N/A"):
        numeric_value = self._coerce_summary_float(value)
        if numeric_value is None:
            return default
        return f"{numeric_value:.{decimals}f}"

    def _format_optional_report_number(self, value, decimals=2, suffix="", default="N/A"):
        numeric_value = self._coerce_summary_float(value)
        if numeric_value is None:
            return default
        return f"{numeric_value:.{decimals}f}{suffix}"

    def _format_optional_report_value(self, value, target_column, target_type, decimals=2, default="N/A"):
        numeric_value = self._coerce_summary_float(value)
        if numeric_value is None:
            return default
        return format_display_value(numeric_value, target_column, target_type, decimals=decimals)

    def _plain_summary_text(self, value) -> str:
        if value is None:
            return ""
        text = re.sub(r"<[^>]+>", " ", str(value))
        text = text.replace("•", " ").replace("&bull;", " ")
        text = html.unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    def _trim_summary_words(self, text: str, max_words: int = 500) -> str:
        words = str(text or "").split()
        if len(words) <= max_words:
            return str(text or "").strip()
        return " ".join(words[:max_words]).rstrip(",;:-") + "."

    def _get_execution_mode(self) -> str:
        return str(self.request.session.get("sxi_execution_mode") or "full").strip().lower()

    def _build_comparison_only_result(self, result):
        return make_json_safe({
            "success": True,
            "sxi_score": result.get("sxi_avg"),
            "current_outcome": result.get("curent_tv"),
            "correlation_type": result.get("rltyp"),
            "model_performance": self._get_performance_metrics(result),
        })

    def _build_report_section_context(
        self,
        *,
        plot_target_dist=None,
        plot_vrm_dist=None,
        edaparagraph1=None,
        edaparagraph2=None,
        corr_plot=None,
        Corr_explanation=None,
        correlation_section_text=None,
        Currentdt=None,
        CurrentRate=None,
        CurrentSXI=None,
        Targetdt=None,
        target_mean=None,
        perc1=None,
        perc2=None,
        perc3=None,
        value1=None,
        value2=None,
        value3=None,
        sxi1=None,
        sxi2=None,
        sxi3=None,
        R2_score_lnm=None,
        Accuracy_lnm=None,
        feature1=None,
        feature2=None,
        feature3=None,
        feature4=None,
        feature5=None,
        feature6=None,
        feature7=None,
        feature8=None,
        feature9=None,
        feature10=None,
        featurevalue1=None,
        featurevalue2=None,
        featurevalue3=None,
        featurevalue4=None,
        featurevalue5=None,
        featurevalue6=None,
        featurevalue7=None,
        featurevalue8=None,
        featurevalue9=None,
        featurevalue10=None,
        decision_0_current=None,
        decision_1_current=None,
        decision_0_tr=None,
        decision_1_tr=None,
    ):
        current_decision_logic_html = "\n\n".join(
            part for part in [
                "<h3>Current decision logic (current behavior):</h3>",
                decision_0_current,
                decision_1_current,
            ] if part
        )
        target_decision_logic_html = "\n\n".join(
            part for part in [
                "<h3>Target decision logic (ideal behavior):</h3>",
                decision_0_tr,
                decision_1_tr,
            ] if part
        )

        return make_json_safe({
            "plot_target_dist": plot_target_dist,
            "plot_vrm_dist": plot_vrm_dist,
            "edaparagraph1": edaparagraph1,
            "edaparagraph2": edaparagraph2,
            "corr_plot": corr_plot,
            "Corr_explanation": Corr_explanation,
            "correlation_section_text": correlation_section_text,
            "Currentdt": Currentdt,
            "CurrentRate": CurrentRate,
            "CurrentSXI": CurrentSXI,
            "Targetdt": Targetdt,
            "target_mean": target_mean,
            "perc1": perc1,
            "perc2": perc2,
            "perc3": perc3,
            "value1": value1,
            "value2": value2,
            "value3": value3,
            "sxi1": sxi1,
            "sxi2": sxi2,
            "sxi3": sxi3,
            "R2_score_lnm": R2_score_lnm,
            "Accuracy_lnm": Accuracy_lnm,
            "feature1": feature1,
            "feature2": feature2,
            "feature3": feature3,
            "feature4": feature4,
            "feature5": feature5,
            "feature6": feature6,
            "feature7": feature7,
            "feature8": feature8,
            "feature9": feature9,
            "feature10": feature10,
            "featurevalue1": featurevalue1,
            "featurevalue2": featurevalue2,
            "featurevalue3": featurevalue3,
            "featurevalue4": featurevalue4,
            "featurevalue5": featurevalue5,
            "featurevalue6": featurevalue6,
            "featurevalue7": featurevalue7,
            "featurevalue8": featurevalue8,
            "featurevalue9": featurevalue9,
            "featurevalue10": featurevalue10,
            "decision_0_current": decision_0_current,
            "decision_1_current": decision_1_current,
            "decision_0_tr": decision_0_tr,
            "decision_1_tr": decision_1_tr,
            "current_decision_logic_html": current_decision_logic_html,
            "target_decision_logic_html": target_decision_logic_html,
        })

    def _get_processed_dataset_details(self, prefer_encoded=True):
        import glob

        csv_master_info = self.request.session.get("csv_master_info") or {}
        run_id = str(self.request.session.get("run_id") or "").strip()
        source_csv_path = self.request.session.get("active_report_source_csv_path") or ""
        dataset_name_override = self.request.session.get("active_report_dataset_name") or ""
        preferred_csv = None

        if not prefer_encoded and source_csv_path and os.path.exists(source_csv_path):
            processed_path = source_csv_path
        elif self.dataframe_path and os.path.exists(self.dataframe_path):
            processed_path = self.dataframe_path
        elif prefer_encoded:
            preferred_csv = csv_master_info.get("master_df_encoded") or csv_master_info.get("csv")
        else:
            preferred_csv = csv_master_info.get("master_df_encoded") or csv_master_info.get("csv")

        if "processed_path" not in locals():
            processed_path = None

        if preferred_csv:
            resolved_path = resolve_media_path(preferred_csv)
            if os.path.exists(resolved_path):
                processed_path = resolved_path

        if processed_path is None:
            processed_path = self.dataframe_path

        df = self._read_tabular_file(processed_path)
        rows = int(df.shape[0]) if df is not None else 0
        cols = int(df.shape[1]) if df is not None else 0

        if dataset_name_override:
            dataset_name = dataset_name_override
        else:
            dataset_name = Path(os.path.basename(processed_path)).stem
            if dataset_name.startswith("master_"):
                dataset_name = dataset_name
            dataset_name = dataset_name.split("_raw_merged")[0]
            dataset_name = dataset_name.replace("master_", "")
            dataset_name = dataset_name.replace("_", " ").title()

        return df, dataset_name, rows, cols

    def _get_report_eda_dataframe(self) -> pd.DataFrame:
        """
        Return the dataset that should be described in the PDF EDA section.
        For a merged flow, use the raw merged artifact before preprocessing.
        For a single uploaded dataset, use the original uploaded file.
        """
        try:
            csv_master_info = self.request.session.get("csv_master_info") or {}
            merged_markers = [
                csv_master_info.get("master_df_encoded"),
                csv_master_info.get("csv"),
            ]
            has_raw_merged_artifact = any(
                isinstance(value, str) and "_raw_merged_" in value
                for value in merged_markers
            )

            if has_raw_merged_artifact:
                merged_df, _, _, _ = self._get_processed_dataset_details(prefer_encoded=False)
                if isinstance(merged_df, pd.DataFrame) and not merged_df.empty:
                    print(
                        "PDF SXI EDA source: current session raw merged dataset "
                        f"({len(merged_df)} rows, {len(merged_df.columns)} columns)"
                    )
                    return merged_df

            uploaded_df = self._read_tabular_file(self.dataframe_path)
            print(
                "PDF SXI EDA source: original uploaded dataset "
                f"({len(uploaded_df)} rows, {len(uploaded_df.columns)} columns)"
            )
            return uploaded_df
        except Exception as exc:
            print(f"Report EDA source fallback failed: {exc}")
            return self._read_tabular_file(self.dataframe_path)

    def _prepare_report_eda_dataframe(self, eda_df: pd.DataFrame, sxified_df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep EDA values from the raw/merged source, but copy SXI score only for
        below/above current SXI counts when the transformed result aligns by row.
        """
        if not isinstance(eda_df, pd.DataFrame):
            return sxified_df

        report_df = eda_df.copy()
        if (
            isinstance(sxified_df, pd.DataFrame)
            and "composite_dxi" in sxified_df.columns
            and len(report_df) == len(sxified_df)
            and "composite_dxi" not in report_df.columns
        ):
            report_df["composite_dxi"] = sxified_df["composite_dxi"].to_numpy()

        return report_df

    def _get_after_outlier_dataframe(self) -> pd.DataFrame:
        """
        Load the post-outlier-treatment dataset from the saved master artifact.
        This keeps report EDA independent from any in-memory dataframe mutations.
        """
        try:
            master_df, _, _, _ = self._get_processed_dataset_details(prefer_encoded=True)
            if isinstance(master_df, pd.DataFrame) and not master_df.empty:
                print(
                    "PDF SXI EDA after-outlier source: encoded master dataset "
                    f"({len(master_df)} rows, {len(master_df.columns)} columns)"
                )
                return master_df
        except Exception as exc:
            print(f"Report EDA after-outlier source fallback failed: {exc}")
        return pd.DataFrame()

    def _build_pdf_prof_user_summary_fallback(self, context: dict) -> str:
        dataset_name = context.get("dataset_name", "Dataset")
        rows = context.get("rows", 0)
        cols = context.get("cols", 0)
        target_variable_name = context.get("target_variable_name", "target variable")
        objective_text = context.get("objective_text", "improve the target outcome")
        current_rate_text = context.get("current_rate_text", "N/A")
        current_sxi_text = context.get("current_sxi_text", "N/A")
        current_outcome_text = context.get("current_outcome_text", "N/A")
        eda_points = context.get("eda_points", [])
        current_drivers = context.get("current_drivers", [])
        target_drivers = context.get("target_drivers", [])
        recommendations = context.get("recommendation_points", [])
        performance_points = context.get("performance_points", [])

        summary_lines = [
            "Overview",
            (
                f"{dataset_name} includes {rows:,} rows across {cols:,} features, with {target_variable_name} used as the target variable. "
                f"The analysis was built to {objective_text.lower()} and translate the model findings into practical business actions."
            ),
            "",
            "Key Findings",
            f"- Current DXI is {current_sxi_text}, while current target performance is {current_rate_text}.",
            f"- Current observed {target_variable_name} is {current_outcome_text}.",
        ]

        summary_lines.extend(f"- {point}" for point in eda_points[:3] if point)
        summary_lines.extend(["", "Model Performance"])
        summary_lines.extend(f"- {point}" for point in performance_points[:4] if point)
        summary_lines.extend(["", "Key Drivers / Factors"])

        for name, value in current_drivers[:3]:
            if name:
                summary_lines.append(f"- {name} is a major current-state driver (importance: {value}).")
        for name, value in target_drivers[:2]:
            if name:
                summary_lines.append(f"- {name} remains influential in the target-state path (importance: {value}).")

        summary_lines.extend([
            "",
            "Business Impact & Recommendations",
            (
                f"The findings show a clear path to improve {target_variable_name} by focusing on the few factors that move SXI and outcome performance the most. "
                "This helps teams prioritize actions with stronger business impact and monitor progress against staged targets."
            ),
        ])
        summary_lines.extend(f"- {point}" for point in recommendations[:3] if point)

        return "\n".join(summary_lines).strip()

    def generate_pdf_prof_user_summary(self, context: dict) -> str:
        report_context = f"""
Dataset overview:
- Dataset name: {context.get("dataset_name")}
- Size: {context.get("rows"):,} rows and {context.get("cols"):,} columns
- Target variable: {context.get("target_variable_name")}
- Target summary: {context.get("target_mean")}

Objective / business goal:
- {context.get("objective_text")}

Key metrics and results:
{chr(10).join(f"- {point}" for point in context.get("performance_points", [])) or "- Not available"}

Important insights from analysis:
{chr(10).join(f"- {point}" for point in context.get("eda_points", [])) or "- Not available"}

Correlation and trend interpretation:
- {context.get("corr_explanation") or "Not available"}

Current key drivers:
{chr(10).join(f"- {name}: {value}" for name, value in context.get("current_drivers", []) if name) or "- Not available"}

Target-state key drivers:
{chr(10).join(f"- {name}: {value}" for name, value in context.get("target_drivers", []) if name) or "- Not available"}

Targets, benchmarks, or projections:
{chr(10).join(f"- {point}" for point in context.get("projection_points", [])) or "- Not available"}

Business impact and improvement opportunities:
{chr(10).join(f"- {point}" for point in context.get("recommendation_points", [])) or "- Not available"}
""".strip()

        system_prompt = """
You are a senior business and data analyst.

Your task is to generate a concise, one-page executive summary of the report genrated by def generate_pdf_prof_user.

STRICT INSTRUCTIONS:
- The summary MUST fit within one page (~300-500 words).
- Use a mix of short paragraphs and bullet points where appropriate.
- Keep language business-friendly, clear, and non-technical where possible.
- Capture ALL critical information from the report, including:
  - Dataset overview (size, features, target variable)
  - Objective / business goal
  - Key metrics and results (accuracy, performance, distributions)
  - Important insights from analysis (EDA, correlations, trends)
  - Model performance comparison (if present)
  - Key drivers / feature importance
  - Business impact and improvement opportunities
  - Any targets, benchmarks, or projections

FORMAT STRUCTURE:
1. Overview
2. Key Findings
3. Model Performance
4. Key Drivers / Factors
5. Business Impact & Recommendations

STYLE GUIDELINES:
- Avoid raw tables unless necessary -> convert to readable text
- Do NOT copy text directly -> summarize intelligently
- Highlight numbers only where impactful
- Keep flow natural
- Prioritize clarity over completeness if space is tight

OUTPUT:
Return only the final one-page summary.
Do not include explanations or meta text.
""".strip()

        user_prompt = f"Use the report details below to write the final one-page summary.\n\n{report_context}"

        try:
            response = (self._llm_request(system_prompt, user_prompt, max_tokens=700) or "").strip()
            if not response and self._llm_out_of_credits():
                return self._build_pdf_prof_user_summary_fallback(context)

            cleaned = re.sub(r"```(?:markdown|text)?", "", response, flags=re.IGNORECASE).replace("```", "").strip()
            if len(cleaned.split()) < 120:
                return self._build_pdf_prof_user_summary_fallback(context)
            return self._trim_summary_words(cleaned, max_words=500)
        except Exception as exc:
            print(f"Summary generation failed: {exc}")
            return self._build_pdf_prof_user_summary_fallback(context)

    def generate_report_summary_pdf(self, filename, summary_text, dataset_name=None):
        logo_path = self._get_report_asset("Sriya_new_logo.png")

        doc = SimpleDocTemplate(
            filename,
            pagesize=LETTER,
            rightMargin=28,
            leftMargin=28,
            topMargin=42,
            bottomMargin=28,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "SummaryTitle",
            parent=styles["Title"],
            fontSize=16,
            leading=19,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#1F4E79"),
            spaceAfter=6,
        )
        subtitle_style = ParagraphStyle(
            "SummarySubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=12,
            textColor=colors.HexColor("#4F81BD"),
            spaceAfter=10,
        )
        heading_style = ParagraphStyle(
            "SummaryHeading",
            parent=styles["Heading2"],
            fontSize=11,
            leading=13,
            textColor=colors.HexColor("#1F4E79"),
            spaceBefore=5,
            spaceAfter=4,
        )
        normal_style = ParagraphStyle(
            "SummaryNormal",
            parent=styles["Normal"],
            fontSize=9.5,
            leading=11.5,
            spaceAfter=4,
        )
        bullet_style = ParagraphStyle(
            "SummaryBullet",
            parent=normal_style,
            leftIndent=12,
            firstLineIndent=-7,
            spaceAfter=3,
        )
        note_style = ParagraphStyle(
            "SummaryNote",
            parent=normal_style,
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#5B5B5B"),
            spaceBefore=8,
            italic=True,
        )

        def draw_summary_header(canvas, pdf_doc):
            if not logo_path or not os.path.exists(logo_path):
                return

            canvas.saveState()
            page_width, page_height = pdf_doc.pagesize
            logo_width = 110
            logo_height = 32
            logo_x = page_width - pdf_doc.rightMargin - logo_width
            logo_y = page_height - logo_height - 18

            canvas.setFillColorRGB(0.25, 0.45, 0.65)
            canvas.rect(
                logo_x - 10,
                logo_y - 6,
                logo_width + 20,
                logo_height + 12,
                stroke=0,
                fill=1,
            )

            canvas.drawImage(
                logo_path,
                logo_x,
                logo_y,
                width=logo_width,
                height=logo_height,
                preserveAspectRatio=True,
                mask='auto',
            )
            canvas.restoreState()

        def to_reportlab_markup(text: str) -> str:
            text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", str(text or ""))
            return self._sanitize_reportlab_markup(text)

        heading_pattern = re.compile(
            r"^(?:\d+\.\s*)?(?:\*\*)?(Overview|Key Findings|Model Performance|Key Drivers(?:\s*/\s*Factors)?|Business Impact\s*&\s*Recommendations)(?:\*\*)?:?\s*$",
            flags=re.IGNORECASE,
        )

        elements = [Paragraph("Executive Summary", title_style)]
        if dataset_name:
            elements.append(Paragraph(self._sanitize_reportlab_markup(dataset_name), subtitle_style))

        for raw_line in str(summary_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                elements.append(Spacer(1, 3))
                continue

            heading_match = heading_pattern.match(line)
            if heading_match:
                elements.append(Paragraph(heading_match.group(1), heading_style))
                continue

            if re.match(r"^[-*•]\s+", line):
                content = re.sub(r"^[-*•]\s+", "", line)
                elements.append(Paragraph(f"&bull; {to_reportlab_markup(content)}", bullet_style))
                continue

            elements.append(Paragraph(to_reportlab_markup(line), normal_style))

        elements.append(Spacer(1, 8))
        elements.append(Paragraph("Note: Please refer full report for more details.", note_style))

        doc.build(elements, onFirstPage=draw_summary_header, onLaterPages=draw_summary_header)

    def _determine_sxi_direction(self, current_sxi, target_sxi):
        try:
            current_value = float(current_sxi)
            target_value = float(target_sxi)
        except (TypeError, ValueError):
            return None

        if target_value > current_value:
            return "increase"
        if target_value < current_value:
            return "decrease"
        return "remain the same"

    def _build_sxi_transition_text(self, from_label, to_label, current_sxi, target_sxi):
        try:
            current_value = float(current_sxi)
            target_value = float(target_sxi)
        except (TypeError, ValueError):
            return (
                f"DXI needs to adjust from the {from_label} DXI "
                f"to the {to_label} target SXI."
            )

        direction_word = self._determine_sxi_direction(current_value, target_value)

        if direction_word == "remain the same":
            return (
                f"DXI needs to remain the same from the {from_label} DXI of {current_value:.2f} "
                f"to the {to_label} target SXI of {target_value:.2f}."
            )

        return (
            f"DXI needs to {direction_word} from the {from_label} DXI of {current_value:.2f} "
            f"to the {to_label} target SXI of {target_value:.2f}."
        )

    def _enforce_sxi_transition_sentences(
        self,
        blocks,
        current_sxi,
        immediate_sxi,
        mid_sxi,
        long_sxi,
    ):
        expected_sentences = {
            "immediate_text": self._build_sxi_transition_text(
                "current", "immediate-term", current_sxi, immediate_sxi
            ),
            "mid_text": self._build_sxi_transition_text(
                "immediate-term", "mid-term", immediate_sxi, mid_sxi
            ),
            "long_text": self._build_sxi_transition_text(
                "mid-term", "long-term", mid_sxi, long_sxi
            ),
        }

        normalized_blocks = {}
        for key, text in (blocks or {}).items():
            expected_sentence = expected_sentences.get(key)
            candidate = str(text or "").strip()
            parts = candidate.split("<br/>")

            if len(parts) >= 3 and expected_sentence:
                normalized_blocks[key] = "<br/>".join(parts[:2] + [expected_sentence])
            elif expected_sentence:
                normalized_blocks[key] = f"{candidate}<br/>{expected_sentence}" if candidate else expected_sentence
            else:
                normalized_blocks[key] = candidate

        return normalized_blocks

    def _default_improvement_text_blocks(
        self,
        target_variable_name,
        improvement_direction,
        movement_word,
        action_verb,
        perc1,
        perc2,
        perc3,
        initial_target_value,
        mid_target_value,
        long_target_value,
        current_sxi,
        immediate_sxi,
        mid_sxi,
        long_sxi,
    ):
        return {
            "immediate_text": (
                "<b>Immediate improvement:</b><br/>"
                f"A {perc1:.2f}% {improvement_direction} brings {target_variable_name} "
                f"{movement_word} to about {initial_target_value}.<br/>"
                f"{self._build_sxi_transition_text('current', 'immediate-term', current_sxi, immediate_sxi)}"
            ),
            "mid_text": (
                "<b>Mid-term improvement:</b><br/>"
                f"A {perc2:.2f}% {improvement_direction} brings {target_variable_name} "
                f"to about {mid_target_value}.<br/>"
                f"{self._build_sxi_transition_text('immediate-term', 'mid-term', immediate_sxi, mid_sxi)}"
            ),
            "long_text": (
                "<b>Long-term improvement:</b><br/>"
                f"A {perc3:.2f}% {improvement_direction} brings {target_variable_name} "
                f"to about {long_target_value}.<br/>"
                f"{self._build_sxi_transition_text('mid-term', 'long-term', mid_sxi, long_sxi)}"
            ),
        }

    def _generate_improvement_text_blocks(
        self,
        target_variable_name,
        improvement_direction,
        movement_word,
        action_verb,
        perc1,
        perc2,
        perc3,
        initial_target_value,
        mid_target_value,
        long_target_value,
        current_sxi,
        immediate_sxi,
        mid_sxi,
        long_sxi,
    ):
        def _fmt_sxi(value):
            try:
                return f"{float(value):.2f}"
            except (TypeError, ValueError):
                return "N/A"

        fallback = self._default_improvement_text_blocks(
            target_variable_name=target_variable_name,
            improvement_direction=improvement_direction,
            movement_word=movement_word,
            action_verb=action_verb,
            perc1=perc1,
            perc2=perc2,
            perc3=perc3,
            initial_target_value=initial_target_value,
            mid_target_value=mid_target_value,
            long_target_value=long_target_value,
            current_sxi=current_sxi,
            immediate_sxi=immediate_sxi,
            mid_sxi=mid_sxi,
            long_sxi=long_sxi,
        )

        current_sxi_text = _fmt_sxi(current_sxi)
        immediate_sxi_text = _fmt_sxi(immediate_sxi)
        mid_sxi_text = _fmt_sxi(mid_sxi)
        long_sxi_text = _fmt_sxi(long_sxi)

        system_prompt = """
You generate PDF-ready business report snippets.

Follow these rules exactly:
- Return valid JSON only.
- Do not wrap the JSON in markdown or code fences.
- Output exactly three keys: immediate_text, mid_text, long_text.
- Each value must be a single HTML snippet.
- Each HTML snippet must use this exact structure:
  <b>Section title:</b><br/>Sentence 1<br/>Sentence 2
- Use only <b> and <br/> tags. Do not use any other HTML tags.
- Keep the section titles exactly:
  Immediate improvement:
  Mid-term improvement:
  Long-term improvement:
- Keep all numbers, percentages, metric names, and projected values exactly as provided.
- Do not invent facts, ranges, metrics, or recommendations.
- Keep the tone professional, concise, and business-ready.
- Follow the wording style of a short business update:
  Sentence 1: "A X% [direction] brings [target] ... to about [value]."
  Sentence 2: "DXI needs to increase/decrease/remain the same from the previous horizon DXI to the current horizon target DXI of [value]."
- Determine direction using ONLY this rule:
  - If target_dxi > current_dxi -> "increase"
  - If target_dxi < current_dxi -> "decrease"
  - If target_dxi = current_dxi -> "remain the same"
- Mention the target DXI explicitly in the second sentence of each snippet.
- Immediate should sound like a short-term update.
- Mid-term should sound like a continuing trend update.
- Long-term should sound like a sustained impact update.
- Keep the wording directionally consistent with the provided action verb and improvement direction.
- Do not mention the prompt, rules, JSON, or model.
""".strip()

        user_prompt = f"""
Create three short report snippets for the target variable improvement section.

Use the exact data below:
- Target variable name: {target_variable_name}
- Improvement direction word: {improvement_direction}
- Movement word for immediate section: {movement_word}
- Action verb: {action_verb}
- Immediate percentage: {perc1:.2f}%
- Immediate projected value: {initial_target_value}
- Current DXI: {current_sxi_text}
- Immediate target DXI: {immediate_sxi_text}
- Mid-term percentage: {perc2:.2f}%
- Mid-term projected value: {mid_target_value}
- Immediate-term current DXI for mid-term comparison: {immediate_sxi_text}
- Mid-term target DXI: {mid_sxi_text}
- Long-term percentage: {perc3:.2f}%
- Long-term projected value: {long_target_value}
- Mid-term current DXI for long-term comparison: {mid_sxi_text}
- Long-term target DXI: {long_sxi_text}

Reference style to follow closely:
- Immediate improvement: A 20.00% increase brings sellingprice UP to about $14,417.05. DXI needs to increase from the current DXI of 42.30 to the immediate-term target DXI of 45.10.
- Mid-term improvement: A 33.12% increase brings sellingprice to about $15,993.91. DXI needs to increase from the immediate-term DXI of 45.10 to the mid-term target DXI of 51.40.
- Long-term improvement: A 56.25% increase brings sellingprice to about $18,772.19. DXI needs to increase from the mid-term DXI of 51.40 to the long-term target DXI of 58.80.

Formatting requirements for each key:
- immediate_text must begin exactly with <b>Immediate improvement:</b><br/>
- mid_text must begin exactly with <b>Mid-term improvement:</b><br/>
- long_text must begin exactly with <b>Long-term improvement:</b><br/>
- Sentence 1 must clearly describe the percentage change and projected target value, matching the sample style closely.
- Sentence 2 must explain the DXI movement between horizons in one sentence.
- Determine direction using ONLY this rule:
  - Immediate sentence compares Current DXI vs Immediate target DXI
  - Mid-term sentence compares Immediate-term DXI vs Mid-term target DXI
  - Long-term sentence compares Mid-term DXI vs Long-term target DXI
  - If target_dxi > current_dxi, use "increase"
  - If target_dxi < current_dxi, use "decrease"
  - If target_dxi = current_dxi, use "remain the same"
- Keep the text concise and natural.
- Preserve the exact provided numbers and projected values.
- Use "UP" or "DOWN" only when it fits naturally with the provided movement word.
- Do not create contradictory language such as saying DXI drops while the action verb says increase, unless that exact direction is provided.
- Do not change capitalization of the section titles.
""".strip()

        required_specs = {
            "immediate_text": {
                "heading": "<b>Immediate improvement:</b><br/>",
                "required_tokens": [f"{perc1:.2f}%", str(target_variable_name), str(initial_target_value), immediate_sxi_text],
            },
            "mid_text": {
                "heading": "<b>Mid-term improvement:</b><br/>",
                "required_tokens": [f"{perc2:.2f}%", str(target_variable_name), str(mid_target_value), mid_sxi_text],
            },
            "long_text": {
                "heading": "<b>Long-term improvement:</b><br/>",
                "required_tokens": [f"{perc3:.2f}%", str(target_variable_name), str(long_target_value), long_sxi_text],
            },
        }
        out_of_credits_blocks = {
            key: f"{spec['heading']}{self._out_of_credits_text()}"
            for key, spec in required_specs.items()
        }

        try:
            response = self._llm_request(system_prompt, user_prompt, max_tokens=700)
            required_specs = {
                "immediate_text": {
                    "heading": "<b>Immediate improvement:</b><br/>",
                    "required_tokens": [f"{perc1:.2f}%", str(target_variable_name), str(initial_target_value), immediate_sxi_text],
                },
                "mid_text": {
                    "heading": "<b>Mid-term improvement:</b><br/>",
                    "required_tokens": [f"{perc2:.2f}%", str(target_variable_name), str(mid_target_value), mid_sxi_text],
                },
                "long_text": {
                    "heading": "<b>Long-term improvement:</b><br/>",
                    "required_tokens": [f"{perc3:.2f}%", str(target_variable_name), str(long_target_value), long_sxi_text],
                },
            }
            parsed = self._extract_json_object(response)
            if not isinstance(parsed, dict):
                return out_of_credits_blocks if self._llm_out_of_credits() else fallback

            final_blocks = {}
            for key, spec in required_specs.items():
                candidate = str(parsed.get(key, "") or "").strip()
                if not candidate.startswith(spec["heading"]):
                    final_blocks[key] = out_of_credits_blocks[key] if self._llm_out_of_credits() else fallback[key]
                    continue

                if candidate.count("<br/>") < 2:
                    final_blocks[key] = out_of_credits_blocks[key] if self._llm_out_of_credits() else fallback[key]
                    continue

                if any(token not in candidate for token in spec["required_tokens"]):
                    final_blocks[key] = out_of_credits_blocks[key] if self._llm_out_of_credits() else fallback[key]
                    continue

                sanitized = self._sanitize_reportlab_markup(candidate)
                final_blocks[key] = sanitized or (out_of_credits_blocks[key] if self._llm_out_of_credits() else fallback[key])

            return self._enforce_sxi_transition_sentences(
                final_blocks,
                current_sxi=current_sxi,
                immediate_sxi=immediate_sxi,
                mid_sxi=mid_sxi,
                long_sxi=long_sxi,
            )
        except Exception as exc:
            print(f"Improvement text generation failed: {exc}")
            return out_of_credits_blocks if self._llm_out_of_credits() else fallback

    def _default_classification_chart_explanation(self, acc_val, prec_val, rec_val, auc_val):
        return (
            "The confusion matrix illustrates the classification performance by comparing "
            "actual vs predicted classes. Correct predictions appear along the diagonal, "
            "while off-diagonal values indicate misclassifications.<br/><br/>"
            "The ROC curve shows the trade-off between True Positive Rate and False Positive Rate. "
            f"The model achieves an AUC of {auc_val:.2f}, indicating its ability to distinguish between classes.<br/><br/>"
            f"Additionally, the model demonstrates Accuracy = {acc_val:.2f}, Precision = {prec_val:.2f}, "
            f"Recall = {rec_val:.2f}. Higher values across these metrics indicate stronger and more reliable "
            "classification performance."
        )

    def _generate_classification_chart_explanation(
        self,
        acc_val,
        prec_val,
        rec_val,
        auc_val,
        confusion_matrix_path=None,
        roc_curve_path=None,
    ):
        fallback = self._default_classification_chart_explanation(
            acc_val=acc_val,
            prec_val=prec_val,
            rec_val=rec_val,
            auc_val=auc_val,
        )

        image_path = confusion_matrix_path or roc_curve_path
        visible_chart = "confusion matrix" if confusion_matrix_path else "ROC curve" if roc_curve_path else "classification chart"

        system_prompt = """
You generate concise PDF-ready explanations for classification charts.

Rules:
- Use only facts visible in the attached chart and the numeric values provided by the user.
- Keep the response to one short business-ready explanation.
- Mention the attached chart directly and accurately.
- Include the exact values for Accuracy, Precision, Recall, and AUC exactly as provided.
- Do not invent counts, classes, thresholds, or claims that are not visible or provided.
- If the image is unclear, stay conservative and rely on the provided metric values.
- Output plain text with optional <br/><br/> line breaks only.
- Do not use markdown, bullet points, or extra headings.
""".strip()

        user_prompt = f"""
Generate a short explanation for the classification performance visuals in the PDF.

Attached chart type: {visible_chart}

Use these exact metric values:
- Accuracy = {acc_val:.2f}
- Precision = {prec_val:.2f}
- Recall = {rec_val:.2f}
- AUC = {auc_val:.2f}

Requirements:
- Start by explaining what the attached chart visually indicates about model performance.
- Then relate that visual interpretation to the exact metric values above.
- Mention all four metrics exactly once using the same numbers.
- Keep the tone professional, clear, and concise.
- Do not mention the prompt, model, or limitations unless the image is unclear.
""".strip()

        try:
            response = self._llm_request(
                system_prompt,
                user_prompt,
                max_tokens=400,
                image_path=image_path,
            )
            candidate = self._sanitize_reportlab_markup(response).strip()
            required_tokens = [
                f"Accuracy = {acc_val:.2f}",
                f"Precision = {prec_val:.2f}",
                f"Recall = {rec_val:.2f}",
                f"AUC = {auc_val:.2f}",
            ]
            if not candidate or any(token not in candidate for token in required_tokens):
                return self._out_of_credits_text() if self._llm_out_of_credits() else fallback
            return candidate
        except Exception as exc:
            print(f"Classification chart explanation generation failed: {exc}")
            return self._out_of_credits_text() if self._llm_out_of_credits() else fallback

    def _default_confusion_box_lines(
        self,
        good_outcome=None,
        bad_outcome=None,
    ):
        positive_label = str(good_outcome or "positive").strip()
        negative_label = str(bad_outcome or "negative").strip()
        return [
            f"TP (True Positive): correctly predicted {positive_label} cases.",
            f"FP (False Positive): {negative_label} cases incorrectly predicted as {positive_label}.",
            f"TN (True Negative): correctly identified {negative_label} cases.",
            f"FN (False Negative): {positive_label} cases missed by the model.",
        ]

    def _generate_confusion_box_lines(
        self,
        target_variable_name=None,
        good_outcome=None,
        bad_outcome=None,
        acc_val=None,
        prec_val=None,
        rec_val=None,
        auc_val=None,
        confusion_matrix_path=None,
    ):
        fallback = self._default_confusion_box_lines(
            good_outcome=good_outcome,
            bad_outcome=bad_outcome,
        )

        def metric_text(label, value):
            try:
                return f"{label} = {float(value):.2f}"
            except (TypeError, ValueError):
                return f"{label} = N/A"

        system_prompt = """
You generate concise PDF-ready bullet lines for a confusion matrix explanation box.

Rules:
- Output exactly 4 lines and nothing else.
- Each line must begin with exactly one of these prefixes:
  TP (True Positive):
  FP (False Positive):
  TN (True Negative):
  FN (False Negative):
- Keep each line to one short sentence.
- Use only the attached confusion matrix image and the exact labels/metrics provided.
- Do not invent counts, percentages, thresholds, or unsupported claims.
- Keep the wording business-friendly and easy to understand.
- Plain text only. No markdown, no numbering, no extra headings.
""".strip()

        user_prompt = f"""
Write 4 short explanation lines for the confusion matrix box in a PDF report.

Context:
- Target variable: {target_variable_name}
- Positive / focus outcome label: {good_outcome}
- Negative / alternate outcome label: {bad_outcome}
- {metric_text("Accuracy", acc_val)}
- {metric_text("Precision", prec_val)}
- {metric_text("Recall", rec_val)}
- {metric_text("AUC", auc_val)}

Requirements:
- Keep the four standard confusion matrix labels exactly:
  TP (True Positive):
  FP (False Positive):
  TN (True Negative):
  FN (False Negative):
- Explain each one in plain business language using the outcome labels when helpful.
- Do not mention any numeric counts unless directly visible and certain.
- Keep each line concise enough to fit inside a report callout box.
""".strip()

        try:
            response = self._llm_request(
                system_prompt,
                user_prompt,
                max_tokens=220,
                image_path=confusion_matrix_path,
            )
            response = (response or "").replace("<br/>", "\n")

            parsed_lines = []
            for raw_line in response.splitlines():
                line = self._sanitize_reportlab_markup(raw_line).strip()
                if not line:
                    continue
                line = re.sub(r"^[\-\*\s]+", "", line).strip()
                parsed_lines.append(line)

            expected_order = ["TP", "FP", "TN", "FN"]
            ordered_lines = {}
            for line in parsed_lines:
                upper_line = line.upper()
                for prefix in expected_order:
                    if upper_line.startswith(f"{prefix} ") or upper_line.startswith(f"{prefix}:") or upper_line.startswith(f"{prefix} ("):
                        ordered_lines[prefix] = line
                        break

            if all(prefix in ordered_lines for prefix in expected_order):
                return [ordered_lines[prefix] for prefix in expected_order]

            return [self._out_of_credits_text()] if self._llm_out_of_credits() else fallback
        except Exception as exc:
            print(f"Confusion box generation failed: {exc}")
            return [self._out_of_credits_text()] if self._llm_out_of_credits() else fallback

    def _default_correlation_text(
        self,
        tv_type,
        target_variable_name,
        metric_val,
        slope_val=0.0,
        current_target_value=None,
        desired_target_direction=None,
    ):
        try:
            slope_val = float(slope_val)
        except (TypeError, ValueError):
            slope_val = 0.0

        slope_pct = abs(slope_val) * 100
        try:
            current_target_numeric = float(current_target_value)
        except (TypeError, ValueError):
            current_target_numeric = None

        current_target_text = format_display_value(
            current_target_numeric,
            target_variable_name,
            tv_type,
            decimals=2,
        ) if current_target_numeric is not None else "N/A"

        sxi_direction_word = "decreases" if self._is_reduction_direction(desired_target_direction) else "increases"
        sxi_change_factor = -1 if sxi_direction_word == "decreases" else 1

        if slope_val > 0:
            target_change_factor = sxi_change_factor
            target_direction_word = "increase" if target_change_factor > 0 else "decrease"
            projected_target_value = (
                current_target_numeric * (1 + (0.20 * target_change_factor))
                if current_target_numeric is not None else None
            )
            target_change_text = format_display_value(
                projected_target_value,
                target_variable_name,
                tv_type,
                decimals=2,
            ) if projected_target_value is not None else "N/A"
            impact_line = (
                f'Current target feature "{target_variable_name}" is {current_target_text}. '
                f'If SXI {sxi_direction_word} by 20%, target feature "{target_variable_name}" is projected to {target_direction_word} to {target_change_text}.'
            )
            direction_line = "m +ve means both SXI and target feature go in same direction."
        elif slope_val < 0:
            target_change_factor = -sxi_change_factor
            target_direction_word = "increase" if target_change_factor > 0 else "decrease"
            projected_target_value = (
                    current_target_numeric * (1 + (0.20 * target_change_factor))
                if current_target_numeric is not None else None
            )
            target_change_text = format_display_value(
                projected_target_value,
                target_variable_name,
                tv_type,
                decimals=2,
            ) if projected_target_value is not None else "N/A"
            impact_line = (
                f'Current target feature "{target_variable_name}" is {current_target_text}. '
                f'If SXI {sxi_direction_word} by 20%, target feature "{target_variable_name}" is projected to {target_direction_word} to {target_change_text}.'
            )
            direction_line = "m -ve means SXI and target feature go in opposite direction."
        else:
            projected_target_value = current_target_numeric
            target_change_text = format_display_value(
                projected_target_value,
                target_variable_name,
                tv_type,
                decimals=2,
            ) if projected_target_value is not None else "N/A"
            impact_line = (
                f'Current target feature "{target_variable_name}" is {current_target_text}. '
                f'If SXI {sxi_direction_word} by 20%, target feature "{target_variable_name}" is projected to remain the same at {target_change_text}.'
            )
            direction_line = "m = 0 means SXI and target feature do not show directional change together."

        slope_logic_line = (
            f"m = {slope_val:.2f} means that change in Y axis or target feature is {slope_pct:.2f}% of the change in X axis or SXI."
        )

        if tv_type == "Categorical":
            return (
                f"<b>Correlation Coefficient r2 (Categorical): {metric_val:.2f}</b><br/>"
                f"<b>Slope m: {slope_val:.2f}</b><br/><br/>"
            )

        return (
            f"<b>Correlation Coefficient r2: {metric_val:.2f}</b><br/>"
            f"<b>Slope m: {slope_val:.2f}</b><br/><br/>"
            #f"{impact_line}<br/><br/>"
            #f"{direction_line}<br/>"
            #f"{slope_logic_line}"
        )
        """
        if tv_type == "Categorical":
            return (
                f"<b>The relationship between SXI and {target_variable_name} is summarized using R² = {metric_val:.2f}.</b><br/><br/>"
                f"This R² value indicates how much of the variation in {target_variable_name} is explained by SXI. "
                "Higher values indicate stronger explanatory power of SXI for the classification outcome."
            )

        return (
            f"<b>The correlation between SXI and {target_variable_name} is {metric_val:.2f}.</b><br/><br/>"
            f"This value indicates the degree of linear association between SXI and {target_variable_name}. "
            "Higher values indicate stronger explanatory power of SXI for predicting the target variable."
        )
        """

    def _resolve_correlation_metric_value(self, tv_type, result=None, explicit_metric=None):
        candidates = []
        if explicit_metric is not None:
            candidates.append(explicit_metric)

        if isinstance(result, dict):
            if tv_type == "Categorical":
                # Old categorical r2 calculation kept for reference. It
                # preferred result["r2"], which could be the local curve-fit
                # value around 0.05 to 0.60:
                # candidates.extend([result.get("r2"), result.get("r2score")])
                candidates.extend([result.get("r2score"), result.get("r2")])
            else:
                candidates.extend([result.get("r2score"), result.get("r2")])

        for candidate in candidates:
            if candidate is None:
                continue
            return normalize_metric_score(candidate)

        return 0.0

    def _resolve_correlation_slope(self, result=None, explicit_slope=None):
        candidates = []
        if explicit_slope is not None:
            candidates.append(explicit_slope)

        if isinstance(result, dict):
            candidates.extend([
                result.get("correlation_slope"),
                result.get("slope"),
            ])

        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return float(candidate)
            except (TypeError, ValueError):
                continue

        return 0.0

    def _is_reduction_direction(self, direction_text):
        return str(direction_text or "").strip().lower() in {
            "decrease",
            "decreased",
            "decreasing",
            "reduce",
            "reduced",
            "reducing",
            "minimize",
            "minimise",
            "lower",
            "less",
        }

    def _generate_correlation_text(
        self,
        tv_type,
        target_variable_name,
        metric_val,
        slope_val=0.0,
        current_target_value=None,
        corr_plot_path=None,
        desired_target_direction=None,
    ):
        # LLM-based correlation explanation disabled per requirement.
        # fallback = self._default_correlation_text(
        #     tv_type=tv_type,
        #     target_variable_name=target_variable_name,
        #     metric_val=metric_val,
        # )
        # system_prompt = """You generate concise PDF-ready chart explanations for business reports."""
        # user_prompt = f"""Write a short explanation for the SXI relationship chart."""
        # response = self._llm_request(system_prompt, user_prompt, max_tokens=300, image_path=corr_plot_path)
        # candidate = self._sanitize_reportlab_markup(response).strip()
        return self._default_correlation_text(
            tv_type=tv_type,
            target_variable_name=target_variable_name,
            metric_val=metric_val,
            slope_val=slope_val,
            current_target_value=current_target_value,
            desired_target_direction=desired_target_direction,
        )

        fallback = self._default_correlation_text(
            tv_type=tv_type,
            target_variable_name=target_variable_name,
            metric_val=metric_val,
            current_target_value=current_target_value,
        )

        metric_label = "R²" if tv_type == "Categorical" else "correlation"
        metric_meaning = (
            f"how much of the variation in {target_variable_name} is explained by SXI"
            if tv_type == "Categorical"
            else f"the strength of association between SXI and {target_variable_name}"
        )

        system_prompt = """
You generate concise PDF-ready chart explanations for business reports.

Rules:
- Use only facts visible in the attached chart and the exact metric value provided.
- Output plain text with optional <br/><br/> line breaks only.
- Do not use markdown, bullets, or headings beyond any inline <b> text.
- Keep the explanation short, professional, and business-ready.
- Do not invent trends, thresholds, or claims not supported by the chart or metric.
- Preserve the metric label and metric value exactly as provided.
- Preserve the metric label and metric value exactly as provided.
- For categorical targets, use R² language and do not mention AUC.
""".strip()

        user_prompt = f"""
Write a short explanation for the SXI relationship chart.

Context:
- Target variable: {target_variable_name}
- Target type: {tv_type}
- Metric label: {metric_label}
- Metric value: {metric_val:.2f}
- Metric meaning: {metric_meaning}

Requirements:
- Start with a bold first sentence using inline HTML <b>...</b>.
- For categorical targets, the first sentence must mention: R² = {metric_val:.2f}
- For numeric targets, the first sentence must mention: {metric_val:.2f}
- The second part should explain what the metric says about SXI in plain business language.
- Mention SXI and {target_variable_name} naturally.
- Keep the wording concise and suitable for a PDF paragraph.
- If the target type is Categorical, do not mention AUC.
""".strip()

        try:
            response = self._llm_request(
                system_prompt,
                user_prompt,
                max_tokens=300,
                image_path=corr_plot_path,
            )
            candidate = self._sanitize_reportlab_markup(response).strip()
            required_tokens = [str(target_variable_name), f"{metric_val:.2f}"]
            if tv_type == "Categorical":
                required_tokens.append("R²")

            if not candidate or any(token not in candidate for token in required_tokens):
                return self._out_of_credits_text() if self._llm_out_of_credits() else fallback

            return candidate
        except Exception as exc:
            print(f"Correlation text generation failed: {exc}")
            return self._out_of_credits_text() if self._llm_out_of_credits() else fallback

    def _default_regression_model_comparison_explanation(
        self,
        better_model_name,
        better_r2,
        other_r2,
        better_mae_display,
        other_mae_display,
    ):
        if better_model_name == "SXI":
            return (
                "SXI demonstrates stronger predictive performance compared to the ChatGPT model, "
                f"achieving higher accuracy (R² = {better_r2:.2f} vs. {other_r2:.2f}) and lower "
                f"prediction error (MAE = {better_mae_display} vs. {other_mae_display}).<br/><br/>"
                "This indicates that SXI predictions are more closely aligned with actual values "
                "and are more reliable for decision-making."
            )

        return (
            "The ChatGPT model demonstrates stronger predictive performance compared to SXI, "
            f"with higher accuracy (R² = {better_r2:.2f} vs. {other_r2:.2f}) and lower "
            f"prediction error (MAE = {better_mae_display} vs. {other_mae_display}).<br/><br/>"
            "This suggests that, under the current evaluation, ChatGPT provides more reliable predictions."
        )

    def _generate_regression_model_comparison_explanation(
        self,
        target_variable_name,
        sxi_r2,
        chatgpt_r2,
        sxi_mae,
        chatgpt_mae,
        act_vs_pred_image_path=None,
    ):
        sxi_mae_display = format_display_value(sxi_mae, target_variable_name, "Numeric", decimals=2)
        chatgpt_mae_display = format_display_value(chatgpt_mae, target_variable_name, "Numeric", decimals=2)

        sxi_is_better = sxi_r2 > chatgpt_r2 and sxi_mae < chatgpt_mae
        better_model_name = "SXI" if sxi_is_better else "ChatGPT"
        better_r2 = sxi_r2 if sxi_is_better else chatgpt_r2
        other_r2 = chatgpt_r2 if sxi_is_better else sxi_r2
        better_mae_display = sxi_mae_display if sxi_is_better else chatgpt_mae_display
        other_mae_display = chatgpt_mae_display if sxi_is_better else sxi_mae_display

        fallback = self._default_regression_model_comparison_explanation(
            better_model_name=better_model_name,
            better_r2=better_r2,
            other_r2=other_r2,
            better_mae_display=better_mae_display,
            other_mae_display=other_mae_display,
        )

        system_prompt = """
You generate concise PDF-ready model comparison explanations for regression reports.

Rules:
- Use only the exact metric values and image context provided.
- Output plain text with optional <br/><br/> line breaks only.
- Keep the explanation short, professional, and business-ready.
- State which model performs better based on higher R² and lower MAE.
- Preserve all metric values and MAE units exactly as provided.
- Do not invent additional metrics, thresholds, or unsupported claims.
- If an image is attached, use it only as supporting visual context.
""".strip()

        user_prompt = f"""
Write a short regression model comparison explanation.

Context:
- Target variable: {target_variable_name}
- Better model based on the provided rules: {better_model_name}
- SXI R² = {sxi_r2:.2f}
- ChatGPT R² = {chatgpt_r2:.2f}
- SXI MAE = {sxi_mae_display}
- ChatGPT MAE = {chatgpt_mae_display}

Requirements:
- Clearly state which model performs better in this evaluation.
- Mention both R² values exactly as provided.
- Mention both MAE values exactly as provided, keeping any units or currency symbols.
- End with one concise business implication sentence.
- Keep the tone suitable for a PDF report.
""".strip()

        try:
            response = self._llm_request(
                system_prompt,
                user_prompt,
                max_tokens=300,
                image_path=act_vs_pred_image_path,
            )
            candidate = self._sanitize_reportlab_markup(response).strip()
            required_tokens = [
                better_model_name,
                f"{sxi_r2:.2f}",
                f"{chatgpt_r2:.2f}",
                str(sxi_mae_display),
                str(chatgpt_mae_display),
            ]
            if not candidate or any(token not in candidate for token in required_tokens):
                return self._out_of_credits_text() if self._llm_out_of_credits() else fallback
            return candidate
        except Exception as exc:
            print(f"Regression comparison explanation generation failed: {exc}")
            return self._out_of_credits_text() if self._llm_out_of_credits() else fallback

    def _build_tree_feature_importance_rows(
        self,
        feature_importance_df=None,
        primary_features=None,
        primary_weights=None,
        max_rows=5,
        value_label="Tree Importance",
    ):
        rows = []
        seen = set()

        def coerce_float(value):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            return numeric

        def add_row(feature, weight):
            feature_name = "" if feature is None else str(feature).strip()
            if not feature_name or feature_name in seen:
                return

            numeric_weight = coerce_float(weight)
            rows.append({"feature": feature_name, "weight": numeric_weight})
            seen.add(feature_name)

        if isinstance(feature_importance_df, pd.DataFrame) and not feature_importance_df.empty:
            working_df = feature_importance_df.copy()
            value_column = next(
                (
                    col for col in [
                        "RF_Importance",
                        "Importance",
                        "Tree Importance",
                        "Tree_Importance",
                    ]
                    if col in working_df.columns
                ),
                None,
            )

            sort_columns = [
                col for col in [value_column, "SXI_Weights", "SXI_Weight"]
                if col and col in working_df.columns
            ]
            if sort_columns:
                working_df = working_df.sort_values(
                    by=sort_columns,
                    ascending=[False] * len(sort_columns),
                    na_position="last",
                ).reset_index(drop=True)

            for _, record in working_df.iterrows():
                feature_label = record.get("Feature")
                category_breakdown = str(record.get("Category_Breakdown") or "").strip()
                if feature_label and category_breakdown:
                    feature_label = (
                        f"{feature_label}\n" +
                        "\n".join(f"- {part.strip()}" for part in category_breakdown.split(";") if part.strip())
                    )
                add_row(feature_label, record.get(value_column) if value_column else None)
                if len(rows) >= max_rows:
                    break

        if len(rows) < max_rows and primary_features is not None and primary_weights is not None:
            for feature, weight in zip(primary_features, primary_weights):
                add_row(feature, weight)
                if len(rows) >= max_rows:
                    break

        while len(rows) < max_rows:
            rows.append({"feature": None, "weight": None})

        total_weight = sum(
            row["weight"] for row in rows
            if isinstance(row["weight"], (int, float)) and row["weight"] is not None
        ) or 1.0

        output = [["Feature", value_label, "% Contribution"]]
        for row in rows:
            weight = row["weight"]
            contribution = (
                f"{((weight / total_weight) * 100):.2f}%"
                if isinstance(weight, (int, float)) and weight is not None
                else ""
            )
            output.append([
                row["feature"],
                f"{weight:.2f}" if isinstance(weight, (int, float)) and weight is not None else None,
                contribution,
            ])

        return output

    def _normalize_feature_importance_dataframe(
        self,
        dataframe,
        required_columns,
        aliases=None,
        label="feature importance",
    ):
        aliases = aliases or {}

        if isinstance(dataframe, pd.Series):
            working_df = dataframe.rename("value").reset_index()
        elif isinstance(dataframe, pd.DataFrame):
            working_df = dataframe.copy()
        else:
            try:
                working_df = pd.DataFrame(dataframe)
            except Exception:
                print(f"SXI warning: {label} dataframe missing, using empty fallback.")
                working_df = pd.DataFrame()

        if working_df.empty and not list(working_df.columns):
            working_df = pd.DataFrame(columns=required_columns)

        for canonical_col, alias_cols in aliases.items():
            if canonical_col in working_df.columns:
                continue
            alias_col = next((col for col in alias_cols if col in working_df.columns), None)
            if alias_col is not None:
                working_df[canonical_col] = working_df[alias_col]

        if "Feature" in required_columns and "Feature" not in working_df.columns:
            if not isinstance(working_df.index, pd.RangeIndex) and len(working_df.index) == len(working_df):
                working_df["Feature"] = working_df.index
            else:
                working_df["Feature"] = pd.NA

        for column in required_columns:
            if column not in working_df.columns:
                working_df[column] = pd.NA

        numeric_columns = [col for col in required_columns if col != "Feature"]
        for column in numeric_columns:
            working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

        return working_df

    def _default_actual_vs_predicted_explanation(
        self,
        target_variable_name,
        within_pct=None,
        r2_score=None,
        mae_display=None,
    ):
        metric_bits = []
        if within_pct is not None:
            metric_bits.append(
                f"{within_pct:.2f}% of predictions fall within +/-10% of the actual {target_variable_name} values"
            )
        if r2_score is not None:
            metric_bits.append(f"R² = {r2_score:.2f}")
        if mae_display:
            metric_bits.append(f"MAE = {mae_display}")

        metric_sentence = ""
        if metric_bits:
            metric_sentence = " " + ", ".join(metric_bits) + "."

        return (
            f"The Actual vs Predicted chart shows how closely SXI predictions track observed {target_variable_name} values."
            f"{metric_sentence}<br/><br/>"
            "A tighter clustering around the diagonal indicates more reliable predictions and lower error across the evaluated records."
        )

    def _generate_actual_vs_predicted_explanation(
        self,
        target_variable_name,
        within_pct=None,
        r2_score=None,
        mae_value=None,
        act_vs_pred_image_path=None,
    ):
        mae_display = (
            format_display_value(mae_value, target_variable_name, "Numeric", decimals=2)
            if mae_value is not None
            else None
        )
        fallback = self._default_actual_vs_predicted_explanation(
            target_variable_name=target_variable_name,
            within_pct=within_pct,
            r2_score=r2_score,
            mae_display=mae_display,
        )

        metric_lines = []
        required_tokens = []

        if within_pct is not None:
            within_token = f"{within_pct:.2f}%"
            metric_lines.append(f"- Within +/-10% = {within_token}")
            required_tokens.append(within_token)
        if r2_score is not None:
            r2_token = f"{r2_score:.2f}"
            metric_lines.append(f"- R² = {r2_token}")
            required_tokens.append(r2_token)
        if mae_display:
            metric_lines.append(f"- MAE = {mae_display}")
            required_tokens.append(str(mae_display))

        system_prompt = """
You generate concise PDF-ready explanations for regression Actual vs Predicted charts.

Rules:
- Use only the visible chart and the exact metrics provided.
- Output plain text with optional <br/><br/> line breaks only.
- Keep the explanation short, business-ready, and accurate.
- Mention what the chart visually indicates about model fit.
- Preserve every provided numeric value exactly as given.
- Do not invent thresholds, trends, or unsupported claims.
""".strip()

        user_prompt = f"""
Write a short explanation for the Actual vs Predicted regression chart in the PDF.

Context:
- Target variable: {target_variable_name}
{chr(10).join(metric_lines) if metric_lines else "- No additional metrics were provided"}

Requirements:
- Start by describing what the chart visually shows about predicted versus actual values.
- If metric values are provided, mention them exactly as shown.
- Keep the explanation concise and professional.
- Do not mention the prompt, model, or limitations unless the image is unclear.
""".strip()

        try:
            response = self._llm_request(
                system_prompt,
                user_prompt,
                max_tokens=220,
                image_path=act_vs_pred_image_path,
            )
            candidate = self._sanitize_reportlab_markup(response).strip()
            if not candidate:
                return self._out_of_credits_text() if self._llm_out_of_credits() else fallback
            if any(token not in candidate for token in required_tokens):
                return self._out_of_credits_text() if self._llm_out_of_credits() else fallback
            return candidate
        except Exception as exc:
            print(f"Actual vs Predicted explanation generation failed: {exc}")
            return self._out_of_credits_text() if self._llm_out_of_credits() else fallback

    def _default_classification_model_comparison_explanation(
        self,
        better_model_name,
        sxi_acc,
        sxi_prec,
        sxi_rec,
        sxi_auc,
        chatgpt_acc,
        chatgpt_prec,
        chatgpt_recall,
        chatgpt_auc,
    ):
        if better_model_name == "SXI":
            return (
                "SXI classification model shows stronger predictive performance compared to the ChatGPT AutoML model. "
                f"SXI achieves Accuracy ({sxi_acc:.2f} vs. {chatgpt_acc:.2f}), Precision ({sxi_prec:.2f} vs. {chatgpt_prec:.2f}), "
                f"Recall ({sxi_rec:.2f} vs. {chatgpt_recall:.2f}), and AUC ({sxi_auc:.2f} vs. {chatgpt_auc:.2f}), "
                "indicating more reliable classification predictions."
            )

        return (
            "The ChatGPT AutoML model shows stronger predictive performance compared to SXI. "
            f"ChatGPT achieves Accuracy ({chatgpt_acc:.2f} vs. {sxi_acc:.2f}), Precision ({chatgpt_prec:.2f} vs. {sxi_prec:.2f}), "
            f"Recall ({chatgpt_recall:.2f} vs. {sxi_rec:.2f}), and AUC ({chatgpt_auc:.2f} vs. {sxi_auc:.2f}), "
            "indicating more reliable classification predictions in the current evaluation."
        )

    def _generate_classification_model_comparison_explanation(
        self,
        sxi_acc,
        sxi_prec,
        sxi_rec,
        sxi_auc,
        chatgpt_acc,
        chatgpt_prec,
        chatgpt_recall,
        chatgpt_auc,
    ):
        sxi_score = sum([
            1 if sxi_acc > chatgpt_acc else 0,
            1 if sxi_prec > chatgpt_prec else 0,
            1 if sxi_rec > chatgpt_recall else 0,
            1 if sxi_auc > chatgpt_auc else 0,
        ])
        chatgpt_score = sum([
            1 if chatgpt_acc > sxi_acc else 0,
            1 if chatgpt_prec > sxi_prec else 0,
            1 if chatgpt_recall > sxi_rec else 0,
            1 if chatgpt_auc > sxi_auc else 0,
        ])

        better_model_name = "SXI" if sxi_score >= chatgpt_score else "ChatGPT"

        fallback = self._default_classification_model_comparison_explanation(
            better_model_name=better_model_name,
            sxi_acc=sxi_acc,
            sxi_prec=sxi_prec,
            sxi_rec=sxi_rec,
            sxi_auc=sxi_auc,
            chatgpt_acc=chatgpt_acc,
            chatgpt_prec=chatgpt_prec,
            chatgpt_recall=chatgpt_recall,
            chatgpt_auc=chatgpt_auc,
        )

        system_prompt = """
You generate concise PDF-ready model comparison explanations for classification reports.

Rules:
- Use only the exact metric values provided.
- Output plain text with optional <br/><br/> line breaks only.
- Keep the explanation short, professional, and business-ready.
- Compare SXI and ChatGPT using Accuracy, Precision, Recall, and AUC.
- Preserve all metric values exactly as provided.
- Do not invent additional metrics, thresholds, or unsupported claims.
""".strip()

        user_prompt = f"""
Write a short classification model comparison explanation.

Context:
- Better overall model based on the provided metrics: {better_model_name}
- SXI Accuracy = {sxi_acc:.2f}
- SXI Precision = {sxi_prec:.2f}
- SXI Recall = {sxi_rec:.2f}
- SXI AUC = {sxi_auc:.2f}
- ChatGPT Accuracy = {chatgpt_acc:.2f}
- ChatGPT Precision = {chatgpt_prec:.2f}
- ChatGPT Recall = {chatgpt_recall:.2f}
- ChatGPT AUC = {chatgpt_auc:.2f}

Requirements:
- Clearly state which model performs better in this evaluation.
- Mention all four SXI metrics exactly as provided.
- Mention all four ChatGPT metrics exactly as provided.
- End with one concise business implication sentence.
- Keep the tone suitable for a PDF report.
""".strip()

        try:
            response = self._llm_request(
                system_prompt,
                user_prompt,
                max_tokens=300,
            )
            candidate = self._sanitize_reportlab_markup(response).strip()
            required_tokens = [
                better_model_name,
                f"{sxi_acc:.2f}",
                f"{sxi_prec:.2f}",
                f"{sxi_rec:.2f}",
                f"{sxi_auc:.2f}",
                f"{chatgpt_acc:.2f}",
                f"{chatgpt_prec:.2f}",
                f"{chatgpt_recall:.2f}",
                f"{chatgpt_auc:.2f}",
            ]
            if not candidate or any(token not in candidate for token in required_tokens):
                return self._out_of_credits_text() if self._llm_out_of_credits() else fallback
            return candidate
        except Exception as exc:
            print(f"Classification comparison explanation generation failed: {exc}")
            return self._out_of_credits_text() if self._llm_out_of_credits() else fallback

    def _default_objective_text(self, perce_inc_text, tv_inc, objective_outcome):
        return (
            f"• Minimum <b>{perce_inc_text}%</b> {tv_inc} in {objective_outcome} "
            f"from current levels using <b>Sriya’s SXI</b>."
        )

    def _generate_objective_text(
        self,
        perce_inc_text,
        tv_inc,
        objective_outcome,
        target_variable_name=None,
    ):
        fallback = self._default_objective_text(perce_inc_text, tv_inc, objective_outcome)
        target_info = self.target_info if isinstance(self.target_info, dict) else {}

        system_prompt = """
You generate one PDF-ready business objective line.

Rules:
- Output exactly one sentence only.
- Start the sentence with the bullet symbol: •
- Use valid inline HTML with only <b> tags when needed.
- Keep the exact percentage, action word, and outcome label exactly as provided.
- Preserve the wording style of a short objective statement.
- Do not add extra commentary, headings, markdown, or line breaks.
- End with "using <b>Sriya’s SXI</b>."
""".strip()

        user_prompt = f"""
Write one concise objective line for a business report.

Target metadata:
- Target Outcome: {target_info.get('Target Outcome', target_variable_name)}
- Target Outcome Type: {target_info.get('Target Outcome Type', '')}
- Good Outcome: {target_info.get('Good Outcome', '')}
- Bad Outcome: {target_info.get('Bad Outcome', '')}
- Good Outcome Value: {target_info.get('Good Outcome Value', '')}
- Bad Outcome Value: {target_info.get('Bad Outcome Value', '')}
- Target Outcome Improvement: {target_info.get('Target Outcome Improvement', '')}
- Task Type: {target_info.get('Task Type', '')}
- is_timeseries: {target_info.get('is_timeseries', '')}

Use these exact values in the final sentence:
- Percentage: {perce_inc_text}%
- Action phrase: {tv_inc}
- Outcome label: {objective_outcome}

Sample style to follow closely:
• Minimum <b>20.00%</b> reduced in Canceled from current levels using <b>Sriya’s SXI</b>

Requirements:
- Follow the sample style closely.
- Keep the exact percentage with two decimals.
- Keep the exact action phrase "{tv_inc}".
- Keep the exact outcome label "{objective_outcome}".
- Include <b>{perce_inc_text}%</b> exactly.
- End exactly with: using <b>Sriya’s SXI</b>.
""".strip()

        try:
            response = self._llm_request(system_prompt, user_prompt, max_tokens=120)
            candidate = self._sanitize_reportlab_markup(response).strip()
            required_tokens = [
                "•",
                f"{perce_inc_text}%",
                str(tv_inc),
                str(objective_outcome),
                "Sriya",
                "SXI",
            ]
            if not candidate or any(token not in candidate for token in required_tokens):
                return fallback
            return candidate
        except Exception as exc:
            print(f"Objective text generation failed: {exc}")
            return fallback

    def _extract_tree_class_label(self, path_str, class_name) -> str:
        if path_str:
            match = re.search(r"Best Path for Class \d+\s*:\s*(.+)", str(path_str))
            if match:
                return match.group(1).strip()

        if isinstance(class_name, (list, tuple, set)):
            for item in class_name:
                if item not in (None, ""):
                    return str(item).strip()
            return "Target Outcome"

        if class_name not in (None, ""):
            return str(class_name).strip()

        return "Target Outcome"

    def _labels_match(self, left, right) -> bool:
        normalize = lambda value: re.sub(r"\s+", " ", str(value or "").strip().lower())
        return normalize(left) == normalize(right)

    def _normalize_tree_label(self, label: str) -> str:
        return re.sub(r"\s+", " ", str(label or "").strip().lower())

    def _target_outcome_display_name(self) -> str:
        target_name = ""
        if isinstance(getattr(self, "target_info", None), dict):
            target_name = str(self.target_info.get("Target Outcome") or "").strip()

        if not target_name:
            return "Target Outcome"

        display_name = self._tree_feature_display_name(target_name)
        return display_name.title() if display_name else "Target Outcome"

    def _is_generic_regression_label(self, class_label: str) -> bool:
        normalized = self._normalize_tree_label(class_label)
        return normalized in {
            "above mean",
            "below mean",
            "above_mean",
            "below_mean",
            "high above mean",
            "low below mean",
        }

    def _tree_heading_for_label(self, class_label: str) -> str:
        label = str(class_label or "").strip()
        lower_label = self._normalize_tree_label(label)
        target_display = self._target_outcome_display_name()

        if self._is_generic_regression_label(label):
            if "above" in lower_label or "high" in lower_label:
                return f"High {target_display}"
            if "below" in lower_label or "low" in lower_label:
                return f"Low {target_display}"

        if "above" in lower_label or "high" in lower_label:
            return f"High {label}"
        if "below" in lower_label or "low" in lower_label:
            return f"Low {label}"
        return label or "Decision Tree Interpretation"

    def _class_value_phrase(self, class_label: str) -> str:
        label = str(class_label or "").strip()
        lower_label = self._normalize_tree_label(label)
        target_display = self._target_outcome_display_name().lower()

        if self._is_generic_regression_label(label):
            if "above" in lower_label or "high" in lower_label:
                return f"higher {target_display} values"
            if "below" in lower_label or "low" in lower_label:
                return f"lower {target_display} values"

        if "above" in lower_label or "high" in lower_label:
            return "above-mean values"
        if "below" in lower_label or "low" in lower_label:
            return "below-mean values"
        return f"the {label} outcome" if label else target_display

    def _tree_branch_direction_label(self, operator: str) -> str:
        if operator in {"<=", "<", "="}:
            return "Main left branch node"
        if operator in {">", ">=", "!="}:
            return "Main right branch node"
        return "Branch node"

    def _select_representative_tree_conditions(self, conditions, max_conditions=3):
        """
        Pick the path nodes that explain the branch, not just the shared prefix.
        Tree paths often share the first splits for both outcomes and diverge near
        the leaf, so use the root plus the last distinct branch nodes.
        """
        unique_conditions = []
        seen = set()
        for condition in conditions or []:
            key = (
                self._normalize_tree_feature_key(condition.get("feature")),
                str(condition.get("operator") or "").strip(),
                str(condition.get("threshold") or "").strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            unique_conditions.append(condition)

        if len(unique_conditions) <= max_conditions:
            return unique_conditions

        tail_count = max_conditions - 1
        return [unique_conditions[0]] + unique_conditions[-tail_count:]

    def _path_to_bullets(self, path_str, class_label: str) -> str:
        heading = self._tree_heading_for_label(class_label)
        if not path_str:
            # Prefer no placeholder copy on Executive Dashboard (tree image only)
            return ""
        if self._is_business_ready_tree_text(path_str):
            return str(path_str).strip()

        bullets = []
        for condition in self._extract_tree_path_conditions(path_str):
            phrase = self._normalize_business_condition_text(
                self._semantic_tree_condition_text(condition, class_label)
            )
            if phrase:
                bullets.append(f"• {phrase} is associated with {class_label}.")

        if not bullets:
            preserved_bullets = self._business_ready_bullet_lines(path_str)
            if preserved_bullets:
                return "\n".join([heading] + preserved_bullets[:3])
            return ""

        return "\n".join([heading] + bullets[:3])

    def _normalize_tree_feature_key(self, feature: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(feature or "").strip().lower()).strip()

    def _humanize_generic_feature_text(self, value: str) -> str:
        text = str(value or "").strip().replace("_", " ")
        if not text:
            return ""
        words = []
        for word in text.split():
            if word.lower() == "sxi":
                words.append("SXI")
            else:
                words.append(word.capitalize())
        return " ".join(words)

    def _get_encoded_feature_meta(self, feature: str):
        encoded_feature = str(feature or "").strip()
        if not encoded_feature:
            return None
        meta = getattr(self, "encoded_feature_mapping", {}).get(encoded_feature)
        if meta:
            return meta
        return self._infer_encoded_feature_meta_from_schema(encoded_feature)

    def _infer_encoded_feature_meta_from_schema(self, encoded_feature: str):
        grouped_columns, base_feature = self._schema_encoded_group_for_feature(encoded_feature)
        if not grouped_columns or not base_feature:
            return None

        prefix = f"{base_feature}_"
        if not str(encoded_feature).startswith(prefix):
            return None

        category = str(encoded_feature)[len(prefix):]
        top_categories = [
            str(column)[len(prefix):]
            for column in grouped_columns
            if str(column).startswith(prefix)
            and not str(column).lower().endswith("_others")
        ]
        return {
            "base_feature": base_feature,
            "category": "Others" if category.lower() == "others" else category,
            "base_feature_display": self._humanize_generic_feature_text(base_feature),
            "category_display": (
                "Others"
                if category.lower() == "others"
                else self._humanize_generic_feature_text(category)
            ),
            "top_categories": top_categories[:5],
        }

    def _schema_encoded_group_for_feature(self, encoded_feature: str):
        schema_columns = (
            list(getattr(self, "_encoded_schema_columns", []) or [])
            or self._load_encoded_master_columns()
        )
        encoded_feature = str(encoded_feature or "").strip()
        if not encoded_feature or not schema_columns:
            return [], ""

        candidates = []
        for column in schema_columns:
            column_text = str(column or "").strip()
            if not column_text.lower().endswith("_others"):
                continue
            base_feature = column_text[:-len("_Others")]
            prefix = f"{base_feature}_"
            if encoded_feature == column_text or encoded_feature.startswith(prefix):
                grouped = [
                    str(candidate or "").strip()
                    for candidate in schema_columns
                    if str(candidate or "").strip().startswith(prefix)
                ]
                if grouped:
                    candidates.append((base_feature, grouped))

        if not candidates:
            return [], ""

        base_feature, grouped_columns = max(candidates, key=lambda item: len(item[0]))
        return grouped_columns, base_feature

    def _category_label_from_encoded_column(self, encoded_column: str, base_feature: str) -> str:
        prefix = f"{base_feature}_"
        category = str(encoded_column or "").strip()
        if category.startswith(prefix):
            category = category[len(prefix):]
        return self._humanize_generic_feature_text(category) or category
        return None

    def _extract_tree_path_conditions(self, path_str):
        conditions = []
        if not path_str:
            return conditions

        for raw_line in str(path_str).splitlines():
            line = raw_line.strip().lstrip("•").strip()
            if not line or line.lower().startswith("best path for class") or line.lower().startswith("leaf =>"):
                continue

            line = re.sub(r"\[.*?\]", "", line).strip()
            line = re.sub(r",\s*value\s*=\s*[-+]?\d*\.?\d+", "", line, flags=re.IGNORECASE).strip()
            line = line.strip("()")

            match = re.match(r"(.+?)\s*(<=|>=|<|>)\s*([-+]?\d*\.?\d+)", line)
            if match:
                feature, operator, threshold = match.groups()
                conditions.append({
                    "feature": self._strip_tree_bullet_prefix(feature),
                    "operator": operator,
                    "threshold": threshold,
                })

        return conditions

    def _extract_tree_path_conditions(self, path_str):
        conditions = []
        if not path_str:
            return conditions

        for raw_line in str(path_str).splitlines():
            line = raw_line.strip().lstrip("â€¢").lstrip("•").strip()
            if not line or line.lower().startswith("best path for class") or line.lower().startswith("leaf =>"):
                continue
            if line.lower().startswith(("prediction range:", "coverage:", "node samples:", "mean target value:")):
                continue

            line = re.sub(r"\[.*?\]", "", line).strip()
            line = re.sub(r",\s*value\s*=\s*[-+]?\d*\.?\d+", "", line, flags=re.IGNORECASE).strip()
            line = line.strip("()")

            match = re.match(r"(.+?)\s*(<=|>=|<|>)\s*([-+]?\d*\.?\d+)", line)
            if match:
                feature, operator, threshold = match.groups()
                conditions.append({
                    "feature": self._strip_tree_bullet_prefix(feature),
                    "operator": operator,
                    "threshold": threshold,
                })
                continue

            category_match = re.match(r"(.+?)\s*(=|!=)\s*(.+)", line)
            if category_match:
                feature, operator, threshold = category_match.groups()
                conditions.append({
                    "feature": feature.strip(),
                    "operator": operator,
                    "threshold": threshold.strip(),
                    "categorical": True,
                })

        return conditions

    def _extract_tree_path_metadata(self, path_str):
        metadata = {}
        if not path_str:
            return metadata
        for raw_line in str(path_str).splitlines():
            line = raw_line.strip().lstrip("â€¢").lstrip("•").strip()
            if ":" not in line:
                continue
            key, value = [part.strip() for part in line.split(":", 1)]
            key_norm = key.lower()
            if key_norm in {"prediction range", "coverage", "node samples", "mean target value"}:
                metadata[key_norm.replace(" ", "_")] = value
        return metadata

    def _coerce_tree_metric_float(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _lookup_tree_feature_metrics(self, feature_importance_df, feature: str):
        metrics = {"sxi_weight": None, "importance": None}
        if (
            not isinstance(feature_importance_df, pd.DataFrame)
            or feature_importance_df.empty
            or "Feature" not in feature_importance_df.columns
        ):
            return metrics

        target_keys = {
            self._normalize_tree_feature_key(feature),
            self._normalize_tree_feature_key(self._tree_feature_display_name(feature)),
        }

        matched_row = None
        for _, record in feature_importance_df.iterrows():
            record_feature = record.get("Feature")
            record_keys = {
                self._normalize_tree_feature_key(record_feature),
                self._normalize_tree_feature_key(self._tree_feature_display_name(record_feature)),
            }
            if target_keys.intersection(record_keys):
                matched_row = record
                break

        if matched_row is None:
            return metrics

        for column in ["SXI_Weights", "SXI_Weight", "SXI Weight"]:
            if column in feature_importance_df.columns:
                metrics["sxi_weight"] = self._coerce_tree_metric_float(matched_row.get(column))
                break

        for column in ["Importance", "RF_Importance", "Tree Importance", "Tree_Importance"]:
            if column in feature_importance_df.columns:
                metrics["importance"] = self._coerce_tree_metric_float(matched_row.get(column))
                break

        return metrics

    def _format_tree_metric(self, value):
        return f"{value:.2f}" if isinstance(value, (int, float)) else "N/A"

    def _strip_tree_bullet_prefix(self, value: str) -> str:
        return re.sub(r"^(?:\s|â€¢|Ã¢â‚¬Â¢|•|\?|\*|-)+", "", str(value or "")).strip()

    def _is_tree_metadata_line(self, line: str) -> bool:
        return str(line or "").strip().lower().startswith((
            "prediction range:",
            "coverage:",
            "node samples:",
            "mean target value:",
        ))

    def _regression_outcome_direction(self, class_label: str) -> str:
        lower_label = self._normalize_tree_label(class_label)
        if "above" in lower_label or "high" in lower_label:
            return "higher"
        return "lower"

    def _operator_to_business_phrase(self, operator: str, threshold: str) -> str:
        if operator == "<=":
            return f"{threshold} or lower"
        if operator == ">":
            return f"above {threshold}"
        if operator == "<":
            return f"below {threshold}"
        if operator == ">=":
            return f"{threshold} or higher"
        if operator == "=":
            return str(threshold)
        if operator == "!=":
            return f"not {threshold}"
        return str(threshold)

    def _condition_to_business_sentence(self, condition, class_label: str, index: int) -> str:
        direction = self._regression_outcome_direction(class_label)
        target_display = self._target_outcome_display_name().lower()
        if condition.get("fallback"):
            feature = self._tree_feature_display_name(self._strip_tree_bullet_prefix(condition.get("feature")))
            readable = f"{feature} is one of the strongest decision-tree drivers"
        else:
            readable = self._humanize_tree_condition(
                condition.get("feature"),
                str(condition.get("operator") or "").strip(),
                str(condition.get("threshold") or "").strip(),
            )
        readable = readable[:1].lower() + readable[1:] if readable else "this condition is met"
        node_prefix = "Root node" if index == 0 else self._tree_branch_direction_label(condition.get("operator"))

        if index == 0:
            return (
                f"{node_prefix}: records where {readable} are the first group moving toward "
                f"{direction} {target_display} values."
            )
        if index == 1:
            return (
                f"{node_prefix}: when {readable} also applies, the profile becomes more clearly aligned "
                f"with {direction} {target_display} values."
            )
        return (
            f"Child node: the final decision point, where {readable}, helps identify the records most "
            f"likely to end in {direction} {target_display} values."
        )

    def _regression_path_header(self, path_str, class_label: str) -> str:
        explicit_label = self._extract_tree_class_label(path_str, class_label)
        lower_label = self._normalize_tree_label(explicit_label)
        class_number = "1" if "above" in lower_label or "high" in lower_label else "0"
        label = "Above Mean" if class_number == "1" else "Below Mean"
        return f"Best Path for Class {class_number}: {label}"

    def _fallback_tree_feature_conditions(self, feature_importance_df, max_conditions=3):
        if not isinstance(feature_importance_df, pd.DataFrame) or feature_importance_df.empty:
            return []

        feature_col = "Feature" if "Feature" in feature_importance_df.columns else feature_importance_df.columns[0]
        score_col = next(
            (
                col for col in [
                    "Importance",
                    "RF_Importance",
                    "Tree Importance",
                    "Tree_Importance",
                    "SXI_Weights",
                    "SXI_Weight",
                    "SXI Weight",
                ]
                if col in feature_importance_df.columns
            ),
            None,
        )
        ranked_df = feature_importance_df
        if score_col:
            ranked_df = feature_importance_df.sort_values(score_col, ascending=False)

        conditions = []
        for _, row in ranked_df.iterrows():
            feature = self._strip_tree_bullet_prefix(row.get(feature_col))
            if not feature or self._is_tree_metadata_line(feature):
                continue
            conditions.append({
                "feature": feature,
                "operator": "",
                "threshold": "",
                "fallback": True,
            })
            if len(conditions) == max_conditions:
                break
        return conditions

    def _regression_path_html(self, path_str, class_label: str, feature_importance_df=None) -> str:
        heading = self._tree_heading_for_label(class_label)
        conditions = self._select_representative_tree_conditions(
            self._extract_tree_path_conditions(path_str),
            max_conditions=3,
        )
        if not conditions and feature_importance_df is not None:
            conditions = self._fallback_tree_feature_conditions(feature_importance_df, max_conditions=3)
        if not conditions:
            return ""
        if conditions:
            list_items = "\n".join(
                f"<li>\n{self._condition_to_business_sentence(condition, class_label, index)}\n</li>"
                for index, condition in enumerate(conditions)
            )
        else:
            list_items = ""

        return "\n\n".join([
            f"<h4>{html.escape(heading)}</h4>",
            "<ul>\n" + list_items + "\n</ul>",
        ])

    def _humanize_tree_condition(self, feature: str, operator: str, threshold: str) -> str:
        if operator in {"=", "!="}:
            feature_name = self._tree_feature_display_name(feature)
            category = self._humanize_generic_feature_text(threshold)
            category_key = str(threshold or "").strip().lower()
            if category_key == "others":
                top_categories_text = self._encoded_top_categories_text(feature)
                if operator == "=":
                    return f"{feature_name.capitalize()} is Others ({top_categories_text})"
                return f"{feature_name.capitalize()} is not Others ({top_categories_text})"
            if operator == "=":
                return f"{feature_name.capitalize()} is {category}"
            return f"{feature_name.capitalize()} is not {category}"

        encoded_meta = self._get_encoded_feature_meta(feature)
        if encoded_meta:
            base_feature = encoded_meta.get("base_feature_display") or self._humanize_generic_feature_text(
                encoded_meta.get("base_feature")
            )
            category = encoded_meta.get("category_display") or self._humanize_generic_feature_text(
                encoded_meta.get("category")
            )
            category_key = str(encoded_meta.get("category") or category).strip().lower()
            top_categories_text = self._encoded_top_categories_text(feature, encoded_meta)
            if operator == "<=":
                if category_key == "others":
                    return f"{base_feature} is one of the top 5 categories ({top_categories_text})"
                return f"{base_feature} is NOT {category}"
            if category_key == "others":
                return f"{base_feature} is outside the top 5 categories ({top_categories_text})"
            return f"{base_feature} is {category}"

        feature_name = self._tree_feature_display_name(feature)
        feature_key = self._normalize_tree_feature_key(feature_name)

        if feature_key == "year" or feature_key.endswith(" year") or "model year" in feature_key:
            return "Older model years" if operator in {"<=", "<"} else "More recent model years"

        if "month of year" in feature_key:
            return "Earlier months of the year" if operator in {"<=", "<"} else "Later months of the year"

        if "sxi" in feature_key:
            return "Lower SXI scores" if operator in {"<=", "<"} else "Higher SXI scores"

        direction = "Lower" if operator in {"<=", "<"} else "Higher"
        return f"{direction} {feature_name.lower()} levels"

    def _tree_label_semantic_polarity(self, class_label: str) -> str:
        label = self._normalize_tree_label(class_label)
        negative_markers = {
            "not",
            "non",
            "no",
            "bad",
            "negative",
            "fail",
            "failed",
            "failure",
            "fraud",
            "risk",
            "churn",
            "defect",
            "reject",
            "rejected",
            "denied",
            "cancel",
            "cancelled",
            "lost",
            "lower",
            "below",
        }
        positive_markers = {
            "yes",
            "good",
            "positive",
            "success",
            "successful",
            "approve",
            "approved",
            "purchase",
            "purchased",
            "buyer",
            "converted",
            "conversion",
            "retain",
            "retained",
            "higher",
            "above",
        }
        tokens = set(re.findall(r"[a-z0-9]+", label))
        if tokens.intersection(negative_markers) or any(marker in label for marker in negative_markers):
            return "negative"
        if tokens.intersection(positive_markers) or any(marker in label for marker in positive_markers):
            return "positive"
        return "positive"

    def _tree_feature_metric_direction(self, feature: str) -> str:
        key = self._normalize_tree_feature_key(self._tree_feature_display_name(feature))
        raw_key = self._normalize_tree_feature_key(feature)
        combined = f"{raw_key} {key}"
        reverse_markers = [
            "days since",
            "dayssincelast",
            "last purchase days ago",
            "lastpurchasedaysago",
            "days ago",
            "churn risk",
            "churnrisk",
            "complaint count",
            "complaintcount",
            "return count",
            "returncount",
            "abandonment score",
            "abandonmentscore",
            "delinquency",
            "default risk",
            "risk score",
            "error count",
            "defect count",
        ]
        positive_markers = [
            "revenue",
            "purchase frequency",
            "purchasefrequency",
            "frequency",
            "engagement score",
            "engagementscore",
            "customer satisfaction",
            "customersatisfaction",
            "satisfaction",
            "session count",
            "sessioncount",
            "number of purchases",
            "numberofpurchases",
            "purchase count",
            "loyalty score",
            "loyaltyscore",
            "income",
            "sales",
            "rating",
            "score",
        ]
        if any(marker in combined for marker in reverse_markers):
            return "reverse"
        if any(marker in combined for marker in positive_markers):
            return "positive"
        return "neutral"

    def _tree_condition_feature_key(self, condition) -> str:
        feature = condition.get("feature") if isinstance(condition, dict) else condition
        encoded_meta = self._get_encoded_feature_meta(feature)
        if encoded_meta:
            feature = encoded_meta.get("base_feature") or feature
        return self._normalize_tree_feature_key(self._tree_feature_display_name(feature))

    def _tree_metric_sort_score(self, metrics) -> float:
        if not metrics:
            return 0.0
        values = []
        for key in ("importance", "sxi_weight"):
            value = metrics.get(key)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_value):
                values.append(numeric_value)
        return max(values) if values else 0.0

    def _semantic_tree_condition_text(self, condition, class_label: str) -> str:
        feature = condition.get("feature")
        operator = condition.get("operator")
        threshold = condition.get("threshold")
        encoded_meta = self._get_encoded_feature_meta(feature)

        if condition.get("categorical") or operator in {"=", "!="} or encoded_meta:
            return self._humanize_tree_condition(feature, operator, threshold)

        feature_name = self._tree_feature_display_name(feature)
        feature_key = self._normalize_tree_feature_key(feature_name)
        polarity = self._tree_label_semantic_polarity(class_label)
        metric_direction = self._tree_feature_metric_direction(feature)

        if metric_direction == "reverse":
            if polarity == "positive":
                if "last purchase" in feature_key or "days since" in feature_key or "days ago" in feature_key:
                    return "more recent purchase activity"
                return f"lower {feature_name.lower()} levels"
            if "last purchase" in feature_key or "days since" in feature_key or "days ago" in feature_key:
                return "longer time since the last purchase"
            return f"higher {feature_name.lower()} levels"

        if metric_direction == "positive":
            if polarity == "positive":
                if "frequency" in feature_key:
                    return "more frequent purchase activity"
                return f"higher {feature_name.lower()} levels"
            if "frequency" in feature_key:
                return "less frequent purchase activity"
            return f"lower {feature_name.lower()} levels"

        if operator in {"<=", "<"}:
            return f"lower {feature_name.lower()} levels"
        return f"higher {feature_name.lower()} levels"

    def _select_semantic_tree_conditions(self, conditions, class_label: str, feature_importance_df=None, max_conditions=3):
        feature_records = {}
        for order, condition in enumerate(conditions or []):
            if not isinstance(condition, dict):
                continue
            feature_key = self._tree_condition_feature_key(condition)
            if not feature_key:
                continue
            metrics = self._lookup_tree_feature_metrics(feature_importance_df, condition.get("feature"))
            record = feature_records.setdefault(feature_key, {
                "condition": condition,
                "metrics": metrics,
                "count": 0,
                "first_order": order,
                "score": self._tree_metric_sort_score(metrics),
                "feature_key": feature_key,
            })
            record["count"] += 1
            score = self._tree_metric_sort_score(metrics)
            if score > record["score"]:
                record["condition"] = condition
                record["metrics"] = metrics
                record["score"] = score

        ranked_records = sorted(
            feature_records.values(),
            key=lambda record: (
                record.get("score", 0.0),
                record.get("count", 0),
                -record.get("first_order", 0),
            ),
            reverse=True,
        )
        if not any(record.get("score", 0.0) for record in ranked_records):
            ranked_records = sorted(
                feature_records.values(),
                key=lambda record: (record.get("count", 0), -record.get("first_order", 0)),
                reverse=True,
            )

        selected = []
        seen_text = set()
        for record in ranked_records:
            condition = record["condition"]
            text = self._semantic_tree_condition_text(condition, class_label)
            normalized_text = self._normalize_tree_feature_key(text)
            if not text or normalized_text in seen_text:
                continue
            seen_text.add(normalized_text)
            selected.append({
                "text": text,
                "strong": self._tree_metric_sort_score(record.get("metrics")) >= 20,
                "branch": self._tree_branch_direction_label(condition.get("operator")),
                "condition": condition,
                "feature_key": record.get("feature_key"),
            })
            if len(selected) == max_conditions:
                break
        return selected

    def _weighted_tree_path_to_bullets(self, path_str, class_label: str, feature_importance_df=None) -> str:
        if self._is_generic_regression_label(class_label):
            return self._regression_path_html(path_str, class_label, feature_importance_df=feature_importance_df)

        heading = self._tree_heading_for_label(class_label)
        outcome_phrase = self._class_value_phrase(class_label)
        readable_conditions = self._select_semantic_tree_conditions(
            self._extract_tree_path_conditions(path_str),
            class_label,
            feature_importance_df=feature_importance_df,
            max_conditions=3,
        )

        if not readable_conditions:
            return ""

        bullets = []
        for index, condition in enumerate(readable_conditions[:3]):
            readable_condition = condition["text"]
            readable_lower = (
                readable_condition[:1].lower() + readable_condition[1:]
                if readable_condition else
                "this condition is met"
            )
            driver_note = " This is one of the strongest drivers in the tree." if condition["strong"] else ""
            node_prefix = "Root node" if index == 0 else condition["branch"] if index == 1 else "Child node"
            if index == 0:
                sentence = (
                    f"• {node_prefix}: records where {readable_lower} are the first clear signal "
                    f"for {outcome_phrase}.{driver_note}"
                )
            elif index == 1:
                sentence = (
                    f"• {node_prefix}: when {readable_lower} also applies, the decision profile "
                    f"becomes more strongly aligned with {outcome_phrase}.{driver_note}"
                )
            else:
                sentence = (
                    f"• {node_prefix}: where {readable_lower}, helps decide which "
                    f"records are most likely to land in {outcome_phrase}.{driver_note}"
                )
            bullets.append(sentence)

        return "\n".join([heading] + bullets)

    def _tree_feature_display_name(self, feature: str) -> str:
        encoded_meta = self._get_encoded_feature_meta(feature)
        if encoded_meta:
            return encoded_meta.get("base_feature_display") or self._humanize_generic_feature_text(
                encoded_meta.get("base_feature")
            )

        feature = str(feature or "").strip().replace("_", " ")
        return {
            "composite dxi": "SXI score",
            "yr mth": "month of year",
            "year month": "month of year",
            "odometer": "odometer readings",
            "condition": "condition level",
            "sellingprice": "selling price",
            "mmr": "MMR value",
            "purchasefrequency": "purchase frequency",
            "engagementscore": "engagement score",
            "customersatisfaction": "customer satisfaction",
            "lastpurchasedaysago": "days since last purchase",
            "dayssincelastpurchase": "days since last purchase",
            "numberofpurchases": "number of purchases",
            "sessioncount": "session count",
            "loyaltyscore": "loyalty score",
            "churnrisk": "churn risk",
            "complaintcount": "complaint count",
            "returncount": "return count",
            "abandonmentscore": "abandonment score",
        }.get(feature.lower(), feature)

    def _encoded_top_categories_text(self, feature: str, encoded_meta=None) -> str:
        normalized_feature = self._normalize_tree_feature_key(feature)
        meta = encoded_meta or self._get_encoded_feature_meta(feature) or {}
        base_feature = str(meta.get("base_feature") or feature or "").strip()
        base_display = self._humanize_generic_feature_text(base_feature).lower() or "feature"
        top_categories = []

        for category in (meta.get("top_categories") or []):
            if str(category).strip().lower() == "others":
                continue
            top_categories.append(self._humanize_generic_feature_text(category))

        if not top_categories:
            grouped_columns = list(getattr(self, "grouped_encoded_features", {}).get(base_feature, []) or [])
            if not grouped_columns:
                for group_name, columns in getattr(self, "grouped_encoded_features", {}).items():
                    if self._normalize_tree_feature_key(group_name) == normalized_feature:
                        grouped_columns = list(columns or [])
                        break

            for encoded_col in grouped_columns:
                grouped_meta = self._get_encoded_feature_meta(encoded_col) or {}
                category = grouped_meta.get("category")
                if category is not None and str(category).strip().lower() == "others":
                    continue
                top_categories.append(str(encoded_col))

        if not top_categories:
            grouped_columns, inferred_base_feature = self._schema_encoded_group_for_feature(feature)
            if not grouped_columns and base_feature:
                grouped_columns, inferred_base_feature = self._schema_encoded_group_for_feature(f"{base_feature}_Others")
            if grouped_columns:
                base_feature = inferred_base_feature or base_feature
                base_display = self._humanize_generic_feature_text(base_feature).lower() or base_display
                for column in grouped_columns:
                    column_text = str(column or "").strip()
                    if column_text.lower().endswith("_others"):
                        continue
                    top_categories.append(self._category_label_from_encoded_column(column_text, base_feature))
                    if len(top_categories) == 5:
                        break

        if not top_categories:
            for encoded_col, mapped_meta in getattr(self, "encoded_feature_mapping", {}).items():
                if self._normalize_tree_feature_key(mapped_meta.get("base_feature")) != normalized_feature:
                    continue
                for category in mapped_meta.get("top_categories") or []:
                    if str(category).strip().lower() != "others":
                        top_categories.append(self._humanize_generic_feature_text(category))
                if top_categories:
                    break

        if not top_categories:
            for encoded_col, mapped_meta in getattr(self, "encoded_feature_mapping", {}).items():
                if self._normalize_tree_feature_key(mapped_meta.get("base_feature")) != normalized_feature:
                    continue
                category = mapped_meta.get("category")
                if category is None or str(category).strip().lower() == "others":
                    continue
                top_categories.append(str(encoded_col))

        seen = set()
        unique_categories = []
        for category in top_categories:
            key = self._normalize_tree_feature_key(category)
            if not key or key in seen:
                continue
            seen.add(key)
            unique_categories.append(category)

        if not unique_categories:
            return "top 5 categories unavailable"
        return f"top 5 {base_display}: {', '.join(unique_categories[:5])}"

    def _encoded_category_column_name(self, base_feature: str, category) -> str:
        base = str(base_feature or "").strip()
        token = str(category if category is not None else "").strip()
        if not base:
            return self._humanize_generic_feature_text(token)
        if not token:
            token = "blank"
        token = re.sub(r"\s+", "_", token)
        token = re.sub(r"[^0-9A-Za-z_]+", "_", token)
        token = re.sub(r"_+", "_", token).strip("_") or "blank"
        return f"{base}_{token}"

    def _load_encoded_master_columns(self):
        import glob

        candidates = []
        try:
            csv_master_info = self.request.session.get("csv_master_info") or {}
            run_id = str(self.request.session.get("run_id") or "").strip()
            candidates.extend([
                csv_master_info.get("master_df_encoded"),
                csv_master_info.get("csv"),
                self.dataframe_path,
            ])
            master_dir = os.path.join(settings.MEDIA_ROOT, "master")
            if run_id:
                candidates.append(os.path.join(master_dir, f"master_{run_id}.csv"))
            master_files = glob.glob(os.path.join(master_dir, "master_*.csv"))
            master_files.sort(key=os.path.getmtime, reverse=True)
            candidates.extend(master_files[:3])
        except Exception:
            candidates.append(getattr(self, "dataframe_path", None))

        seen_candidates = set()
        for candidate in candidates:
            if not candidate:
                continue
            try:
                path = resolve_media_path(candidate) if isinstance(candidate, str) else candidate
                path_key = os.path.abspath(str(path))
                if path_key in seen_candidates:
                    continue
                seen_candidates.add(path_key)
                if not path or not os.path.exists(path):
                    continue
                columns = pd.read_csv(path, nrows=0, low_memory=False).columns.tolist()
                if columns:
                    self._encoded_schema_columns = [str(col) for col in columns]
                    return self._encoded_schema_columns
            except Exception as exc:
                print(f"[SXI][TREE_VALIDATION] Could not read encoded master columns from {candidate}: {exc}")

        return []

    def _tree_condition_phrase(self, feature: str, operator: str, threshold: str = None) -> str:
        encoded_meta = self._get_encoded_feature_meta(feature)
        if encoded_meta:
            base_feature = encoded_meta.get("base_feature_display") or self._humanize_generic_feature_text(
                encoded_meta.get("base_feature")
            )
            category = encoded_meta.get("category_display") or self._humanize_generic_feature_text(
                encoded_meta.get("category")
            )
            category_key = str(encoded_meta.get("category") or category).strip().lower()
            top_categories_text = self._encoded_top_categories_text(feature, encoded_meta)
            if operator == "<=":
                if category_key == "others":
                    return f"records within the top 5 categories for {base_feature} ({top_categories_text})"
                return f"records where {base_feature} is not {category}"
            if category_key == "others":
                return f"records outside the top 5 categories for {base_feature} ({top_categories_text})"
            return f"records where {base_feature} is {category}"

        feature_name = self._tree_feature_display_name(feature)
        if operator in {"=", "!="} and str(threshold or "").strip().lower() == "others":
            top_categories_text = self._encoded_top_categories_text(feature)
            if operator == "=":
                return f"records where {feature_name} is Others ({top_categories_text})"
            return f"records where {feature_name} is not Others ({top_categories_text})"

        feature_key = feature_name.lower()

        if feature_key == "year" or feature_key.endswith(" year") or "model year" in feature_key:
            return "older model years" if operator == "<=" else "more recent model years"
        if "month of year" in feature_key:
            return "earlier months of the year" if operator == "<=" else "later months of the year"
        if "odometer" in feature_key or "mileage" in feature_key:
            return "lower odometer readings" if operator == "<=" else "higher odometer readings"
        if "condition" in feature_key:
            return "lower condition levels" if operator == "<=" else "higher condition levels"
        if "sxi" in feature_key:
            return "lower SXI scores" if operator == "<=" else "higher SXI scores"

        direction = "lower" if operator == "<=" else "higher"
        return f"{direction} {feature_name}"

    def _tree_prompt_context(self, path_str) -> str:
        if not path_str:
            return "No readable branch conditions were extracted."

        notes = []
        for raw_line in str(path_str).splitlines():
            line = raw_line.strip().lstrip("•").strip()
            if not line or line.lower().startswith("best path for class") or line.lower().startswith("leaf =>"):
                continue
            line = re.sub(r"\[.*?\]", "", line).strip()
            line = re.sub(r",\s*value\s*=\s*[-+]?\d*\.?\d+", "", line, flags=re.IGNORECASE).strip()
            line = line.strip("()")
            match = re.match(r"(.+?)\s*(<=|>)\s*([-+]?\d*\.?\d+)", line)
            if match:
                feature, operator, threshold = match.groups()
                safe_phrase = self._tree_condition_phrase(feature, operator, threshold)
                notes.append(f"- {feature.strip()} {operator} {threshold} -> {safe_phrase}")

        if not notes:
            return "No readable branch conditions were extracted."

        return "\n".join(notes[:5])

    def _business_feature_label(self, feature: str) -> str:
        label = str(self._tree_feature_display_name(feature) or feature or "").strip()
        if not label:
            return "this feature"
        if label.upper() == label:
            return label
        if label.lower().startswith("sxi"):
            return "SXI" + label[3:]
        return label[:1].lower() + label[1:]

    def _normalize_business_condition_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", str(text or "").strip())
        text = re.sub(r"\s*\([^)]*top 5[^)]*\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\btop 5 categories\b", "common categories", text, flags=re.IGNORECASE)
        text = text.replace(" NOT ", " not ")
        text = text.replace(" is NOT ", " is not ")
        text = text.rstrip(".")
        if not text:
            return ""
        return text[:1].upper() + text[1:]

    def _tree_pair_class_label(self, path_str, class_names, index: int) -> str:
        label = self._extract_tree_class_label(path_str, None)
        if label and label != "Target Outcome":
            return label
        if isinstance(class_names, (list, tuple)) and len(class_names) > index and class_names[index] not in (None, ""):
            return str(class_names[index]).strip()
        return f"Class {index}"

    def _opposite_tree_operator(self, operator: str) -> str:
        operator = str(operator or "").strip()
        return {
            "<=": ">",
            "<": ">=",
            ">": "<=",
            ">=": "<",
            "=": "!=",
            "!=": "=",
        }.get(operator, operator)

    def _tree_business_condition_item(self, condition, class_label: str = None, reverse: bool = False):
        feature = self._strip_tree_bullet_prefix(condition.get("feature"))
        operator = str(condition.get("operator") or "").strip()
        threshold = str(condition.get("threshold") or "").strip()
        if reverse:
            operator = self._opposite_tree_operator(operator)

        encoded_meta = self._get_encoded_feature_meta(feature)
        is_categorical = bool(encoded_meta) or bool(condition.get("categorical")) or operator in {"=", "!="}

        if is_categorical:
            phrase = self._normalize_business_condition_text(
                self._humanize_tree_condition(feature, operator, threshold)
            )
            category_key = ""
            if encoded_meta:
                category_key = self._normalize_tree_feature_key(
                    f"{encoded_meta.get('base_feature')} {encoded_meta.get('category')}"
                )
            else:
                category_key = self._normalize_tree_feature_key(f"{feature} {threshold}")
            return {
                "key": f"categorical:{category_key}",
                "type": "categorical",
                "phrase": phrase,
            }

        if class_label:
            phrase = self._normalize_business_condition_text(
                self._semantic_tree_condition_text(
                    {**condition, "feature": feature, "operator": operator, "threshold": threshold},
                    class_label,
                )
            )
            return {
                "key": f"numeric:{self._normalize_tree_feature_key(feature)}",
                "type": "numeric",
                "phrase": phrase,
            }

        direction = "Lower" if operator in {"<=", "<"} else "Higher"
        feature_label = self._business_feature_label(feature)
        return {
            "key": f"numeric:{self._normalize_tree_feature_key(feature_label)}",
            "type": "numeric",
            "direction": direction,
            "feature_label": feature_label,
            "phrase": f"{direction} {feature_label}",
        }

    def _tree_business_sentence(self, item, class_label: str) -> str:
        label = str(class_label or "").strip() or "this class"
        phrase = str(item.get("phrase") or "").strip()
        if not phrase:
            return ""
        return f"• {phrase} is associated with {label}."

    def _is_business_ready_tree_text(self, text: str) -> bool:
        text = str(text or "").strip()
        if not text:
            return False
        if not re.search(r"(?m)^\s*(?:â€¢|Ã¢â‚¬Â¢|•|\*|-)\s+", text):
            return False
        if re.search(r"\b(?:less than|greater than|threshold|gini|entropy|samples)\b", text, flags=re.IGNORECASE):
            return False
        if re.search(r"(?:<=|>=|(?<!\w)<(?!\w)|(?<!\w)>(?!\w))", text):
            return False
        return bool(re.search(r"\b(?:associated with|higher|lower|more|less|longer|recent|segment|premium|regular)\b", text, flags=re.IGNORECASE))

    def _business_ready_bullet_lines(self, text: str):
        if not text:
            return []
        lines = []
        for raw_line in str(text).splitlines():
            line = raw_line.strip()
            if not re.match(r"^(?:â€¢|Ã¢â‚¬Â¢|•|\*|-)\s+", line):
                continue
            clean = self._strip_tree_bullet_prefix(line)
            if not clean:
                continue
            if re.search(r"\b(?:less than|greater than|threshold|gini|entropy|samples)\b", clean, flags=re.IGNORECASE):
                continue
            if re.search(r"(?:<=|>=|(?<!\w)<(?!\w)|(?<!\w)>(?!\w))", clean):
                continue
            if re.search(r"\b(?:associated with|higher|lower|more|less|longer|recent|segment|premium|regular)\b", clean, flags=re.IGNORECASE):
                lines.append(f"• {clean}")
        return lines

    def _tree_business_interpretation_pair(
        self,
        path_0,
        path_1,
        class_names,
        feature_importance_df=None,
        tree_image_path=None,
        max_bullets=3,
    ):
        """
        Build both class explanations together so numeric directions cannot
        contradict each other across the two target classes.
        """
        path_0_text = str(path_0 or "").strip()
        path_1_text = str(path_1 or "").strip()
        path_0_ready = self._is_business_ready_tree_text(path_0_text)
        path_1_ready = self._is_business_ready_tree_text(path_1_text)
        if path_0_text and not path_1_text and path_0_ready:
            return path_0_text, ""
        if path_1_text and not path_0_text and path_1_ready:
            return "", path_1_text
        if path_0_ready or path_1_ready:
            fallback_0 = (
                path_0_text
                if path_0_ready
                else self.expl_dt_pth(path_0, class_names, tree_image_path=tree_image_path, feature_importance_df=feature_importance_df) if path_0 else ""
            )
            fallback_1 = (
                path_1_text
                if path_1_ready
                else self.expl_dt_pth(path_1, class_names, tree_image_path=tree_image_path, feature_importance_df=feature_importance_df) if path_1 else ""
            )
            return fallback_0, fallback_1

        class_0_label = self._tree_pair_class_label(path_0, class_names, 0)
        class_1_label = self._tree_pair_class_label(path_1, class_names, 1)
        heading_0 = self._tree_heading_for_label(class_0_label)
        heading_1 = self._tree_heading_for_label(class_1_label)

        conditions_0 = self._select_semantic_tree_conditions(
            self._extract_tree_path_conditions(path_0),
            class_0_label,
            feature_importance_df=feature_importance_df,
            max_conditions=max_bullets,
        )
        conditions_1 = self._select_semantic_tree_conditions(
            self._extract_tree_path_conditions(path_1),
            class_1_label,
            feature_importance_df=feature_importance_df,
            max_conditions=max_bullets,
        )

        def build_bullets(selected_conditions, class_label):
            bullets = []
            seen = set()
            for selected in selected_conditions:
                condition = selected.get("condition") or {}
                item = self._tree_business_condition_item(condition, class_label=class_label)
                key = item.get("key") or selected.get("feature_key")
                text_key = self._normalize_tree_feature_key(item.get("phrase"))
                if not item.get("phrase") or (key, text_key) in seen:
                    continue
                seen.add((key, text_key))
                sentence = self._tree_business_sentence(item, class_label)
                if sentence:
                    bullets.append(sentence)
                if len(bullets) == max_bullets:
                    break
            return bullets

        bullets_0 = build_bullets(conditions_0, class_0_label)
        bullets_1 = build_bullets(conditions_1, class_1_label)

        if not bullets_0 or not bullets_1:
            fallback_0 = self.expl_dt_pth(path_0, class_names, tree_image_path=tree_image_path, feature_importance_df=feature_importance_df) if path_0 else ""
            fallback_1 = self.expl_dt_pth(path_1, class_names, tree_image_path=tree_image_path, feature_importance_df=feature_importance_df) if path_1 else ""
            if not fallback_0 and path_0_text:
                preserved_0 = self._business_ready_bullet_lines(path_0_text)
                if preserved_0:
                    fallback_0 = "\n".join([heading_0] + preserved_0[:max_bullets])
            if not fallback_1 and path_1_text:
                preserved_1 = self._business_ready_bullet_lines(path_1_text)
                if preserved_1:
                    fallback_1 = "\n".join([heading_1] + preserved_1[:max_bullets])
            if not fallback_0:
                fallback_0 = self._weighted_tree_path_to_bullets("", class_0_label, feature_importance_df=feature_importance_df)
            if not fallback_1:
                fallback_1 = self._weighted_tree_path_to_bullets("", class_1_label, feature_importance_df=feature_importance_df)
            return fallback_0, fallback_1

        return "\n".join([heading_0] + bullets_0[:max_bullets]), "\n".join([heading_1] + bullets_1[:max_bullets])

    
    def prepare_execution_params(self):
        """Prepare parameters for model_execution"""
        
        # Build mapping for categorical targets
        mapping = None
        task_type = str(self.target_info.get('Task Type') or "").strip().lower()
        target_type = str(self.target_info.get('Target Outcome Type') or "").strip().lower()
        if target_type == 'categorical' or task_type == "classification":
            good_value = self._normalize_binary_code(self.target_info.get('Good Outcome Value'))
            bad_value = self._normalize_binary_code(self.target_info.get('Bad Outcome Value'))
            if good_value is None or bad_value is None:
                raise ValueError(
                    "Categorical SXI execution requires binary outcome codes after normalization. "
                    f"Received Good Outcome Value={self.target_info.get('Good Outcome Value')!r}, "
                    f"Bad Outcome Value={self.target_info.get('Bad Outcome Value')!r}."
                )
            mapping = {
                good_value: self.target_info['Good Outcome'],
                bad_value: self.target_info['Bad Outcome']
            }
        
        # Determine toggle direction
        target_imp = int(self.target_info.get('Target Outcome Improvement', 20))
        if abs(target_imp) == 10:
            target_imp = 20 if target_imp > 0 else -20
        toggle = "Decrease" if target_imp < 0 else "Increase"
        
        return mapping, toggle
    
    def execute(self, generate_reports=True, include_report_context=False):
        """
        Execute the full SXI pipeline
        
        Returns:
            dict: Results dictionary with all SXI outputs
        """
        self.check_cancelled()
        try:
            reset_llm_usage_tracking(request=self.request, stage="execution")

            # Load data
            report_eda_source_df = self._get_report_eda_dataframe()
            df = self.load_dataframe()
            dataset_quality_error = self._dataset_quality_error_message(df)
            if dataset_quality_error:
                return {
                    "error": True,
                    "message": dataset_quality_error,
                    "error_type": "DATASET_QUALITY",
                }
            
            # Prepare parameters
            mapping, toggle = self.prepare_execution_params()
            
            # Store buyerid in session for the SXI classes
            try:
                self.request.session['buyerid'] = self.buyerid
                self.request.session.modified = True
            except Exception:
                pass

            if generate_reports:
                self._update_report_progress(
                    status="running",
                    full_report_path="",
                    summary_report_path="",
                    error="",
                )

            self.check_cancelled()
            # import time
            # delay = 15
            # print(f"Waiting for {delay} seconds...")
            # time.sleep(delay)
            # Execute SXI
            executor = model_execution(
                request=self.request,
                dataframe=df,
                values_exe=self.target_info,
                mapping=mapping,
                toogle_val=toggle
            )
            executor.encoded_feature_mapping = dict(self.encoded_feature_mapping or {})
            executor.grouped_encoded_features = dict(self.grouped_encoded_features or {})
            
            result = executor.sxi_execution()
            if not result or result.get("fulldata_sxi") is None:
                detail = result.get("message") if isinstance(result, dict) else None
                if detail:
                    raise ValueError(f"SXI model execution failed before producing `fulldata_sxi`: {detail}")
                raise ValueError(
                    "SXI model execution did not produce `fulldata_sxi`. "
                    "Check the earlier `sxi_exe.py` traceback for the root cause."
                )
            proxy_warning = self._problem_contract_quality_warning(df)
            result["target_source"] = (
                (self.target_info or {}).get("Target Source")
                or (self.request.session.get("target_source") if self.request else None)
            )
            result["data_quality"] = (
                (self.target_info or {}).get("Data Quality")
                or (self.request.session.get("data_quality") if self.request else None)
                or {}
            )
            result["proxy_warning"] = proxy_warning
            if proxy_warning:
                print(f"[SXI] {proxy_warning}")
            sxi_avg = result.get("sxi_avg")
            fulldata_sxi = result.get("fulldata_sxi")
            report_eda_source_df = self._prepare_report_eda_dataframe(
                report_eda_source_df,
                fulldata_sxi,
            )
            curent_tv = result.get("curent_tv")
            locdist = result.get("locdist")
            selcls = result.get("selcls")
            twds = result.get("twds")
            r2 = result.get("r2")
            lty = result.get("lty")
            ltx = result.get("ltx")
            ltchg = result.get("ltchg")
            updt_corr_plt = result.get("updt_corr_plt")
            tgsxi = result.get("tgsxi")
            midsxi = result.get("midsxi")
            immediateval = result.get("immediateval")
            midtermval = result.get("midtermval")
            longtermval = result.get("longtermval")
            perchg = result.get("perchg")
            midper = result.get("midper")
            curbdcnt = result.get("curbdcnt")
            tgout = result.get("tgout")
            midout = result.get("midout")
            plotly_fig = result.get("plotly_fig")
            loc_curr_tree = result.get("loc_curr_tree")
            loc_target_tree = result.get("loc_target_tree")
            best_path_0_str = result.get("best_path_0_str")
            best_path_1_str = result.get("best_path_1_str")
            best_path_1_str_tr = result.get("best_path_1_str_tr")
            best_path_0_str_tr = result.get("best_path_0_str_tr")
            auc_best = result.get("auc_best")
            sxiacc = result.get("sxiacc")
            sxiprec = result.get("sxiprec")
            cma = result.get("cma")
            dap = result.get("dap")
            actlocs = result.get("actlocs")
            sximae = result.get("sximae")
            rltyp = result.get("rltyp")
            accind = result.get("accind")
            sxi_auc_value = result.get("sxi_auc_value")
            locauc = result.get("locauc")
            cm = result.get("cm")
            print({
                "selected_outcome_ui": self.target_info.get("Selected Outcome"),
                "selected_outcome_backend": result.get("optimization_target_class"),
                "selected_outcome_meaning": self.target_info.get("Selected Outcome Meaning"),
                "good_outcome": self.target_info.get("Good Outcome Value"),
                "bad_outcome": self.target_info.get("Bad Outcome Value"),
                "target_change": self.target_info.get("Target Outcome change") or self._target_change_from_info(),
                "optimization_target_class": result.get("optimization_target_class"),
                "slope": result.get("correlation_slope"),
                "current_sxi": sxi_avg,
                "target_sxi": tgsxi,
                "correlation_direction": rltyp,
                "iteration_selected": self.target_info.get("iteration_selected"),
            })
            edaplots = result.get("edaplots")
            sxi_within_percentage = result.get("sxi_within_percentage")
            mi_score = result.get("mi_score")
            mi_score_target = result.get("mi_score_target")
            mi_score_raw = result.get("mi_score_raw")
            if mi_score_raw is None:
                mi_score_raw = mi_score
            mi_score_target_raw = result.get("mi_score_target_raw")
            if mi_score_target_raw is None:
                mi_score_target_raw = mi_score_target
            self.encoded_feature_mapping = (
                result.get("encoded_feature_mapping")
                or self.encoded_feature_mapping
                or {}
            )
            self.grouped_encoded_features = (
                result.get("grouped_encoded_features")
                or self.grouped_encoded_features
                or {}
            )
            result["encoded_feature_mapping"] = self.encoded_feature_mapping
            result["grouped_encoded_features"] = self.grouped_encoded_features
            cols = result.get("cols")
            schema_columns = []
            if isinstance(df, pd.DataFrame):
                schema_columns.extend([str(col) for col in df.columns])
            if cols:
                schema_columns.extend([str(col) for col in cols])
            self._encoded_schema_columns = list(dict.fromkeys(schema_columns))
            r2score = result.get("r2score")
            sxirecall = result.get("sxirecall")
            combined_weights = result.get("combined_weights")
            sxi_error = result.get("error_pct")
            # print("cols",cols)
            print("r2score",r2score)
            print("sxiacc",sxiacc)
            print("sxirecall",sxirecall)
            print('SXIAUC',accind)
            print("sximae",sximae)
            print('Conf',cm)
            print('Conf Vals',cma)
            print("sxiavg",sxi_avg)
            print("fulldata_sxi",fulldata_sxi)
            print("curent_tv",curent_tv)
            print("locdist",locdist)
            print("selcls",selcls)
            print("twds",twds)
            print("r2",r2)
            print("lty",lty)
            print("ltx",ltx)
            print("ltchg",ltchg)
            print("updt_corr_plt",updt_corr_plt)
            print("tgsxi",tgsxi)
            print("midsxi",midsxi)
            print("immediateval",immediateval)
            print("midtermval",midtermval)
            print("longtermval",longtermval)
            print("perchg",perchg)
            print("midper",midper)
            print("curbdcnt",curbdcnt)
            print("tgout",tgout)
            output_dict=self.target_info
            print("output_dict",output_dict)
            print("sxi_within_percentage",sxi_within_percentage)
            tv=output_dict['Target Outcome']
            tvtype=output_dict.get('Target Outcome Type')
            task_type = str(output_dict.get('Task Type', '')).lower()
            if task_type == 'classification':
                tvtype = 'Categorical'
                if mapping is None:
                    good_value = self._normalize_binary_code(output_dict.get('Good Outcome Value'))
                    bad_value = self._normalize_binary_code(output_dict.get('Bad Outcome Value'))
                    if good_value is not None and bad_value is not None:
                        mapping = {
                            good_value: output_dict.get('Good Outcome', 'yes'),
                            bad_value: output_dict.get('Bad Outcome', 'no'),
                        }
            elif task_type in {'regression', 'time_series'}:
                tvtype = 'Numeric'
            elif str(tvtype).lower() == 'continuous':
                tvtype = 'Numeric'
            output_dict['Target Outcome Type'] = tvtype
            result["sxi_avg"] = round(sxi_avg, 3) if sxi_avg is not None else 0.0
            execution_mode = self._get_execution_mode()
            if execution_mode == "comparison_only":
                finalize_llm_usage_tracking(request=self.request)
                return self._build_comparison_only_result(result)
            print("<--------actualvspredicted--------->",actlocs)
            print('Curr Trees',loc_curr_tree, 'Target Trees',loc_target_tree)
            print('Auc Best',locauc)
            if combined_weights is not None:
                print('Combined Weights',combined_weights.head())
            else:
                print('Combined Weights: None')
            print('sxi mape error',sxi_error)
            # print("edaplot",edaplots)

            # response = pipe(prompt)[0]['generated_text']

            # sxi_avg=round(sxi_avg,2)
            sxi_avg = round(sxi_avg, 3) if sxi_avg is not None else 0.0
            # Save the model execution response as a second chat
            
            tv = output_dict['Target Outcome']
            print("Columns in fulldata_sxi:", fulldata_sxi.columns)
            # FIX-3 (add this)
            #df = fulldata_sxi

            if tvtype == 'Categorical':
                classes = list(mapping.values()) if mapping else [
                    output_dict.get('Bad Outcome', 'no'),
                    output_dict.get('Good Outcome', 'yes'),
                ]
                target_counts = fulldata_sxi[tv].value_counts()
                bad_code = self._normalize_binary_code(output_dict.get('Bad Outcome Value'))
                good_code = self._normalize_binary_code(output_dict.get('Good Outcome Value'))
                clas1 = round((target_counts.get(bad_code, 0) / len(fulldata_sxi)) * 100, 2) if len(fulldata_sxi) else 0.0
                clas2 = round((target_counts.get(good_code, 0) / len(fulldata_sxi)) * 100, 2) if len(fulldata_sxi) else 0.0
            else:
                classes = ['Below Mean', 'Above Mean']
                clas1 = 0.0
                clas2 = 0.0
            ## SXI Distribution text
            text = self.chkdelinss(tv, fulldata_sxi, sxi_avg, selcls, twds, clas1, clas2, classes)


            toi = output_dict.get('Target Outcome Improvement', 20)

            # Convert string → numeric safely
            if isinstance(toi, str):
                toi = toi.replace('%', '').strip()

            toi = float(toi)
            optimization_target_class = output_dict.get(
                "Optimization Target Class",
                output_dict.get("optimization_target_class"),
            )
            selected_meaning = str(output_dict.get("Selected Outcome Meaning") or "").strip().lower()
            selected_is_bad = selected_meaning in {"no", "bad", "negative", "fraud", "risk", "failure", "defect"}
            if output_dict['Target Outcome Type'] == 'Numeric':
                print("<--------------inside Numeric---------->")

                if optimization_target_class not in (None, ""):
                    outchos = optimization_target_class
                elif toi < 0:
                    outchos = output_dict['Bad Outcome']
                else:
                    outchos = output_dict['Good Outcome']
                improvstr = f"{abs(toi)}% {'reduction' if toi < 0 else 'increase'}"

                original_tv_col = f"{tv}_original"
                source_col = original_tv_col if original_tv_col in df.columns else tv
                current_tv = round(float(df[source_col].mean()), 2)
            else:
                print("<--------------inside Categorical---------->")
                outchos = optimization_target_class
                if outchos in (None, ""):
                    outchos = output_dict['Bad Outcome Value'] if toi < 0 else output_dict['Good Outcome Value']
                improvstr = f"{abs(toi)}% {'reduction' if toi < 0 else 'increase'}"

                current_tv = round(
                    (df[tv].value_counts().get(int(outchos), 0) / len(df)) * 100, 2
                )

            # Print or return the results
            print(f"Outcome Chosen: {outchos}")
            # print(f"c {current_tv}%")
            print(f"Improvement String: {improvstr}")
            print('Corr Del\n', text)
            if not generate_reports and not include_report_context:
                finalize_llm_usage_tracking(request=self.request)
                return self._format_results(result)
            ### chatgpt classification and regression
            send_to_llm = self.request.session.get("send_to_llm", False)
            roc_plot = None
            act_vs_pred_gpt = None
            # Initialize all variables to None
            acc = prec = recall = aucscore = random_algo = z = loc_conf = None
            gpt_mae = gpt_mape = gpt_r2 = within_percentage = None
            
            if send_to_llm and tvtype == 'Categorical':
                print('Classification')
                try:
                    acc, prec, recall, aucscore, random_algo, z, loc_conf, roc_plot, _loc_conf_html = executor.gptmlclsf()
                except Exception as e:
                    acc = prec = recall = aucscore = random_algo = z = loc_conf = roc_plot = _loc_conf_html = None
                    print(f"Error during classification: {e}")
                
                print("Accuracy (acc):", acc if acc is not None else "None")
                print("Precision (prec):", prec if prec is not None else "None")
                print("AUC Score (aucscore):", aucscore if aucscore is not None else "None")
                print("Random Algorithm (random_algo):", random_algo if random_algo is not None else "None")
                print("Z Value (z):", z if z is not None else "None")
                print(f"Accuracy: {acc:.2%}" if acc is not None else "Accuracy: None")
                print(f"Precision: {prec:.2%}" if prec is not None else "Precision: None")
                print(f"AUC: {aucscore:.2f}" if aucscore is not None else "AUC: None")
                print(f"Random Algorithm: {random_algo}, Z Value: {z}, Location: {loc_conf}")
                print("Roc plot location ==>", roc_plot if roc_plot is not None else "None")
                print("confusion matrix:", loc_conf if loc_conf is not None else "None")

            elif send_to_llm and tvtype == 'Numeric':
                print('Regression')
                try:
                    gpt_mae,gpt_mape, gpt_r2, within_percentage, random_algo, act_vs_pred_gpt = executor.gptmlreg()
                except Exception as e:
                    gpt_mae = gpt_mape = gpt_r2 = within_percentage = random_algo = act_vs_pred_gpt = None
                    print(f"Error during regression: {e}")
                
                # gpt_mae = 325.4567 if gpt_mae is not None else None
                # gpt_r2 = 0.7890 if gpt_r2 is not None else None
                # within_percentage = 76.54 if within_percentage is not None else None
                # random_algo = "Linear Regression" if random_algo is not None else None
                # act_vs_pred_gpt = [[1000, 1200], [1100, 1150]] if act_vs_pred_gpt is not None else None 
                
                print("Mean Absolute Error (MAE):", gpt_mae if gpt_mae is not None else "None")
                print("Mean Absolute Percentage Error (MAPE):", gpt_mape if gpt_mape is not None else "None")
                print("R-squared (r2):", gpt_r2 if gpt_r2 is not None else "None")
                print("Within Percentage:", within_percentage if within_percentage is not None else "None")
                print("Random Algorithm:", random_algo if random_algo is not None else "None")
                print("confusion matrix:", act_vs_pred_gpt if act_vs_pred_gpt is not None else "None")
                print(f"MAE: {gpt_mae:.2f}" if gpt_mae is not None else "MAE: None")
                print(f"R-squared: {gpt_r2:.2f}" if gpt_r2 is not None else "R-squared: None")
                print(f"Within Percentage: {within_percentage:.2f}" if within_percentage is not None else "Within Percentage: None")

            if output_dict['Target Outcome Type'] == 'Numeric':
                if optimization_target_class not in (None, ""):
                    outchosclass = output_dict.get('Bad Outcome') if selected_is_bad else output_dict.get('Good Outcome')
                    outchos = optimization_target_class

                    original_tv_col = f"{tv}_original"
                    source_col = original_tv_col if original_tv_col in df.columns else tv
                    current_tv = round(float(df[source_col].mean()), 2)

                    improvstr = f"{abs(toi)}% {'reduction' if toi < 0 else 'increase'}"
                    change = 'Decrease' if toi < 0 else 'Increase'

                elif toi < 0:
                    outchosclass = output_dict['Bad Outcome']
                    outchos = output_dict['Bad Outcome Value']

                    # if outchosclass.lower() in ['below mean', 'below_mean']:
                    #     current_tv = round((df[df[tv] < df[tv].mean()].shape[0] / len(df)) * 100, 2)
                    # elif outchosclass.lower() in ['above mean', 'above_mean']:
                    original_tv_col = f"{tv}_original"
                    source_col = original_tv_col if original_tv_col in df.columns else tv
                    current_tv = round(float(df[source_col].mean()), 2)

                    improvstr = f"{abs(toi)}% reduction"
                    change = 'Decrease'

                else:
                    outchosclass = output_dict['Good Outcome']
                    outchos = output_dict['Good Outcome Value']

                    # if outchosclass.lower() in ['below mean', 'below_mean']:
                    #     current_tv = round((df[df[tv] < df[tv].mean()].shape[0] / len(df)) * 100, 2)
                    # elif outchosclass.lower() in ['above mean', 'above_mean']:
                    original_tv_col = f"{tv}_original"
                    source_col = original_tv_col if original_tv_col in df.columns else tv
                    current_tv = round(float(df[source_col].mean()), 2)

                    improvstr = f"{abs(toi)}% increase"
                    change = 'Increase'

            else:
                if optimization_target_class not in (None, ""):
                    outchos = optimization_target_class
                    outchosclass = output_dict.get('Bad Outcome') if selected_is_bad else output_dict.get('Good Outcome')

                    current_tv = round(
                        (df[tv].value_counts().get(int(outchos), 0) / len(df)) * 100, 2
                    )

                    improvstr = f"{abs(toi)}% {'reduction' if toi < 0 else 'increase'}"
                    change = 'Decrease' if toi < 0 else 'Increase'

                elif toi < 0:
                    outchos = output_dict['Bad Outcome Value']
                    outchosclass = output_dict['Bad Outcome']

                    current_tv = round(
                        (df[tv].value_counts().get(int(outchos), 0) / len(df)) * 100, 2
                    )

                    improvstr = f"{abs(toi)}% reduction"
                    change = 'Decrease'

                else:
                    outchos = output_dict['Good Outcome Value']
                    outchosclass = output_dict['Good Outcome']

                    current_tv = round(
                        (df[tv].value_counts().get(int(outchos), 0) / len(df)) * 100, 2
                    )

                    improvstr = f"{abs(toi)}% increase"
                    change = 'Increase'
            
            print(f"Outcome Chosen: {outchos}")
            print(f"Outcome Class: {outchosclass}")
            print(
                "Current Target Value: "
                f"{format_display_value(current_tv, tv, output_dict['Target Outcome Type'], decimals=2)}"
            )
            print(f"Improvement String: {improvstr}")
            print(f"Change Type: {change}")

            print('sxiacc, sxiprec, sxirecall, cma, dap, actlocs, sxiacc, sximae:',sxiacc, sxiprec,sxirecall, cma, actlocs, accind, sximae)
            
            rand = random.randint(1, 9999999)

            def normalize_report_path(value, pick_first=False):
                if pick_first and isinstance(value, (list, tuple)):
                    value = value[0] if value else None
                if value in (None, "", "None"):
                    return None
                return str(value).replace('\\', '/')

            # --- Setup ---
            # Keep plots and CSV artifacts in the per-session output directory.
            PLOT_DIR = str(self.output_dir)
            updt_corr_plt = normalize_report_path(updt_corr_plt)
            print('updt_corr_plt:', updt_corr_plt)

            edaplots = normalize_report_path(edaplots, pick_first=True)
            print('edaplots:', edaplots)

            loc_target_tree = normalize_report_path(loc_target_tree)
            print('loc_target_tree:', loc_target_tree)

            loc_curr_tree = normalize_report_path(loc_curr_tree)
            print('loc_curr_tree:', loc_curr_tree)

            locdist = normalize_report_path(locdist)
            roc_plot = normalize_report_path(roc_plot)
            locauc = normalize_report_path(locauc)
            cm = normalize_report_path(cm)
            auc_best = normalize_report_path(auc_best)
            actlocs = normalize_report_path(actlocs)

            data_columns = cols
            column_name = "Features"

            blue_banner = self._get_report_asset("blue.png")
            sriya_logo = self._get_report_asset("Sriya_new_logo.png")
            logo_path = sriya_logo
            print("logo_path:", logo_path)
            print("blue_banner:", blue_banner)

            first_row = [[column_name]]
            user_data = [data_columns[i:i + 2] for i in range(0, len(data_columns), 2)]
            table_data = first_row + user_data

            target_variable_name = output_dict['Target Outcome']
            tv_type = tvtype
            target_count = df[target_variable_name].value_counts()
            tv_class = classes

            target_mean = None
            model_accuracy = None
            precision_score = None
            area_under_curve = None
            confusion_matrix = None
            MAE_Perf = None
            R2_Perf = None
            Error_Perf = None
            improvementcard3 = None
            improvementcardvalue3 = None
            improvementcard6 = None
            improvementcardvalue6 = None
            Accuracy_lnm = None
            Precision_lnm = None
            Recall_lnm = None
            MAE_lnm = None
            R2_score_lnm = None
            area_under_curve_lnm = None
            Actual_fig=None
            sxiauc=None
            Recall_score=None

            # --- Target Mean / Summary ---
            if tv_type == "Categorical":
                formatted_counts = []
                target_count_lookup = target_count.to_dict()
                categorical_pairs = []
                good_label = self.target_info.get('Good Outcome')
                bad_label = self.target_info.get('Bad Outcome')
                good_value = self._normalize_binary_code(self.target_info.get('Good Outcome Value'))
                bad_value = self._normalize_binary_code(self.target_info.get('Bad Outcome Value'))

                if good_label is not None:
                    categorical_pairs.append((good_label, good_value))
                if bad_label is not None:
                    categorical_pairs.append((bad_label, bad_value))

                if not categorical_pairs:
                    categorical_pairs = [(class_label, None) for class_label in tv_class[:2]]

                for class_label, class_code in categorical_pairs:
                    class_count_value = target_count_lookup.get(class_label)
                    if class_count_value is None and class_code is not None:
                        class_count_value = target_count_lookup.get(class_code)
                    class_count_value = int(class_count_value) if class_count_value is not None else 0
                    formatted_counts.append(f"{class_label}: {class_count_value:,}")

                target_mean = " & ".join(formatted_counts) if formatted_counts else "Not available"
            else:
                target_mean = df[target_variable_name].mean()

            # --- Performance Change Interpretation ---
            perce_inc = abs(perchg)
            bad_outcome = output_dict['Bad Outcome']
            good_outcome = output_dict['Good Outcome']

            tv_inc = change.lower()

            corr_plot = updt_corr_plt
            dist_plot = locdist
            eda_path = edaplots
            roc_plot = roc_plot if tvtype == 'Categorical' else None    
            curr_tree = loc_curr_tree
            targ_tree = loc_target_tree
            loc_auc = locauc if tvtype == 'Categorical' else None
            c_m = cm if tvtype == 'Categorical' else None
            print('Conf',cm)
            print('Conf Vals',cma)

            if tvtype == 'Numeric':
                print("<-------actual vs predicted1-------->")  
                print(actlocs)
                print(act_vs_pred_gpt)
            else:
                pass

            import unicodedata
            
            tveda = eda_path
            distplot = dist_plot

            plot_target_dist = tveda
            plot_vrm_dist = distplot
            Targetdt = targ_tree       
            Currentdt = curr_tree
            Actual_fig=auc_best

            # --- Model Performance Metrics (Chatgpt) ---
            if tv_type == "Categorical":
                model_accuracy = acc
                precision_score = prec
                Recall_score = recall
                area_under_curve = aucscore
                confusion_matrix = cm
            else:
                MAE_Perf = gpt_mae
                R2_Perf = round(float(gpt_r2),3) if gpt_r2 is not None else 0
                Error_Perf = gpt_mape

            AUC = auc_best  # Common AUC plot
            confusion_matrix = cm

            ClassRate = tv_class[1]
            if tv_type == "Categorical":
                CurrentRate = df[tv].value_counts()[int(outchos)] / len(df) * 100
            else:
                original_tv_col = f"{tv}_original"
                source_col = original_tv_col if original_tv_col in fulldata_sxi.columns else tv
                CurrentRate = round(float(fulldata_sxi[source_col].mean()), 3)
            # Good_Outcome_outchos = output_dict['Good Outcome Value']
            # Good_Outcome_Rate = df[tv].value_counts()[int(Good_Outcome_outchos)]/len(df) * 100
            print("Current Rate:", CurrentRate)
            # print("Good Outcome:", Good_Outcome_Rate)
            CurrentSXI = sxi_avg

            value1 = round(tgout,3)
            value2 = round(midout,3)
            value3 = round(lty,3)  # assumed static
            if tv_type == "Numeric" and CurrentRate not in (None, 0):
                perc1 = round(abs(((value1 - CurrentRate) / CurrentRate) * 100), 3)
                perc2 = round(abs(((value2 - CurrentRate) / CurrentRate) * 100), 3)
                perc3 = round(abs(((value3 - CurrentRate) / CurrentRate) * 100), 3)
            else:
                perc1 = round(abs(perchg), 3)
                perc2 = round(abs(midper), 3)
                perc3 = round(abs(ltchg), 3)
            # print(perc1,perc2,perc3)
            if output_dict['Target Outcome Type'] == 'Numeric':
                # Handle Numeric targets dynamically based on Bad Outcome and Good Outcome
                if toi < 0:
                    change_type = 'Decreased'
                else:
                    change_type = 'Increased'
            else:
                # Handle categorical targets
                if toi < 0:
                    change_type = 'Reduced'
                else:
                    change_type = 'Increased'

            tv_inc = change_type.lower()
            is_reduction_case = self._is_reduction_direction(tv_inc)
            improvement_direction = "decrease" if is_reduction_case else "increase"
            movement_word = "DOWN" if is_reduction_case else "UP"
            action_verb = "reduce" if is_reduction_case else "increase"
            # tv_inc = change_type  # already lowercase

            if tv_type == "Categorical":
                if tv_inc in ["decreased", "reduced"]:
                    title = f"{ClassRate.capitalize()} Risk Reduction"
                else:
                    title = f"{ClassRate.capitalize()} Success Score"
            else:
                if tv_inc == "decreased":
                    title = f"{target_variable_name.capitalize()} Risk Reduction"
                else:
                    title = f"{target_variable_name.capitalize()} Success Score"

            risk_index = 'bank fraud risk index' if tv_type == "Categorical" else 'customer lifetime value risk index'

            print("Risk Index:", risk_index)
            print("Title:", title)

            title = title
            val = risk_index

            # --- Target Status ---
            name_imp = f"{tv} {change_type.lower()}"
            init = "Immediate"
            mid = "Mid-Term"
            end = "Long-Term"

            conv = target_variable_name
            sxi1 = tgsxi
            sxi2 = midsxi
            sxi3 = ltx

            # --- Improvement Cards ---
            improvementcard1 = "Current DXI"
            improvementcardvalue1 = self._format_optional_float(sxi_avg, 2)

            improvementcard2 = f"Current {ClassRate}"
            improvementcardvalue2 = self._format_optional_float(CurrentRate, 2)

            normalized_r2score = normalize_metric_score(r2score) if r2score is not None else None

            if tv_type == "Categorical":
                improvementcard3 = f"{ClassRate} Accuracy"
                improvementcardvalue3 = self._format_optional_float(sxiacc, 2)
                improvementcard6 = f"{ClassRate} Precision"
                improvementcardvalue6 = self._format_optional_float(sxiprec, 3)
            else:
                improvementcard3 = f"{ClassRate} R² Score"
                improvementcardvalue3 = self._format_optional_float(normalized_r2score, 3)
                improvementcard6 = f"{ClassRate} MAE"
                improvementcardvalue6 = self._format_optional_float(sximae, 3)

            improvementcard4 = "Target DXI"
            improvementcardvalue4 = self._format_optional_float(tgsxi, 3)

            improvementcard5 = f"Target {ClassRate}"
            improvementcardvalue5 = self._format_optional_float(tgout, 3)

            # --- LNM Performance Metrics (SXI) ---
            if tv_type == "Categorical":
                Accuracy_lnm = self._normalize_classification_percentage(sxiacc)
                Precision_lnm = self._normalize_classification_percentage(sxiprec)
                Recall_lnm = self._normalize_classification_percentage(sxirecall)
                area_under_curve_lnm = self._coerce_summary_float(
                    sxi_auc_value if sxi_auc_value is not None else accind
                )
            else:
                MAE_lnm = self._coerce_summary_float(sximae)
                R2_score_lnm = normalized_r2score
                #area_under_curve_lnm = AUC
                area_under_curve_lnm = sxi_error
            mi_score = self._normalize_feature_importance_dataframe(
                mi_score,
                required_columns=["Feature", "Importance", "SXI_Weights"],
                aliases={
                    "Feature": ["Features", "feature", "Variable", "Column", "index"],
                    "Importance": ["RF_Importance", "Tree Importance", "Tree_Importance", "MI Score", "MI_Score", "value"],
                    "SXI_Weights": ["SXI_Weight", "SXI Weight"],
                },
                label="current weight",
            )
            mi_score_target = self._normalize_feature_importance_dataframe(
                mi_score_target,
                required_columns=["Feature", "RF_Importance", "SXI_Weight"],
                aliases={
                    "Feature": ["Features", "feature", "Variable", "Column", "index"],
                    "RF_Importance": ["Importance", "Tree Importance", "Tree_Importance", "MI Score", "MI_Score", "value"],
                    "SXI_Weight": ["SXI_Weights", "SXI Weight"],
                },
                label="target weight",
            )
            combined_weights = self._normalize_feature_importance_dataframe(
                combined_weights,
                required_columns=["Feature", "Lasso_Weight", "MI_Weight", "PCA_Weight", "NB_Weight", "XGB_Weight"],
                aliases={
                    "Feature": ["Features", "feature", "Variable", "Column", "index"],
                    "Lasso_Weight": ["Lasso Weight", "Lasso"],
                    "MI_Weight": ["MI Weight", "MI"],
                    "PCA_Weight": ["PCA Weight", "PCA"],
                    "NB_Weight": ["NB Weight", "Naive Bayes", "Naive_Bayes"],
                    "XGB_Weight": ["XGB Weight", "XGBoost", "XG Boost", "XGB"],
                },
                label="combined algorithm weight",
            )
            result["mi_score"] = mi_score
            result["mi_score_target"] = mi_score_target
            result["combined_weights"] = combined_weights

            self.output_dir.mkdir(parents=True, exist_ok=True)
            if not mi_score.empty:
                mi_score.to_csv(self.output_dir / "current_weights.csv", index=False)
            if not mi_score_target.empty:
                mi_score_target.to_csv(self.output_dir / "target_weights.csv", index=False)
            # combined_weights.to_csv(self.output_dir / "combined_weights.csv", index=False)

            mi_score_sorted = mi_score.copy()
            if isinstance(mi_score_sorted, pd.DataFrame) and not mi_score_sorted.empty:
                feature_labels = mi_score_sorted["Feature"]
                mi_score_sorted = mi_score_sorted[
                    feature_labels.notna() & feature_labels.astype(str).str.strip().ne("")
                ].copy()
                sort_cols = [col for col in ['Importance', 'SXI_Weights'] if col in mi_score_sorted.columns]
                if sort_cols:
                    mi_score_sorted = mi_score_sorted.sort_values(
                        by=sort_cols,
                        ascending=[False] * len(sort_cols),
                    ).reset_index(drop=True)

            top_n = min(5, len(mi_score_sorted))
            top_features = mi_score_sorted['Feature'].head(top_n).tolist()
            top_values = mi_score_sorted['Importance'].head(top_n).tolist()
            top_values_sxi = mi_score_sorted['SXI_Weights'].head(top_n).tolist()
            # algo1=combined_weights.iloc[0,0]
            # algo2=combined_weights.iloc[1,0]
            # algo3=combined_weights.iloc[2,0]
            # algo4=combined_weights.iloc[3,0]
            # algo5=combined_weights.iloc[4,0]
            # val1 = combined_weights.iloc[0, 1]
            # val2 = combined_weights.iloc[1, 1]
            # val3 = combined_weights.iloc[2, 1]
            # val4 = combined_weights.iloc[3, 1]
            # val5 = combined_weights.iloc[4, 1]
            # algo_value_map = {
            #     algo1: val1,
            #     algo2: val2,
            #     algo3: val3,
            #     algo4: val4,
            #     algo5: val5
            # }
            # matched_output = []

            # for feature in top_features:
            #     for algo, val in algo_value_map.items():
            #         matched_output.append({
            #             'Feature': feature,
            #             'Algorithm': algo,
            #             'Value': val
            #         })
            # matched_df = pd.DataFrame(matched_output)
            # SAFELY FILTER COMBINED WEIGHTS
            matched_df = combined_weights[
                combined_weights['Feature'].isin(top_features)
            ].copy()
            print('ALGOS and VALS \n', matched_df)
            
            max_features = 5
            df_weights = matched_df.head(max_features).copy().reset_index(drop=True)

            # Pad rows if less than 5
            while len(df_weights) < max_features:
                df_weights.loc[len(df_weights)] = [None] * len(df_weights.columns)

            lasso_vals = df_weights['Lasso_Weight'].tolist()
            mi_vals    = df_weights['MI_Weight'].tolist()
            pca_vals   = df_weights['PCA_Weight'].tolist()
            nb_vals    = df_weights['NB_Weight'].tolist()
            xgb_vals   = df_weights['XGB_Weight'].tolist()
            print('TOP FEATURES \n', top_features)
            print('Lasso Vals \n', lasso_vals)
            print('MI Vals \n', mi_vals)

            # Pad with None if fewer than 5
            while len(top_features) < 5:
                top_features.append(None)
                top_values.append(None)

            # Assign to variables
            feature1, feature2, feature3, feature4, feature5 = top_features
            featurevalue1, featurevalue2, featurevalue3, featurevalue4, featurevalue5 = top_values


            # Step 2: Get top 5 features and values for secondary target
            mi_score_target_sorted = mi_score_target.copy()
            if isinstance(mi_score_target_sorted, pd.DataFrame) and not mi_score_target_sorted.empty:
                feature_labels_target = mi_score_target_sorted["Feature"]
                mi_score_target_sorted = mi_score_target_sorted[
                    feature_labels_target.notna() & feature_labels_target.astype(str).str.strip().ne("")
                ].copy()
                sort_cols_target = [col for col in ['RF_Importance', 'SXI_Weight'] if col in mi_score_target_sorted.columns]
                if sort_cols_target:
                    mi_score_target_sorted = mi_score_target_sorted.sort_values(
                        by=sort_cols_target,
                        ascending=[False] * len(sort_cols_target),
                    ).reset_index(drop=True)

            top_n_target = min(5, len(mi_score_target_sorted))
            top_features_target = mi_score_target_sorted['Feature'].head(top_n_target).tolist()
            top_val_tv_sxi = mi_score_target_sorted['SXI_Weight'].head(top_n_target).tolist()
            top_values_target = mi_score_target_sorted['RF_Importance'].head(top_n_target).tolist()
            # algo1=combined_weights.iloc[0,0]
            # algo2=combined_weights.iloc[1,0]
            # algo3=combined_weights.iloc[2,0]
            # algo4=combined_weights.iloc[3,0]
            # algo5=combined_weights.iloc[4,0]
            # val1 = combined_weights.iloc[0, 1]
            # val2 = combined_weights.iloc[1, 1]
            # val3 = combined_weights.iloc[2, 1]
            # val4 = combined_weights.iloc[3, 1]
            # val5 = combined_weights.iloc[4, 1]
            # algo_value_map = {
            #     algo1: val1,
            #     algo2: val2,
            #     algo3: val3,
            #     algo4: val4,
            #     algo5: val5
            # }
            # matched_output_tr = []

            # for feature in top_features_target:
            #     for algo, val in algo_value_map.items():
            #         matched_output_tr.append({
            #             'Feature': feature,
            #             'Algorithm': algo,
            #             'Value': val
            #         })
            # matched_df_tr = pd.DataFrame(matched_output_tr)
            matched_df_tr = combined_weights[
                combined_weights['Feature'].isin(top_features_target)
            ].copy()
            print('ALGOS and VALS TR \n', matched_df_tr)
            max_features = 5
            df_weights_tr = matched_df_tr.head(max_features).copy().reset_index(drop=True)

            # Pad rows if less than 5
            while len(df_weights_tr) < max_features:
                df_weights_tr.loc[len(df_weights_tr)] = [None] * len(df_weights_tr.columns)

            lasso_vals_tr = df_weights_tr['Lasso_Weight'].tolist()
            mi_vals_tr    = df_weights_tr['MI_Weight'].tolist()
            pca_vals_tr   = df_weights_tr['PCA_Weight'].tolist()
            nb_vals_tr    = df_weights_tr['NB_Weight'].tolist()
            xgb_vals_tr   = df_weights_tr['XGB_Weight'].tolist()
            print('TOP FEATURES TR \n', top_features_target)
            print('Lasso Vals TR \n', lasso_vals_tr)
            print('MI Vals TR \n', mi_vals_tr)
            print('PCA Vals TR \n', pca_vals_tr)
            print('NB Vals TR \n', nb_vals_tr)
            print('SXI Wt TR \n', featurevalue1, featurevalue2, featurevalue3, featurevalue4, featurevalue5)

            # Pad with None if fewer than 5
            while len(top_features_target) < 5:
                top_features_target.append(None)
                top_values_target.append(None)

            # Assign to variables
            feature6, feature7, feature8, feature9, feature10 = top_features_target
            featurevalue6, featurevalue7, featurevalue8, featurevalue9, featurevalue10 = top_values_target

            name2 = 'Report_prof' + str(rand)
            name1='Report_user' + str(rand)
            name_2 = name2 + '.pdf'
            name_1 = name1 + '.pdf'
            os.makedirs(f'media/files/chatbot/{self.buyerid}/pdf/', exist_ok=True)
            pdf_loc2 = f'media/files/chatbot/{self.buyerid}/pdf/{name2}.pdf'
            #pdf_loc1 = f'media/files/chatbot/{self.buyerid}/pdf/{name_1}'
            import uuid

            # Create unique folder inside MEDIA_ROOT/reports/
            unique_folder = str(uuid.uuid4())
            report_folder = os.path.join(settings.MEDIA_ROOT,"reports",unique_folder)

            os.makedirs(report_folder, exist_ok=True)
            pdf_loc1 = os.path.join(report_folder, name_1)
            summary_pdf_loc = os.path.join(report_folder, f"{name1}_summary.pdf")
            
            # fpdf.output(pdf_loc, "F")
            eda1 = ''
        
            step_start = time.perf_counter()
            print("Starting generate_sxi_distribution_interpretation")
            before_outlier_eda_df = report_eda_source_df
            after_outlier_eda_df = self._get_after_outlier_dataframe()
            if after_outlier_eda_df.empty:
                after_outlier_eda_df = df
            valid_bullets = self.generate_sxi_distribution_interpretation(
                before_outlier_eda_df,
                target_variable_name,
                tv_type=tv_type,
                target_count_summary=target_mean,
                current_sxi=CurrentSXI,
                after_outlier_df=after_outlier_eda_df,
                sxi_df=fulldata_sxi,
            )#,client=client
            print(f"generate_sxi_distribution_interpretation took {time.perf_counter() - step_start:.2f}s")
 
            step_start = time.perf_counter()
            print("Starting Corr_explanation")
            Corr_explanation = self.generate_corr_intprt(
                        target_variable=target_variable_name,
                        current_score=sxi_avg,
                        current_rate=CurrentRate,
                        mid_score=midsxi,
                        mid_rate=midtermval,
                        long_score=ltx,
                        long_rate=lty,
                        target_type=tv_type,
                        # model_name="gpt-4",  # or "gpt-3.5-turbo"
                        # client=client,
                        )
            print(f"Corr_explanation took {time.perf_counter() - step_start:.2f}s")

            step_start = time.perf_counter()
            print("Starting current decision-tree business interpretation pair")
            decision_0_current, decision_1_current = self._tree_business_interpretation_pair(
                best_path_0_str,
                best_path_1_str,
                classes,
                feature_importance_df=mi_score_raw,
                tree_image_path=Currentdt,
            )
            print(f"current decision-tree business interpretation pair took {time.perf_counter() - step_start:.2f}s")

            step_start = time.perf_counter()
            print("Starting target decision-tree business interpretation pair")
            decision_0_tr, decision_1_tr = self._tree_business_interpretation_pair(
                best_path_0_str_tr,
                best_path_1_str_tr,
                classes,
                feature_importance_df=mi_score_target_raw,
                tree_image_path=Targetdt,
            )
            print(f"target decision-tree business interpretation pair took {time.perf_counter() - step_start:.2f}s")

            preferred_label = output_dict.get('Good Outcome') or output_dict.get('Bad Outcome')
            if tvtype == 'Categorical':
                if self._labels_match(self._extract_tree_class_label(best_path_1_str, classes), preferred_label):
                    decision_0_current, decision_1_current = decision_1_current, decision_0_current
                if self._labels_match(self._extract_tree_class_label(best_path_1_str_tr, classes), preferred_label):
                    decision_0_tr, decision_1_tr = decision_1_tr, decision_0_tr

            correlation_section_text = self._generate_correlation_text(
                tv_type=tv_type,
                target_variable_name=target_variable_name,
                metric_val=self._resolve_correlation_metric_value(tv_type=tv_type, result=result),
                slope_val=self._resolve_correlation_slope(result=result),
                current_target_value=CurrentRate,
                corr_plot_path=corr_plot,
                desired_target_direction=improvement_direction,
            )
            report_section_context = self._build_report_section_context(
                plot_target_dist=plot_target_dist,
                plot_vrm_dist=plot_vrm_dist,
                edaparagraph1=eda1,
                edaparagraph2=valid_bullets,
                corr_plot=corr_plot,
                Corr_explanation=Corr_explanation,
                correlation_section_text=correlation_section_text,
                Currentdt=Currentdt,
                CurrentRate=CurrentRate,
                CurrentSXI=CurrentSXI,
                target_mean=target_mean,
                perc1=perc1,
                perc2=perc2,
                perc3=perc3,
                value1=value1,
                value2=value2,
                value3=value3,
                sxi1=sxi1,
                sxi2=sxi2,
                sxi3=sxi3,
                R2_score_lnm=R2_score_lnm,
                Accuracy_lnm=Accuracy_lnm,
                feature1=feature1,
                feature2=feature2,
                feature3=feature3,
                feature4=feature4,
                feature5=feature5,
                feature6=feature6,
                feature7=feature7,
                feature8=feature8,
                feature9=feature9,
                feature10=feature10,
                featurevalue1=featurevalue1,
                featurevalue2=featurevalue2,
                featurevalue3=featurevalue3,
                featurevalue4=featurevalue4,
                featurevalue5=featurevalue5,
                featurevalue6=featurevalue6,
                featurevalue7=featurevalue7,
                featurevalue8=featurevalue8,
                featurevalue9=featurevalue9,
                featurevalue10=featurevalue10,
                Targetdt=Targetdt,
                decision_0_current=decision_0_current,
                decision_1_current=decision_1_current,
                decision_0_tr=decision_0_tr,
                decision_1_tr=decision_1_tr,
            )
            if execution_mode == "report_context_only":
                finalize_llm_usage_tracking(request=self.request)
                return make_json_safe({
                    "success": True,
                    "sxi_score": result.get("sxi_avg"),
                    "model_performance": self._get_performance_metrics(result),
                    "report_context": report_section_context,
                })
            report_section_overrides = {}
            if self.request.session.get("sxi_report_use_original_sections"):
                report_section_overrides = dict(self.request.session.get("sxi_original_report_context") or {})
            
            step_start = time.perf_counter()
            print("Starting generate_pdf_prof")
            self.generate_pdf_prof(
                filename=pdf_loc2,
                target_variable_name=target_variable_name,
                target_mean=target_mean,
                table_data=table_data,
                sriya_logo=sriya_logo,
                plot_target_dist=plot_target_dist,
                plot_vrm_dist=plot_vrm_dist,
                perce_inc=perce_inc,
                tv_inc=tv_inc,
                bad_outcome=bad_outcome,
                good_outcome=good_outcome,
                Targetdt=Targetdt,
                Currentdt=Currentdt,
                risk_index=risk_index,
                blue_banner=blue_banner,
                tv_type=tv_type,
                model_accuracy=model_accuracy,
                precision_score=precision_score,
                area_under_curve=area_under_curve,
                MAE_Perf=MAE_Perf,
                R2_Perf=R2_Perf,
                Error_Perf=Error_Perf,
                Actual_fig=Actual_fig,
                AUC=AUC,
                confusion_matrix=confusion_matrix,
                ClassRate=ClassRate,
                CurrentRate=CurrentRate,
                CurrentSXI=CurrentSXI,
                perc1=perc1,
                perc2=perc2,
                perc3=perc3,
                value1=value1,
                value2=value2,
                value3=value3,
                name=name_imp,
                init=init,
                mid=mid,
                end=end,
                conv=conv,
                sxi1=sxi1,
                sxi2=sxi2,
                sxi3=sxi3,
                improvementcard1=improvementcard1,
                improvementcard2=improvementcard2,
                improvementcardvalue2=improvementcardvalue2,
                improvementcard3=improvementcard3,
                improvementcardvalue3=improvementcardvalue3,
                improvementcard4=improvementcard4,
                improvementcardvalue4=improvementcardvalue4,
                improvementcard5=improvementcard5,
                improvementcardvalue5=improvementcardvalue5,
                improvementcard6=improvementcard6,
                improvementcardvalue6=improvementcardvalue6,
                MAE_lnm=MAE_lnm,
                R2_score_lnm=R2_score_lnm,
                area_under_curve_lnm=area_under_curve_lnm,
                Accuracy_lnm=Accuracy_lnm,
                Precision_lnm=Precision_lnm,
                Recall_lnm=Recall_lnm,
                Recall_score=Recall_score,
                feature1=feature1,
                feature2=feature2,
                feature3=feature3,
                feature4=feature4,
                feature5=feature5,
                feature6=feature6,
                feature7=feature7,
                feature8=feature8,
                feature9=feature9,
                feature10=feature10,
                featurevalue1=featurevalue1,
                featurevalue2=featurevalue2,
                featurevalue3=featurevalue3,
                featurevalue4=featurevalue4,
                featurevalue5=featurevalue5,
                featurevalue6=featurevalue6,
                featurevalue7=featurevalue7,
                featurevalue8=featurevalue8,
                featurevalue9=featurevalue9,
                featurevalue10=featurevalue10,
                corr_plot=corr_plot,
                edaparagraph1=eda1,
                edaparagraph2=valid_bullets,
                decision_0_current=decision_0_current,
                decision_1_current=decision_1_current,
                decision_0_tr=decision_0_tr,
                decision_1_tr=decision_1_tr,
                title=title,
                Corr_explanation=Corr_explanation,
                report_section_overrides=report_section_overrides,

                lasso_vals_tr=lasso_vals_tr,
                mi_vals_tr=mi_vals_tr,
                pca_vals_tr=pca_vals_tr,
                nb_vals_tr=nb_vals_tr,
                xgb_vals_tr=xgb_vals_tr,

                lasso_vals=lasso_vals,
                mi_vals=mi_vals,
                pca_vals=pca_vals,
                nb_vals=nb_vals,
                xgb_vals=xgb_vals,
                
                top_val_tv_sxi=top_val_tv_sxi,
                top_values_sxi=top_values_sxi,
                # feat1=feat1,
                # feat2=feat2,
                # feat3=feat3,
                # feat4=feat4,
                # feat5=feat5,

                # prcnt1=prcnt1,
                # prcnt2=prcnt2,
                # prcnt3=prcnt3,
                # prcnt4=prcnt4,
                # prcnt5=prcnt5,
                
                # coef1=coef1,
                # coef2=coef2,
                # coef3=coef3,
                # coef4=coef4,
                # coef5=coef5,
                
                # vv1=vv1,
                # vv2=vv2,
                # vv3=vv3,
                # vv4=vv4,
                # vv5=vv5,
                
                # v1=v1,
                # v2=v2,  
                # v3=v3,
                # v4=v4,
                # v5=v5,
                # prediff=prediff,
                # adjusted_target_value=prednew,
                # first_target_value=predold,
                # prediffperc=prediffperc,
            )
            print(f"generate_pdf_prof took {time.perf_counter() - step_start:.2f}s")

            step_start = time.perf_counter()
            print("Starting generate_pdf_prof_user")
            self.generate_pdf_prof_user(
                filename=pdf_loc1,
                actlocs=actlocs,
                result=result,
                combined_weights=combined_weights,
                target_variable_name=target_variable_name,
                target_mean=target_mean,
                table_data=table_data,
                sriya_logo=sriya_logo,
                plot_target_dist=plot_target_dist,
                plot_vrm_dist=plot_vrm_dist,
                perce_inc=perce_inc,
                tv_inc=tv_inc,
                bad_outcome=bad_outcome,
                good_outcome=good_outcome,
                Targetdt=Targetdt,
                Currentdt=Currentdt,
                risk_index=risk_index,
                blue_banner=blue_banner,
                tv_type=tv_type,
                model_accuracy=model_accuracy,
                precision_score=precision_score,
                area_under_curve=area_under_curve,
                MAE_Perf=MAE_Perf,
                R2_Perf=R2_Perf,
                Error_Perf=Error_Perf,
                Actual_fig=Actual_fig,
                AUC=AUC,
                confusion_matrix=confusion_matrix,
                ClassRate=ClassRate,
                CurrentRate=CurrentRate,
                CurrentSXI=CurrentSXI,
                perc1=perc1,
                perc2=perc2,
                perc3=perc3,
                value1=value1,
                value2=value2,
                value3=value3,
                name=name_imp,
                init=init,
                mid=mid,
                end=end,
                conv=conv,
                sxi1=sxi1,
                sxi2=sxi2,
                sxi3=sxi3,
                improvementcard1=improvementcard1,
                improvementcard2=improvementcard2,
                improvementcardvalue2=improvementcardvalue2,
                improvementcard3=improvementcard3,
                improvementcardvalue3=improvementcardvalue3,
                improvementcard4=improvementcard4,
                improvementcardvalue4=improvementcardvalue4,
                improvementcard5=improvementcard5,
                improvementcardvalue5=improvementcardvalue5,
                improvementcard6=improvementcard6,
                improvementcardvalue6=improvementcardvalue6,
                MAE_lnm=MAE_lnm,
                R2_score_lnm=R2_score_lnm,
                area_under_curve_lnm=area_under_curve_lnm,
                Accuracy_lnm=Accuracy_lnm,
                Precision_lnm=Precision_lnm,
                Recall_lnm=Recall_lnm,
                Recall_score=Recall_score,
                feature1=feature1,
                feature2=feature2,
                feature3=feature3,
                feature4=feature4,
                feature5=feature5,
                feature6=feature6,
                feature7=feature7,
                feature8=feature8,
                feature9=feature9,
                feature10=feature10,
                featurevalue1=featurevalue1,
                featurevalue2=featurevalue2,
                featurevalue3=featurevalue3,
                featurevalue4=featurevalue4,
                featurevalue5=featurevalue5,
                featurevalue6=featurevalue6,
                featurevalue7=featurevalue7,
                featurevalue8=featurevalue8,
                featurevalue9=featurevalue9,
                featurevalue10=featurevalue10,
                corr_plot=corr_plot,
                edaparagraph1=eda1,
                edaparagraph2=valid_bullets,
                decision_0_current=decision_0_current,
                decision_1_current=decision_1_current,
                decision_0_tr=decision_0_tr,
                decision_1_tr=decision_1_tr,
                title=title,
                Corr_explanation=Corr_explanation,
                report_section_overrides=report_section_overrides,

                lasso_vals_tr=lasso_vals_tr,
                mi_vals_tr=mi_vals_tr,
                pca_vals_tr=pca_vals_tr,
                nb_vals_tr=nb_vals_tr,
                xgb_vals_tr=xgb_vals_tr,

                lasso_vals=lasso_vals,
                mi_vals=mi_vals,
                pca_vals=pca_vals,
                nb_vals=nb_vals,
                xgb_vals=xgb_vals,
                
                top_val_tv_sxi=top_val_tv_sxi,
                top_values_sxi=top_values_sxi,
                # feat1=feat1,
                # feat2=feat2,
                # feat3=feat3,
                # feat4=feat4,
                # feat5=feat5,

                # prcnt1=prcnt1,
                # prcnt2=prcnt2,
                # prcnt3=prcnt3,
                # prcnt4=prcnt4,
                # prcnt5=prcnt5,
                
                # coef1=coef1,
                # coef2=coef2,
                # coef3=coef3,
                # coef4=coef4,
                # coef5=coef5,
                
                # vv1=vv1,
                # vv2=vv2,
                # vv3=vv3,
                # vv4=vv4,
                # vv5=vv5,
                
                # v1=v1,
                # v2=v2,  
                # v3=v3,
                # v4=v4,
                # v5=v5,
                # prediff=prediff,
                # adjusted_target_value=prednew,
                # first_target_value=predold,
                # prediffperc=prediffperc,
            )
            print(f"generate_pdf_prof_user took {time.perf_counter() - step_start:.2f}s")
            self._update_report_progress(
                status="running",
                full_report_path=pdf_loc1,
                error="",
            )

            summary_generated = False
            try:
                _, dataset_name, rows, cols = self._get_processed_dataset_details()
                objective_outcome = good_outcome if str(tv_inc).strip().lower() in ("increase", "increased") else bad_outcome
                perce_inc_num = self._coerce_summary_float(perce_inc)
                perce_inc_text = f"{perce_inc_num:.2f}" if perce_inc_num is not None else str(perce_inc)

                def build_driver_pairs(features, values):
                    pairs = []
                    for feature, value in zip(features, values):
                        if feature in (None, "", "None"):
                            continue
                        numeric_value = self._coerce_summary_float(value)
                        display_value = f"{numeric_value:.3f}" if numeric_value is not None else str(value)
                        pairs.append((str(feature).replace("_", " ").title(), display_value))
                    return pairs

                current_drivers = build_driver_pairs(
                    [feature1, feature2, feature3, feature4, feature5],
                    [featurevalue1, featurevalue2, featurevalue3, featurevalue4, featurevalue5],
                )
                target_drivers = build_driver_pairs(
                    [feature6, feature7, feature8, feature9, feature10],
                    [featurevalue6, featurevalue7, featurevalue8, featurevalue9, featurevalue10],
                )

                performance_points = []
                if tv_type == "Numeric":
                    sxi_r2 = self._coerce_summary_float(R2_score_lnm)
                    sxi_mape = self._coerce_summary_float(area_under_curve_lnm)
                    if sxi_r2 is not None:
                        performance_points.append(f"SXI regression performance delivered an R2 of {sxi_r2:.2f}.")
                    if MAE_lnm is not None:
                        performance_points.append(
                            f"SXI mean absolute error was {format_display_value(MAE_lnm, target_variable_name, tv_type, decimals=2)}."
                        )
                    if sxi_mape is not None:
                        performance_points.append(f"SXI MAPE was {sxi_mape:.2f}.")
                    if send_to_llm and R2_Perf is not None:
                        comparison_r2 = self._coerce_summary_float(R2_Perf)
                        comparison_mae = format_display_value(MAE_Perf, target_variable_name, tv_type, decimals=2)
                        if comparison_r2 is not None:
                            performance_points.append(
                                f"Comparison model R2 was {comparison_r2:.2f} with MAE of {comparison_mae}."
                            )
                else:
                    sxi_acc = self._coerce_summary_float(Accuracy_lnm)
                    sxi_prec = self._coerce_summary_float(Precision_lnm)
                    sxi_rec = self._coerce_summary_float(Recall_lnm)
                    sxi_auc = self._coerce_summary_float(area_under_curve_lnm)
                    if sxi_acc is not None:
                        performance_points.append(f"SXI classification accuracy reached {sxi_acc:.2f}.")
                    if sxi_prec is not None and sxi_rec is not None:
                        performance_points.append(f"Precision and recall were {sxi_prec:.2f} and {sxi_rec:.2f}, respectively.")
                    if sxi_auc is not None:
                        performance_points.append(f"AUC performance was {sxi_auc:.2f}.")
                    if send_to_llm and model_accuracy is not None:
                        comparison_acc = self._coerce_summary_float(model_accuracy)
                        comparison_auc = self._coerce_summary_float(area_under_curve)
                        if comparison_acc is not None:
                            comparison_text = f"Comparison model accuracy was {comparison_acc:.2f}"
                            if comparison_auc is not None:
                                comparison_text += f" with AUC of {comparison_auc:.2f}"
                            performance_points.append(comparison_text + ".")

                projection_points = []
                current_sxi_num = self._coerce_summary_float(CurrentSXI)
                immediate_sxi_num = self._coerce_summary_float(sxi1)
                mid_sxi_num = self._coerce_summary_float(sxi2)
                long_sxi_num = self._coerce_summary_float(sxi3)
                if current_sxi_num is not None:
                    projection_points.append(f"Current DXI is {current_sxi_num:.2f}.")
                if immediate_sxi_num is not None or value1 is not None:
                    immediate_sxi_text = f"{immediate_sxi_num:.2f}" if immediate_sxi_num is not None else "N/A"
                    projection_points.append(
                        f"Immediate target projects SXI of {immediate_sxi_text} and outcome of "
                        f"{format_display_value(value1, target_variable_name, tv_type, decimals=2)}."
                    )
                if mid_sxi_num is not None or value2 is not None:
                    mid_sxi_text = f"{mid_sxi_num:.2f}" if mid_sxi_num is not None else "N/A"
                    projection_points.append(
                        f"Mid-term target projects SXI of {mid_sxi_text} and outcome of "
                        f"{format_display_value(value2, target_variable_name, tv_type, decimals=2)}."
                    )
                if long_sxi_num is not None or value3 is not None:
                    long_sxi_text = f"{long_sxi_num:.2f}" if long_sxi_num is not None else "N/A"
                    projection_points.append(
                        f"Long-term target projects SXI of {long_sxi_text} and outcome of "
                        f"{format_display_value(value3, target_variable_name, tv_type, decimals=2)}."
                    )

                recommendation_points = []
                if current_drivers:
                    recommendation_points.append(
                        f"Focus first on the highest-weighted current-state drivers: {', '.join(name for name, _ in current_drivers[:3])}."
                    )
                recommendation_points.extend([
                    "Use the immediate and mid-term SXI targets as operational checkpoints to track progress toward the long-term goal.",
                    "Review the top drivers regularly so teams can intervene early when the most influential variables drift.",
                ])

                summary_context = {
                    "dataset_name": dataset_name,
                    "rows": rows,
                    "cols": cols,
                    "target_variable_name": target_variable_name,
                    "target_mean": target_mean,
                    "objective_text": self._plain_summary_text(
                        self._generate_objective_text(
                            perce_inc_text=perce_inc_text,
                            tv_inc=tv_inc,
                            objective_outcome=objective_outcome,
                            target_variable_name=target_variable_name,
                        )
                    ),
                    "corr_explanation": self._plain_summary_text(Corr_explanation),
                    "eda_points": [self._plain_summary_text(point) for point in self._split_report_lines(valid_bullets)[:4]],
                    "current_drivers": current_drivers,
                    "target_drivers": target_drivers,
                    "performance_points": [point for point in performance_points if point],
                    "projection_points": [point for point in projection_points if point],
                    "recommendation_points": [point for point in recommendation_points if point],
                    "current_rate_text": format_display_value(CurrentRate, target_variable_name, tv_type, decimals=2),
                    "current_sxi_text": f"{current_sxi_num:.2f}" if current_sxi_num is not None else "N/A",
                    "current_outcome_text": format_display_value(Currentdt, target_variable_name, tv_type, decimals=2),
                }

                summary_text = self.generate_pdf_prof_user_summary(summary_context)
                if getattr(settings, 'GENERATE_SUMMARY_REPORT', False):
                    self.generate_report_summary_pdf(summary_pdf_loc, summary_text, dataset_name=dataset_name)
                    summary_generated = True
                    self._update_report_progress(
                        status="completed",
                        full_report_path=pdf_loc1,
                        summary_report_path=summary_pdf_loc,
                        error="",
                    )
                    print(f"Summary PDF saved as {summary_pdf_loc}")
                else:
                    summary_generated = False
                    self._update_report_progress(
                        status="completed",
                        full_report_path=pdf_loc1,
                        summary_report_path=None,
                        error="",
                    )
                    print("Summary PDF generation skipped per backend toggle.")
            except Exception as summary_exc:
                self._update_report_progress(
                    status="completed",
                    full_report_path=pdf_loc1,
                    error="",
                )
                print(f"Summary PDF generation failed: {summary_exc}")

            finalize_llm_usage_tracking(request=self.request)
            print_llm_usage_summary(request=self.request, prefix="[REPORT LLM TOKENS]")

            print('Report Generated')    

            if isinstance(result, dict):
                result["pdf_report_path"] = pdf_loc1
                result["report_context"] = report_section_context
                if summary_text:
                    result["explanation"] = summary_text
                if summary_generated:
                    result["summary_pdf_path"] = summary_pdf_loc
            else:
                result = {
                    "data": result,
                    "pdf_report_path": pdf_loc1
                }
                if summary_text:
                    result["explanation"] = summary_text
                if summary_generated:
                    result["summary_pdf_path"] = summary_pdf_loc

            print("PDF Path Stored:", pdf_loc1)
            # Extract results
            return self._format_results(result)
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            message = (
                self._friendly_dataset_error_message(fallback_error=e, df=locals().get("df"))
                if self._looks_like_dataset_quality_error(e)
                else str(e)
            )
            if generate_reports:
                self._update_report_progress(
                    status="error",
                    error=f"Report generation failed: {message}",
                )
            print(f"SXI Execution Error: {error_trace}")
            
            return {
                'error': True,
                'message': message,
                'error_type': 'DATASET_QUALITY' if message.startswith("Dataset error:") else 'SXI_EXECUTION',
                'trace': error_trace
            }
    
    def _format_results(self, result):
        """Format the results for frontend consumption"""
        
        if not result:
            return {
                'error': True,
                'message': 'No results returned from SXI execution'
            }
        
        # -------- Decision paths (TEXT FILES) --------
        current_paths_text = (
            f"{result.get('best_path_0_str', '')}\n\n"
            f"{result.get('best_path_1_str', '')}"
        )
        print("Current Paths Text:", current_paths_text)  # Debug print

        target_paths_text = (
            f"{result.get('best_path_0_str_tr', '')}\n\n"
            f"{result.get('best_path_1_str_tr', '')}"
        )
        print("Target Paths Text:", target_paths_text)  # Debug print

        current_path_file = self._save_text_artifact(
            "decision_paths_current.txt",
            current_paths_text
        )

        target_path_file = self._save_text_artifact(
            "decision_paths_target.txt",
            target_paths_text
        )
        
        # Extract key metrics
        formatted = {
            'success': True,
            'sxi_score': result.get('sxi_avg'),
            'current_outcome': result.get('curent_tv'),
            'correlation_type': result.get('rltyp'),
            'r2_score': result.get('r2'),
            
            # Improvement targets
            'immediate': {
                'sxi': result.get('tgsxi'),
                'outcome': result.get('tgout'),
                'improvement_pct': result.get('perchg')
            },
            'midterm': {
                'sxi': result.get('midsxi'),
                'outcome': result.get('midout'),
                'improvement_pct': result.get('midper')
            },
            'longterm': {
                'sxi': result.get('ltx'),
                'outcome': result.get('lty'),
                'improvement_pct': result.get('ltchg')
            },
            
            # Visualizations
            'plots': {
                'distribution': result.get('locdist'),
                'correlation': result.get('plotly_fig'),
                'current_tree': result.get('loc_curr_tree'),
                'target_tree': result.get('loc_target_tree'),
                'eda_plots': result.get('edaplots', [])
            },
            
            # Model performance
            'model_performance': self._get_performance_metrics(result),
            
            # Decision paths
            "decision_paths_files": {
                "current": current_path_file,
                "target": target_path_file,
            },
            
            # Feature importance
            'feature_importance': {
                'current': result.get('mi_score'),
                'target': result.get('mi_score_target')
            },
            'fulldata_path': result.get('fulldata_path') or '',
            'fulldata_url': result.get('fulldata_url') or '',
            'fulldata_category': result.get('fulldata_category') or '',
        }

        from agent2.utils.json_utils import make_json_safe
        # 🔥 PRESERVE PDF PATH IF EXISTS
        if "pdf_report_path" in result:
            formatted["pdf_report_path"] = result["pdf_report_path"]
        if "summary_pdf_path" in result:
            formatted["summary_pdf_path"] = result["summary_pdf_path"]
        if "report_context" in result:
            formatted["report_context"] = result["report_context"]

        print("Formatted SXI Results:", formatted)  # Debug print

        return make_json_safe(formatted)

        
        # return formatted
    
    def _get_performance_metrics(self, result):
        """Extract performance metrics based on target type"""
        model_used = result.get('ml_model', 'Unknown')
        
        if self.target_info.get('Target Outcome Type') == 'Categorical':
            return {
                'type': 'classification',
                'model_used': model_used,
                'accuracy': result.get('sxiacc'),
                'precision': result.get('sxiprec'),
                'recall': result.get('sxirecall'),
                'auc': result.get('sxi_auc_value'),
                'auc_raw': result.get('auc_best'),
                'confusion_matrix': result.get('cm')
            }
        else:
            return {
                'type': 'regression',
                'model_used': model_used,
                'r2_score': result.get('r2score'),
                'mae': result.get('sximae'),
                'within_10pct': result.get('sxi_within_percentage'),
                'actual_vs_predicted': result.get('actlocs')
            }
        


    def generate_pdf_prof(self,filename,target_variable_name=None, target_mean=None, table_data=None, sriya_logo=None, plot_target_dist=None,
                    plot_vrm_dist=None,perce_inc=None, tv_inc=None, bad_outcome=None,good_outcome=None,Targetdt=None, Currentdt=None, risk_index=None,blue_banner=None,
                    tv_type=None, model_accuracy=None, precision_score=None, area_under_curve=None, MAE_Perf=None,R2_Perf=None, Error_Perf=None,
                    Actual_fig=None, AUC=None, confusion_matrix=None, ClassRate=None, CurrentRate=None, CurrentSXI=None,
                    perc1=None, perc2=None, perc3=None, value1=None, value2=None, value3=None, name=None, init=None, mid=None, end=None, 
                    conv=None, sxi1=None, sxi2=None, sxi3=None, 
                    improvementcard1=None, improvementcard2=None, improvementcardvalue2=None, improvementcard3=None,improvementcardvalue3=None,
                    improvementcard4=None, improvementcardvalue4=None, improvementcard5=None, improvementcardvalue5=None,improvementcard6=None,improvementcardvalue6=None,
                    MAE_lnm=None, R2_score_lnm=None, area_under_curve_lnm=None, Accuracy_lnm=None, Precision_lnm=None, Recall_lnm=None, Recall_score=None,
                    feature1=None,feature2=None,feature3=None, feature4=None, feature5=None,feature6=None,feature7=None,feature8=None, feature9=None, 
                    feature10=None,featurevalue1=None,featurevalue2=None,featurevalue3=None,featurevalue4=None,featurevalue5=None,
                    featurevalue6=None,featurevalue7=None,featurevalue8=None,featurevalue9=None,featurevalue10=None,corr_plot=None,
                    edaparagraph2=None,edaparagraph1=None,decision_1_current=None, decision_0_current=None, 
                    decision_0_tr=None, decision_1_tr=None, title=None,Corr_explanation=None, 
                    lasso_vals_tr=None, mi_vals_tr=None, pca_vals_tr=None, nb_vals_tr=None, xgb_vals_tr=None,
                    lasso_vals=None, mi_vals=None, pca_vals=None, nb_vals=None, xgb_vals=None, top_val_tv_sxi=None, top_values_sxi=None,
                    feat1=None, feat2=None, feat3=None, feat4=None, feat5=None, 
                    prcnt1=None, prcnt2=None, prcnt3=None, prcnt4=None, prcnt5=None,
                    coef1=None, coef2=None, coef3=None, coef4=None, coef5=None,
                    vv1=None, vv2=None, vv3=None, vv4=None, vv5=None,
                    v1=None, v2=None, v3=None, v4=None, v5=None, report_section_overrides=None,
                    prediff=None, adjusted_target_value=None, first_target_value=None,prediffperc=None):

        print(f"Generating PDF: {filename}")
        blue_banner = self._get_report_asset("blue.png")
        sriya_logo = self._get_report_asset("Sriya_new_logo.png")
        doc = SimpleDocTemplate(filename, pagesize=LETTER,
                                rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)

        styles = getSampleStyleSheet()
        subtitle_style = ParagraphStyle('SubtitleStyle',parent=styles['Heading2'],  # You can base it on another style, like 'Heading2'
        fontSize=14,  # Adjust the font size
        textColor=colors.darkblue,  # Change the color to dark blue
        spaceAfter=12  # Space after the subtitle
        )
        styles.add(ParagraphStyle(name='SmallBulletStyle',fontSize=9.5, leading=11.5, spaceAfter=4))
        styles.add(subtitle_style)
        styles.add(ParagraphStyle(name='TitleStyle', fontSize=20, alignment=1, spaceAfter=20, leading=28))
        styles.add(ParagraphStyle(name='HeadingStyle', fontSize=14, alignment=1, spaceAfter=10, leading=20))
        styles.add(ParagraphStyle(name='NormalStyle', fontSize=10, leading=15))
        styles.add(ParagraphStyle(name='MicrosoftSansSerif',fontName='MicrosoftSansSerif',fontSize=12,leading=15,textColor=colors.black)
        )
        page1_dataset_title_style = ParagraphStyle(
            "page1_dataset_title",
            parent=styles["Normal"],
            fontSize=20,
            leading=22,
            textColor=colors.HexColor("#2E3192"),
            alignment=TA_LEFT,
        )

        def format_two_decimals(value, default="N/A"):
            try:
                return f"{float(value):.2f}"
            except (TypeError, ValueError):
                return default if value is None else str(value)

        def format_report_display_value(value, display_target_name=None, display_target_type=None):
            return format_display_value(
                value,
                display_target_name if display_target_name is not None else target_variable_name,
                display_target_type if display_target_type is not None else tv_type,
                decimals=2,
            )

        send_to_llm = bool(self.request.session.get("send_to_llm", False))
        

        elements = []

        report_section_overrides = dict(report_section_overrides or {})

        def pick_report_value(key, default):
            override_value = report_section_overrides.get(key)
            if override_value is None:
                return default
            if isinstance(override_value, str) and override_value == "":
                return default
            if isinstance(override_value, np.ndarray) and override_value.size == 0:
                return default
            if isinstance(override_value, (pd.Series, pd.DataFrame)) and override_value.empty:
                return default
            if isinstance(override_value, (list, tuple, set, dict)) and len(override_value) == 0:
                return default
            return override_value

        # Sriya logo
        if sriya_logo:
            logo = Image(sriya_logo, width=100, height=40)
            logo.hAlign = 'RIGHT'
        else:
            logo = Spacer(1, 40)
        if blue_banner:
            background = Image(blue_banner, width=8.5*inch, height=1.5*inch)
        else:
            background = Spacer(1, 1.5*inch)

        # Black CASE STUDY box
        # case_image = Image("case.png", width=2.5*inch, height=1.8*inch)
        # case_image.hAlign = 'LEFT'
        # elements.append(Spacer(1, 1.8*inch))  # Push content below blue banner
        # elements.append(case_image)  # Adjust size

        # Blue Box
        doc = SimpleDocTemplate(filename, pagesize=LETTER)
        def draw_background(canvas, doc):
            self._draw_pdf_header_banner(
                canvas,
                blue_banner,
                x=0,
                y=LETTER[1] - 1.7 * inch,
                width=LETTER[0],
                height=1.7 * inch,
            )
            
        # Build PDF with background
        build_start = time.perf_counter()
        print("generate_pdf_prof: starting header/background doc.build")
        doc.build(elements, onFirstPage=draw_background)
        print(f"generate_pdf_prof: header/background doc.build took {time.perf_counter() - build_start:.2f}s")



        # Create a table with the background image in cell and logo over it
        overlay_table = Table([[background]])
        overlay_table.setStyle(TableStyle([
            # ('LEFTPADDING', (0, 0), (-1, -1), 0),
            # ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            # ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        logo_table = Table([[logo]])
        logo_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            # ('LEFTPADDING', (0, 0), (-1, -1), 0),
            # ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            # ('TOPPADDING', (0, 0), (-1, -1), 0),
            # ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))



        # Sriya Logo
        header_table = Table([[logo]], colWidths=[6.9*inch, 0.5*inch])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, 0), 'RIGHT'),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
        ]))
        elements.append(header_table)



        # Title
        # Subtitle
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(f"<b><i>{title}</i></b>", styles['HeadingStyle']))
        # elements.append(Paragraph(title, styles['TitleStyle']))
        elements.append(Spacer(1, 6))

        # Blue line
        elements.append(Table([['']], colWidths=[6.5*inch], rowHeights=[2], style=[
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0000FF'))
        ]))
        elements.append(Spacer(1, 12))

        # if table_data:
        #     # Calculate column width
        #     page_width = LETTER[0] - doc.leftMargin - doc.rightMargin
        #     num_columns = max(len(row) for row in table_data)
        #     col_width = page_width / num_columns

        #     # Create the table
        #     main_table = Table(table_data, colWidths=[col_width] * num_columns)

        #     # Apply styling
        #     main_table.setStyle(TableStyle([
        #         ('BACKGROUND', (0, 0), (-1, 0), colors.midnightblue),
        #         ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        #         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        #         ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        #         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        #         ('FONTSIZE', (0, 0), (-1, 0), 12),
        #         ('FONTSIZE', (0, 1), (-1, -1), 8),
        #         ('TOPPADDING', (0, 0), (-1, -1), 6),
        #         ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        #         ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        #         ('BACKGROUND', (0, 1), (-1, -1), colors.skyblue),
        #         ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
        #     ]))

        #     # Wrap the table in another table for centering
        #     centered_table = Table([[main_table]], colWidths=[page_width])
        #     centered_table.setStyle(TableStyle([
        #         ('ALIGN', (0, 0), (-1, -1), 'CENTER')
        #     ]))

        #     elements.append(centered_table)
        #     elements.append(Spacer(1, 20))

        latest_run_id = (
            UploadedFile.objects
            .order_by("-uploaded_at")
            .values_list("run_id", flat=True)
            .first()
        )

        latest_run_files = (
            UploadedFile.objects
            .filter(run_id=latest_run_id)
            .order_by("uploaded_at")
        )
        file_names = list(
            latest_run_files.values_list("original_name", flat=True)
        )
        file_paths = [f.file.path for f in latest_run_files]
        
        dataframes = {}
        for f in latest_run_files:
            print(f.file, f.original_name)
            file_path = f.file.path
            file_name = f.original_name 
            df = self._read_tabular_file(file_path)
            dataframes[file_name] = df

            print(f"Loaded {file_name} | rows={df.shape[0]} cols={df.shape[1]}")

        elements.append(
            Paragraph(
                f"<b>No. of Files:</b> {len(dataframes)}",
                styles['NormalStyle']
            )
        )
        elements.append(Spacer(1, 3))

        for idx, (file_name, df) in enumerate(dataframes.items(), start=1):
            elements.append(
                Paragraph(
                    f"<b>File {idx}: {file_name}</b>",
                    styles['NormalStyle']
                )
            )
            elements.append(
                Paragraph(
                    f"• No of Columns: {df.shape[1]}",
                    styles['SmallBulletStyle']
                )
            )

            elements.append(
                Paragraph(
                    f"• No of Rows: {df.shape[0]}",
                    styles['SmallBulletStyle']
                )
            )

            elements.append(Spacer(1, 6))


        # elements.append(Paragraph("<b>Total</b>", styles['NormalStyle']))
        # elements.append(Paragraph("• Columns: 120", styles['SmallBulletStyle']))
        # elements.append(Paragraph("• Rows: 61502", styles['SmallBulletStyle']))
        # elements.append(Spacer(1, 10))
        total_rows = sum(df.shape[0] for df in dataframes.values())
        total_columns = sum(df.shape[1] for df in dataframes.values())

        elements.append(Paragraph("<b>Total</b>", styles['NormalStyle']))
        elements.append(
            Paragraph(f"• Columns: {total_columns}", styles['SmallBulletStyle'])
        )
        elements.append(
            Paragraph(f"• Rows: {total_rows}", styles['SmallBulletStyle'])
        )
        elements.append(Spacer(1, 3))

        normal_style = styles["Normal"]
        step_style = ParagraphStyle(
            "StepStyle",
            parent=normal_style,
            fontSize=10,
            spaceBefore=8,
            spaceAfter=4,
            leftIndent=0,
            alignment=TA_LEFT
        )

        sub_bullet_style = ParagraphStyle(
            "SubBulletStyle",
            parent=normal_style,
            fontSize=10,
            leftIndent=18,
            spaceBefore=2,
            spaceAfter=2
        )


        if target_variable_name:
            target_info = f"""
                <b>Target Variable:</b> {target_variable_name}<br/>
                <b>Target Count: </b> {target_mean if target_mean is not None else 'N/A'}<br/>
            """
            elements.append(Paragraph(target_info, styles['NormalStyle']))

        elements.append(Paragraph("<b>Step 1: Tabular Column Detection</b>", styles['NormalStyle']))
        elements.append(Paragraph("– Tabular Column detected.", styles['SmallBulletStyle']))
        elements.append(Paragraph("– Task type is set to <b>Tabular [Numerical]</b> modality.", styles['SmallBulletStyle']))
        elements.append(Paragraph("– Date Column Detection:", styles['SmallBulletStyle']))
        elements.append(Paragraph("  o No timestamp/date column suitable for forecasting.", styles['SmallBulletStyle']))

        elements.append(Spacer(1, 2))
        elements.append(Paragraph("<b>Step 2: Text Column Detection</b>", styles['NormalStyle']))
        elements.append(Paragraph("– No major text column was detected.", styles['SmallBulletStyle']))
        elements.append(Paragraph("– Text was skipped.", styles['SmallBulletStyle']))

        elements.append(Spacer(1, 2))
        elements.append(Paragraph("<b>Final Decision</b>", styles['NormalStyle']))
        elements.append(Paragraph("– Modality: Tabular [Numerical]", styles['SmallBulletStyle']))
        elements.append(Paragraph(f"– Target Variable: Detected {target_variable_name} column.", styles['SmallBulletStyle']))

        # ---------- AGENT SELECTED TABULAR PREPROCESSING ----------
        elements.append(Spacer(1, 2))
        elements.append(Paragraph(
            "• <b>Agent Selected Tabular Data Preprocessing Steps</b>",
            step_style
        ))

        # ---- Missing Values ----
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(
            "– <b>Handling missing values (imputation or removal)</b>",
            sub_bullet_style
        ))
        elements.append(Paragraph(
            "▪ Missing data is either filled with meaningful values or removed to avoid incorrect model learning.",
            ParagraphStyle(
                "explain",
                parent=sub_bullet_style,
                leftIndent=32
            )
        ))

        # ---- Outliers ----
        elements.append(Spacer(1, 2))
        elements.append(Paragraph(
            "– <b>Outlier detection and treatment</b>",
            sub_bullet_style
        ))
        elements.append(Paragraph(
            "▪ Extreme or abnormal values were identified using percentile-based thresholds, "
            "where values below the 10th percentile and above the 80th percentile were removed "
            "to reduce distortion in model results.",
            ParagraphStyle(
                "explain",
                parent=sub_bullet_style,
                leftIndent=32
            )
        ))

        # ---- Encoding ----
        elements.append(Spacer(1, 2))
        elements.append(Paragraph(
            "– <b>Categorical encoding (Label / One-Hot Encoding)</b>",
            sub_bullet_style
        ))
        elements.append(Paragraph(
            "▪ Categorical text values are converted into numerical format so machine learning models can process them.",
            ParagraphStyle(
                "explain",
                parent=sub_bullet_style,
                leftIndent=32
            )
        ))

        # ---- Scaling ----
        elements.append(Spacer(1, 2))
        elements.append(Paragraph(
            "– <b>Feature scaling</b>",
            sub_bullet_style
        ))
        elements.append(Paragraph(
            "▪ Numerical values are scaled to a common range to ensure no single feature dominates the model.",
            ParagraphStyle(
                "explain",
                parent=sub_bullet_style,
                leftIndent=32
            )
        ))

        # ---- Feature Selection ----
        elements.append(Spacer(1, 2))
        elements.append(Paragraph(
            "– <b>Feature selection</b>",
            sub_bullet_style
        ))
        elements.append(Paragraph(
            "▪ Only the most relevant features are retained to improve model accuracy and reduce noise.",
            ParagraphStyle(
                "explain",
                parent=sub_bullet_style,
                leftIndent=32
            )
        ))

        def draw_background(canvas, doc):
            self._draw_pdf_header_banner(
                canvas,
                blue_banner,
                x=0,
                y=LETTER[1] - 1.5 * inch,
                width=LETTER[0],
                height=1.5 * inch,
            )


        # ---- Page 2 Starts Here -----------------------------------------------------------------------------------------------------------
        

    #  ------------------Page 3-------------------------------------------------------------------------------------------------------------------
        elements.append(PageBreak())
        #Objective
        elements.append(Paragraph("<u><b>Sriya Results</b></u>", styles['TitleStyle']))
        elements.append(Paragraph("<u><b>Objective</b></u>", styles['HeadingStyle']))
        objective_outcome = good_outcome if str(tv_inc).strip().lower() in ("increase", "increased") else bad_outcome
        perce_inc_text = format_two_decimals(perce_inc, default=str(perce_inc))
        elements.append(Paragraph(f"• Minimum <b>{perce_inc_text}%</b> {tv_inc} in {objective_outcome} from current levels using <b>Sriya’s SXI</b>.", styles['SmallBulletStyle']))
        elements.append(Spacer(1, 12))

        #Definitions
        elements.append(Paragraph("<u><b>Definitions</b></u>", styles['HeadingStyle']))
        definitions = [
        f"• SXI: {risk_index} derived by dynamic weighted average of all important features. 5-7 Machine Learning algorithms in a proprietary formula gives this score. Every ID will get an SXI Score.",
        "• SXI: SXI enhanced by Deep Neural Network based Reinforced Learning.",
        "• LNM: Large Numerical Model powered by SXI. LNMs are “curated pre-trained and pre-validated models capable of predicting unseen data with very high accuracy and precision.”"
        ]
        for defn in definitions:
            elements.append(Paragraph(defn, styles['SmallBulletStyle']))
            elements.append(Spacer(1, 3))

        discussion_plot_vrm = pick_report_value("plot_vrm_dist", plot_vrm_dist)
        discussion_plot_target = pick_report_value("plot_target_dist", plot_target_dist)
        eda_bullets = pick_report_value("edaparagraph2", edaparagraph2)
        corr_plot_for_section = pick_report_value("corr_plot", corr_plot)
        corr_explanation_for_section = pick_report_value("Corr_explanation", Corr_explanation)
        current_tree_image = pick_report_value("Currentdt", Currentdt)
        current_tree_rate = pick_report_value("CurrentRate", CurrentRate)
        current_tree_sxi = pick_report_value("CurrentSXI", CurrentSXI)
        target_tree_image = pick_report_value("Targetdt", Targetdt)
        decision_0_current_for_section = pick_report_value("decision_0_current", decision_0_current)
        decision_1_current_for_section = pick_report_value("decision_1_current", decision_1_current)
        decision_0_tr_for_section = pick_report_value("decision_0_tr", decision_0_tr)
        decision_1_tr_for_section = pick_report_value("decision_1_tr", decision_1_tr)
        
        # Discussion
        elements.append(Paragraph("<u><b>Discussion & Results</b></u>", styles['HeadingStyle']))
        elements.append(Spacer(1, 6))

        # Subtitles for charts
        centered = ParagraphStyle(name="centered", parent=styles['NormalStyle'], alignment=TA_CENTER, fontSize=12,leading=10, fontName="Helvetica-Bold",)
        SXI=Paragraph(f"<b>DXI Distribution of {target_variable_name}</b>",centered)
        Target=Paragraph("<b>Target Distribution</b>",centered)
        chart_titles = Table([
            [SXI,Target],], colWidths=[250, 250])
        chart_titles.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(chart_titles)
        elements.append(Spacer(1, 12))

        # Chart placeholders — replace with actual images
        try:
            chart1 = Image(discussion_plot_vrm, width=300, height=200)
            chart2 = Image(discussion_plot_target, width=300, height=200)
            chart_table = Table([[chart1, chart2]], colWidths=[250, 250])
            chart_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
            elements.append(chart_table)
            elements.append(Spacer(1, 3))
        except Exception as e:
            elements.append(Paragraph("Charts not available.", styles['NormalStyle']))

        # SXI EDA Section
        elements.append(Paragraph("<u><b>SXI Exploratory Data Analysis</b></u>", styles['HeadingStyle']))
        # eda_text = f"""
        # {edaparagraph1}
        # """
        # elements.append(Paragraph(eda_text, styles['SmallBulletStyle']))
        # elements.append(Spacer(1, 3))


        self._append_eda_content(elements, eda_bullets, styles)

        elements.append(PageBreak())
        # ----- Page 4 Starts Here -----------------------------------------------------------------------------------------------------------
        
        # pot_fig="potential.png" 
        # chartp = Image(pot_fig, width=550, height=150)
            # elements.append(chartp)
        center = ParagraphStyle(
            name="center",
            alignment=TA_CENTER,
            fontSize=9,
            leading=12,
            fontName="Helvetica-Bold",
            textColor=colors.black,
        )

        right_white = ParagraphStyle(
            name="right_white",
            alignment=TA_CENTER,
            fontSize=9,
            leading=12,
            fontName="Helvetica-Bold",
            textColor=colors.white,
        )

        heading = ParagraphStyle(
            name="heading",
            alignment=TA_CENTER,
            fontSize=14,
            leading=16,
            fontName="Helvetica-Bold",
            textColor=colors.darkblue,
        )

        # ---------------------------------------------------
        # 1. Header
        # ---------------------------------------------------

        elements.append(Paragraph("<b>Potential Business Improvement</b>", heading))
        elements.append(Spacer(1, 14))

        current_target_display = format_report_display_value(CurrentRate)
        current_target_heading = (
            f"Current {ClassRate} Rate"
            if tv_type == "Categorical"
            else f"Current {target_variable_name}"
        )
        elements.append(
            Paragraph(f"{current_target_heading}: <b>{current_target_display}</b>", center)
        )
        elements.append(Paragraph(f"Current DXI: <b>{CurrentSXI}</b>", center))
        elements.append(Spacer(1, 20))

        # ---------------------------------------------------
        # 2. BOX MAKER FUNCTION
        # ---------------------------------------------------

        if tv_inc in ["decreased", "reduced"]:
            bad=bad_outcome
        else:
            bad=good_outcome
        
        print("bad pdf generation:", bad)

        def make_business_box(label, perc, value, sxi):
            heading_map = {
                "Immediate": "Initial Improvement",
                "Mid-Term": "Mid Term Improvement",
                "Long-Term": "Long Term Improvement",
            }
            left_heading = heading_map.get(label, f"{label} Improvement")
            perc_text = format_two_decimals(abs(perc))
            value_text = format_report_display_value(value)

            left_box = Paragraph(
                f"""
                <b>{left_heading}</b><br/>
                {perc_text}% {tv_inc}<br/>
                Projected {target_variable_name}: {value_text}
                """,
                center
            )

            right_box = Paragraph(
                f"""
                <b>SXI: {sxi}</b>
                """,
                right_white
            )

            table = Table(
                [[left_box, right_box]],
                colWidths=[170, 90],
                rowHeights=[45]
            )

            table.setStyle(TableStyle([
                ('GRID', (0,0), (0,0), 1, colors.black),
                ('BACKGROUND', (1,0), (1,0), colors.royalblue),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ]))

            return table


        # ---------------------------------------------------
        # 3. THREE-LEVEL STACK (Immediate → Mid → Long)
        # ---------------------------------------------------

        elements.append(make_business_box("Immediate", abs(perc1), value1, sxi1))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph("↓", center))
        elements.append(Spacer(1, 8))

        elements.append(make_business_box("Mid-Term", perc2, value2, sxi2))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph("↓", center))
        elements.append(Spacer(1, 8))

        elements.append(make_business_box("Long-Term", perc3, value3, sxi3))
        elements.append(Spacer(1, 20))

        
        data = [
            [f"{improvementcard1}: {CurrentSXI}","",f"{improvementcard2}: {improvementcardvalue2}","",f"{improvementcard3}: {improvementcardvalue3}"],
        ]
        # table = Table(data,colWidths=[2.2*inch]*5,rowHeights=[0.3*inch]*4)
        # table = Table(data, colWidths=[150], rowHeights=[60]) 
        table = Table(data, colWidths=[160, 10, 160, 10, 160],rowHeights=[60]) 
        
        style = TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), colors.skyblue), # Header row background
            ('BACKGROUND', (2, 0), (2, 0), colors.mediumaquamarine), # Header row background
            ('BACKGROUND', (4, 0), (4, 0), colors.orange), # Header row background
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), # Header row text color
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'), # Align all cells to center
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOX', (0,0), (0,0), 1, colors.white),
            ('BOX', (2,0), (2,0), 1, colors.white),
            ('BOX', (4,0), (4,0), 1, colors.white),

            
        ])
        
        table.setStyle(style)
        elements.append(table)


        # pot_fig= corr_plot
        print("corr_plot", corr_plot)
        pot_fig = corr_plot_for_section
        print("pot_fig", pot_fig)
        chartp = Image(pot_fig, width=450, height=250)
        elements.append(chartp)
        elements.append(Spacer(1, 3))

        elements.append(PageBreak())
        definitions = [Corr_explanation]

        lines = str(corr_explanation_for_section or "").replace("<ul>", "").replace("</ul>", "").split("<li>")
        lines = [line.replace("</li>", "").strip() for line in lines if line.strip()]

        bullet_list = ListFlowable(
            [ListItem(Paragraph(line, styles['Normal'])) for line in lines],
            bulletType='bullet'
        )

        elements.append(bullet_list)
        elements.append(Spacer(1, 12))

        data = [
            [f"{improvementcard4}: {improvementcardvalue4}","",f"{improvementcard5}: {improvementcardvalue5}","",f"{improvementcard6}: {improvementcardvalue6}"],
        ]
        # table = Table(data,colWidths=[2.2*inch]*5,rowHeights=[0.3*inch]*4)
        # table = Table(data, colWidths=[150], rowHeights=[60]) 
        table = Table(data, colWidths=[160, 10, 160, 10, 160],rowHeights=[60]) 
        
        style = TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), colors.salmon), # Header row background
            ('BACKGROUND', (2, 0), (2, 0), colors.skyblue), # Header row background
            ('BACKGROUND', (4, 0), (4, 0), colors.mediumaquamarine), # Header row background
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), # Header row text color
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'), # Align all cells to center
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('RIGHTPADDING', (1, 0), (1, -1), 20),  # Add space after column 2
            ('BOX', (0,0), (0,0), 1, colors.white),
            ('BOX', (2,0), (2,0), 1, colors.white),
            ('BOX', (4,0), (4,0), 1, colors.white),

            
        ])
        
        table.setStyle(style)
        elements.append(table)

        elements.append(Spacer(1,12))
        

    #----- Page 5 Starts Here --------------------------------------------------------------------------

        #Metrics
        elements.append(Paragraph("<u><b> SXI Performance Metrics Table</b></u>", styles['HeadingStyle']))


        if tv_type == "Categorical":
            data = [
                ['Metrics', 'Values'],
                ['Accuracy',  Accuracy_lnm],
                ['Precision', Precision_lnm],
                ['Recall', Recall_lnm],
                ['AUC', area_under_curve_lnm]
            ]
            

            table = Table(data, colWidths=[200, 120, 120, 100]) 
            
            style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.midnightblue), # Header row background
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), # Header row text color
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'), # Align all cells to center
                ('GRID', (0, 0), (-1, -1), 1, colors.black), # Add grid lines
            ])
            
            table.setStyle(style)
            elements.append(table)
            elements.append(Spacer(1,20))

            if self.request.session.get("send_to_llm", False):

                elements.append(Paragraph("<b>CHATGPT Metrics Table</b>", styles['HeadingStyle']))
                chatgpt_pdf_acc = self._metric(
                    self._normalize_classification_percentage(model_accuracy)
                )
                chatgpt_pdf_prec = self._metric(
                    self._normalize_classification_percentage(precision_score)
                )
                chatgpt_pdf_recall = self._metric(
                    self._normalize_classification_percentage(Recall_score)
                )
                chatgpt_pdf_auc = self._metric(area_under_curve)

                automl_data = [
                    ['Metrics', 'CHATGPT-ML'],
                    ['Accuracy', f"{chatgpt_pdf_acc:.2f}"],
                    ['Precision', f"{chatgpt_pdf_prec:.2f}"],
                    ['Recall', f"{chatgpt_pdf_recall:.2f}"],
                    ['AUC', f"{chatgpt_pdf_auc:.2f}"]
                ]

                automl_table = Table(automl_data, colWidths=[200, 120])
                automl_table.setStyle(style)
                elements.append(automl_table)
            #Confuson Matrix
            figure_heading = ParagraphStyle(
                name="FigureHeading",
                fontSize=12,
                alignment=TA_CENTER,
                fontName="Helvetica-Bold",
                spaceAfter=6
            )

            if confusion_matrix and AUC:
                # Images
                conf_img = Image(confusion_matrix, width=150, height=150)
                auc_img = Image(AUC, width=150, height=150)

                # Headings
                conf_title = Paragraph("Confusion Matrix", figure_heading)
                auc_title = Paragraph("AUC ROC Curve", figure_heading)

                # Table layout: headings row + image row
                chart_table = Table(
                    [
                        [conf_title, auc_title],
                        [conf_img, auc_img]
                    ],
                    colWidths=[250, 250],
                    rowHeights=[30, 220]
                )
            else:
                chart_table = Table(
                    [[Paragraph(
                        "Confusion matrix and ROC curve images were unavailable for this run.",
                        styles['NormalStyle']
                    )]],
                    colWidths=[500]
                )
            bullet_style = ParagraphStyle(
                name="ConfusionBullet",
                fontSize=9,
                leading=12,
                leftIndent=8
            )
            confusion_box_lines = self._generate_confusion_box_lines(
                target_variable_name=target_variable_name,
                good_outcome=good_outcome,
                bad_outcome=bad_outcome,
                acc_val=chatgpt_pdf_acc if self.request.session.get("send_to_llm", False) else model_accuracy,
                prec_val=chatgpt_pdf_prec if self.request.session.get("send_to_llm", False) else precision_score,
                rec_val=chatgpt_pdf_recall if self.request.session.get("send_to_llm", False) else Recall_score,
                auc_val=chatgpt_pdf_auc if self.request.session.get("send_to_llm", False) else area_under_curve,
                confusion_matrix_path=confusion_matrix,
            )

            confusion_box = Table(
                [[
                    Paragraph("• TP (True Positive): correctly detected fraud cases.", bullet_style),
                ], [
                    Paragraph("• FP (False Positive): genuine cases incorrectly flagged as fraud.", bullet_style),
                ], [
                    Paragraph("• TN (True Negative): genuine cases correctly identified as non-fraud.", bullet_style),
                ], [
                    Paragraph("• FN (False Negative): fraud cases missed by the model.", bullet_style),
                ]],
                colWidths=[500]
            )

            confusion_box = Table(
                [[Paragraph(f"&bull; {line}", bullet_style)] for line in confusion_box_lines],
                colWidths=[500]
            )
            confusion_box.setStyle(TableStyle([
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ]))

            elements.append(
                KeepTogether([
                    chart_table,
                    Spacer(1, 10),
                    confusion_box,
                    Spacer(1, 12)
                ])
            )
            chart_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

                # Optional border (comment out if not needed)
                ('BOX', (0, 0), (-1, -1), 1, colors.black),

                # Padding
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))

            # # Keep everything on one page
            # elements.append(
            #     KeepTogether([
            #         chart_table,
            #         Spacer(1, 12)
            #     ])
            # )
            elements.append(PageBreak())

            chart_titles = Table(
                [
                    ["Current Decision Tree"],
                    [Paragraph(
                        f"Current Fraud: {CurrentRate}% | SXI: {round(CurrentSXI,3)}",
                        ParagraphStyle(
                            name="SubTitle",
                            fontSize=10,
                            alignment=TA_CENTER,
                            leading=12
                        )
                    )]
                ],
                colWidths=[400],
                rowHeights=[28, 22]
            )

            chart_titles.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

                # Border
                ('BOX', (0, 0), (-1, -1), 1, colors.black),

                # Title style
                ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (0, 0), 12),

                # Subtitle style
                ('FONTSIZE', (0, 1), (0, 1), 10),
                ('TEXTCOLOR', (0, 1), (0, 1), colors.black),

                # Padding
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))

            elements.append(chart_titles)
            elements.append(Spacer(1, 12))
            if current_tree_image:
                chart1 = Image(current_tree_image, width=500, height=300)
                chart_table = Table([[chart1]], colWidths=[250, 250])
                chart_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
                elements.append(chart_table)
            else:
                elements.append(Paragraph("Current decision tree image was unavailable for this run.", styles['NormalStyle']))
            elements.append(Spacer(1, 20))

        elif tv_type == "Numeric":
            data = [
                ['Metrics', 'Values'],
                ['Mean Absolute Error (MAE)', self._format_optional_report_value(MAE_lnm, target_variable_name, tv_type, decimals=2)],
                ['R² Score', self._format_optional_report_number(R2_score_lnm, decimals=2)],
                ['Error %', self._format_optional_report_number(area_under_curve_lnm, decimals=2, suffix="%")]
            ]
            
            table = Table(data, colWidths=[200, 120, 120, 100]) 
            
            style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.midnightblue), # Header row background
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), # Header row text color
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'), # Align all cells to center
                ('GRID', (0, 0), (-1, -1), 1, colors.black), # Add grid lines
            ])
            
            table.setStyle(style)
            elements.append(table)
            elements.append(Spacer(1,20))
            if self.request.session.get("send_to_llm", False):

                elements.append(Paragraph("<b>CHATGPT Metrics Table</b>", styles['HeadingStyle']))
                print("DEBUG -> send_to_llm:", self.request.session.get("send_to_llm", False))
                print("DEBUG -> MAE_Perf:", MAE_Perf)
                print("DEBUG -> R2_Perf:", R2_Perf)
                print("DEBUG -> Error_Perf:", Error_Perf)
                automl_data = [
                    ['Metrics',  'Values'],
                    ['MAE', f"{float(MAE_Perf):.2f}" if MAE_Perf is not None else "N/A"],
                    ['R² Score', R2_Perf],
                    ['Error %',  f"{float(Error_Perf):.2f}%" if Error_Perf is not None else "N/A"]
                ]

                automl_table = Table(automl_data, colWidths=[200, 120])
                automl_table.setStyle(style)
                elements.append(automl_table)

            elements.append(PageBreak())

            Currentdt=Currentdt
            chart_titles = Table(
                [
                    ["Current Decision Tree"],
                    [Paragraph(
                        f"Current {target_variable_name}: {float(current_tree_rate):.2f} | SXI: {current_tree_sxi}",
                        ParagraphStyle(
                            name="SubTitle",
                            fontSize=10,
                            alignment=TA_CENTER,
                            leading=12
                        )
                    )]
                ],
                colWidths=[400],
                rowHeights=[28, 22]
            )

            chart_titles.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

                # Border
                ('BOX', (0, 0), (-1, -1), 1, colors.black),

                # Title style
                ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (0, 0), 16),

                # Subtitle style
                ('FONTSIZE', (0, 1), (0, 1), 10),
                ('TEXTCOLOR', (0, 1), (0, 1), colors.black),

                # Padding
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))

            elements.append(chart_titles)
            elements.append(Spacer(1, 12))
            if current_tree_image:
                chart1 = Image(current_tree_image, width=500, height=300)
                chart_table = Table([[chart1]], colWidths=[250, 250])
                chart_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
                elements.append(chart_table)
            else:
                elements.append(Paragraph("Current decision tree image was unavailable for this run.", styles['NormalStyle']))
            elements.append(Spacer(1, 20))
            elements.append(PageBreak())

        elements.append(Spacer(1, 12))


    #  ------------ Page 6 Starts Here -----------------------------------------------------------------------------------------------------------

        # elements.append("Not Fraud")
    #     dt_not = """

    # • ELEVATORS_MEDI > 0.07: Suggests the individual resides in a better-equipped building, indicating stable applicants and lower fraud risk.
    # • DAYS_EMPLOYED > 135.5: Implies stable employment history, further reducing fraud likelihood.
    # • NONLIVINGAPARTMENTS_MEDI ≤ 0.001: Indicates standard housing with few non-living features, typically linked with lower fraud cases.
    # """
        # Decision path explanations - no bullets, just line-separated text
        # Define bullet-style lines using HTML formatting
        def extract_bullet_points(decision_text):
            return [line.strip().lstrip("• ").strip() for line in decision_text.split('\n') if line.strip()]

        
        def extract_bullet_points(decision_text):
            bullets = []
            for raw_line in (decision_text or "").split('\n'):
                line = raw_line.strip()
                if not line:
                    continue
                if line.lower().startswith("low ") or line.lower().startswith("high "):
                    continue
                bullets.append(line.lstrip("• ").strip())
            return bullets
        
        def extract_bullet_points(decision_text):
            bullets = []
            for raw_line in (decision_text or "").split('\n'):
                line = raw_line.strip()
                if not line:
                    continue
                if line.lower().startswith("low ") or line.lower().startswith("high "):
                    continue
                line = line.lstrip("•").lstrip("-").strip()
                bullets.append(line)
            return bullets

        # def wrap_imp(val):
        #     return Paragraph("" if val is None else str(val), styles["Normal"])
        
        header_cell_style = ParagraphStyle(
            "FeatureImportanceHeaderCell",
            parent=styles["Normal"],
            textColor=colors.white,
            fontName="Helvetica-Bold",
        )

        def wrap_imp(val, is_header=False):
            style = header_cell_style if is_header else styles["Normal"]
            return Paragraph("" if val is None else str(val), style)
        def safe_get(lst, idx, decimals=None):
            try:
                value = lst[idx]
                if decimals is not None and value is not None:
                    return round(value, decimals)
                return value
            except (IndexError, TypeError):
                return None

        # Extract bullet lines
        dt_not_bullets = extract_bullet_points(decision_0_current_for_section)
        dt_yes_bullets = extract_bullet_points(decision_1_current_for_section)
        
        # GOOD outcome section
        elements.append(Paragraph(f"<b>{good_outcome}</b>", styles['NormalStyle']))
        for bullet in dt_not_bullets:
            elements.append(Paragraph(f"• {bullet}", styles['SmallBulletStyle']))
            elements.append(Spacer(1, 3))

        elements.append(Spacer(1, 7))  # Spacing between good and bad outcome sections

        # BAD outcome section
        elements.append(Paragraph(f"<b>{bad_outcome}</b>", styles['NormalStyle']))
        for bullet in dt_yes_bullets:
            elements.append(Paragraph(f"• {bullet}", styles['SmallBulletStyle']))
            elements.append(Spacer(1, 3))
        #Feature Importance
        elements.append(Paragraph("<u><b>Feature Importance- Current Decision Tree </b></u>", styles['HeadingStyle']))
        elements.append(Spacer(1, 12))
        # data = [
        #         ['Features', 'SXI Weights','Algo1','Algo2','Algo3','Algo4','Algo5','Importance'],
        #         [feature1, round(top_values_sxi[0],2), round(lasso_vals[0],3), round(mi_vals[0],3), round(pca_vals[0],3), round(nb_vals[0],3), round(xgb_vals[0],3), featurevalue1],
        #         [feature2, round(top_values_sxi[1],2), round(lasso_vals[1],3), round(mi_vals[1],3), round(pca_vals[1],3), round(nb_vals[1],3), round(xgb_vals[1],3), featurevalue2],
        #         [feature3, round(top_values_sxi[2],2), round(lasso_vals[2],3), round(mi_vals[2],3), round(pca_vals[2],3), round(nb_vals[2],3), round(xgb_vals[2],3), featurevalue3],
        #         [feature4, round(top_values_sxi[3],2), round(lasso_vals[3],3), round(mi_vals[3],3), round(pca_vals[3],3), round(nb_vals[3],3), round(xgb_vals[3],3), featurevalue4],
        #         [feature5, round(top_values_sxi[4],2), round(lasso_vals[4],3), round(mi_vals[4],3), round(pca_vals[4],3), round(nb_vals[4],3), round(xgb_vals[4],3), featurevalue5]
        #         ]

        data = [
            [
                feature1,
                safe_get(top_values_sxi, 0, 2),
                safe_get(xgb_vals, 0, 3),
                safe_get(pca_vals, 0, 3),
                safe_get(lasso_vals, 0, 3),
                safe_get(mi_vals, 0, 3),
                safe_get(nb_vals, 0, 3),
                featurevalue1 if 'featurevalue1' in locals() else None,
            ],

            [
                feature2,
                safe_get(top_values_sxi, 1, 2),
                safe_get(xgb_vals, 1, 3),
                safe_get(pca_vals, 1, 3),
                safe_get(lasso_vals, 1, 3),
                safe_get(mi_vals, 1, 3),
                safe_get(nb_vals, 1, 3),
                featurevalue2 if 'featurevalue2' in locals() else None,
            ],

            [
                feature3,
                safe_get(top_values_sxi, 2, 2),
                safe_get(xgb_vals, 2, 3),
                safe_get(pca_vals, 2, 3),
                safe_get(lasso_vals, 2, 3),
                safe_get(mi_vals, 2, 3),
                safe_get(nb_vals, 2, 3),
                featurevalue3 if 'featurevalue3' in locals() else None,
            ],

            [
                feature4,
                safe_get(top_values_sxi, 3, 2),
                safe_get(xgb_vals, 3, 3),
                safe_get(pca_vals, 3, 3),
                safe_get(lasso_vals, 3, 3),
                safe_get(mi_vals, 3, 3),
                safe_get(nb_vals, 3, 3),
                featurevalue4 if 'featurevalue4' in locals() else None,
            ],

            [
                feature5,
                safe_get(top_values_sxi, 4, 2),
                safe_get(xgb_vals, 4, 3),
                safe_get(pca_vals, 4, 3),
                safe_get(lasso_vals, 4, 3),
                safe_get(mi_vals, 4, 3),
                safe_get(nb_vals, 4, 3),
                featurevalue5 if 'featurevalue5' in locals() else None,
            ],
        ]
        print("Data for feature importance table:", data)     
        # table = Table(data,colWidths=[2.2*inch]*5,rowHeights=[0.3*inch]*4)
        # data = [[wrap_imp(cell) for cell in row] for row in data]
        data = [
            [wrap_imp(cell, is_header=(row_index == 0)) for cell in row]
            for row_index, row in enumerate(data)
        ]

        from reportlab.lib.pagesizes import A4
        page_width = A4[0] - 72  # 36 left + 36 right margin

        col_widths = [
            page_width * 0.22,  # Features
            page_width * 0.10,  # SXI Weights
            page_width * 0.09,  # Algo1
            page_width * 0.09,  # Algo2
            page_width * 0.09,  # Algo3
            page_width * 0.09,  # Algo4
            page_width * 0.09,  # Algo5
            page_width * 0.13   # Importance
        ]

        table = Table(data, colWidths=col_widths, repeatRows=1)

        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.midnightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),

            ('ALIGN', (1, 1), (-2, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),

            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ])

        table.setStyle(style)
        elements.append(table)

        elements.append(Spacer(1, 20))
        Targetdt=Targetdt
        chart_titles = Table(
            [
                ["Target Decision Tree"],
                [Paragraph(
                    f"Target {target_variable_name}: {value1:.2f} | SXI: {sxi1}",
                    ParagraphStyle(
                        name="TargetSubTitle",
                        fontSize=10,
                        alignment=TA_CENTER,
                        leading=12
                    )
                )]
            ],
            colWidths=[400],
            rowHeights=[28, 22]
        )

        chart_titles.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

            # Border
            ('BOX', (0, 0), (-1, -1), 1, colors.black),

            # Title style
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, 0), 16),

            # Subtitle style
            ('FONTSIZE', (0, 1), (0, 1), 10),
            ('TEXTCOLOR', (0, 1), (0, 1), colors.black),

            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))

        elements.append(chart_titles)
        elements.append(Spacer(1, 12))
        # Chart placeholders — replace with actual images
        if target_tree_image:
            chart1 = Image(target_tree_image, width=500, height=300)
            chart_table = Table([[chart1]], colWidths=[250, 250])
            chart_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
            elements.append(chart_table)
        else:
            elements.append(Paragraph("Target decision tree image was unavailable for this run.", styles['NormalStyle']))
        elements.append(Spacer(1, 20))
        
        dt_not_bullets = extract_bullet_points(decision_0_tr_for_section)
        dt_yes_bullets = extract_bullet_points(decision_1_tr_for_section)

        # Add GOOD outcome bullets
        elements.append(Paragraph(f"<b>{good_outcome}</b>", styles['NormalStyle']))
        for bullet in dt_not_bullets:
            elements.append(Paragraph(f"• {bullet}", styles['SmallBulletStyle']))
            elements.append(Spacer(1, 3))  # Adjust spacing as needed

        elements.append(Spacer(1, 12))  # Space between good and bad outcome sections

        # Add BAD outcome bullets
        elements.append(Paragraph(f"<b>{bad_outcome}</b>", styles['NormalStyle']))
        for bullet in dt_yes_bullets:
            elements.append(Paragraph(f"• {bullet}", styles['SmallBulletStyle']))
            elements.append(Spacer(1, 3))
        #Feature Importance
        elements.append(Paragraph("<u><b>Feature Importance- Target Decision Tree </b></u>", styles['HeadingStyle']))
        elements.append(Spacer(1, 12))

        # data = [
        #         ['Features', 'SXI Weights','Algo1','Algo2','Algo3','Algo4','Algo5','Importance'],
        #         [feature6, round(top_val_tv_sxi[0],3),   round(lasso_vals_tr[0],3), round(mi_vals_tr[0],3), round(pca_vals_tr[0],3), round(nb_vals_tr[0],3), round(xgb_vals_tr[0],3), featurevalue6],
        #         [feature7, round(top_val_tv_sxi[1],3),   round(lasso_vals_tr[1],3), round(mi_vals_tr[1],3), round(pca_vals_tr[1],3), round(nb_vals_tr[1],3), round(xgb_vals_tr[1],3), featurevalue7],
        #         [feature8, round(top_val_tv_sxi[2],3),   round(lasso_vals_tr[2],3), round(mi_vals_tr[2],3), round(pca_vals_tr[2],3), round(nb_vals_tr[2],3), round(xgb_vals_tr[2],3), featurevalue8],
        #         [feature9, round(top_val_tv_sxi[3],3),   round(lasso_vals_tr[3],3), round(mi_vals_tr[3],3), round(pca_vals_tr[3],3), round(nb_vals_tr[3],3), round(xgb_vals_tr[3],3), featurevalue9],
        #         [feature10, round(top_val_tv_sxi[4],3),  round(lasso_vals_tr[4],3), round(mi_vals_tr[4],3), round(pca_vals_tr[4],3), round(nb_vals_tr[4],3), round(xgb_vals_tr[4],3), featurevalue10]
        #         ]

        data = [
            ['Features', 'SXI Weights', 'XG Boost', 'PCA', 'Lasso', 'MI', 'Naive Bayes','Importance'],

            [feature6, 
            safe_get(top_val_tv_sxi, 0, 3),
            safe_get(xgb_vals_tr, 0, 3),           
            safe_get(pca_vals_tr, 0, 3),           
            safe_get(lasso_vals_tr, 0, 3),         
            safe_get(mi_vals_tr, 0, 3),            
            safe_get(nb_vals_tr, 0, 3),
            featurevalue6 if 'featurevalue6' in locals() else None,
            ],         

            [feature7, 
            safe_get(top_val_tv_sxi, 1, 3),
            safe_get(xgb_vals_tr, 1, 3),
            safe_get(pca_vals_tr, 1, 3),
            safe_get(lasso_vals_tr, 1, 3),
            safe_get(mi_vals_tr, 1, 3),
            safe_get(nb_vals_tr, 1, 3),
            featurevalue7 if 'featurevalue7' in locals() else None,
            ],

            [feature8,
            safe_get(top_val_tv_sxi, 2, 3),
            safe_get(xgb_vals_tr, 2, 3),
            safe_get(pca_vals_tr, 2, 3),
            safe_get(lasso_vals_tr, 2, 3),
            safe_get(mi_vals_tr, 2, 3),
            safe_get(nb_vals_tr, 2, 3),
            featurevalue8 if 'featurevalue8' in locals() else None,
            ],

            [feature9,
            safe_get(top_val_tv_sxi, 3, 3),
            safe_get(xgb_vals_tr, 3, 3),
            safe_get(pca_vals_tr, 3, 3),
            safe_get(lasso_vals_tr, 3, 3),
            safe_get(mi_vals_tr, 3, 3),
            safe_get(nb_vals_tr, 3, 3),
            featurevalue9 if 'featurevalue9' in locals() else None,
            ],

            [feature10,
            safe_get(top_val_tv_sxi, 4, 3),
            safe_get(xgb_vals_tr, 4, 3),
            safe_get(pca_vals_tr, 4, 3),
            safe_get(lasso_vals_tr, 4, 3),
            safe_get(mi_vals_tr, 4, 3),
            safe_get(nb_vals_tr, 4, 3),
            featurevalue10 if 'featurevalue10' in locals() else None,
            ],
        ]


        # table = Table(data,colWidths=[2.2*inch]*5,rowHeights=[0.3*inch]*4)
        data = [
            [wrap_imp(cell, is_header=(row_index == 0)) for cell in row]
            for row_index, row in enumerate(data)
        ]
        page_width = A4[0] - 72  # 36 left + 36 right margin

        col_widths = [
            page_width * 0.22,  # Features
            page_width * 0.10,  # SXI Weights
            page_width * 0.09,  # Algo1
            page_width * 0.09,  # Algo2
            page_width * 0.09,  # Algo3
            page_width * 0.09,  # Algo4
            page_width * 0.09,  # Algo5
            page_width * 0.13   # Importance
        ]

        table = Table(data, colWidths=col_widths, repeatRows=1)

        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.midnightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),

            ('ALIGN', (1, 1), (-2, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),

            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ])

        table.setStyle(style)
        elements.append(table)



    #-------------------------------------------------------------------------------------------------------------------------
    #------------ChatGPT Last Page---------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------------



    # elif categorical:
    #     pass
        elements.append(PageBreak())


    # except Exception as e:
    #             print(f"Error generating PDF: {e}")



            # # Save the PDF
            # try:

        
        build_start = time.perf_counter()
        print("generate_pdf_prof: starting final doc.build")
        doc.build(elements)
        print(f"generate_pdf_prof: final doc.build took {time.perf_counter() - build_start:.2f}s")
        print(f"PDF saved as {filename}")
            # except Exception as e:
            #     print(f"Error generating PDF: {e}")


    def build_actual_predicted_table(self, actlocs, target_variable):

        table_data = [
            [f"Actual {target_variable}", f"Predicted {target_variable}"]
        ]

        if actlocs is None:
            print("actlocs is None")
            return table_data

        try:

            if isinstance(actlocs, list):

                for row in actlocs[:10]:

                    if isinstance(row, (list, tuple)) and len(row) >= 2:

                        actual = round(float(row[0]))
                        pred = round(float(row[1]))

                        table_data.append([actual, pred])

        except Exception as e:
            print("Actual vs Predicted table error:", e)

        return table_data

    def generate_pdf_prof_user(self,filename,target_variable_name=None, target_mean=None, table_data=None, sriya_logo=None, plot_target_dist=None,
                    plot_vrm_dist=None,perce_inc=None, tv_inc=None, bad_outcome=None,good_outcome=None,Targetdt=None, Currentdt=None, risk_index=None,blue_banner=None,
                    tv_type=None, model_accuracy=None, precision_score=None, area_under_curve=None, MAE_Perf=None,R2_Perf=None, Error_Perf=None,
                    Actual_fig=None, AUC=None, confusion_matrix=None, ClassRate=None, CurrentRate=None, CurrentSXI=None,
                    perc1=None, perc2=None, perc3=None, value1=None, value2=None, value3=None, name=None, init=None, mid=None, end=None, 
                    conv=None, sxi1=None, sxi2=None, sxi3=None, 
                    improvementcard1=None, improvementcard2=None, improvementcardvalue2=None, improvementcard3=None,improvementcardvalue3=None,
                    improvementcard4=None, improvementcardvalue4=None, improvementcard5=None, improvementcardvalue5=None,improvementcard6=None,improvementcardvalue6=None,
                    MAE_lnm=None, R2_score_lnm=None, area_under_curve_lnm=None, Accuracy_lnm=None, Precision_lnm=None, Recall_lnm=None, Recall_score=None,
                    feature1=None,feature2=None,feature3=None, feature4=None, feature5=None,feature6=None,feature7=None,feature8=None, feature9=None, 
                    feature10=None,featurevalue1=None,featurevalue2=None,featurevalue3=None,featurevalue4=None,featurevalue5=None,
                    featurevalue6=None,featurevalue7=None,featurevalue8=None,featurevalue9=None,featurevalue10=None,corr_plot=None,
                    edaparagraph2=None,edaparagraph1=None,decision_1_current=None, decision_0_current=None, 
                    decision_0_tr=None, decision_1_tr=None, title=None,Corr_explanation=None, 
                    lasso_vals_tr=None, mi_vals_tr=None, pca_vals_tr=None, nb_vals_tr=None, xgb_vals_tr=None,
                    lasso_vals=None, mi_vals=None, pca_vals=None, nb_vals=None, xgb_vals=None, top_val_tv_sxi=None, top_values_sxi=None,
                    feat1=None, feat2=None, feat3=None, feat4=None, feat5=None, 
                    prcnt1=None, prcnt2=None, prcnt3=None, prcnt4=None, prcnt5=None,
                    coef1=None, coef2=None, coef3=None, coef4=None, coef5=None,
                    vv1=None, vv2=None, vv3=None, vv4=None, vv5=None,
                    v1=None, v2=None, v3=None, v4=None, v5=None, report_section_overrides=None,
                    prediff=None, adjusted_target_value=None, first_target_value=None,prediffperc=None, actlocs=None, result=None, combined_weights=None):

        
        print(f"Generating PDF: {filename}")
        blue_banner = self._get_report_asset("blue.png")
        sriya_logo = self._get_report_asset("Sriya_new_logo.png")
        doc = SimpleDocTemplate(filename, pagesize=LETTER,
                                rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
        #------header footer 
        styles = getSampleStyleSheet()
        subtitle_style = ParagraphStyle('SubtitleStyle',parent=styles['Heading2'],  # You can base it on another style, like 'Heading2'
        fontSize=14,  # Adjust the font size
        textColor=colors.darkblue,  # Change the color to dark blue
        spaceAfter=12  # Space after the subtitle
        )
        styles.add(ParagraphStyle(name='SmallBulletStyle',fontSize=9.5, leading=11.5, spaceAfter=4))
        styles.add(subtitle_style)
        styles.add(ParagraphStyle(name='TitleStyle', fontSize=20, alignment=1, spaceAfter=20, leading=28))
        styles.add(ParagraphStyle(name='HeadingStyle', fontSize=14, alignment=1, spaceAfter=10, leading=20))
        styles.add(ParagraphStyle(name='NormalStyle', fontSize=10, leading=15))
        styles.add(ParagraphStyle(name='MicrosoftSansSerif',fontName='MicrosoftSansSerif',fontSize=12,leading=15,textColor=colors.black)
        )
        page1_dataset_title_style = ParagraphStyle(
            "page1_dataset_title",
            parent=styles["Normal"],
            fontSize=20,
            leading=22,
            textColor=colors.HexColor("#2E3192"),
            alignment=TA_LEFT,
        )

        def format_two_decimals(value, default="N/A"):
            try:
                return f"{float(value):.2f}"
            except (TypeError, ValueError):
                return default if value is None else str(value)

        def format_report_display_value(value, display_target_name=None, display_target_type=None):
            return format_display_value(
                value,
                display_target_name if display_target_name is not None else target_variable_name,
                display_target_type if display_target_type is not None else tv_type,
                decimals=2,
            )

        send_to_llm = bool(self.request.session.get("send_to_llm", False))

       
        # def draw_footer_only(canvas, doc):

        #     canvas.saveState()

        #     # -------- HEADER BLUE BACKGROUND --------
        #     blue_banner = r"D:\work\SriyaAI-Agent-System\AGENT12\sriya_chatbot\static\images\blue.png"

        #     canvas.drawImage(
        #         blue_banner,
        #         x=0,          # start from left
        #         y=720,        # vertical position
        #         width=612,    # full page width (LETTER)
        #         height=90,    # banner height
        #         mask='auto'
        #     )

        #     # -------- FOOTER --------
        #     footer_text = "HIGHLY CONFIDENTIAL – Sriya.AI copyright © 2025"

        #     canvas.setFont("Helvetica", 9)
        #     canvas.drawString(30, 20, footer_text)

        #     canvas.restoreState()
        blue_banner = self._get_report_asset("blue.png")
        sriya_logo = self._get_report_asset("Sriya_new_logo.png")

        def draw_footer_only(canvas, doc):

            canvas.saveState()

            page_width, page_height = doc.pagesize

            # -------- BLUE HEADER BACKGROUND --------
            self._draw_pdf_header_banner(
                canvas,
                blue_banner,
                0,
                page_height - 95,
                page_width,
                95,
            )

            # -------- SRIYA LOGO --------
            self._draw_pdf_logo(
                canvas,
                sriya_logo,
                page_width - 180,
                page_height - 80,
                150,
                45,
            )

            safe_dataset_name = self._sanitize_reportlab_markup(dataset_name or "Dataset")
            title_para = Paragraph(f"<b><i>{safe_dataset_name}</i></b>", page1_dataset_title_style)
            title_x = doc.leftMargin + 2
            title_y = page_height - 72
            title_width = page_width - title_x - 220
            title_para.wrap(title_width, 36)
            title_para.drawOn(canvas, title_x, title_y)

            canvas.setStrokeColor(colors.red)
            canvas.setLineWidth(2)
            canvas.line(25, page_height - 118, page_width - 120, page_height - 118)

            # -------- FOOTER --------
            footer_text = f"HIGHLY CONFIDENTIAL – Sriya.AI copyright © {datetime.now().year}"

            canvas.setFont("Helvetica", 9)
            canvas.drawString(30, 20, footer_text)

            canvas.restoreState()
        def draw_header_footer(canvas, doc):

            canvas.saveState()

            # -------- LIGHT BLUE BACKGROUND BEHIND LOGO --------
            canvas.setFillColorRGB(0.25, 0.45, 0.65)

            canvas.rect(
                25,
                750,
                160,
                45,
                stroke=0,
                fill=1
            )

            # -------- LOGO --------
            self._draw_pdf_logo(
                canvas,
                sriya_logo,
                30,
                755,
                150,
                40,
            )

            # -------- FOOTER --------
            footer_text = f"HIGHLY CONFIDENTIAL – Sriya.AI copyright © {datetime.now().year}"

            canvas.setFillColorRGB(0,0,0)
            canvas.setFont("Helvetica", 9)
            canvas.drawString(30, 20, footer_text)

            page_num = canvas.getPageNumber()

            if page_num > 1:
                canvas.drawCentredString(300, 20, str(page_num - 1))

            canvas.restoreState()
        elements = []
    #------- page 1-----------------

        def draw_page1_background(canvas, doc):
            self._draw_pdf_header_banner(
                canvas,
                blue_banner,
                0,
                LETTER[1] - 1.6 * inch,
                LETTER[0],
                1.6 * inch,
            )

        elements = []

        report_section_overrides = dict(report_section_overrides or {})

        def pick_report_value(key, default):
            override_value = report_section_overrides.get(key)
            if override_value is None:
                return default
            if isinstance(override_value, str) and override_value == "":
                return default
            if isinstance(override_value, np.ndarray) and override_value.size == 0:
                return default
            if isinstance(override_value, (pd.Series, pd.DataFrame)) and override_value.empty:
                return default
            if isinstance(override_value, (list, tuple, set, dict)) and len(override_value) == 0:
                return default
            return override_value

        # ---------------- DATASET TITLE ----------------
        import glob

        # ---------------- LOAD ACTIVE ANALYSIS DATASET ----------------
        df, dataset_name, rows, cols = self._get_processed_dataset_details()
        original_df, _, _, original_cols = self._get_processed_dataset_details(prefer_encoded=False)
        feature_df = original_df if original_df is not None else df
        description_df = feature_df if feature_df is not None else df
        feature_cols_count = int(feature_df.shape[1]) if feature_df is not None else int(original_cols or cols)
        elements.append(Spacer(1, 110))


        # ---------------- DATA DESCRIPTION ----------------

        section_style = ParagraphStyle(
            "section",
            parent=styles["Heading2"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#2E3192")
        )

        elements.append(Paragraph("<u><b>Data Description</b></u>", section_style))
        elements.append(Spacer(1, 10))

        if description_df is not None:
            rows = description_df.shape[0]
            cols = description_df.shape[1]
        else:
            rows = 0
            cols = 0

        elements.append(
            Paragraph(
                f"The dataset consists of {rows} rows and {cols} columns",
                styles["NormalStyle"]
            )
        )

        elements.append(Spacer(1, 15))


        # ---------------- DATASET TABLE ----------------

        data_table = [
            ["Dataset", "Number of Rows", "Number of Columns"],
            [dataset_name, str(rows), str(cols)]
        ]

        table = Table(data_table, colWidths=[150,150,150])

        table.setStyle(TableStyle([

        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#4F81BD")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),

        ('BACKGROUND',(0,1),(0,1),colors.HexColor("#4F81BD")),
        ('TEXTCOLOR',(0,1),(0,1),colors.white),

        ('BACKGROUND',(1,1),(-1,1),colors.HexColor("#DCE6F1")),

        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('GRID',(0,0),(-1,-1),1,colors.white)

        ]))

        elements.append(table)

        elements.append(Spacer(1, 25))


        # ---------------- TARGET VARIABLE ----------------

        elements.append(Paragraph("<u><b>Target Variable</b></u>", styles['HeadingStyle']))
        elements.append(Spacer(1, 6))

        elements.append(
            Paragraph(
                f"Target Variable Used: <u>{target_variable_name}</u>",
                styles["NormalStyle"]
            )
        )

        elements.append(Spacer(1, 20))


        # ---------------- FEATURE LIST ----------------

        elements.append(Paragraph("<u><b>Feature (Column) Names</b></u>", styles['HeadingStyle']))

        elements.append(
            Paragraph(
                f"The processed dataset includes the following {feature_cols_count} original columns:",
                styles["NormalStyle"]
            )
        )

        elements.append(Spacer(1, 10))


        # ---------------- FEATURE LIST ----------------

        clean_columns = [str(col) for col in feature_df.columns]

        feature_rows = []

        for i in range(0, len(clean_columns), 2):
            col1 = clean_columns[i]
            col2 = clean_columns[i+1] if i+1 < len(clean_columns) else ""
            feature_rows.append([col1, col2])

        feature_table = Table(feature_rows, colWidths=[225, 225])

        feature_table.setStyle(TableStyle([
            ('BOX',(0,0),(-1,-1),1,colors.black),

            ('LEFTPADDING',(0,0),(-1,-1),10),
            ('RIGHTPADDING',(0,0),(-1,-1),10),
            ('TOPPADDING',(0,0),(-1,-1),5),
            ('BOTTOMPADDING',(0,0),(-1,-1),5),

            ('ALIGN',(0,0),(-1,-1),'LEFT')
        ]))

        elements.append(feature_table)
    # ------------------Page 2-------------------------------------------------------------------------------------------------------------------
        elements.append(PageBreak())
        # elements.append(Paragraph("<u><b>Objective</b></u>", styles['HeadingStyle']))
        # objective_outcome = good_outcome if str(tv_inc).strip().lower() in ("increase", "increased") else bad_outcome
        # perce_inc_text = format_two_decimals(perce_inc, default=str(perce_inc))
        # objective_text = self._generate_objective_text(
        #     perce_inc_text=perce_inc_text,
        #     tv_inc=tv_inc,
        #     objective_outcome=objective_outcome,
        #     target_variable_name=target_variable_name,
        # )
        # elements.append(Paragraph(objective_text, styles['SmallBulletStyle']))
        # elements.append(Spacer(1, 12))

        elements.append(Paragraph("<u><b>Definitions</b></u>", styles['HeadingStyle']))
        definitions = [
        f"• SXI: SXI risk index derived by dynamic weighted average of all important features. 5-7 Machine Learning algorithms in a proprietary formula gives this score. Every ID will get an SXI Score.",
        "• SXI: SXI enhanced by Deep Neural Network based Reinforced Learning.",
        "• LNM: Large Numerical Model powered by SXI. LNMs are “curated pre-trained and pre-validated models capable of predicting unseen data with very high accuracy and precision.”"
        ]
        for defn in definitions:
            elements.append(Paragraph(defn, styles['SmallBulletStyle']))
            elements.append(Spacer(1, 3))

        discussion_plot_vrm = pick_report_value("plot_vrm_dist", plot_vrm_dist)
        discussion_plot_target = pick_report_value("plot_target_dist", plot_target_dist)
        eda_bullets = pick_report_value("edaparagraph2", edaparagraph2)
        corr_plot_for_section = pick_report_value("corr_plot", corr_plot)
        corr_explanation_for_section = pick_report_value("Corr_explanation", Corr_explanation)
        corr_text_for_section = pick_report_value("correlation_section_text", None)
        current_tree_image = pick_report_value("Currentdt", Currentdt)
        current_tree_rate = pick_report_value("CurrentRate", CurrentRate)
        current_tree_sxi = pick_report_value("CurrentSXI", CurrentSXI)
        target_tree_image = pick_report_value("Targetdt", Targetdt)
        target_mean_for_original_sections = pick_report_value("target_mean", target_mean)
        perc1_for_original_sections = pick_report_value("perc1", perc1)
        perc2_for_original_sections = pick_report_value("perc2", perc2)
        perc3_for_original_sections = pick_report_value("perc3", perc3)
        value1_for_original_sections = pick_report_value("value1", value1)
        value2_for_original_sections = pick_report_value("value2", value2)
        value3_for_original_sections = pick_report_value("value3", value3)
        sxi1_for_original_sections = pick_report_value("sxi1", sxi1)
        sxi2_for_original_sections = pick_report_value("sxi2", sxi2)
        sxi3_for_original_sections = pick_report_value("sxi3", sxi3)
        R2_score_for_original_sections = pick_report_value("R2_score_lnm", R2_score_lnm)
        Accuracy_for_original_sections = pick_report_value("Accuracy_lnm", Accuracy_lnm)
        current_tree_features_for_section = [
            pick_report_value("feature1", feature1),
            pick_report_value("feature2", feature2),
            pick_report_value("feature3", feature3),
            pick_report_value("feature4", feature4),
            pick_report_value("feature5", feature5),
        ]
        current_tree_feature_values_for_section = [
            pick_report_value("featurevalue1", featurevalue1),
            pick_report_value("featurevalue2", featurevalue2),
            pick_report_value("featurevalue3", featurevalue3),
            pick_report_value("featurevalue4", featurevalue4),
            pick_report_value("featurevalue5", featurevalue5),
        ]
        target_tree_features_for_section = [
            pick_report_value("feature6", feature6),
            pick_report_value("feature7", feature7),
            pick_report_value("feature8", feature8),
            pick_report_value("feature9", feature9),
            pick_report_value("feature10", feature10),
        ]
        target_tree_feature_values_for_section = [
            pick_report_value("featurevalue6", featurevalue6),
            pick_report_value("featurevalue7", featurevalue7),
            pick_report_value("featurevalue8", featurevalue8),
            pick_report_value("featurevalue9", featurevalue9),
            pick_report_value("featurevalue10", featurevalue10),
        ]
        decision_0_current_for_section = pick_report_value("decision_0_current", decision_0_current)
        decision_1_current_for_section = pick_report_value("decision_1_current", decision_1_current)
        decision_0_tr_for_section = pick_report_value("decision_0_tr", decision_0_tr)
        decision_1_tr_for_section = pick_report_value("decision_1_tr", decision_1_tr)
        
        # Discussion
        elements.append(Paragraph("<u><b>Discussion & Results</b></u>", styles['HeadingStyle']))
        elements.append(Spacer(1, 6))

        # Subtitles for charts
        centered = ParagraphStyle(name="centered", parent=styles['NormalStyle'], alignment=TA_CENTER, fontSize=12,leading=10, fontName="Helvetica-Bold",)
        SXI=Paragraph(f"<b>DXI Distribution of {target_variable_name}</b>",centered)
        Target=Paragraph("<b>Target Distribution</b>",centered)
        chart_titles = Table([
            [SXI,Target],], colWidths=[250, 250])
        chart_titles.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(chart_titles)
        elements.append(Spacer(1, 12))

        # Chart placeholders — replace with actual images
        try:
            chart1 = Image(discussion_plot_vrm, width=300, height=200)
            chart2 = Image(discussion_plot_target, width=300, height=200)
            chart_table = Table([[chart1, chart2]], colWidths=[250, 250])
            chart_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
            elements.append(chart_table)
            elements.append(Spacer(1, 3))
        except Exception as e:
            elements.append(Paragraph("Charts not available.", styles['NormalStyle']))

        # SXI EDA Section
        elements.append(Paragraph("<u><b>SXI Exploratory Data Analysis</b></u>", styles['HeadingStyle']))

        self._append_eda_content(elements, eda_bullets, styles)

    # ----- Page 3 Starts Here -----------------------------------------------------------------------------------------------------------
        elements.append(PageBreak())
        # Title
        elements.append(Paragraph("<u><b>Performance matrix</b></u>", styles['HeadingStyle']))
        elements.append(Spacer(1, 12))


        # ---------------- PERFORMANCE TABLE ----------------
        # =========================================================
        # REGRESSION PERFORMANCE TABLE
        # =========================================================
        if tv_type == "Numeric":

            chatgpt_r2 = self._coerce_summary_float(R2_Perf)
            chatgpt_mae = self._coerce_summary_float(MAE_Perf)
            chatgpt_mape = self._coerce_summary_float(Error_Perf)

            sxi_r2 = self._coerce_summary_float(R2_score_lnm)
            sxi_mae = self._coerce_summary_float(MAE_lnm)
            sxi_mape = self._coerce_summary_float(area_under_curve_lnm)

            chatgpt_r2_display = self._format_optional_report_number(chatgpt_r2, decimals=2)
            sxi_r2_display = self._format_optional_report_number(sxi_r2, decimals=2)
            chatgpt_mae_display = self._format_optional_report_value(chatgpt_mae, target_variable_name, tv_type, decimals=2)
            sxi_mae_display = self._format_optional_report_value(sxi_mae, target_variable_name, tv_type, decimals=2)
            chatgpt_mape_display = self._format_optional_report_number(chatgpt_mape, decimals=2)
            sxi_mape_display = self._format_optional_report_number(sxi_mape, decimals=2)
            perf_data = [
                ["Model", "R²", "MAE", "MAPE"],
                ["DXI", sxi_r2_display, sxi_mae_display, sxi_mape_display],
            ]
            if send_to_llm:
                perf_data.insert(1, ["ChatGPT", chatgpt_r2_display, chatgpt_mae_display, chatgpt_mape_display])

            perf_table = Table(
                perf_data,
                colWidths=[150, 120, 150, 120],
                hAlign="CENTER"
            )

            perf_table.setStyle(TableStyle([

                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#4F81BD")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),

                ('BACKGROUND', (0,1), (0,-1), colors.HexColor("#4F81BD")),
                ('TEXTCOLOR', (0,1), (0,-1), colors.white),

                ('BACKGROUND', (1,1), (-1,-1), colors.HexColor("#DCE6F1")),

                ('ALIGN',(0,0),(-1,-1),'CENTER'),
                ('GRID',(0,0),(-1,-1),1,colors.white),

            ]))

            elements.append(perf_table)
            elements.append(Spacer(1,15))


            # ---------------- PERFORMANCE EXPLANATION ----------------

            if send_to_llm and all(
                value is not None
                for value in [sxi_r2, chatgpt_r2, sxi_mae, chatgpt_mae]
            ):
                explanation = self._generate_regression_model_comparison_explanation(
                    target_variable_name=target_variable_name,
                    sxi_r2=sxi_r2,
                    chatgpt_r2=chatgpt_r2,
                    sxi_mae=sxi_mae,
                    chatgpt_mae=chatgpt_mae,
                    act_vs_pred_image_path=actlocs,
                )
                elements.append(Paragraph(explanation, styles['NormalStyle']))
            elif send_to_llm:
                elements.append(Paragraph(
                    "SXI regression performance metrics were unavailable for this run, so the table shows N/A instead of a numeric score.",
                    styles['NormalStyle']
                ))
            else:
                elements.append(Paragraph("SXI-only performance is shown for this run.", styles['NormalStyle']))

            elements.append(Spacer(1, 15))
        
        # =========================================================
        # CLASSIFICATION PERFORMANCE TABLE
        # =========================================================
        else:

            chatgpt_acc = self._metric(
                self._normalize_classification_percentage(model_accuracy)
            )
            chatgpt_prec = self._metric(
                self._normalize_classification_percentage(precision_score)
            )
            chatgpt_recall = self._metric(
                self._normalize_classification_percentage(Recall_score)
            )
            chatgpt_auc = self._metric(area_under_curve)

            sxi_acc = self._normalize_classification_percentage(Accuracy_lnm)
            sxi_prec = self._normalize_classification_percentage(Precision_lnm)
            sxi_rec = self._normalize_classification_percentage(Recall_lnm)
            sxi_auc = area_under_curve_lnm if area_under_curve_lnm else 0

            perf_data = [
                ["Model", "Accuracy", "Precision", "Recall", "AUC"],
                ["DXI",
                f"{float(sxi_acc):.2f}",
                f"{float(sxi_prec):.2f}",
                f"{float(sxi_rec):.2f}",
                f"{float(sxi_auc):.2f}"]
            ]
            if send_to_llm:
                perf_data.insert(1, [
                    "ChatGPT",
                    f"{float(chatgpt_acc):.2f}",
                    f"{float(chatgpt_prec):.2f}",
                    f"{float(chatgpt_recall):.2f}",
                    f"{float(chatgpt_auc):.2f}"
                ])

            perf_table = Table(
                perf_data,
                colWidths=[120,90,90,90,90],
                hAlign="CENTER"
            )

            perf_table.setStyle(TableStyle([

                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#4F81BD")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),

                ('BACKGROUND', (0,1), (0,-1), colors.HexColor("#4F81BD")),
                ('TEXTCOLOR', (0,1), (0,-1), colors.white),

                ('BACKGROUND', (1,1), (-1,-1), colors.HexColor("#DCE6F1")),

                ('ALIGN',(0,0),(-1,-1),'CENTER'),
                ('GRID',(0,0),(-1,-1),1,colors.white),

            ]))

            elements.append(perf_table)
            elements.append(Spacer(1,15))

            if send_to_llm:
                explanation = self._generate_classification_model_comparison_explanation(
                    sxi_acc=sxi_acc,
                    sxi_prec=sxi_prec,
                    sxi_rec=sxi_rec,
                    sxi_auc=sxi_auc,
                    chatgpt_acc=chatgpt_acc,
                    chatgpt_prec=chatgpt_prec,
                    chatgpt_recall=chatgpt_recall,
                    chatgpt_auc=chatgpt_auc,
                )
                elements.append(Paragraph(explanation, styles['NormalStyle']))
            else:
                elements.append(Paragraph("SXI-only performance is shown for this run.", styles['NormalStyle']))
        elements.append(Spacer(1, 15))

        # VISUALIZATION SECTION (REGRESSION / CLASSIFICATION)
        # ---------------------------------------------------------
        # REGRESSION CASE--
        if tv_type == "Numeric":

            # ---------------- ACTUAL VS PREDICTED GRAPH ----------------
            elements.append(Paragraph("<b>Actual vs Predicted</b>", styles['HeadingStyle']))
            elements.append(Spacer(1, 10))

            try:
                if actlocs:
                    scatter_plot = Image(actlocs, width=450, height=300)
                    scatter_plot.hAlign = "CENTER"
                    elements.append(scatter_plot)
                else:
                    elements.append(
                        Paragraph("Actual vs Predicted plot not available.", styles['NormalStyle'])
                    )
            except Exception:
                elements.append(
                    Paragraph("Actual vs Predicted plot not available.", styles['NormalStyle'])
                )

            elements.append(Spacer(1, 12))


            # ---------------- WITHIN 10% EXPLANATION ----------------

            def safe_float(val, default=0.0):
                try:
                    return float(val)
                except:
                    return default

            within_pct = safe_float(result.get("sxi_within_percentage"))

            within_text = f"""
            SXI produces predictions within ±10% of the actual values
            for {within_pct:.2f}% of the evaluated data points.

            This metric indicates the proportion of predictions that closely
            align with observed outcomes, reflecting model reliability.
            """

            actual_vs_predicted_explanation = self._generate_actual_vs_predicted_explanation(
                target_variable_name=target_variable_name,
                within_pct=within_pct,
                r2_score=R2_score_lnm if R2_score_lnm is not None else None,
                mae_value=MAE_lnm if MAE_lnm is not None else None,
                act_vs_pred_image_path=actlocs,
            )
            elements.append(Paragraph(actual_vs_predicted_explanation, styles['NormalStyle']))
            elements.append(Spacer(1, 20))

        # CLASSIFICATION CASE
        else:


            # ---------------- ROC CURVE ----------------
            #Confuson Matrix
            figure_heading = ParagraphStyle(
                name="FigureHeading",
                fontSize=12,
                alignment=TA_CENTER,
                fontName="Helvetica-Bold",
                spaceAfter=6
            )

            if confusion_matrix and AUC:
                # Images
                conf_img = Image(confusion_matrix, width=150, height=150)
                auc_img = Image(AUC, width=150, height=150)

                # Headings
                conf_title = Paragraph("Confusion Matrix", figure_heading)
                auc_title = Paragraph("AUC ROC Curve", figure_heading)

                # Table layout: headings row + image row
                chart_table = Table(
                    [
                        [conf_title, auc_title],
                        [conf_img, auc_img]
                    ],
                    colWidths=[250, 250],
                    rowHeights=[30, 220]
                )
            else:
                chart_table = Table(
                    [[Paragraph(
                        "Confusion matrix and ROC curve images were unavailable for this run.",
                        styles['NormalStyle']
                    )]],
                    colWidths=[500]
                )
            bullet_style = ParagraphStyle(
                name="ConfusionBullet",
                fontSize=9,
                leading=12,
                leftIndent=8
            )
            confusion_box_lines = self._generate_confusion_box_lines(
                target_variable_name=target_variable_name,
                good_outcome=good_outcome,
                bad_outcome=bad_outcome,
                acc_val=Accuracy_lnm,
                prec_val=Precision_lnm,
                rec_val=Recall_lnm,
                auc_val=area_under_curve_lnm,
                confusion_matrix_path=confusion_matrix,
            )

            confusion_box = Table(
                [[
                    Paragraph("• TP (True Positive): correctly detected fraud cases.", bullet_style),
                ], [
                    Paragraph("• FP (False Positive): genuine cases incorrectly flagged as fraud.", bullet_style),
                ], [
                    Paragraph("• TN (True Negative): genuine cases correctly identified as non-fraud.", bullet_style),
                ], [
                    Paragraph("• FN (False Negative): fraud cases missed by the model.", bullet_style),
                ]],
                colWidths=[500]
            )
            confusion_box = Table(
                [[Paragraph(f"&bull; {line}", bullet_style)] for line in confusion_box_lines],
                colWidths=[500]
            )

            confusion_box.setStyle(TableStyle([
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ]))

            elements.append(
                KeepTogether([
                    chart_table,
                    Spacer(1, 10),
                    confusion_box,
                    Spacer(1, 12)
                ])
            )
            chart_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

                # Optional border (comment out if not needed)
                ('BOX', (0, 0), (-1, -1), 1, colors.black),

                # Padding
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            # ---------------- CLASSIFICATION EXPLANATION ----------------

            def safe_float(val, default=0.0):
                try:
                    return float(val)
                except:
                    return default
            sxi_acc = Accuracy_lnm if Accuracy_lnm else 0
            sxi_prec = Precision_lnm if Precision_lnm else 0
            sxi_rec = Recall_lnm if Recall_lnm else 0
            sxi_auc = area_under_curve_lnm if area_under_curve_lnm else 0
            acc_val = safe_float(sxi_acc)
            prec_val = safe_float(sxi_prec)
            rec_val = safe_float(sxi_rec)
            auc_val = safe_float(sxi_auc)
            explanation = self._generate_classification_chart_explanation(
                acc_val=acc_val,
                prec_val=prec_val,
                rec_val=rec_val,
                auc_val=auc_val,
                confusion_matrix_path=confusion_matrix,
                roc_curve_path=AUC,
            )

            elements.append(Paragraph(explanation, styles['NormalStyle']))
            elements.append(Spacer(1, 20))
    #----- Page 4 Starts Here --------------------------------------------------------------------------
        elements.append(PageBreak())
        elements.append(Spacer(1, 40))


        # ---------------- KPI CARDS (TOP) ----------------

        metric_card_label = f"{target_variable_name} R² Score"
        metric_card_value = (
            R2_score_for_original_sections
            if R2_score_for_original_sections is not None
            else 0.0
        )
        if tv_type == "Categorical":
            metric_card_label = f"{target_variable_name} Accuracy"
            metric_card_value = (
                Accuracy_for_original_sections
                if Accuracy_for_original_sections is not None
                else 0.0
            )
        current_sxi_card_value = self._coerce_summary_float(current_tree_sxi)
        if current_sxi_card_value is None:
            current_sxi_card_value = 0.0

        current_target_value = format_report_display_value(current_tree_rate)
        initial_target_value = format_report_display_value(value1_for_original_sections)
        mid_target_value = format_report_display_value(value2_for_original_sections)
        long_target_value = format_report_display_value(value3_for_original_sections)

        card_data = [
        [
        Paragraph("<b>Current DXI</b>", styles['NormalStyle']),
        Paragraph(f"<b>{current_sxi_card_value:.2f}</b>", styles['HeadingStyle'])
        ],
        [
        Paragraph(f"<b>Current {target_variable_name}</b>", styles['NormalStyle']),
        Paragraph(f"<b>{current_target_value}</b>", styles['HeadingStyle'])
        ],
        [
        Paragraph(f"<b>{metric_card_label}</b>", styles['NormalStyle']),
        Paragraph(f"<b>{metric_card_value:.2f}</b>", styles['HeadingStyle'])
        ]
        ]


        card_table = Table([card_data], colWidths=[180,180,180])


        card_table.setStyle(TableStyle([

        ('BACKGROUND',(0,0),(0,0),colors.HexColor("#4A90E2")),
        ('BACKGROUND',(1,0),(1,0),colors.HexColor("#48C9B0")),
        ('BACKGROUND',(2,0),(2,0),colors.HexColor("#F5B041")),

        ('TEXTCOLOR',(0,0),(-1,-1),colors.white),

        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),

        ('BOX',(0,0),(-1,-1),0.5,colors.white),

        ('BOTTOMPADDING',(0,0),(-1,-1),15),
        ('TOPPADDING',(0,0),(-1,-1),15)

        ]))

        elements.append(card_table)

        elements.append(Spacer(1,40))

        # ---------------- KPI CARDS (BOTTOM) ----------------

        bottom_cards = [
        [
        Paragraph("<b>Target DXI</b>", styles['NormalStyle']),
        Paragraph(f"<b>{sxi1_for_original_sections:.2f}</b>", styles['HeadingStyle'])
        ],
        [
        Paragraph(f"<b>Target {target_variable_name}</b>", styles['NormalStyle']),
        Paragraph(f"<b>{initial_target_value}</b>", styles['HeadingStyle'])
        ],
        [
        Paragraph(f"<b>{target_variable_name} Change</b>", styles['NormalStyle']),
        Paragraph(f"<b>{perc1_for_original_sections:.2f}%</b>", styles['HeadingStyle'])
        ]
        ]


        bottom_table = Table([bottom_cards], colWidths=[180,180,180])


        bottom_table.setStyle(TableStyle([

        ('BACKGROUND',(0,0),(0,0),colors.HexColor("#FF5A6E")),
        ('BACKGROUND',(1,0),(1,0),colors.HexColor("#4A90E2")),
        ('BACKGROUND',(2,0),(2,0),colors.HexColor("#48C9B0")),

        ('TEXTCOLOR',(0,0),(-1,-1),colors.white),

        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),

        ('BOX',(0,0),(-1,-1),0.5,colors.white),

        ('BOTTOMPADDING',(0,0),(-1,-1),15),
        ('TOPPADDING',(0,0),(-1,-1),15)

        ]))

        elements.append(bottom_table)

        elements.append(Spacer(1,20))
        elements.append(PageBreak())
        #------------------------------------------------
        elements.append(Spacer(1, 20))
        # ---------------- CORRELATION TEXT ----------------
        normalized_tv_inc = (tv_inc or "").strip().lower()
        is_reduction_case = self._is_reduction_direction(normalized_tv_inc)
        improvement_direction = "decrease" if is_reduction_case else "increase"
        movement_word = "DOWN" if is_reduction_case else "UP"
        action_verb = "reduce" if is_reduction_case else "increase"

        metric_val = self._resolve_correlation_metric_value(
            tv_type=tv_type,
            result=result,
        )
        slope_val = self._resolve_correlation_slope(result=result)
        corr_text = corr_text_for_section or self._generate_correlation_text(
            tv_type=tv_type,
            target_variable_name=target_variable_name,
            metric_val=metric_val,
            slope_val=slope_val,
            current_target_value=current_tree_rate,
            corr_plot_path=corr_plot_for_section,
            desired_target_direction=improvement_direction,
        )

        elements.append(Paragraph(corr_text, styles['NormalStyle']))
        elements.append(Spacer(1,15))

        # ---------------- CORRELATION GRAPH ----------------
        try:

            corr_chart = Image(corr_plot_for_section, width=450, height=300)
            corr_chart.hAlign = "CENTER"

            elements.append(corr_chart)

        except:

            elements.append(
                Paragraph("Correlation graph not available.", styles['NormalStyle'])
            )


        elements.append(Spacer(1,20))


        # ---------------- IMPROVEMENT TEXT ----------------
        improvement_texts = self._generate_improvement_text_blocks(
            target_variable_name=target_variable_name,
            improvement_direction=improvement_direction,
            movement_word=movement_word,
            action_verb=action_verb,
            perc1=perc1_for_original_sections,
            perc2=perc2_for_original_sections,
            perc3=perc3_for_original_sections,
            initial_target_value=initial_target_value,
            mid_target_value=mid_target_value,
            long_target_value=long_target_value,
            current_sxi=current_tree_sxi,
            immediate_sxi=sxi1_for_original_sections,
            mid_sxi=sxi2_for_original_sections,
            long_sxi=sxi3_for_original_sections,
        )
        immediate_text = improvement_texts["immediate_text"]
        mid_text = improvement_texts["mid_text"]
        long_text = improvement_texts["long_text"]


        elements.append(Paragraph(immediate_text, styles['NormalStyle']))
        elements.append(Spacer(1,6))

        elements.append(Paragraph(mid_text, styles['NormalStyle']))
        elements.append(Spacer(1,6))

        elements.append(Paragraph(long_text, styles['NormalStyle']))

        elements.append(Spacer(1,20))


        
    #-------Page 5 ------
        elements.append(PageBreak())
        elements.append(Spacer(1, 40))
        # ---------------------------------------------------
        # STYLES
        # ---------------------------------------------------

        heading = ParagraphStyle(
            "heading",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=colors.darkblue,
            alignment=1
        )

        center = ParagraphStyle(
            "center",
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            alignment=1
        )

        right_white = ParagraphStyle(
            "right_white",
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            alignment=1,
            textColor=colors.white
        )

        improvement_left_style = ParagraphStyle(
            "improvement_left_style",
            parent=center,
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=1
        )

        # ---------------------------------------------------
        # TITLE
        # ---------------------------------------------------
        target_label = (
            target_variable_name.replace("_", " ").title()
            if target_variable_name
            else "Target Value"
        )
        elements.append(Paragraph("Potential Business Improvement", heading))
        elements.append(Spacer(1, 6))

        if isinstance(target_mean_for_original_sections, (int, float)):
            mean_value = format_report_display_value(target_mean_for_original_sections)
        else:
            mean_value = (
                target_mean_for_original_sections
                if target_mean_for_original_sections is not None
                else "N/A"
            )

        summary_metric_label = "Count" if tv_type == "Categorical" else "Mean"
        elements.append(
            Paragraph(f"<b>{summary_metric_label} {target_label}: {mean_value}</b>", center)
        )

        elements.append(Spacer(1, 14))

        # ---------------------------------------------------
        # BOX FUNCTION
        # ---------------------------------------------------
        def _legacy_make_business_box(perc, value, sxi, label):
            heading_map = {
                "Initial": "Initial Improvement",
                "Mid-Term": "Mid Term Improvement",
                "Long-Term": "Long Term Improvement",
            }
            left_heading = heading_map.get(label, f"{label} Improvement")

            perc = "N/A" if perc is None else f"{abs(float(perc)):.2f}"
            value_text = format_report_display_value(value)
            value = value_text
            sxi = "N/A" if sxi is None else f"{float(sxi):.2f}"

            left = Paragraph(
                (
                    f"<b>{left_heading}</b><br/>"
                    f"{perc}% {tv_inc}<br/>"
                    f"Projected {target_label}: {value}"
                ),
                center
            )

            right = Paragraph(
                f"{label} Target DXI:<br/> {sxi}",
                right_white
            )

            table = Table(
                [[left, right]],
                colWidths=[2.1*inch, 1.2*inch],
                rowHeights=[0.6*inch]
            )

            table.setStyle(TableStyle([
                ("BACKGROUND",(1,0),(1,0),colors.HexColor("#10A6C9")),
                ("BOX",(0,0),(-1,-1),2,colors.HexColor("#10A6C9")),
                ("LEFTPADDING",(0,0),(-1,-1),8),
                ("RIGHTPADDING",(0,0),(-1,-1),8),
                ("TOPPADDING",(0,0),(-1,-1),6),
                ("BOTTOMPADDING",(0,0),(-1,-1),6),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ]))

            return table

        def make_business_box(perc, value, sxi, label):
            heading_map = {
                "Initial": "Initial Improvement",
                "Mid-Term": "Mid Term Improvement",
                "Long-Term": "Long Term Improvement",
            }
            left_heading = heading_map.get(label, f"{label} Improvement")

            perc = "N/A" if perc is None else f"{abs(float(perc)):.2f}"
            value_text = format_report_display_value(value)
            value = value_text
            sxi = "N/A" if sxi is None else f"{float(sxi):.2f}"

            available_width = doc.width
            right_col_width = 1.55 * inch
            left_col_width = available_width - right_col_width

            left = Paragraph(
                (
                    f"<b>{left_heading}</b><br/>"
                    f"&#9632; {perc}% {tv_inc}<br/>"
                    f"Projected {target_label}: {value}"
                ),
                improvement_left_style
            )

            right = Paragraph(
                f"{label} Target DXI:<br/>{sxi}",
                right_white
            )

            table = Table(
                [[left, right]],
                colWidths=[left_col_width, right_col_width]
            )

            table.setStyle(TableStyle([
                ("BACKGROUND",(1,0),(1,0),colors.HexColor("#10A6C9")),
                ("BOX",(0,0),(-1,-1),2,colors.HexColor("#10A6C9")),
                ("LEFTPADDING",(0,0),(-1,-1),8),
                ("RIGHTPADDING",(0,0),(-1,-1),8),
                ("TOPPADDING",(0,0),(-1,-1),8),
                ("BOTTOMPADDING",(0,0),(-1,-1),8),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ]))

            return table

        # ---------------------------------------------------
        # CREATE BOXES
        # ---------------------------------------------------

        elements.append(make_business_box(
            perc1_for_original_sections,
            value1_for_original_sections,
            sxi1_for_original_sections,
            "Initial",
        ))
        elements.append(Spacer(1, 12))
        elements.append(make_business_box(
            perc2_for_original_sections,
            value2_for_original_sections,
            sxi2_for_original_sections,
            "Mid-Term",
        ))
        elements.append(Spacer(1, 12))
        elements.append(make_business_box(
            perc3_for_original_sections,
            value3_for_original_sections,
            sxi3_for_original_sections,
            "Long-Term",
        ))
        elements.append(Spacer(1,20))
        elements.append(PageBreak())
        elements.append(Spacer(1, 40))
        # =========================================================
        # ACTUAL vs PREDICTED TABLE (FINAL - NO CORE CHANGE)
        # =========================================================
    
        elements.append(Paragraph("<b>Actual vs Predicted Values</b>", heading))
        elements.append(Spacer(1, 10))


        def safe_float(val, default=0.0):
            try:
                return float(val)
            except:
                return default

        def pick_prediction_columns(df, target_name=None):
            if df is None or not hasattr(df, "columns"):
                return None, None

            columns = list(df.columns)

            actual_candidates = []
            predicted_candidates = []

            if target_name:
                actual_candidates.extend([
                    f"Actual {target_name}",
                    f"Actual {str(target_name).replace('_', ' ')}",
                ])
                predicted_candidates.extend([
                    f"Predicted {target_name}",
                    f"Predicted {str(target_name).replace('_', ' ')}",
                ])

            actual_candidates.extend(["Actual Values", "Actual"])
            predicted_candidates.extend(["Predicted Values", "Predicted"])

            actual_col = next((col for col in actual_candidates if col in columns), None)
            predicted_col = next((col for col in predicted_candidates if col in columns), None)

            if actual_col is None:
                actual_col = next((col for col in columns if str(col).lower().startswith("actual")), None)
            if predicted_col is None:
                predicted_col = next((col for col in columns if str(col).lower().startswith("predicted")), None)

            return actual_col, predicted_col

        def format_table_value(val):
            if val is None:
                return "N/A"
            try:
                if pd.isna(val):
                    return "N/A"
            except Exception:
                pass
            try:
                return format_report_display_value(val)
            except (ValueError, TypeError):
                return str(val)


        # ---------------------------------------------------------
        # REGRESSION (USES dap)
        # ---------------------------------------------------------
        if tv_type == "Numeric":

            table_data = [
                [f"Actual {target_variable_name}", f"Predicted {target_variable_name}"]
            ]

            try:
                df = result.get("dap") if result else None

                if df is not None:
                    actual_col = f"Actual {target_variable_name}"
                    pred_col = f"Predicted {target_variable_name}"

                    for _, row in df.head(10).iterrows():
                        table_data.append([
                            format_table_value(row.get(actual_col)),
                            format_table_value(row.get(pred_col))
                        ])

            except Exception as e:
                print("Regression table error:", e)

            actual_table = Table(table_data, colWidths=[250, 250])


        # ---------------------------------------------------------
        # CLASSIFICATION (USING CONFUSION MATRIX)
        # ---------------------------------------------------------
        else:

            table_data = [[f"Actual {target_variable_name}", f"Predicted {target_variable_name}"]]

            try:
                df = result.get("dap") if isinstance(result, dict) else None

                if df is not None and not df.empty:
                    actual_col, pred_col = pick_prediction_columns(df, target_variable_name)

                    if actual_col and pred_col:
                        for _, row in df.head(10).iterrows():
                            table_data.append([
                                format_table_value(row.get(actual_col)),
                                format_table_value(row.get(pred_col)),
                            ])
                    else:
                        print("Classification table error: Actual/Predicted columns not found in dap")

                # fallback if something fails
                if len(table_data) == 1:
                    table_data.append(["Data not available", "Data not available"])

            except Exception as e:
                print("Classification table error:", e)
                table_data.append(["Error", "Error"])

            actual_table = Table(table_data, colWidths=[250, 250])


        # ---------------------------------------------------------
        # TABLE STYLING
        # ---------------------------------------------------------
        actual_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))


        # ---------------------------------------------------------
        # ADD TO PDF
        # ---------------------------------------------------------
        elements.append(actual_table)
        elements.append(Spacer(1, 20))
        # -------- ADD TABLE TO PDF -------- #


    #----Page 6th -------
        header_cell_style = ParagraphStyle(
            "TreeImportanceHeaderCell",
            parent=styles["Normal"],
            textColor=colors.white,
            fontName="Helvetica-Bold",
        )

        def wrap_imp(val, is_header=False):
            style = header_cell_style if is_header else styles["Normal"]
            return Paragraph("" if val is None else str(val), style)

        def safe_get(lst, idx, decimals=None):
            try:
                value = lst[idx]
                if decimals is not None and value is not None:
                    return round(value, decimals)
                return value
            except (IndexError, TypeError):
                return None

        from reportlab.lib.pagesizes import A4
        page_width = A4[0] - 72

        col_widths_feat = [
            page_width * 0.50,  # Feature
            page_width * 0.25,  # SXI Weights
            page_width * 0.25,  # % Contribution
        ]

        feat_table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.midnightblue),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('ALIGN',      (1, 0), (-1, -1), 'CENTER'),
            ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID',       (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME',   (0, 0), (-1,  0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0, 0), (-1, -1), 9),
            ('TOPPADDING',    (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ])

        def build_feature_importance_rows(
            primary_features,
            primary_weights,
            fallback_df=None,
            max_rows=5,
            value_label='Tree Importance',
        ):
            return self._build_tree_feature_importance_rows(
                feature_importance_df=fallback_df,
                primary_features=primary_features,
                primary_weights=primary_weights,
                max_rows=max_rows,
                value_label=value_label,
            )

        def render_decision_logic_rows(decision_text):
            for row_type, line in self._decision_logic_report_rows(decision_text):
                safe_line = self._sanitize_reportlab_markup(line)
                if not safe_line or re.fullmatch(r"(?:â€¢|•|-|\s)+", safe_line):
                    continue
                if row_type == "heading":
                    elements.append(Spacer(1, 6))
                    elements.append(Paragraph(f"<b>{safe_line}</b>", styles['NormalStyle']))
                    elements.append(Spacer(1, 4))
                elif row_type == "path_heading":
                    elements.append(Paragraph(f"<b>{safe_line}</b>", styles['NormalStyle']))
                    elements.append(Spacer(1, 4))
                else:
                    clean_line = self._strip_tree_bullet_prefix(safe_line)
                    if clean_line:
                        elements.append(Paragraph(f"- {clean_line}", styles['SmallBulletStyle']))
                        elements.append(Spacer(1, 3))

        # ── CURRENT DECISION TREE — Page 6 ───────────────────────────────────────
        elements.append(PageBreak())
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("<u><b>Current Decision Tree — Business Interpretation</b></u>", styles['HeadingStyle']))
        elements.append(Spacer(1, 8))        

        if current_tree_image:
            chart1 = Image(current_tree_image, width=500, height=300)
            chart_table = Table([[chart1]], colWidths=[500])
            chart_table.setStyle(TableStyle([('ALIGN', (0,0),(-1,-1),'CENTER')]))
            elements.append(chart_table)
            elements.append(Spacer(1, 8))

        decision_0_current = "\n\n".join(
            [text for text in [decision_0_current_for_section, decision_1_current_for_section] if text]
        )
        render_decision_logic_rows(decision_0_current)
        raw_lines = []

        for line in raw_lines:
            if line.lower().startswith("low "):
                elements.append(Spacer(1, 6))
                elements.append(Paragraph(f"<b>{line}</b>", styles['NormalStyle']))
                elements.append(Spacer(1, 4))
            elif line.lower().startswith("high "):
                elements.append(Spacer(1, 10))
                elements.append(Paragraph(f"<b>{line}</b>", styles['NormalStyle']))
                elements.append(Spacer(1, 4))
            else:
                clean = line.lstrip("•").strip()
                if clean:
                    elements.append(Paragraph(f"• {clean}", styles['SmallBulletStyle']))
                    elements.append(Spacer(1, 3))

        elements.append(Spacer(1, 10))

        elements.append(PageBreak())
        elements.append(Spacer(1, 40))
        elements.append(Paragraph("<u><b>Feature Importance — Current Decision Tree</b></u>", styles['HeadingStyle']))
        elements.append(Spacer(1, 10))

        current_feature_importance_df = (
            None
            if report_section_overrides
            else result.get("mi_score") if isinstance(result, dict) else None
        )
        curr_feat_data = build_feature_importance_rows(
            current_tree_features_for_section,
            current_tree_feature_values_for_section,
            fallback_df=current_feature_importance_df,
            value_label='Tree Importance',
        )
        curr_feat_data = [
            [wrap_imp(cell, is_header=(row_index == 0)) for cell in row]
            for row_index, row in enumerate(curr_feat_data)
        ]
        curr_feat_table = Table(curr_feat_data, colWidths=col_widths_feat, repeatRows=1)
        curr_feat_table.setStyle(feat_table_style)
        elements.append(curr_feat_table)
        elements.append(Spacer(1, 15))

        # ── TARGET DECISION TREE — Page 6  ────────────────────────────────────────
        elements.append(PageBreak())
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("<u><b>Target Decision Tree — Business Interpretation</b></u>", styles['HeadingStyle']))
        elements.append(Spacer(1, 8))

        if target_tree_image:
            chart2 = Image(target_tree_image, width=500, height=300)
            chart_table2 = Table([[chart2]], colWidths=[500])
            chart_table2.setStyle(TableStyle([('ALIGN', (0,0),(-1,-1),'CENTER')]))
            elements.append(chart_table2)
            elements.append(Spacer(1, 10))

        decision_0_tr = "\n\n".join(
            [text for text in [decision_0_tr_for_section, decision_1_tr_for_section] if text]
        )
        render_decision_logic_rows(decision_0_tr)
        raw_lines_tr = []

        for line in raw_lines_tr:
            if line.lower().startswith("low "):
                elements.append(Spacer(1, 6))
                elements.append(Paragraph(f"<b>{line}</b>", styles['NormalStyle']))
                elements.append(Spacer(1, 4))
            elif line.lower().startswith("high "):
                elements.append(Spacer(1, 10))
                elements.append(Paragraph(f"<b>{line}</b>", styles['NormalStyle']))
                elements.append(Spacer(1, 4))
            else:
                clean = line.lstrip("•").strip()
                if clean:
                    elements.append(Paragraph(f"• {clean}", styles['SmallBulletStyle']))
                    elements.append(Spacer(1, 3))

        elements.append(Spacer(1, 10))

        elements.append(PageBreak())
        elements.append(Spacer(1, 40))
        elements.append(Paragraph("<u><b>Feature Importance — Target Decision Tree</b></u>", styles['HeadingStyle']))
        elements.append(Spacer(1, 10))

        target_feature_importance_df = (
            None
            if report_section_overrides
            else result.get("mi_score_target") if isinstance(result, dict) else None
        )
        targ_feat_data = build_feature_importance_rows(
            target_tree_features_for_section,
            target_tree_feature_values_for_section,
            fallback_df=target_feature_importance_df,
            value_label='Tree Importance',
        )
        targ_feat_data = [
            [wrap_imp(cell, is_header=(row_index == 0)) for cell in row]
            for row_index, row in enumerate(targ_feat_data)
        ]
        targ_feat_table = Table(targ_feat_data, colWidths=col_widths_feat, repeatRows=1)
        targ_feat_table.setStyle(feat_table_style)
        elements.append(targ_feat_table)
        elements.append(Spacer(1, 20))

        build_start = time.perf_counter()
        print("generate_pdf_prof_user: starting final doc.build")
        doc.build(
            elements,
            onFirstPage=draw_footer_only,
            onLaterPages=draw_header_footer
        )
        print(f"generate_pdf_prof_user: final doc.build took {time.perf_counter() - build_start:.2f}s")
        print(f"PDF saved as {filename}")
            # except Exception as e:
            #     print(f"Error generating PDF: {e}")

    # def generate_sxi_distribution_interpretation(
    #     self,
    #     df,
    #     target_column: str,
    #     tv_type: str = None,
    #     target_count_summary: str = None,
    # ) -> str:
    #     """
    #     Generates SXI Exploratory Data Analysis using dataset values.
    #     Clean, business-friendly, and PDF-safe format.
    #     """

    #     try:
    #         # ---------------- VALIDATION ----------------
    #         if target_column not in df.columns:
    #             raise ValueError(f"{target_column} not found in dataframe")

    #         analysis_series = pd.to_numeric(df[target_column], errors='coerce').dropna()
    #         if analysis_series.empty:
    #             raise ValueError(f"{target_column} does not contain numeric values")

    #         original_target_column = f"{target_column}_original"
    #         stats_column = original_target_column if original_target_column in df.columns else target_column
    #         stats_series = pd.to_numeric(df[stats_column], errors='coerce').dropna()
    #         if stats_series.empty:
    #             if tv_type == "Categorical":
    #                 stats_series = analysis_series.copy()
    #             else:
    #                 raise ValueError(f"{stats_column} does not contain numeric values")

    #         # ---------------- CALCULATIONS ----------------
    #         total = len(analysis_series)

    #         mean_val = analysis_series.mean()
    #         min_val = stats_series.min()
    #         max_val = stats_series.max()
    #         actual_mean_val = stats_series.mean()
    #         std_val = stats_series.std()

    #         below_count = (analysis_series < mean_val).sum()
    #         above_count = (analysis_series >= mean_val).sum()

    #         # Keep the first two lines consistent across task types, but show class counts
    #         # on line 3 for classification targets.
    #         lines = [
    #             f"{total:,} valid instances were analyzed for {target_column}.",
    #             f"1. {target_column}: {below_count:,} instances Below Mean (Mean: {mean_val:.2f})",
    #             f"2. {target_column}: {above_count:,} instances Above Mean (Mean: {mean_val:.2f})",
    #         ]

    #         if tv_type == "Categorical":
    #             class_count_text = target_count_summary
    #             if not class_count_text:
    #                 stats_value_counts = df[stats_column].fillna("Unknown").value_counts()
    #                 class_count_text = " & ".join(
    #                     f"{label}: {count}" for label, count in stats_value_counts.items()
    #                 ) or "Not available"
    #             lines.append(f"3. Target Count: {class_count_text}")
    #         else:
    #             lines.append(f"3. {target_column} Minimum: {min_val:.2f}")
    #             lines.extend([
    #                 f"4. {target_column} Maximum: {max_val:.2f}",
    #                 f"5. {target_column} Mean: {actual_mean_val:.2f}",
    #                 f"6. {target_column} SD: {std_val:.2f}",
    #             ])

    #         return "\n".join(lines)

    #     except Exception as e:
    #         print("SXI interpretation error:", e)

    #         # ---------------- FALLBACK ----------------
    #         lines = [
    #             f"Data distribution details could not be generated for {target_column}.",
    #             f"1. {target_column}: Below Mean data not available",
    #             f"2. {target_column}: Above Mean data not available",
    #             f"3. {'Target Count: Not available' if tv_type == 'Categorical' else f'{target_column} Minimum: Not available'}",
    #         ]
    #         if tv_type != "Categorical":
    #             lines.extend([
    #                 f"4. {target_column} Maximum: Not available",
    #                 f"5. {target_column} Mean: Not available",
    #                 f"6. {target_column} SD: Not available",
    #             ])
    #         return "\n".join(lines)

    def _get_target_outlier_summary(self, target_column: str):
        csv_master_info = self.request.session.get("csv_master_info") or {}
        summary = csv_master_info.get("summary") or {}
        outlier_info = csv_master_info.get("outlier") or summary.get("outlier") or {}
        target_summary = outlier_info.get("target")
        if isinstance(target_summary, dict) and target_summary.get("method"):
            return target_summary

        columns = outlier_info.get("columns") or {}
        normalized_target = self._normalize_column_match_key(target_column)
        for column_name, column_summary in columns.items():
            if self._normalize_column_match_key(column_name) == normalized_target:
                return column_summary
        return {}

    def _format_target_outlier_method_line(self, target_column: str) -> str:
        outlier_summary = self._get_target_outlier_summary(target_column)
        method = str(outlier_summary.get("method") or "").strip()
        definition = str(outlier_summary.get("definition") or "").strip()
        if not method or not definition:
            return ""

        condition = str(outlier_summary.get("condition") or "").strip()
        m_value = outlier_summary.get("m_value")
        if m_value is None:
            m_text = "M = infinity"
        else:
            try:
                m_text = f"M = {float(m_value):.4f}"
            except (TypeError, ValueError):
                m_text = ""

        qualifiers = [item for item in [condition, m_text] if item]
        qualifier_text = f" ({'; '.join(qualifiers)})" if qualifiers else ""
        return (
            f"Technique used : {method}{qualifier_text}\n"
            f"Definition: {definition}"
        )

    def generate_sxi_distribution_interpretation(
        self,
        df,
        target_column: str,
        tv_type: str = None,
        target_count_summary: str = None,
        current_sxi: float = None,
        after_outlier_df=None,
        sxi_df=None,
    ) -> str:
        """
        Generates SXI Exploratory Data Analysis using dataset values.
        Clean, business-friendly, and PDF-safe format.
        """

        try:
            # ---------------- VALIDATION ----------------
            before_stats_column = self._resolve_dataframe_column(df, target_column)
            after_stats_column = self._resolve_dataframe_column(after_outlier_df, target_column)
            if before_stats_column is None and after_stats_column is None:
                raise ValueError(f"{target_column} not found in before or after dataframe")
            before_source_df = df if before_stats_column is not None and isinstance(df, pd.DataFrame) else after_outlier_df
            if before_stats_column is None:
                before_stats_column = after_stats_column
            if after_stats_column is None:
                after_stats_column = before_stats_column

            original_target_column = f"{target_column}_original"
            stats_column = (
                original_target_column
                if isinstance(before_source_df, pd.DataFrame) and original_target_column in before_source_df.columns
                else before_stats_column
            )
            target_values = before_source_df[stats_column].dropna()
            is_categorical = str(tv_type or "").strip().lower() == "categorical"

            if target_values.empty:
                raise ValueError(f"{stats_column} does not contain usable values")

            if is_categorical:
                analysis_series = target_values
                stats_series = target_values
            else:
                analysis_series = self._numeric_target_series(before_source_df, target_column)
                if analysis_series.empty:
                    analysis_series = self._numeric_target_series(after_outlier_df, target_column)
                if analysis_series.empty:
                    raise ValueError(f"{stats_column} does not contain numeric values")
                stats_series = analysis_series

            # Use SXI values for the below/above current SXI split when available.
            # Classification targets are often label-encoded in `target_column`; comparing
            # that 0/1 series against the current SXI makes the split meaningless.
            distribution_series = pd.Series(dtype=float)
            for candidate_df in (before_source_df, after_outlier_df, sxi_df):
                if not isinstance(candidate_df, pd.DataFrame) or candidate_df.empty:
                    continue
                sxi_column = self._resolve_dataframe_column(candidate_df, "composite_dxi")
                if sxi_column is None:
                    continue
                candidate_series = self._coerce_numeric_like_series(candidate_df[sxi_column]).dropna()
                if not candidate_series.empty:
                    distribution_series = candidate_series
                    break

            # ---------------- CALCULATIONS ----------------
            total = len(analysis_series)

            default_sxi_mean = distribution_series.mean() if not distribution_series.empty else np.nan
            current_sxi_number = self._coerce_summary_float(current_sxi)
            if current_sxi_number is not None and pd.notna(current_sxi_number):
                sxi_reference = current_sxi_number
            elif pd.notna(default_sxi_mean):
                sxi_reference = float(default_sxi_mean)
            else:
                sxi_reference = None

            def format_count_summary_text(summary_text):
                if not summary_text:
                    return summary_text
                return re.sub(
                    r"\b\d[\d,]*\b",
                    lambda match: f"{int(match.group(0).replace(',', '')):,}",
                    str(summary_text),
                )

            if sxi_reference is not None and pd.notna(sxi_reference) and not distribution_series.empty:
                below_count = int((distribution_series < sxi_reference).sum())
                above_count = int((distribution_series >= sxi_reference).sum())
                sxi_reference_text = f"{sxi_reference:.2f}"
            else:
                below_count = above_count = None
                sxi_reference_text = "not available"

            # ---------------- FINAL OUTPUT ----------------
            if is_categorical:
                below_text = f"{below_count:,}" if below_count is not None else "Not available"
                above_text = f"{above_count:,}" if above_count is not None else "Not available"
                lines = [
                    f"{total:,} valid instances were analyzed for {target_column}.",
                    f"1. {target_column}: {below_text} instances Below Current DXI (Current DXI: {sxi_reference_text})",
                    f"2. {target_column}: {above_text} instances Above Current DXI (Current DXI: {sxi_reference_text})",
                ]

                stats_value_counts = before_source_df[stats_column].fillna("Unknown").value_counts()
                class_count_text = " & ".join(
                    f"{label}: {count:,}" for label, count in stats_value_counts.items()
                ) or format_count_summary_text(target_count_summary) or "Not available"
                lines.append(f"3. Target Count: {class_count_text}")
            else:
                after_series = self._numeric_target_series(after_outlier_df, target_column)
                if after_series.empty:
                    after_series = stats_series

                before_stats = self._numeric_distribution_stats(stats_series)
                after_stats = self._numeric_distribution_stats(after_series)
                total_after = len(after_series)
                outlier_method_line = self._format_target_outlier_method_line(target_column)

                lines = [
                    f"{total:,} valid instances before outlier treatment and {total_after:,} valid instances after outlier treatment were analyzed for {target_column}.",
                ]
                if outlier_method_line:
                    lines.append(outlier_method_line)
                lines.extend([
                    "| Metric | Before outlier treatment | After outlier treatment |",
                    "|---|---|---|",
                    f"| Minimum | {self._format_distribution_stat(before_stats['min'], target_column, tv_type)} | {self._format_distribution_stat(after_stats['min'], target_column, tv_type)} |",
                    f"| Maximum | {self._format_distribution_stat(before_stats['max'], target_column, tv_type)} | {self._format_distribution_stat(after_stats['max'], target_column, tv_type)} |",
                    f"| Mean | {self._format_distribution_stat(before_stats['mean'], target_column, tv_type)} | {self._format_distribution_stat(after_stats['mean'], target_column, tv_type)} |",
                    f"| SD | {self._format_distribution_stat(before_stats['std'], target_column, tv_type)} | {self._format_distribution_stat(after_stats['std'], target_column, tv_type)} |",
                ])

            return "\n".join(lines)

        except Exception as e:
            print("SXI interpretation error:", e)

            # ---------------- FALLBACK ----------------
            if tv_type == "Categorical":
                lines = [
                    f"Data distribution details could not be generated for {target_column}.",
                    f"1. {target_column}: Below Mean data not available",
                    f"2. {target_column}: Above Mean data not available",
                    "3. Target Count: Not available",
                ]
            else:
                lines = [
                    f"Data distribution details could not be generated for {target_column}.",
                    "| Metric | Before outlier treatment | After outlier treatment |",
                    "|---|---|---|",
                    "| Minimum | Not available | Not available |",
                    "| Maximum | Not available | Not available |",
                    "| Mean | Not available | Not available |",
                    "| SD | Not available | Not available |",
                ]
            return "\n".join(lines)

    def _resolve_dataframe_column(self, df, target_column):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        if target_column in df.columns:
            return target_column
        normalized_target = self._normalize_column_match_key(target_column)
        for column in df.columns:
            normalized_column = self._normalize_column_match_key(column)
            if normalized_column == normalized_target:
                return column
        compact_target = normalized_target.replace("_", "")
        for column in df.columns:
            normalized_column = self._normalize_column_match_key(column)
            if normalized_column.replace("_", "") == compact_target:
                return column
        return None

    def _normalize_column_match_key(self, value) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^0-9a-z]+", "_", text)
        return re.sub(r"_+", "_", text).strip("_")

    def _numeric_distribution_stats(self, series):
        numeric = self._coerce_numeric_like_series(pd.Series(series)).dropna()
        return {
            "min": numeric.min() if not numeric.empty else np.nan,
            "max": numeric.max() if not numeric.empty else np.nan,
            "mean": numeric.mean() if not numeric.empty else np.nan,
            "std": numeric.std() if len(numeric) > 1 else np.nan,
        }

    def _numeric_target_series(self, df, target_column):
        column = self._resolve_dataframe_column(df, target_column)
        if column is None:
            return pd.Series(dtype=float)
        return self._coerce_numeric_like_series(df[column]).dropna()

    def _format_distribution_stat(self, value, target_column, tv_type):
        if pd.isna(value):
            return "Not available"
        return format_display_value(value, target_column, tv_type, decimals=2)
    
    def generate_corr_intprt(
            self,
            target_variable: str,
            current_score: float,
            current_rate: float,
            mid_score: float,
            mid_rate: float,
            long_score: float,
            long_rate: float,
            target_type: str = None,
        ) -> str:
        """
        Generates a strictly formatted PDF-ready Correlation Graph Interpretation
        with enforced bullet-point structure.
        Uses Claude API (claude-sonnet-4-20250514).
        """
        current_score = round(current_score, 2)
        mid_score = round(mid_score, 2)
        long_score = round(long_score, 2)
        metric_label = "Rate" if str(target_type or "").lower() == "categorical" else "Value"
        current_rate_text = format_display_value(current_rate, target_variable, target_type, decimals=2)
        mid_rate_text = format_display_value(mid_rate, target_variable, target_type, decimals=2)
        long_rate_text = format_display_value(long_rate, target_variable, target_type, decimals=2)
        text_stats = f"""
Target Variable: {target_variable}

Current:
SXI = {current_score}, {metric_label} = {current_rate_text}

Mid-Term:
SXI = {mid_score}, {metric_label} = {mid_rate_text}

Long-Term:
SXI = {long_score}, {metric_label} = {long_rate_text}
"""

        system_prompt = "You strictly follow formatting rules. Never merge bullets."

        user_prompt = f"""
You are generating a PDF-ready business interpretation.

STRICT FORMATTING RULES (DO NOT VIOLATE):
- Output MUST start with exactly this heading format:
- Leave ONE blank line after heading
- Strategic horizons MUST be bullet points using symbol: •
- Each bullet MUST be on a new line
- Do NOT merge bullets into paragraphs
- Do NOT remove bullets
- Do NOT add extra commentary
- Keep professional analytical tone
- Keep numeric values EXACTLY as provided
- Output MUST be valid HTML
- Use <h3> for heading
- Use <p> for paragraphs
- Use <ul> and <li> for bullet points
- Each strategic horizon MUST be a separate <li>
- Do NOT merge <li> items
- Do NOT return plain text
- Do NOT add markdown

STRUCTURE TO FOLLOW EXACTLY:

<h3>Correlation Graph Interpretation (SXI Score vs Target Variable – {target_variable})</h3>

<p>Paragraph explaining overall relationship and R-Square importance.</p>

<p>The graph highlights four key data points that represent different strategic horizons:</p>

<ul>
<li><b>Blue (Current):</b> State current SXI = {current_score} and {metric_label} = {current_rate_text} clearly.</li>
<li><b>Purple (Immediate):</b> Explain short-term directional impact relative to current.</li>
<li><b>Black (Mid-Term):</b> State projected SXI = {mid_score} and {metric_label} = {mid_rate_text} clearly.</li>
<li><b>Dark Green (Long-Term):</b> State projected SXI = {long_score} and {metric_label} = {long_rate_text} clearly.</li>
</ul>

NOW GENERATE USING ONLY:

{text_stats}
""".strip()

        try:
            response = self._llm_request(system_prompt, user_prompt, max_tokens=900)
            if not (response or "").strip() and self._llm_out_of_credits():
                return self._out_of_credits_text()
            response = response.replace("•", "\n•")
            response = response.replace("###", "\n###")
            return response.strip()
        except Exception as e:
            print(f"❌ Error in generate_corr_intprt (Claude): {e}")
            return self._out_of_credits_text() if self._llm_out_of_credits() else ""

    def chkdelinss(self,tv,df1,sxi,selcls,twds,clas1,clas2,classes):
        #Count of class1 in above SXI
        d1=df1.loc[(df1['composite_dxi'] >= sxi) & (df1[tv] == 1)]
        #Count of class0 in above SXI
        d2=df1.loc[(df1['composite_dxi'] >= sxi) & (df1[tv] == 0)]
        #Count of class1 in below SXI
        d3=df1.loc[(df1['composite_dxi'] < sxi) & (df1[tv] == 1)]
        #Count of class0 in below SXI
        d4=df1.loc[(df1['composite_dxi'] < sxi) & (df1[tv] == 0)]
        uy = selcls+twds
        if uy == 'cls1abv' or uy == 'cls2bel':
            clas1fnl = round((len(d1)/(len(d1) +len(d2)))*100,2)
            clas2fnl = round((len(d4)/(len(d3) +len(d4)))*100,2) 
            clsafnl = round((len(d1)/len(df1[tv]))*100,2)
            clsbfnl = round((len(d4)/len(df1[tv]))*100,2)
            text = (
                    f"{classes[1]} Above SXI Score of {sxi} - {clas1fnl}%\n"
                    f"{classes[0]} Below SXI Score of {sxi} -  {clas2fnl}%\n\n"
                    f"{classes[1]} Above SXI Score of {sxi} - {clsafnl}% of {clas1}%\n"
                    f"{classes[0]} Below SXI Score of {sxi} - {clsbfnl}% of {clas2}%"
                )
        else:
            clas1fnl = (len(d2)/(len(d1) +len(d2)))*100
            clas2fnl = (len(d3)/(len(d3) +len(d4)))*100 
            clsafnl = (len(d2)/len(df1[tv]))*100
            clsbfnl = (len(d3)/len(df1[tv]))*100
            text = (
                    f"{classes[0]} Above SXI Score of {sxi} - {clas2fnl}%\n"
                    f"{classes[1]} Below SXI Score of {sxi} - {clas1fnl}%\n\n"
                    f"{classes[0]} Above SXI Score of {sxi} - {clsafnl}% of {clas2}%\n"
                    f"{classes[1]} Below SXI Score of {sxi} - {clsbfnl}% of {clas1}%"
                )
        return text

    def _expl_dt_pth_legacy_structured(self, path_str, class_name, tree_image_path=None):
        """
        Generate a structured, business-readable interpretation
        from one decision tree path using Claude API.
        Produces TWO sections: Low <class> and High <class> — matching PDF format.
        """
        class_label = self._extract_tree_class_label(path_str, class_name)
        return self._path_to_bullets(path_str, class_label)
    def _expl_dt_pth_legacy_image(self, path_str, class_name, tree_image_path=None):
        """
        Generate a business-readable tree interpretation.
        Path-based only. The report must never invent a branch summary from image
        interpretation or LLM guesses.
        """
        class_label = self._extract_tree_class_label(path_str, class_name)
        return self._path_to_bullets(path_str, class_label)

    def _narrative_path_to_bullets(self, path_str, class_label: str) -> str:
        if not path_str:
            return ""
        return self._path_to_bullets(path_str, class_label)

    def expl_dt_pth(self, path_str, class_name, tree_image_path=None, feature_importance_df=None):
        """
        Final decision-tree explanation formatter.
        Returns path-derived bullets only. No image fallback and no LLM guessing.
        """
        path_text = str(path_str or "").strip()
        business_markers = [
            "## - 0 - - -",
            "## - 1 - - -",
            "TOP 3 HIGH VALUE PATHS",
            "TOP 3 LOW VALUE PATHS",
            "TOP 3 STRONGEST",
            "Best Business Paths:",
            "Business Recommendation:",
        ]
        if path_text and any(marker in path_text for marker in business_markers):
            return path_text
        if self._is_business_ready_tree_text(path_text):
            return path_text

        class_label = self._extract_tree_class_label(path_str, class_name)
        if not path_str:
            return ""

        weighted_branch_text = self._weighted_tree_path_to_bullets(
            path_str,
            class_label,
            feature_importance_df=feature_importance_df,
        )
        if weighted_branch_text:
            return weighted_branch_text
        return self._path_to_bullets(path_str, class_label)
