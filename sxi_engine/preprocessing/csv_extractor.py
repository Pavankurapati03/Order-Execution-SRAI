# ==========================================================
# IMPORTS 
# ==========================================================
import os
import json
import re
import glob
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
try:
    from django.conf import settings
except ImportError:
    class MockSettings:
        MEDIA_ROOT = os.path.join(os.getcwd(), 'media')
        MEDIA_URL = '/media/'
    settings = MockSettings()

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

try:
    from pipeline.models import UploadedFile, GridFSFile
    from pipeline.utils import gridfs_download_to_temp, gemini_detect_primary_keys
    from pipeline.metadata_extractors.custom_preprocessing import run_custom_preprocessing
except ImportError:
    UploadedFile = None
    GridFSFile = None
    gridfs_download_to_temp = None
    gemini_detect_primary_keys = None
    run_custom_preprocessing = None
import builtins

def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    try:
        builtins.print(*args, **kwargs)
    except (OSError, UnicodeEncodeError):
        try:
            safe_args = [str(arg).encode('ascii', errors='replace').decode('ascii') for arg in args]
            builtins.print(*safe_args, **kwargs)
        except Exception:
            pass


# ==========================================================

# 1️⃣ SAFE FILE READER
# ==========================================================
def _looks_like_collapsed_delimited_csv(df: Optional[pd.DataFrame]) -> bool:
    if df is None or df.empty or len(df.columns) != 1:
        return False

    header = str(df.columns[0] or "")
    return any(delim in header for delim in [";", "\t", "|"])


def _read_csv_with_fallbacks(path: str, encoding: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, encoding=encoding)
    if not _looks_like_collapsed_delimited_csv(df):
        return df

    for sep in [";", "\t", "|"]:
        try:
            retry_df = pd.read_csv(
                path,
                low_memory=False,
                encoding=encoding,
                sep=sep,
            )
            if len(retry_df.columns) > 1:
                print(f"[CSV] Detected delimiter '{sep}' for {os.path.basename(path)}")
                return retry_df
        except Exception:
            continue

    try:
        retry_df = pd.read_csv(
            path,
            low_memory=False,
            encoding=encoding,
            sep=None,
            engine="python",
        )
        if len(retry_df.columns) > 1:
            print(f"[CSV] Auto-detected delimiter for {os.path.basename(path)}")
            return retry_df
    except Exception:
        pass

    return df


def safe_read_file(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None

    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".csv":
            encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
            for encoding in encodings_to_try:
                try:
                    return _read_csv_with_fallbacks(path, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            return pd.read_csv(
                path,
                low_memory=False,
                encoding="latin1",
                encoding_errors="ignore",
                sep=None,
                engine="python",
            )
        elif ext in (".xls", ".xlsx"):
            return pd.read_excel(path)
        elif ext == ".json":
            return pd.read_json(path)
        return None
    except Exception as e:
        print("❌ File read error:", e)
        return None


# ==========================================================
# 2️⃣ COLUMN NORMALIZATION
# ==========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def coerce_numeric_text_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series

    cleaned = (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "none": np.nan, "null": np.nan})
        .str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def coerce_numeric_like_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if not target_column or target_column not in df.columns:
        return df

    coerced = coerce_numeric_text_series(df[target_column])
    if coerced.notna().sum() == 0:
        return df

    non_null_original = df[target_column].notna().sum()
    valid_ratio = (coerced.notna().sum() / non_null_original) if non_null_original else 0
    if valid_ratio < 0.8:
        return df

    df = df.copy()
    df[target_column] = coerced
    print(f"[TARGET] Coerced numeric-like target '{target_column}' to numeric (valid ratio {valid_ratio:.0%})")
    return df

def _humanize_feature_text(value: Any) -> str:
    text = str(value or "").strip().replace("_", " ")
    if not text:
        return ""
    return " ".join(word.upper() if word.lower() == "sxi" else word.capitalize() for word in text.split())


def _sanitize_category_token(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        text = "blank"
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "blank"


def _build_encoded_column_name(base_feature: str, category_value: Any, used_names: set[str]) -> str:
    token = _sanitize_category_token(category_value)
    candidate = f"{base_feature}_{token}"
    suffix = 2
    while candidate in used_names:
        candidate = f"{base_feature}_{token}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _build_5_plus_1_encoded_columns(series: pd.Series, column_name: str, top_n: int = 5):
    normalized = (
        series.fillna("Missing")
        .astype(str)
        .str.strip()
        .replace({"": "Missing", "nan": "Missing", "none": "Missing", "null": "Missing"})
    )
    value_counts = normalized.value_counts(dropna=False)
    top_categories = value_counts.head(top_n).index.tolist()
    used_others = normalized.nunique(dropna=False) > top_n
    collapsed = normalized.where(normalized.isin(top_categories), "Others") if used_others else normalized

    ordered_categories = list(top_categories)
    if used_others and "Others" not in ordered_categories:
        ordered_categories.append("Others")

    encoded_df = pd.DataFrame(index=series.index)
    feature_mapping = {}
    encoded_columns = []
    used_names: set[str] = set()

    for category in ordered_categories:
        encoded_col = _build_encoded_column_name(column_name, category, used_names)
        encoded_df[encoded_col] = (collapsed == category).astype(int)
        feature_mapping[encoded_col] = {
            "base_feature": column_name,
            "category": "Others" if str(category).strip().lower() == "others" else str(category),
            "base_feature_display": _humanize_feature_text(column_name),
            "category_display": (
                "Others"
                if str(category).strip().lower() == "others"
                else str(category)
            ),
            "top_categories": [str(item) for item in top_categories],
        }
        encoded_columns.append(encoded_col)

    return encoded_df, feature_mapping, encoded_columns, {
        "top_categories": [str(item) for item in top_categories],
        "used_others": used_others,
        "encoded_columns": encoded_columns,
    }
#  IMPROVED SMART MERGE LOGIC

def smart_merge_datasets(dfs: List[pd.DataFrame], primary_keys: List[str] = None) -> pd.DataFrame:
    """
    Smart merge logic:

    1️⃣ If primary key exists → perform OUTER JOIN
    2️⃣ If no primary key → perform SAFE ROW STACK
    3️⃣ Align schemas before stacking
    """

    if not dfs:
        return None

    if len(dfs) == 1:
        return dfs[0].copy()

    primary_keys = primary_keys or []

    # ------------------------------------------------------
    # CASE 1: PRIMARY KEY BASED MERGE
    # ------------------------------------------------------
    if primary_keys:
        print(f"🔑 Merging using primary keys: {primary_keys}")

        merged = dfs[0].copy()

        for df in dfs[1:]:
            common_keys = [pk for pk in primary_keys if pk in df.columns and pk in merged.columns]

            if common_keys:
                merged = pd.merge(
                    merged,
                    df,
                    on=common_keys,
                    how="outer",
                    suffixes=("", "_dup")
                )
            else:
                print("⚠ No common primary key found. Falling back to row stack.")
                merged = pd.concat([merged, df], axis=0, ignore_index=True)

        return merged

    # ------------------------------------------------------
    # CASE 2: SAFE ROW STACK
    # ------------------------------------------------------
    print("📦 No primary key → Performing safe row stack")

    # Align columns
    all_columns = set()
    for df in dfs:
        all_columns.update(df.columns)

    aligned_dfs = []
    for df in dfs:
        missing_cols = all_columns - set(df.columns)
        for col in missing_cols:
            df[col] = np.nan
        aligned_dfs.append(df)

    merged = pd.concat(aligned_dfs, axis=0, ignore_index=True)

    return merged


def _common_primary_keys(left: pd.DataFrame, right: pd.DataFrame, primary_keys: List[str]) -> List[str]:
    return [
        pk for pk in (primary_keys or [])
        if pk in left.columns and pk in right.columns
    ]


def _coalesce_duplicate_merge_columns(df: pd.DataFrame, suffix: str = "_baseline") -> pd.DataFrame:
    df = df.copy()
    for baseline_col in [col for col in df.columns if col.endswith(suffix)]:
        final_col = baseline_col[:-len(suffix)]
        if final_col in df.columns:
            df[final_col] = df[final_col].combine_first(df[baseline_col])
            df.drop(columns=[baseline_col], inplace=True)
        else:
            df.rename(columns={baseline_col: final_col}, inplace=True)
    return df


def merge_baseline_with_uploaded_data(
    baseline_df: pd.DataFrame,
    uploaded_df: pd.DataFrame,
    primary_keys: List[str] = None,
) -> pd.DataFrame:
    """
    Merge dashboard baseline data with uploaded data.

    If primary keys are available in both datasets, merge row records by those
    keys. Without primary keys, merge columns side-by-side by row position; any
    remaining missing values are handled later by the existing missing-value
    preprocessing step.
    """
    baseline_df = baseline_df.copy().reset_index(drop=True)
    uploaded_df = uploaded_df.copy().reset_index(drop=True)
    common_keys = _common_primary_keys(baseline_df, uploaded_df, primary_keys or [])

    if common_keys:
        print(f"[BASELINE MERGE] Primary key found {common_keys} -> key-based row merge")
        merged = pd.merge(
            baseline_df,
            uploaded_df,
            on=common_keys,
            how="outer",
            suffixes=("_baseline", ""),
        )
        return _coalesce_duplicate_merge_columns(merged)

    print("[BASELINE MERGE] No primary key -> column-wise merge by row position")
    max_rows = max(len(baseline_df), len(uploaded_df))
    baseline_aligned = baseline_df.reindex(range(max_rows))
    uploaded_aligned = uploaded_df.reindex(range(max_rows))

    merged = uploaded_aligned.copy()
    for col in baseline_aligned.columns:
        if col in merged.columns:
            merged[col] = merged[col].combine_first(baseline_aligned[col])
        else:
            merged[col] = baseline_aligned[col]

    return merged


# ==========================================================
# 4️⃣ MERGE VALIDATION
# ==========================================================
def validate_master_merge(master: pd.DataFrame, dfs: List[pd.DataFrame]) -> Dict[str, Any]:

    report = {"status": "ok", "issues": [], "checks": {}}

    # Boolean pollution
    bool_cols = [
        col for col in master.columns
        if master[col].dropna().isin([True, False]).all()
    ]
    if bool_cols:
        report["status"] = "warning"
        report["issues"].append(f"Boolean values detected: {bool_cols}")

    # Case duplicate check
    lower_map = {}
    for col in master.columns:
        lower_map.setdefault(col.lower(), []).append(col)

    duplicates = {k: v for k, v in lower_map.items() if len(v) > 1}
    if duplicates:
        report["status"] = "error"
        report["issues"].append(f"Duplicate logical columns: {duplicates}")

    # High null columns
    high_null_cols = [
        col for col in master.columns
        if master[col].isna().mean() > 0.95
    ]
    if high_null_cols:
        report["issues"].append(f">95% null columns: {high_null_cols}")

    report["checks"]["rows"] = len(master)
    report["checks"]["columns"] = len(master.columns)

    return report


# ==========================================================
# 5️⃣ RAW MERGED DATASET BUILDER
# ==========================================================
def build_raw_merged_dataset(run_id: str, request=None) -> Optional[Dict[str, Any]]:

    if request:
        source_type = request.session.get("source_type", "file_upload")
        if source_type == "bigquery_ga4":
            division_dir = os.path.join(settings.MEDIA_ROOT, "feature_divisions", f"session_{run_id}")
            existing = [
                f for f in os.listdir(division_dir)
                if f.endswith(f"_raw_merged_{run_id}.csv")
            ] if os.path.isdir(division_dir) else []
            if existing:
                raw_filename = existing[0]
                df_temp = pd.read_csv(os.path.join(division_dir, raw_filename))
                return {
                    "csv": f"{settings.MEDIA_URL}feature_divisions/session_{run_id}/{raw_filename}",
                    "file_name": raw_filename,
                    "summary": {
                        "rows": len(df_temp),
                        "cols": list(df_temp.columns),
                        "files_used": 1,
                    }
                }

    dfs = []
    files = UploadedFile.objects.filter(run_id=run_id).order_by("uploaded_at")
    first_tabular_file_name = None

    for uf in files:
        modality = uf.confirmed_modality or uf.detected_modality or "tabular"
        if modality != "tabular":
            continue

        if first_tabular_file_name is None:
            first_tabular_file_name = os.path.basename(uf.file.name)

        abs_path = getattr(uf.file, "path", None)
        if not abs_path:
            continue

        df = safe_read_file(abs_path)
        if df is None or df.empty:
            continue

        df = normalize_columns(df)
        dfs.append(df)

    #raw_merged = dfs[0] if len(dfs) == 1 else strict_column_merge(dfs)
    # Detect primary keys from uploaded files
    primary_keys = []
    for uf in UploadedFile.objects.filter(run_id=run_id):
        if uf.primary_keys:
            primary_keys.extend(
                [c.strip().lower() for c in uf.primary_keys.split(",")]
            )

    # -------------------------------------------------------------
    # 🌟 LOAD PERSISTENT ACTIVE MODULE STATE (SXI Baseline Merging)
    # -------------------------------------------------------------
    import json
    selected_model_slug = None
    data_usage_choice = None
    # Prefer the active Django session. The license app stores model slugs
    # like "transaction-fraud", not display labels like "Transaction".
    if request:
        selected_model_slug = request.session.get("selected_model_slug")
        data_usage_choice = request.session.get("data_usage_choice")

    # Fallback to persistent active_module_state.json only for non-request callers.
    # A live request without a dashboard choice must not inherit stale module data.
    if request is None:
        try:
            state_file_root = os.path.join(settings.BASE_DIR, "active_module_state.json")
            state_file_license = os.path.join(settings.BASE_DIR, "sriya_license", "active_module_state.json")
            for sf in [state_file_root, state_file_license]:
                if os.path.exists(sf):
                    with open(sf, "r") as f:
                        state = json.load(f)
                        if not selected_model_slug:
                            selected_model_slug = state.get("selected_model_slug")
                        if not data_usage_choice:
                            data_usage_choice = state.get("data_usage_choice")
        except Exception:
            pass

    slug_to_file = {
        "transaction-fraud": "transaction Base.csv",
        "fake-claims": "Claim fraud.csv",
        "late-payments": "Claim_details.csv",
        # Backward-compatible labels used by older hard-coded copies.
        "Transaction": "transaction Base.csv",
        "Claims": "Claim fraud.csv",
        "Payments": "Claim_details.csv",
    }
    
    baseline_filename = slug_to_file.get(selected_model_slug)
    raw_merged = None

    if data_usage_choice == "use" and baseline_filename:
        print(f"[BASELINE MERGE] Active module: {selected_model_slug} | Target baseline: {baseline_filename}")
        data_dir = os.path.join(settings.BASE_DIR, "sriya_license", "licenses", "static", "licenses", "Data")
        os.makedirs(data_dir, exist_ok=True)
        baseline_path = os.path.join(data_dir, baseline_filename)
        
    
        baseline_df = safe_read_file(baseline_path)
            
        if baseline_df is not None and not baseline_df.empty:
            baseline_df = normalize_columns(baseline_df)
            print(f"[SUCCESS] Baseline dataset loaded: {baseline_df.shape[0]} rows, {baseline_df.shape[1]} columns")
            
            # If user uploaded data exists, merge them!
            if dfs:
                user_raw = smart_merge_datasets(dfs, primary_keys)
                if user_raw is not None and not user_raw.empty:
                    print(f"[MERGE] Merging uploaded dataset ({user_raw.shape[0]}x{user_raw.shape[1]}) with baseline...")
                    raw_merged = merge_baseline_with_uploaded_data(
                        baseline_df,
                        user_raw,
                        primary_keys,
                    )
                    print(f"[MERGE SUCCESS] Combined dataset shape: {raw_merged.shape[0]} rows, {raw_merged.shape[1]} columns")
                else:
                    raw_merged = baseline_df
            else:
                raw_merged = baseline_df
        else:
            if dfs:
                raw_merged = smart_merge_datasets(dfs, primary_keys)
    else:
        # Standard merging if "do_not_use" chosen
        if dfs:
            raw_merged = smart_merge_datasets(dfs, primary_keys)

    if raw_merged is None or raw_merged.empty:
        return None

    # Keep an event-level copy (pre user collapse) so P2/P4/P8 can use full
    # session/event rows (timestamps for repurchase; all rows for promo sensitivity).
    event_level = raw_merged.copy()
    try:
        from pipeline.purchase_history import attach_calendar_gap_days

        event_level = attach_calendar_gap_days(event_level)
        raw_merged = event_level
    except Exception as exc:
        print(f"[MERGE] calendar gap attach skipped: {exc}")

    # Merge repeated user_pseudo_id rows
    user_col = "user_pseudo_id" if "user_pseudo_id" in raw_merged.columns else None
    had_duplicate_users = bool(user_col and raw_merged[user_col].duplicated().any())
    if had_duplicate_users:
        print(f"[MERGE] Aggregating repeated {user_col} rows...")
        temporal_or_id = {
            "date",
            "eventdate",
            "event_date",
            "event_timestamp",
            "order_timestamp",
            "order_date",
            "transactionid",
            "transaction_id",
            "days_until_next_purchase",
            "event_observed",
            "_event_observed",
        }
        agg_dict = {}
        for col in raw_merged.columns:
            if col == user_col:
                continue
            key = str(col).strip().lower()
            if col in ["is_buyer", "target", "purchased"]:
                agg_dict[col] = "max"
            elif key in temporal_or_id or "timestamp" in key:
                agg_dict[col] = "min"
            elif pd.api.types.is_numeric_dtype(raw_merged[col]) or pd.api.types.is_bool_dtype(raw_merged[col]):
                agg_dict[col] = "mean" if key in {"days_until_next_purchase"} else "sum"
            else:
                agg_dict[col] = "first"
        # days_until_next_purchase is the same on every event row of a user
        if "days_until_next_purchase" in agg_dict:
            agg_dict["days_until_next_purchase"] = "min"

        null_mask = raw_merged[user_col].isna()
        null_users = raw_merged[null_mask]
        valid_users = raw_merged[~null_mask]

        if not valid_users.empty:
            valid_users = valid_users.groupby(user_col).agg(agg_dict).reset_index()

        raw_merged = pd.concat([valid_users, null_users], ignore_index=True)
        print(f"[MERGE] After aggregation shape: {raw_merged.shape}")

    master_dir = os.path.join(settings.MEDIA_ROOT, "feature_divisions", f"session_{run_id}")
    os.makedirs(master_dir, exist_ok=True)

    base_name = os.path.splitext(first_tabular_file_name or f"raw_{run_id}.csv")[0]
    raw_filename = f"{base_name}_raw_merged_{run_id}.csv"
    event_filename = f"{base_name}_event_rows_{run_id}.csv"

    raw_path = os.path.join(master_dir, raw_filename)
    raw_merged.to_csv(raw_path, index=False)
    event_rel = None
    if had_duplicate_users:
        event_path = os.path.join(master_dir, event_filename)
        try:
            event_level.to_csv(event_path, index=False)
            event_rel = f"{settings.MEDIA_URL}feature_divisions/session_{run_id}/{event_filename}"
            print(f"[MERGE] Wrote event-level rows for P2/P4/P8: {event_path}")
        except Exception as exc:
            print(f"[MERGE] event-level save failed: {exc}")

    return {
        "csv": f"{settings.MEDIA_URL}feature_divisions/session_{run_id}/{raw_filename}",
        "file_name": raw_filename,
        "event_rows_csv": event_rel,
        "summary": {
            "rows": int(len(raw_merged)),
            "cols": list(raw_merged.columns),
            "files_used": len(dfs) if dfs else 0,
            "event_rows": int(len(event_level)) if had_duplicate_users else int(len(raw_merged)),
        },
    }


# ==========================================================
# 6️⃣ MISSING HANDLING
# ==========================================================
def handle_missing_val(df, protected_cols=set()):

    df = df.copy()

    for col in df.columns:

        # 🔐 Skip protected columns
        if col in protected_cols:
            print(f"[AUTO][MISSING] SKIPPED protected column: {col}")
            continue

        missing_percentage = df[col].isnull().sum() / len(df) * 100
        print(f"[AUTO][MISSING] {col} - Missing percent: {missing_percentage:.2f}%")

        # ---------------------------------------------------
        # Drop if >80% missing
        # ---------------------------------------------------
        if missing_percentage > 80:
            print(f"[AUTO][MISSING] Dropping column: {col}")
            df.drop(columns=[col], inplace=True)
            continue

        # ---------------------------------------------------
        # Categorical → Fill with Mode
        # ---------------------------------------------------
        if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):

            if df[col].mode().empty:
                continue

            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)

        # ---------------------------------------------------
        # Numeric → Mean/Median Based on Skew
        # ---------------------------------------------------
        elif pd.api.types.is_numeric_dtype(df[col]):

            mean = df[col].mean()
            median = df[col].median()

            if df[col].dropna().nunique() > 1:

                skew_val = df[col].skew()

                if isinstance(skew_val, (int, float)) and np.isfinite(skew_val):
                    fill_value = median if abs(skew_val) > 0.5 else mean
                else:
                    fill_value = mean
            else:
                fill_value = mean

            df[col] = df[col].fillna(fill_value)

    return df

# ==========================================================
# 7️⃣ OUTLIER HANDLING  —  Dynamic Outlier Engine
# ==========================================================
#
#  ROUTING LOGIC  (M = SD / |Mean|  →  Coefficient of Variation)
#
#   M < 0.50          →  Method 1 : 3-SD Rule
#                           WHY: Data is stable & near-normal. Parametric
#                           bounds [μ-3σ, μ+3σ] are tight and precise.
#
#   M >= 0.50 or inf   ->  Method 2 : Log-Transform + Z-Score
#                           WHY: Heavy exponential tails / extreme variance.
#                           ln(X+eps) compresses the scale, then standard
#                           Z-score detects outliers in log-space before
#                           back-transforming bounds to original scale.
#
#   Residual outliers →  Method 4 : Winsorization (1st / 99th pct)
#                           WHY: Safety net — after primary removal any
#                           surviving extreme tails are hard-clamped.
#                           Preserves row count; no data loss.
#
#  ROW DROPPING POLICY:
#   • Methods 1, 3      →  Outlier ROWS ARE DROPPED (hard removal).
#   • Method 4          →  Values CLAMPED (no row drop) — fallback only.
#   • Protected columns →  Completely skipped (target, PKs, etc.)
# ==========================================================


# ── Internal helpers ─────────────────────────────────────────────────

def _compute_variance_ratio(series: pd.Series) -> float:
    """M = SD / |Mean|.  Returns np.inf when mean ≈ 0 (zero-mean guard)."""
    clean = series.dropna()
    if len(clean) < 3:
        return np.inf
    mean_val = clean.mean()
    if abs(mean_val) < 1e-9:
        return np.inf
    return clean.std(ddof=1) / abs(mean_val)


def _method1_3sd_bounds(series: pd.Series):
    """
    Method 1 — Parametric 3-SD Rule.
    Bounds: [μ - 3σ,  μ + 3σ]
    Condition: M < 0.50  (low/moderate variance, near-normal distribution)
    """
    mu    = series.mean()
    sigma = series.std(ddof=1)
    lower = mu - 3.0 * sigma
    upper = mu + 3.0 * sigma
    why   = (
        f"M<0.50 → data is stable & near-normal | "
        f"μ={mu:.4f}  σ={sigma:.4f} | "
        f"bounds=[μ-3σ={lower:.4f}, μ+3σ={upper:.4f}]"
    )
    return lower, upper, why




def _method2_log_zscore_bounds(series: pd.Series, z_thresh: float = 3.0):
    """
    Method 2 — Log-Transform + Z-Score.
    Steps: shift to positive → ln(X+ε) → Z-score in log-space → back-transform.
    Condition: M ≥ 0.50  (heavy tails / exponential-like distribution)
    """
    epsilon = 1.0
    offset  = 0.0
    if series.min() <= 0:
        offset = abs(series.min()) + 1.0      # guarantee strictly positive input

    log_vals   = np.log(series + offset + epsilon)
    mu_log     = log_vals.mean()
    sigma_log  = log_vals.std(ddof=1)
    l_log      = mu_log - z_thresh * sigma_log
    u_log      = mu_log + z_thresh * sigma_log

    lower = np.exp(l_log) - offset - epsilon  # back to original scale
    upper = np.exp(u_log) - offset - epsilon
    why   = (
        f"M≥0.50 → heavy tails/high variance | "
        f"offset={offset:.2f}  ε={epsilon:.2f} | "
        f"log-space: μ={mu_log:.4f}  σ={sigma_log:.4f} | "
        f"log-bounds=[{l_log:.4f}, {u_log:.4f}] | "
        f"original-scale bounds=[{lower:.4f}, {upper:.4f}]"
    )
    return lower, upper, why


def _method4_winsorize_inplace(df: pd.DataFrame, col: str) -> int:
    """
    Method 4 — Winsorization (residual safety net).
    Clamps surviving extreme values to [p01, p99].  NO rows dropped.
    Returns count of values clamped.
    """
    p01 = df[col].quantile(0.01)
    p99 = df[col].quantile(0.99)
    mask = (df[col] < p01) | (df[col] > p99)
    count = int(mask.sum())
    df[col] = df[col].clip(lower=p01, upper=p99)
    return count, p01, p99


# ── Main public function ──────────────────────────────────────────────

OUTLIER_METHOD_DEFINITIONS = {
    "method_1_3sd": {
        "method": "3-SD Rule",
        "condition": "M < 0.50",
        "definition": (
            "When M < 0.50, the data has low to moderate variance relative to its mean, "
            "indicating a roughly symmetric (near-normal) distribution. "
            "In a true normal distribution, 99.73% of values fall within +/-3 "
            "standard deviations. Values beyond this boundary are statistically "
            "improbable genuine observations and are most likely noise or errors."
        ),
    },
    "method_2_log_zscore": {
        "method": "Log-Z-Score",
        "condition": "M >= 0.50 or M = infinity",
        "definition": (
            "When M >= 0.50, the data has heavy exponential tails or extreme "
            "variance, a distribution that stretches far to the right, such as "
            "transaction amounts, claim values, or insurance premiums. Applying "
            "standard Z-score or IQR on such data would either keep obvious "
            "outliers or remove too many valid high-value records. The logarithm "
            "compresses the scale by converting multiplicative relationships to "
            "additive ones. After transformation, the data behaves approximately "
            "normally in log-space, where Z-score bounds become valid again. The "
            "bounds are then back-transformed to the original scale for row removal."
        ),
    },
}


def handle_outliers_iqr(df: pd.DataFrame, protected_cols: set = None) -> pd.DataFrame:
    """
    Dynamic outlier removal engine.

    Per-column pipeline:
      1. Compute M = SD / |Mean| → select Method 1 / 2 / 3.
      2. Mark outlier ROWS for that column and DROP them from the DataFrame.
      3. Run Winsorization (Method 4) as a residual safety net (clamp, no drop).

    Terminal output per column:
      • Method chosen + one-line reason (WHY)
      • Exact bounds computed
      • Rows flagged, rows dropped, dataset shape before → after

    Returns cleaned DataFrame (fewer rows, same columns).
    """
    if protected_cols is None:
        protected_cols = set()

    df = df.copy()
    rows_start = len(df)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    total_dropped = 0
    outlier_report = {
        "rows_start": int(rows_start),
        "columns": {},
    }

    print()
    print("=" * 72)
    print("  [OUTLIER ENGINE]  Dynamic 3-Method Outlier Removal — START")
    print(f"  Dataset shape on entry : {df.shape[0]} rows × {df.shape[1]} cols")
    print("=" * 72)

    for col in numeric_cols:

        # ── Skip protected columns ───────────────────────────────────────
        if col in protected_cols:
            print(f"\n  [OUTLIER] ⏭  SKIP  '{col}'  (protected — PK / system column)")
            continue

        series = df[col].dropna()
        if len(series) < 3:
            print(f"\n  [OUTLIER] ⏭  SKIP  '{col}'  (< 3 non-null values)")
            continue

        if series.nunique(dropna=True) <= 2:
            print(f"\n  [OUTLIER] SKIP  '{col}'  (binary/low-cardinality numeric)")
            continue

        M          = _compute_variance_ratio(series)
        rows_before = len(df)

        # ── Select & apply primary method ───────────────────────────────
        if not np.isfinite(M) or M >= 0.50:
            method_id   = "METHOD 2"
            method_label = "Log-Transform + Z-Score"
            method_key = "method_2_log_zscore"
            lower, upper, why = _method2_log_zscore_bounds(series)

        else:
            method_id   = "METHOD 1"
            method_label = "3-SD Rule  (parametric)"
            method_key = "method_1_3sd"
            lower, upper, why = _method1_3sd_bounds(series)

        # ── Flag outlier rows (where this column is out-of-bounds) ───────
        outlier_mask  = (df[col] < lower) | (df[col] > upper)
        flagged_count = int(outlier_mask.sum())

        # ── DROP the outlier rows ─────────────────────────────────────────
        df = df[~outlier_mask].reset_index(drop=True)
        rows_after   = len(df)
        dropped_this = rows_before - rows_after
        total_dropped += dropped_this

        # ── Method 4: Winsorization on survivors (residual clamp) ────────
        winsor_count, p01, p99 = _method4_winsorize_inplace(df, col)
        method_definition = OUTLIER_METHOD_DEFINITIONS.get(method_key, {})
        outlier_report["columns"][col] = {
            "method_key": method_key,
            "method_id": method_id,
            "method": method_definition.get("method", method_label),
            "condition": method_definition.get("condition", ""),
            "definition": method_definition.get("definition", ""),
            "m_value": None if not np.isfinite(M) else float(M),
            "lower_bound": float(lower) if np.isfinite(lower) else None,
            "upper_bound": float(upper) if np.isfinite(upper) else None,
            "flagged_count": int(flagged_count),
            "dropped_count": int(dropped_this),
            "rows_before": int(rows_before),
            "rows_after": int(rows_after),
            "winsorized_count": int(winsor_count),
        }

        # ── Terminal report for this column ──────────────────────────────
        print()
        print(f"  ┌─ COLUMN : '{col}'")
        print(f"  │  M (SD/|Mean|)  = {M:.4f}")
        print(f"  │  ▶ {method_id} — {method_label}")
        print(f"  │  WHY  : {why}")
        print(f"  │  Outlier rows flagged  : {flagged_count}")
        print(f"  │  Outlier rows DROPPED  : {dropped_this}  "
              f"({rows_before} → {rows_after} rows)")
        if winsor_count:
            print(f"  │  Method 4 (Winsorize)  : {winsor_count} residuals clamped "
                  f"to [{p01:.4f}, {p99:.4f}]")
        else:
            print(f"  │  Method 4 (Winsorize)  : 0 residuals — column is clean ✓")
        print(f"  └─ Dataset shape after   : {df.shape[0]} rows × {df.shape[1]} cols")

    # ── Final summary ────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  [OUTLIER ENGINE]  COMPLETE")
    print(f"  Rows on entry    : {rows_start}")
    print(f"  Rows removed     : {total_dropped}")
    print(f"  Rows remaining   : {len(df)}  "
          f"({100 * len(df) / rows_start:.1f}% of original)")
    print(f"  Final shape      : {df.shape[0]} rows × {df.shape[1]} cols")
    print("=" * 72)
    print()

    outlier_report["rows_removed"] = int(total_dropped)
    outlier_report["rows_remaining"] = int(len(df))
    df.attrs["outlier_report"] = _json_safe(outlier_report)

    return df


# ==========================================================
# 8️⃣ ENCODING
# ==========================================================
def preprocess_data(df, target_column, protected_cols=set()):
    df = df.copy()
    encoded_feature_mapping = {}
    grouped_encoded_features = {}
    categorical_encoding_summary = {}
    encoded_columns = []
    categorical_source_columns = []
    validation = {
        "encoded_columns_exist": True,
        "others_columns_exist": True,
        "unencoded_categorical_columns": [],
        "training_schema_matches_master": True,
    }

    print("\n[PREPROCESS] Encoding decision summary:")

    for column in df.columns:

        if column in protected_cols:
            continue

        if column == target_column:
            continue

        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
            unique_values = df[column].nunique(dropna=False)
            print(
                f"[ENCODING] Column: {column} | unique={unique_values} → UNIVERSAL 5+1 ONE-HOT ENCODING"
            )

            encoded_df, feature_mapping, created_columns, encoding_summary = _build_5_plus_1_encoded_columns(
                df[column],
                column,
                top_n=5,
            )
            df = pd.concat([df.drop(columns=[column]), encoded_df], axis=1)
            encoded_feature_mapping.update(feature_mapping)
            grouped_encoded_features[column] = created_columns
            categorical_encoding_summary[column] = encoding_summary
            encoded_columns.extend(created_columns)
            categorical_source_columns.append(column)

    print("[PREPROCESS] Encoding completed\n")
    validation["encoded_columns_exist"] = all(col in df.columns for col in encoded_columns)

    expected_others_columns = []
    for source_col, meta in categorical_encoding_summary.items():
        if meta.get("used_others"):
            others_cols = [
                col for col in grouped_encoded_features.get(source_col, [])
                if str(encoded_feature_mapping.get(col, {}).get("category", "")).strip().lower() == "others"
            ]
            expected_others_columns.extend(others_cols)
    validation["others_columns_exist"] = all(col in df.columns for col in expected_others_columns)

    allowed_unencoded = set(protected_cols)
    if target_column:
        allowed_unencoded.add(target_column)
    validation["unencoded_categorical_columns"] = [
        col for col in df.select_dtypes(include=["object", "category"]).columns
        if col not in allowed_unencoded
    ]

    return df, {
        "encoded_feature_mapping": encoded_feature_mapping,
        "grouped_encoded_features": grouped_encoded_features,
        "categorical_encoding_summary": categorical_encoding_summary,
        "encoded_columns": encoded_columns,
        "categorical_source_columns": categorical_source_columns,
        "expected_others_columns": expected_others_columns,
        "validation": validation,
    }


# ==========================================================
# 9️⃣ CORRELATION FILTER
# ==========================================================
def drop_zero_correlation_features(df, target_column, protected_cols=set()):
    if not target_column or target_column not in df.columns:
        return df

    numeric_cols = df.select_dtypes(include=["number"]).columns
    corr = df[numeric_cols].corr()

    if target_column not in corr.columns:
        return df

    to_drop = [
        col for col in corr[target_column].index
        if col != target_column
        and col not in protected_cols
        and abs(corr[target_column][col]) < 0.01
    ]

    df.drop(columns=to_drop, inplace=True)
    return df


def drop_highly_correlated_features(df, threshold=0.90, protected_cols=set()):
    numeric_cols = df.select_dtypes(include=["number"]).columns
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Create an upper triangle matrix to avoid dropping both correlated features
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [
        column for column in upper.columns 
        if any(upper[column] > threshold) and column not in protected_cols
    ]
    
    if to_drop:
        print(f"[CORRELATION] Dropping highly correlated features (> {threshold}): {to_drop}")
        df.drop(columns=to_drop, inplace=True)
        
    return df


# ==========================================================
# 🔟-A  DATA IMBALANCE HANDLER
# ==========================================================
#
#  DEFINITION:
#    • minority class < 10% of total rows  →  IMBALANCED
#    • minority class >= 10% of total rows →  BALANCED (no action)
#
#  FLOW (classification problems only):
#    1. detect_imbalance()    → checks minority % against 10% threshold
#    2. If IMBALANCED         → handle_data_imbalance() runs:
#         a. Saves the ORIGINAL (imbalanced) dataset FIRST
#            as  unbalanced_before_master_{run_id}.csv
#         b. Then undersamples majority class so that:
#                minority  = 20%  of new total
#                majority  = 80%  of new total
#         c. This single balanced CSV is the one used for master
#    3. If BALANCED           → no extra files, master built from full data
#    4. Called AUTOMATICALLY inside build_master_dataset() BEFORE
#       master CSV is written — target column is never touched by encoding
# ==========================================================


# ------------------------------------------------------------------
# HELPER: Convert all numpy types → native Python for JSON safety
# ------------------------------------------------------------------
def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_json_safe(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj



# ------------------------------------------------------------------
# STEP 1: Detect whether the target column is imbalanced
# ------------------------------------------------------------------
def detect_imbalance(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Reads value_counts() of target_column.
    Returns is_imbalanced=True when minority class is < 10% of total rows.
    All values in the returned dict are plain Python types (JSON-safe).
    """
    if not target_column or target_column not in df.columns:
        print("[IMBALANCE] ⚠️  Target column not found – skipping imbalance check.")
        return {"is_imbalanced": False}

    counts         = df[target_column].value_counts()
    total          = len(df)
    majority_class = counts.index[0]
    minority_class = counts.index[-1]
    majority_count = int(counts.iloc[0])
    minority_count = int(counts.iloc[-1])
    minority_pct   = minority_count / total * 100
    ratio          = majority_count / minority_count if minority_count > 0 else float("inf")
    is_imbalanced  = minority_pct < 10.0

    print(f"\n[IMBALANCE] 📊 Class distribution for '{target_column}':")
    for cls, cnt in counts.items():
        print(f"            class {cls}: {int(cnt)} rows ({int(cnt)/total*100:.1f}%)")
    print(f"[IMBALANCE] Imbalance ratio  (majority/minority): {ratio:.2f}")
    print(f"[IMBALANCE] Minority share   : {minority_pct:.1f}%")
    print(f"[IMBALANCE] Is imbalanced    (minority < 10%)   : {is_imbalanced}")

    return _json_safe({
        "is_imbalanced":      bool(is_imbalanced),
        "majority_class":     str(majority_class),
        "minority_class":     str(minority_class),
        "majority_count":     int(majority_count),
        "minority_count":     int(minority_count),
        "imbalance_ratio":    float(round(ratio, 2)),
        "minority_pct":       float(round(minority_pct, 2)),
        "class_distribution": {str(k): int(v) for k, v in counts.items()},
    })


def _candidate_file_stub(target_share: float) -> str:
    minority_pct = int(round(target_share * 100))
    majority_pct = int(round((1.0 - target_share) * 100))
    return f"smote_{minority_pct}_{majority_pct}"


def build_smote_candidate_artifacts(
    df: pd.DataFrame,
    target_column: str,
    run_id: str,
    master_dir: str,
) -> Dict[str, Any]:
    """
    Build SMOTE candidate artifacts using TWO ITERATIONS (10:90 and 20:80)
    only when the minority class is below 10%.
    
    This function:
    1. Creates two SMOTE-balanced datasets with different ratios
    2. Saves them as CSV files in master_dir
    3. Returns metadata for each candidate for SXI comparison
    
    Args:
        df:             Input DataFrame
        target_column:  Name of target column
        run_id:         Unique run identifier
        master_dir:     Directory to save candidate CSVs
        
    Returns:
        Dict with executed flag, report, and candidates list
    """
    os.makedirs(master_dir, exist_ok=True)

    result: Dict[str, Any] = {
        "executed": False,
        "report": detect_imbalance(df, target_column),
        "candidates": [],
    }

    minority_pct = float(result["report"].get("minority_pct") or 0.0)

    if not result["report"].get("is_imbalanced", False):
        print("[IMBALANCE] Dataset is balanced enough. SMOTE candidate generation skipped.")
        return result

    if minority_pct >= 10.0:
        print(
            "[IMBALANCE] Minority class is already >= 10%. "
            "Skipping SMOTE candidate generation and keeping the original master dataset."
        )
        return result

    # Use handle_data_imbalance to get SMOTE iterations
    imbalance_result = handle_data_imbalance(
        df=df,
        target_column=target_column,
        run_id=run_id,
        master_dir=master_dir,
    )
    
    if not imbalance_result.get("executed"):
        print("[IMBALANCE] Failed to create SMOTE iterations.")
        return result
    
    # Process each SMOTE iteration and create candidates
    for idx, iteration in enumerate(imbalance_result.get("smote_iterations", []), start=1):
        if not iteration.get("success"):
            print(f"[IMBALANCE] Iteration {idx} ({iteration.get('iteration_label')}) failed. Skipping.")
            continue
        
        candidate_df = iteration["balanced_df"].copy()
        target_share = iteration["target_share"]
        iteration_label = iteration["iteration_label"]
        
        # Re-index master_id
        if "master_id" in candidate_df.columns:
            candidate_df.drop(columns=["master_id"], inplace=True)
        candidate_df.insert(0, "master_id", range(1, len(candidate_df) + 1))
        
        # Save to CSV
        safe_label = iteration_label.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "_").lower()
        candidate_filename = f"master_{run_id}_{safe_label}.csv"
        candidate_path = os.path.join(master_dir, candidate_filename)
        candidate_df.to_csv(candidate_path, index=False)
        
        # Get class distribution
        candidate_counts = candidate_df[target_column].value_counts()
        minority_pct = iteration["minority_pct"]
        
        candidate_info = {
            "executed": True,
            "method": "smote",
            "smote_applied": True,
            "file_name": candidate_filename,
            "csv": f"/media/master/{candidate_filename}",
            "master_df_encoded": f"/media/master/{candidate_filename}",
            "target_share": float(target_share),
            "target_ratio_label": iteration["target_ratio_label"],
            "iteration_label": iteration_label,
            "reason": None,
            "class_distribution": iteration.get("class_distribution", {str(k): int(v) for k, v in candidate_counts.items()}),
            "minority_pct": minority_pct,
            "rows": iteration.get("rows", len(candidate_df)),
            "achieved_ratio_label": iteration["achieved_ratio_label"],
        }
        
        result["candidates"].append(candidate_info)
        print(
            f"[IMBALANCE] ✅ {iteration_label} saved | "
            f"rows={candidate_info['rows']} | minority%={minority_pct:.1f}%"
        )
    
    result["executed"] = any(candidate.get("executed") for candidate in result["candidates"])
    print(f"[IMBALANCE] 🎉 {len(result['candidates'])} SMOTE candidates ready for SXI comparison")
    
    return _json_safe(result)


# ------------------------------------------------------------------
# STEP 2: Undersample majority class so minority reaches 20% of total
# ------------------------------------------------------------------
def undersample_to_ratio(
    df: pd.DataFrame,
    target_column: str,
    majority_class,
    minority_count: int,
) -> pd.DataFrame:
    """
    Keeps ALL minority rows intact.
    Reduces majority rows so that:
        minority / (minority + majority_kept) = 20%
        => majority_kept = minority_count * 4   (80:20 ratio)

    random_state=42 ensures reproducibility.
    Works for string labels ('Yes'/'No') and numeric labels (0/1).
    """
    # desired majority count so minority is exactly 20%
    # minority / total = 0.20  =>  total = minority / 0.20  =>  majority = total * 0.80
    desired_majority = int(minority_count * 4)   # 4× minority → 80 % of new total

    majority_mask = df[target_column].astype(str) == str(majority_class)
    majority_df   = df[majority_mask]
    minority_df   = df[~majority_mask]

    if len(majority_df) <= desired_majority:
        print(
            f"[IMBALANCE] Majority ({len(majority_df)}) already "
            f"<= desired ({desired_majority}). No reduction needed."
        )
        return df

    majority_sampled = majority_df.sample(n=desired_majority, random_state=42)
    balanced_df      = pd.concat([majority_sampled, minority_df], axis=0).reset_index(drop=True)

    new_minority_pct = len(minority_df) / len(balanced_df) * 100
    print(
        f"[IMBALANCE] ✅ 80:20 undersample → "
        f"majority kept: {desired_majority} | minority: {len(minority_df)} | "
        f"total: {len(balanced_df)} | minority%: {new_minority_pct:.1f}%"
    )
    return balanced_df


# ------------------------------------------------------------------
# STEP 3: Orchestrator — save unbalanced FIRST, then produce balanced CSV
# ------------------------------------------------------------------
def _apply_smote_iteration(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    target_column: str,
    target_share: float,
    minority_count: int,
    iteration_label: str,
) -> Dict[str, Any]:
    """
    Apply SMOTE for a specific target ratio (e.g., 0.10 for 10:90, 0.20 for 20:80).
    
    Args:
        df:               Full dataframe
        feature_df:       Features only (no target, no master_id)
        target_column:    Target column name
        target_share:     Target minority percentage (e.g., 0.20, 0.10)
        minority_count:   Current minority count
        iteration_label:  Label for logging
        
    Returns:
        Dict with balanced_df, minority_pct, achieved_ratio_label, success flag
    """
    result = {
        "iteration_label": iteration_label,
        "target_share": target_share,
        "target_ratio_label": f"{int(target_share * 100)}:{int((1 - target_share) * 100)}",
        "success": False,
        "balanced_df": None,
        "minority_pct": None,
        "achieved_ratio_label": None,
    }
    
    try:
        # Calculate desired sampling strategy (minority / majority ratio)
        desired_ratio = target_share / max(1e-9, (1.0 - target_share))
        
        # Apply SMOTE
        smote = SMOTE(
            sampling_strategy=desired_ratio,
            random_state=42,
            k_neighbors=min(5, minority_count - 1),
        )
        X_resampled, y_resampled = smote.fit_resample(feature_df, df[target_column])
        
        # Create balanced dataframe
        balanced_df = pd.DataFrame(X_resampled, columns=feature_df.columns)
        balanced_df[target_column] = y_resampled
        balanced_df.reset_index(drop=True, inplace=True)
        
        # Calculate metrics
        balanced_counts = balanced_df[target_column].value_counts()
        minority_pct = float(round(balanced_counts.iloc[-1] / len(balanced_df) * 100, 2))
        
        result["balanced_df"] = balanced_df
        result["minority_pct"] = minority_pct
        result["achieved_ratio_label"] = f"{minority_pct:.2f}:{100.0 - minority_pct:.2f}"
        result["class_distribution"] = {str(k): int(v) for k, v in balanced_counts.items()}
        result["rows"] = len(balanced_df)
        result["success"] = True
        
        print(
            f"[IMBALANCE] ✅ {iteration_label} | "
            f"rows={len(balanced_df)} | minority%={minority_pct:.1f}% | "
            f"ratio={result['achieved_ratio_label']}"
        )
        
    except Exception as e:
        print(f"[IMBALANCE] ❌ {iteration_label} failed: {str(e)}")
        result["error"] = str(e)
    
    return result


def handle_data_imbalance(
    df: pd.DataFrame,
    target_column: str,
    run_id: str,
    master_dir: str,
) -> Dict[str, Any]:
    """
    Balance an imbalanced binary classification dataset using TWO SMOTE ITERATIONS.
    
    This function creates two SMOTE-balanced datasets for comparison:
    1. Iteration 1: 10:90 ratio (10% minority, 90% majority)
    2. Iteration 2: 20:80 ratio (20% minority, 80% majority)
    
    Both datasets are prepared for SXI comparison, and the best one is selected.

    Args:
        df:             Input DataFrame with potential imbalance
        target_column:  Name of the target column
        run_id:         Unique run identifier
        master_dir:     Directory for master files

    Returns:
        The returned dict contains:
        executed        bool
        report          dict from detect_imbalance()
        balanced_df     pd.DataFrame (the best one from SXI comparison)
        smote_iterations  list of both iteration results for comparison
    """
    imbalance_report = detect_imbalance(df, target_column)
    imbalance_report["is_imbalanced"] = False  # Disabled per user request to use raw total.csv data

    result: Dict[str, Any] = {
        "executed":    False,
        "report":      imbalance_report,
        "balanced_df": None,
        "smote_iterations": [],
    }

    if not imbalance_report.get("is_imbalanced", False):
        print("[IMBALANCE] ✅ Dataset balancing (SMOTE) is disabled. Using raw unbalanced dataset.")
        return result

    majority_class = imbalance_report["majority_class"]
    minority_count = imbalance_report["minority_count"]
    
    if minority_count < 2:
        print("[IMBALANCE] ❌ Too few minority samples for SMOTE. Returning original dataset.")
        return result

    # Prepare features for SMOTE
    feature_df = df.drop(columns=[target_column]).copy()
    if "master_id" in feature_df.columns:
        feature_df.drop(columns=["master_id"], inplace=True)
    
    # Check for non-numeric features
    non_numeric = [
        col for col in feature_df.columns
        if not pd.api.types.is_numeric_dtype(feature_df[col]) and not pd.api.types.is_bool_dtype(feature_df[col])
    ]
    if non_numeric:
        print(f"[IMBALANCE] ⚠️  Non-numeric features found after encoding: {non_numeric[:5]}. Returning original dataset.")
        return result
    
    # Convert bool to int
    for col in feature_df.columns:
        if pd.api.types.is_bool_dtype(feature_df[col]):
            feature_df[col] = feature_df[col].astype(int)

    print("\n[IMBALANCE] 🔄 Creating TWO SMOTE iterations for comparison...")

    current_minority_share = minority_count / len(df)

    # ── ITERATION 1: 10:90 (10% minority, 90% majority) ──────────────────────
    print("\n[IMBALANCE] 📊 Iteration 1: SMOTE 10:90 ratio...")
    if current_minority_share >= 0.10:
        print(f"[IMBALANCE] Minority already >= 10% ({current_minority_share*100:.1f}%). Using original for Iteration 1.")
        iteration_1_data = {
            "iteration_label": "Iteration 1 (Original)",
            "target_share": float(current_minority_share),
            "target_ratio_label": "Original",
            "success": True,
            "balanced_df": df.copy(),
            "minority_pct": float(round(current_minority_share * 100, 2)),
            "achieved_ratio_label": f"Original ({float(round(current_minority_share * 100, 2))}%)",
            "class_distribution": imbalance_report.get("class_distribution", {}),
            "rows": len(df)
        }
    else:
        iteration_1_data = _apply_smote_iteration(
            df=df,
            feature_df=feature_df,
            target_column=target_column,
            target_share=0.10,
            minority_count=minority_count,
            iteration_label="Iteration 1 (10:90)"
        )
    result["smote_iterations"].append(iteration_1_data)

    # ── ITERATION 2: 20:80 (20% minority, 80% majority) ──────────────────────
    print("\n[IMBALANCE] 📊 Iteration 2: SMOTE 20:80 ratio...")
    if current_minority_share >= 0.20:
        print(f"[IMBALANCE] Minority already >= 20% ({current_minority_share*100:.1f}%). Using original for Iteration 2.")
        iteration_2_data = {
            "iteration_label": "Iteration 2 (Original)",
            "target_share": float(current_minority_share),
            "target_ratio_label": "Original",
            "success": True,
            "balanced_df": df.copy(),
            "minority_pct": float(round(current_minority_share * 100, 2)),
            "achieved_ratio_label": f"Original ({float(round(current_minority_share * 100, 2))}%)",
            "class_distribution": imbalance_report.get("class_distribution", {}),
            "rows": len(df)
        }
    else:
        iteration_2_data = _apply_smote_iteration(
            df=df,
            feature_df=feature_df,
            target_column=target_column,
            target_share=0.20,
            minority_count=minority_count,
            iteration_label="Iteration 2 (20:80)"
        )
    result["smote_iterations"].append(iteration_2_data)
    
    # Default to first successful iteration's dataframe
    for iteration in result["smote_iterations"]:
        if iteration.get("success"):
            balanced_df = iteration["balanced_df"].copy()
            if "master_id" in balanced_df.columns:
                balanced_df.drop(columns=["master_id"], inplace=True)
            balanced_df.insert(0, "master_id", range(1, len(balanced_df) + 1))
            
            result["balanced_df"] = balanced_df
            result["executed"] = True
            print(f"[IMBALANCE] 🎉 SMOTE iterations prepared for SXI comparison. Defaulting to {iteration['iteration_label']}")
            break
    
    return result



# ==========================================================
# 🔟 MASTER DATASET BUILDER
# ==========================================================
def build_master_dataset(
    run_id: str,
    preprocessing_mode: str = "auto",
    custom_config: dict | None = None,
    input_csv_path: str | None = None,
    output_name: str | None = None,
    request = None,
) -> Optional[Dict[str, Any]]:

    is_total = False
    task_type = (custom_config or {}).get("task_type", "").lower().strip()
    if task_type == "regression" or (input_csv_path and ("total.csv" in os.path.basename(input_csv_path) or "total.csv" in str(input_csv_path).lower() or output_name == "total.csv")):
        is_total = True

    def log_chat_msg(msg):
        if is_total and request and hasattr(request, "session"):
            history = request.session.get("chat_history", [])
            history.append({
                "sender": "bot",
                "text": msg
            })
            request.session["chat_history"] = history
            if hasattr(request.session, "modified"):
                request.session.modified = True
            # Throttled save: keep UI live updates without racing every single line.
            save_counter = int(request.session.get("_preprocess_chat_save_counter") or 0) + 1
            request.session["_preprocess_chat_save_counter"] = save_counter
            if save_counter == 1 or save_counter % 8 == 0:
                try:
                    from pipeline.session_utils import safe_session_save
                    safe_session_save(request.session, label="preprocess_chat")
                except Exception:
                    pass

    custom_config = custom_config or {}
    protected_cols = set(["master_id", "source_file", "run_id"])

    df = safe_read_file(input_csv_path)
    if df is None or df.empty:
        return None

    df = normalize_columns(df)
    master = df.copy()

    pk = custom_config.get("primary_key") or ""
    protected_cols.update([normalize_column_name(c) for c in pk.split(",") if c.strip()])

    encoding_artifacts = {
        "encoded_feature_mapping": {},
        "grouped_encoded_features": {},
        "categorical_encoding_summary": {},
        "encoded_columns": [],
        "categorical_source_columns": [],
        "expected_others_columns": [],
        "validation": {
            "encoded_columns_exist": True,
            "others_columns_exist": True,
            "unencoded_categorical_columns": [],
            "training_schema_matches_master": True,
        },
    }
    outlier_artifacts = {}
    # always initialise target so it is defined in both auto and custom mode
    target = ""

    if preprocessing_mode == "auto":
        
         # Target detection — MUST happen FIRST before any preprocessing
        target = custom_config.get("target_column")
        target = normalize_column_name(target) if target else ""

        # Protect target immediately — before missing, encoding, correlation, and outliers

        # ── IMBALANCE CHECK — RAW data pe, preprocessing se PEHLE (Disabled per request) ────────
        master_dir = os.path.join(settings.MEDIA_ROOT, "master")
        task_type  = custom_config.get("task_type", "").lower().strip()
        is_classification = task_type == "classification"

        imbalance_result = {
            "executed": False,
            "report": {"is_imbalanced": False},
            "candidates": [],
        }
        # ──────────────────────────────────────────────────────────────────


        # Protect target for missing handling / encoding / correlation
        if target and target in master.columns:
            protected_cols.add(target)

        # 1) Missing handling
        log_chat_msg("hey we started missing value handling")
        log_chat_msg("we used median/mode imputation")
        master = handle_missing_val(master, protected_cols)
        log_chat_msg("hey we completed missing value handling")

        if target and target not in master.columns:
            print(f"[WARNING] Target column '{target}' missing after feature engineering.")

        # 2) Universal 5+1 encoding (before correlation + outliers)
        log_chat_msg("hey we started categorical encoding")
        log_chat_msg("we used universal 5+1 encoding")
        master, encoding_artifacts = preprocess_data(
            master,
            target,
            protected_cols
        )
        log_chat_msg("hey we completed categorical encoding")

        if target and target in master.columns:
            protected_cols.add(target)
        protected_cols.update(encoding_artifacts.get("encoded_columns", []))

        # 3) Correlation filtering BEFORE outlier row-drops
        #    so target-proxy / redundant numeric columns cannot wipe a class.
        log_chat_msg("hey we started zero-correlation feature filtering")
        log_chat_msg("we dropped zero-correlation features")
        master = drop_zero_correlation_features(master, target, protected_cols)
        log_chat_msg("hey we completed feature filtering")

        log_chat_msg("hey we started highly-correlated feature filtering")
        master = drop_highly_correlated_features(master, threshold=0.90, protected_cols=protected_cols)
        log_chat_msg("we dropped highly-correlated features (>0.90)")
        log_chat_msg("hey we completed highly-correlated feature filtering")

        # Also drop numeric features that are near-duplicates of the target
        # (pairwise high-corr may keep one purchase/revenue proxy depending on column order).
        if target and target in master.columns:
            numeric_cols = master.select_dtypes(include=["number"]).columns
            if target in numeric_cols:
                target_corr = master[numeric_cols].corr()[target].abs()
                target_proxy_drop = [
                    col for col, val in target_corr.items()
                    if col != target
                    and col not in protected_cols
                    and pd.notna(val)
                    and float(val) >= 0.90
                ]
                if target_proxy_drop:
                    print(
                        f"[CORRELATION] Dropping target-proxy features "
                        f"(|corr({target})| >= 0.90): {target_proxy_drop}"
                    )
                    master.drop(columns=target_proxy_drop, inplace=True, errors="ignore")
                    log_chat_msg(
                        f"we dropped target-proxy features (|corr|>=0.90): "
                        f"{', '.join(target_proxy_drop)}"
                    )

        # 4) Outlier handling AFTER correlation cleanup (row drops on remaining features only)
        # Keep target protected; also skip remaining high-|corr| target proxies if any survived.
        outlier_protected_cols = set(protected_cols)
        if target and target in master.columns:
            outlier_protected_cols.add(target)
            numeric_cols = master.select_dtypes(include=["number"]).columns
            if target in numeric_cols:
                target_corr = master[numeric_cols].corr()[target].abs()
                for col, val in target_corr.items():
                    if col == target or col in outlier_protected_cols:
                        continue
                    if pd.notna(val) and float(val) >= 0.50:
                        outlier_protected_cols.add(col)
        print(
            f"[OUTLIER] Target '{target}' protected; "
            f"{len(outlier_protected_cols)} columns excluded from row-drop outlier engine"
        )
        log_chat_msg("hey we started outlier handling")
        log_chat_msg("we used Dynamic 3-Method Outlier Removal")
        master = handle_outliers_iqr(master, outlier_protected_cols)

        # Log column-specific dynamic outlier methods
        outlier_report = master.attrs.get("outlier_report") or {}
        columns_info = outlier_report.get("columns", {})
        if columns_info:
            for col_name, info in columns_info.items():
                m_val = info.get("m_value")
                method_id = info.get("method_id")
                method_name = info.get("method")
                if m_val is not None:
                    log_chat_msg(f"Column '{col_name}': M-Score={m_val:.4f} → Used {method_id} ({method_name})")
                else:
                    log_chat_msg(f"Column '{col_name}': Used {method_id} ({method_name})")
        log_chat_msg("hey we completed outlier handling")
        outlier_artifacts = dict(master.attrs.get("outlier_report") or {})
        if target:
            outlier_columns = outlier_artifacts.get("columns") or {}
            outlier_artifacts["target"] = outlier_columns.get(target, {})


    else:
        log_chat_msg("hey we started custom preprocessing")
        method_name = custom_config.get("scaling", {}).get("method") or "custom"
        log_chat_msg(f"we used {method_name} method")
        master = run_custom_preprocessing(master, custom_config)
        log_chat_msg("hey we completed custom preprocessing")
        # read target in custom mode too so imbalance check runs
        target = custom_config.get("target_column", "")
        target = normalize_column_name(target) if target else ""
        master_dir = os.path.join(settings.MEDIA_ROOT, "master")
        task_type = custom_config.get("task_type", "").lower().strip()
        is_classification = task_type == "classification"
        imbalance_result = {
            "executed": False,
            "report": {"is_imbalanced": False},
            "candidates": [],
        }
        if target and target in master.columns:
            protected_cols.add(target)
        log_chat_msg("hey we started missing value handling")
        log_chat_msg("we used median/mode imputation")
        master = handle_missing_val(master, protected_cols)
        log_chat_msg("hey we completed missing value handling")
        log_chat_msg("hey we started categorical encoding")
        log_chat_msg("we used universal 5+1 encoding")
        master, encoding_artifacts = preprocess_data(
            master,
            target,
            protected_cols
        )
        log_chat_msg("hey we completed categorical encoding")

    master = master.loc[:, ~master.columns.duplicated()]
    master.dropna(axis=1, how="all", inplace=True)
    master = coerce_numeric_like_target(master, target)
    master_df_encoded = master.copy()

    # Min-Max Normalize all numeric columns except master_id and the target column
    log_chat_msg("hey we started normalization")
    log_chat_msg("we used minmax scalar")
    for col in master_df_encoded.columns:
        if col in ("master_id", target):
            continue
        if pd.api.types.is_numeric_dtype(master_df_encoded[col]):
            # Skip normalization for binary/classification features (<= 2 unique values)
            if master_df_encoded[col].nunique(dropna=True) <= 2:
                continue

            col_min = master_df_encoded[col].min()
            col_max = master_df_encoded[col].max()
            if pd.notna(col_min) and pd.notna(col_max) and (col_max - col_min) > 0:
                master_df_encoded[col] = (master_df_encoded[col] - col_min) / (col_max - col_min)
            else:
                master_df_encoded[col] = 0.0
    log_chat_msg("hey we completed normalization")

    if "master_id" not in master_df_encoded.columns:
        master_df_encoded.insert(0, "master_id", range(1, len(master_df_encoded) + 1))

    validation = dict(encoding_artifacts.get("validation") or {})
    validation["encoded_columns_exist"] = all(
        col in master_df_encoded.columns for col in encoding_artifacts.get("encoded_columns", [])
    )
    validation["others_columns_exist"] = all(
        col in master_df_encoded.columns for col in encoding_artifacts.get("expected_others_columns", [])
    )
    allowed_raw_categorical = set(protected_cols)
    if target:
        allowed_raw_categorical.add(target)
    validation["unencoded_categorical_columns"] = [
        col for col in master_df_encoded.select_dtypes(include=["object", "category"]).columns
        if col not in allowed_raw_categorical
    ]
    validation["training_schema_matches_master"] = bool(
        encoding_artifacts.get("encoded_columns")
        or not validation["unencoded_categorical_columns"]
    )
    encoding_artifacts["validation"] = validation

    os.makedirs(master_dir, exist_ok=True)

    # If balancing was applied, keep only the final master artifact.
    for stale_name in [
        f"unbalanced_before_master_{run_id}.csv",
        f"master_{run_id}_balanced_80_20.csv",
    ]:
        stale_path = os.path.join(master_dir, stale_name)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    folder_name = output_name if output_name else "total"
    master_dir = os.path.join(settings.MEDIA_ROOT, "master", folder_name)
    os.makedirs(master_dir, exist_ok=True)

    if output_name:
        csv_filename = f"master_{run_id}_{output_name}.csv"
        metadata_filename = f"master_{run_id}_{output_name}_encoding.json"
    else:
        csv_filename = f"master_{run_id}.csv"
        metadata_filename = f"master_{run_id}_encoding.json"

    csv_path = os.path.join(master_dir, csv_filename)
    master_df_encoded.to_csv(csv_path, index=False)
    metadata_path = os.path.join(master_dir, metadata_filename)
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(_json_safe(encoding_artifacts), metadata_file, ensure_ascii=False, indent=2)

    return _json_safe({
        "csv": f"/media/master/{folder_name}/{csv_filename}",
        "master_df_encoded": f"/media/master/{folder_name}/{csv_filename}",
        "original_processed_csv": None,
        "encoding_metadata_json": f"/media/master/{folder_name}/{metadata_filename}",
        "encoded_feature_mapping": encoding_artifacts.get("encoded_feature_mapping", {}),
        "grouped_encoded_features": encoding_artifacts.get("grouped_encoded_features", {}),
        "categorical_encoding_summary": encoding_artifacts.get("categorical_encoding_summary", {}),
        "outlier": outlier_artifacts,
        "summary": {
            "rows": int(len(master_df_encoded)),
            "cols": list(master_df_encoded.columns),
            "label_encoded_columns": [],
            "label_mappings": [],
            "encoded_columns": encoding_artifacts.get("encoded_columns", []),
            "categorical_source_columns": encoding_artifacts.get("categorical_source_columns", []),
            "grouped_encoded_features": encoding_artifacts.get("grouped_encoded_features", {}),
            "categorical_encoding_summary": encoding_artifacts.get("categorical_encoding_summary", {}),
            "validation": validation,
            "outlier": outlier_artifacts,
        },
        "imbalance": {
            "executed": imbalance_result["executed"],
            "report": imbalance_result["report"],
            "candidates": imbalance_result.get("candidates", []),
            "final_master": f"master_{run_id}.csv",
        },
    })


def save_master_from_dataframe(df: pd.DataFrame, run_id: str):
    """
    Final safe save for master dataset
    - Removes duplicate columns
    - Drops fully empty columns
    - Inserts master_id
    """

    master = df.copy()

    # 🔐 Ensure unique columns
    if master.columns.duplicated().any():
        dupes = master.columns[master.columns.duplicated()].tolist()
        print("⚠️ Removing duplicate columns before save:", dupes)
        master = master.loc[:, ~master.columns.duplicated()]

    #  Drop fully empty columns
    master = master.dropna(axis=1, how="all")

    #  Insert master_id
    if "master_id" not in master.columns:
        master.insert(0, "master_id", range(1, len(master) + 1))

    master_dir = os.path.join(settings.MEDIA_ROOT, "master")
    os.makedirs(master_dir, exist_ok=True)

    csv_path = os.path.join(master_dir, f"master_{run_id}.csv")

    master.to_csv(csv_path, index=False)

    return {
        "csv": f"{settings.MEDIA_URL}master/master_{run_id}.csv",
        "summary": {
            "rows": int(len(master)),
            "cols": list(master.columns),
        }
    }
