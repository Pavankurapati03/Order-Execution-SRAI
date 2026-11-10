import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_LIKE_COLUMNS = {
    "id",
    "index",
    "i_n_d_e_x",
    "master_id",
    "customer_id",
    "cust_id",
    "user_id",
    "account_id",
    "primary_key",
}

DERIVED_LEAK_COLUMNS = {
    "composite_dxi",
    "composite_dxi_label",
    "gd_bddxi",
    "netqyty_bucket",
}

COMMERCE_POST_OUTCOME = {
    "nextpurchasedate",
    "nextorderdate",
    "nextpurchase",
    "nextorder",
    "futurepurchase",
    "futurerevenue",
    "totalfuturerevenue",
    "ltvproxy",
    "ltvafter",
    "repeatpurchase30d",
    "repeatpurchase60d",
    "repeatpurchase90d",
    "repeatpurchase",
    "daysuntilnextpurchase",
    "daystonextpurchase",
    "daystonext",
    "timetoevent",
    "eventobserved",
    "isrepurchaseready",
    "norders",
    "totalorders",
    "ordercount",
}


def _normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _is_binary_target(series: pd.Series) -> bool:
    unique_values = pd.Series(series).dropna().unique().tolist()
    return len(unique_values) == 2


def detect_suspected_leakage_columns(
    df: pd.DataFrame,
    target_col: str,
    primary_key: Optional[str] = None,
) -> Dict[str, List[str]]:
    target_norm = _normalize_name(target_col)
    primary_key_norm = _normalize_name(primary_key)

    id_like: List[str] = []
    derived_like: List[str] = []
    target_like: List[str] = []
    post_outcome: List[str] = []
    time_like: List[str] = []

    cancelled_target = target_norm in {"cancelled", "cancel", "cancellation"}
    target_token_patterns = [
        f"{target_norm}label",
        f"{target_norm}score",
        f"{target_norm}bucket",
        f"{target_norm}flag",
        f"{target_norm}class",
        f"{target_norm}target",
    ]

    for column in df.columns:
        if column == target_col:
            continue

        normalized = _normalize_name(column)
        if not normalized:
            continue

        if normalized in ID_LIKE_COLUMNS or normalized.endswith(("id", "index", "key")):
            id_like.append(column)
            continue

        if primary_key_norm and normalized == primary_key_norm:
            id_like.append(column)
            continue

        if normalized in DERIVED_LEAK_COLUMNS or normalized.startswith(("compositedxi", "sxi")):
            derived_like.append(column)
            continue

        if target_norm and (
            target_norm in normalized
            or any(token in normalized for token in target_token_patterns)
        ):
            target_like.append(column)
            continue

        if cancelled_target and normalized in {
            "deptime",
            "taxiout",
            "taxiin",
            "wheelsoff",
            "wheelson",
            "airtime",
            "weatherdelay",
            "lateaircraftdelay",
        }:
            post_outcome.append(column)
            continue

        if normalized in COMMERCE_POST_OUTCOME or normalized.startswith(
            ("repeatpurchase", "futurepurchase", "nextpurchase", "nextorder")
        ):
            # Skip when this column is the training target (already continued above).
            post_outcome.append(column)
            continue

        if normalized in {"date", "timestamp", "datetime", "eventdate", "fldate"}:
            time_like.append(column)

    suspected_columns = sorted(
        set(id_like + derived_like + target_like + post_outcome)
    )
    return {
        "suspected_columns": suspected_columns,
        "id_like": sorted(set(id_like)),
        "derived_like": sorted(set(derived_like)),
        "target_like": sorted(set(target_like)),
        "post_outcome": sorted(set(post_outcome)),
        "time_like": sorted(set(time_like)),
    }


def find_high_correlation_columns(
    df: pd.DataFrame,
    target_col: str,
    threshold: float = 0.99,
) -> List[Tuple[str, float]]:
    if target_col not in df.columns:
        return []

    target = pd.to_numeric(df[target_col], errors="coerce")
    if target.nunique(dropna=True) < 2:
        return []

    highly_correlated: List[Tuple[str, float]] = []
    for column in df.columns:
        if column == target_col:
            continue
        feature = pd.to_numeric(df[column], errors="coerce")
        valid_mask = target.notna() & feature.notna()
        if valid_mask.sum() < 3:
            continue
        valid_feature = feature.loc[valid_mask]
        valid_target = target.loc[valid_mask]
        if valid_feature.nunique(dropna=True) < 2:
            continue
        corr = float(valid_feature.corr(valid_target))
        if np.isfinite(corr) and abs(corr) >= threshold:
            highly_correlated.append((column, corr))

    return sorted(highly_correlated, key=lambda item: abs(item[1]), reverse=True)


def build_binary_split_indices(
    df: pd.DataFrame,
    target_col: str,
    *,
    raw_date_col: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Index, pd.Index, Dict[str, Any]]:
    working_df = df.copy()

    if raw_date_col and raw_date_col in working_df.columns:
        parsed_dates = pd.to_datetime(working_df[raw_date_col], errors="coerce")
        valid_mask = parsed_dates.notna() & working_df[target_col].notna()
        working_df = working_df.loc[valid_mask].copy()
        parsed_dates = parsed_dates.loc[valid_mask]
        working_df["_split_date"] = parsed_dates
        working_df = working_df.sort_values("_split_date", kind="stable")
        split_at = max(1, int(len(working_df) * (1.0 - test_size)))
        split_at = min(split_at, len(working_df) - 1)
        train_index = working_df.index[:split_at]
        test_index = working_df.index[split_at:]
        metadata = {
            "strategy": "chronological",
            "date_column": raw_date_col,
            "train_start": str(working_df["_split_date"].iloc[0].date()),
            "train_end": str(working_df.loc[train_index, "_split_date"].max().date()),
            "test_start": str(working_df.loc[test_index, "_split_date"].min().date()),
            "test_end": str(working_df["_split_date"].iloc[-1].date()),
        }
        return train_index, test_index, metadata

    valid_df = working_df.loc[working_df[target_col].notna()].copy()
    y = pd.to_numeric(valid_df[target_col], errors="coerce")
    valid_df = valid_df.loc[y.notna()].copy()
    y = y.loc[valid_df.index].astype(int)
    train_index, test_index = train_test_split(
        valid_df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y if _is_binary_target(y) else None,
    )
    return pd.Index(train_index), pd.Index(test_index), {"strategy": "random_stratified"}


def fit_safe_binary_logistic_pipeline(
    df: pd.DataFrame,
    target_col: str,
    *,
    primary_key: Optional[str] = None,
    raw_date_col: Optional[str] = None,
    extra_drop_columns: Optional[Sequence[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    audit = detect_suspected_leakage_columns(df, target_col, primary_key=primary_key)
    drop_columns = set(audit["suspected_columns"])
    drop_columns.add(target_col)
    if extra_drop_columns:
        drop_columns.update(extra_drop_columns)

    train_index, test_index, split_metadata = build_binary_split_indices(
        df,
        target_col,
        raw_date_col=raw_date_col,
        test_size=test_size,
        random_state=random_state,
    )

    usable_df = df.loc[train_index.union(test_index)].copy()
    y = pd.to_numeric(usable_df[target_col], errors="coerce")
    valid_mask = y.notna()
    usable_df = usable_df.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(int)

    X = usable_df.drop(columns=list(drop_columns), errors="ignore").copy()
    X = X.loc[:, X.notna().any(axis=0)]

    train_index = train_index.intersection(X.index)
    test_index = test_index.intersection(X.index)
    X_train = X.loc[train_index].copy()
    X_test = X.loc[test_index].copy()
    y_train = y.loc[train_index].copy()
    y_test = y.loc[test_index].copy()

    numeric_columns = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [col for col in X_train.columns if col not in numeric_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "auc": float(roc_auc_score(y_test, probabilities)),
    }

    return {
        "audit": audit,
        "drop_columns": sorted(drop_columns),
        "feature_columns": X.columns.tolist(),
        "train_index": train_index,
        "test_index": test_index,
        "split_metadata": split_metadata,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": metrics,
    }
