"""
Adapter Layer: Isolates field-name discrepancies between the raw SXI result dictionary
and the standardized SolutionResult Pydantic contract.
"""

from typing import Dict, Any, List, Optional
from portal.agents.schemas import SolutionResult, FeatureWeight


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts string/number values (even with '%' or '$') to float."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        clean = str(val).replace("%", "").replace("$", "").replace(",", "").strip()
        return float(clean)
    except Exception:
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely converts value to int."""
    if val is None:
        return default
    try:
        return int(_safe_float(val, default))
    except Exception:
        return default


def adapt_sxi_to_solution_result(sxi_dict: Dict[str, Any]) -> SolutionResult:
    """
    Normalizes a raw SXI optimization dictionary into a validated SolutionResult Pydantic model.
    """
    sol_code = str(sxi_dict.get("solution_code") or "E4").upper().strip()
    target_col = str(sxi_dict.get("target_name") or sxi_dict.get("target_column") or "target")
    task_type_raw = str(sxi_dict.get("task_type") or "regression").lower()
    target_type = "binary" if "class" in task_type_raw or "binary" in task_type_raw else "continuous"
    
    kpi_name = str(sxi_dict.get("kpi_name") or sxi_dict.get("target_display") or "Primary KPI")
    b_kpi = _safe_float(sxi_dict.get("baseline_kpi"))
    
    # Extract Level 1, 2, 3 optimized KPIs
    lvl1 = _safe_float(sxi_dict.get("lvl1_kpi", sxi_dict.get("optimized_kpi", b_kpi)))
    lvl2 = _safe_float(sxi_dict.get("lvl2_kpi", lvl1))
    lvl3 = _safe_float(sxi_dict.get("lvl3_kpi", lvl2))
    kpi_optimized = {"L1": lvl1, "L2": lvl2, "L3": lvl3}

    # Extract SXI scores
    sxi_b = _safe_float(sxi_dict.get("current_sxi_score", sxi_dict.get("current_sxi", 6.0)))
    sxi_o = _safe_float(sxi_dict.get("target_sxi_score", sxi_dict.get("lvl2_sxi", 8.5)))

    # Model Performance & Holdout Metrics
    perf = sxi_dict.get("model_performance") or {}
    holdout: Dict[str, float] = {}
    if "accuracy_pct" in perf and perf["accuracy_pct"] is not None:
        holdout["accuracy"] = _safe_float(perf["accuracy_pct"]) / 100.0
    elif "accuracy" in perf:
        holdout["accuracy"] = _safe_float(perf["accuracy"])

    if "r2" in perf and perf["r2"] is not None:
        holdout["r2"] = _safe_float(perf["r2"])
    if "rmse" in perf and perf["rmse"] is not None:
        holdout["rmse"] = _safe_float(perf["rmse"])
    if "mae" in perf and perf["mae"] is not None:
        holdout["mae"] = _safe_float(perf["mae"])
    if "auc_score" in perf and perf["auc_score"] is not None:
        holdout["auc"] = _safe_float(perf["auc_score"])
    if "f1_score" in perf and perf["f1_score"] is not None:
        holdout["f1"] = _safe_float(perf["f1_score"])

    train_samples = _safe_int(perf.get("train_samples", sxi_dict.get("train_count", 0)))
    test_samples = _safe_int(perf.get("test_samples", sxi_dict.get("test_count", 0)))
    primary_model = str(perf.get("model_type") or "XGBoost Ensemble")

    # Features and Leakage Detection
    weights_raw = sxi_dict.get("weights_table") or sxi_dict.get("top_drivers") or []
    features: List[FeatureWeight] = []
    leakage_suspects: List[str] = []

    for item in weights_raw:
        fname = str(item.get("Feature") or item.get("feature") or item.get("name") or item.get("feature_name") or item.get("feature_display") or "").strip()
        if not fname or fname.lower() == "unknown":
            continue
        w_val = item.get("Composite_Weight") or item.get("weight") or item.get("importance") or item.get("importance_score") or item.get("importance_pct") or item.get("pct") or 0.0
        if isinstance(w_val, str):
            w_val = w_val.replace("%", "").strip()
        w_val = _safe_float(w_val)
        # Ensure weight is bounded in [0, 1]
        norm_w = min(max(w_val if w_val <= 1.0 else w_val / 100.0, 0.0), 1.0)
        
        dir_raw = str(item.get("direction") or item.get("Direction") or "unknown").lower()
        direction = "down" if "down" in dir_raw or "dec" in dir_raw else ("up" if "up" in dir_raw or "inc" in dir_raw else "unknown")
        actionable = bool(item.get("actionable") if "actionable" in item else (item.get("Actionability", "Yes") == "Yes"))

        fw = FeatureWeight(
            name=fname,
            weight=norm_w,
            direction=direction,
            actionable=actionable
        )
        features.append(fw)

        # Leakage guard: a single feature with > 60% importance
        if norm_w >= 0.60:
            leakage_suspects.append(fname)

    # Sort features by descending weight
    features.sort(key=lambda x: x.weight, reverse=True)

    # Data Quality Stats
    dq: Dict[str, float] = {}
    if "data_quality" in sxi_dict:
        dq = {k: _safe_float(v) for k, v in sxi_dict["data_quality"].items()}
    else:
        dq = {
            "missing_pct": _safe_float(sxi_dict.get("missing_pct", 0.0)),
            "outlier_pct": _safe_float(sxi_dict.get("outlier_pct", 0.0)),
        }

    return SolutionResult(
        solution=sol_code,
        target_column=target_col,
        target_type=target_type,
        kpi_name=kpi_name,
        kpi_baseline=b_kpi,
        kpi_optimized=kpi_optimized,
        sxi_baseline=sxi_b,
        sxi_optimized=sxi_o,
        models_run=[primary_model],
        primary_model=primary_model,
        holdout=holdout,
        n_train=train_samples,
        n_test=test_samples,
        features=features,
        leakage_suspects=leakage_suspects,
        feature_aliases=sxi_dict.get("feature_aliases", {}),
        data_quality=dq
    )
