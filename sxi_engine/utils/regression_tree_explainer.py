import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.tree import _tree


def humanize_label(value):
    text = str(value or "").strip().replace("_", " ")
    if not text:
        return ""
    return " ".join("SXI" if part.lower() == "sxi" else part.capitalize() for part in text.split())


def _safe_float(value, default=None):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        return default
    return numeric


def _format_number(value):
    value = _safe_float(value)
    if value is None:
        return ""
    if abs(value - round(value)) < 0.01:
        return str(int(round(value)))
    return f"{value:.2f}"


def infer_one_hot_metadata(feature_names, X=None, encoded_feature_mapping=None):
    """Build a generic encoded-column reverse map for get_dummies/OneHotEncoder output."""
    metadata = dict(encoded_feature_mapping or {})
    feature_names = [str(col) for col in feature_names]

    if X is None or not isinstance(X, pd.DataFrame) or X.empty:
        binary_columns = set(feature_names)
    else:
        binary_columns = set()
        for col in feature_names:
            if col not in X.columns:
                continue
            values = pd.to_numeric(X[col], errors="coerce").dropna().unique()
            if len(values) and set(np.round(values, 6)).issubset({0.0, 1.0}):
                binary_columns.add(col)

    prefix_groups = defaultdict(list)
    for col in feature_names:
        if col in metadata or col not in binary_columns or "_" not in col:
            continue
        prefix, category = col.rsplit("_", 1)
        if prefix and category:
            prefix_groups[prefix].append((col, category))

    for prefix, columns in prefix_groups.items():
        if len(columns) < 2:
            col, category = columns[0]
            if category and not re.fullmatch(r"[-+]?\d+(\.\d+)?", category):
                metadata[col] = {
                    "base_feature": prefix,
                    "category": category,
                    "base_feature_display": humanize_label(prefix),
                    "category_display": humanize_label(category),
                }
            continue
        if X is not None and isinstance(X, pd.DataFrame):
            available = [col for col, _ in columns if col in X.columns]
            if available:
                row_sums = X[available].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
                valid_ratio = float((row_sums <= 1.01).mean()) if len(row_sums) else 0.0
                if valid_ratio < 0.90:
                    continue
        for col, category in columns:
            metadata[col] = {
                "base_feature": prefix,
                "category": category,
                "base_feature_display": humanize_label(prefix),
                "category_display": humanize_label(category),
            }

    return metadata


def _condition_signature(condition, one_hot_metadata):
    meta = one_hot_metadata.get(condition["feature"])
    if meta and 0.0 <= condition["threshold"] <= 1.0:
        base = str(meta.get("base_feature") or condition["feature"])
        category = str(meta.get("category") or "")
        if condition["operator"] == ">":
            return ("cat_eq", base, category)
        return ("cat_ne", base, category)
    return ("num", condition["feature"], condition["operator"], round(float(condition["threshold"]), 6))


def simplify_conditions(conditions, one_hot_metadata=None):
    one_hot_metadata = one_hot_metadata or {}
    numeric_bounds = defaultdict(lambda: {"lower": None, "upper": None})
    categorical = defaultdict(lambda: {"positive": set(), "negative": set(), "display": None})
    passthrough = []

    for condition in conditions:
        feature = str(condition.get("feature") or "").strip()
        operator = condition.get("operator")
        threshold = _safe_float(condition.get("threshold"))
        if not feature or operator not in {"<=", ">"} or threshold is None:
            continue

        meta = one_hot_metadata.get(feature)
        if meta and 0.0 <= threshold <= 1.0:
            base = str(meta.get("base_feature") or feature)
            category = str(meta.get("category") or "").strip()
            display = meta.get("base_feature_display") or humanize_label(base)
            categorical[base]["display"] = display
            if operator == ">":
                categorical[base]["positive"].add(category)
            else:
                categorical[base]["negative"].add(category)
            continue

        bounds = numeric_bounds[feature]
        if operator == "<=":
            bounds["upper"] = threshold if bounds["upper"] is None else min(bounds["upper"], threshold)
        else:
            bounds["lower"] = threshold if bounds["lower"] is None else max(bounds["lower"], threshold)

    contradictions = []
    simplified = []
    for feature, bounds in numeric_bounds.items():
        lower = bounds["lower"]
        upper = bounds["upper"]
        if lower is not None and upper is not None and lower >= upper:
            contradictions.append(f"{feature} has incompatible bounds")
            continue
        if lower is not None:
            simplified.append({"feature": feature, "operator": ">", "threshold": lower, "kind": "numeric"})
        if upper is not None:
            simplified.append({"feature": feature, "operator": "<=", "threshold": upper, "kind": "numeric"})

    for base, details in categorical.items():
        positives = {x for x in details["positive"] if x}
        negatives = {x for x in details["negative"] if x}
        if len(positives) > 1:
            contradictions.append(f"{details['display'] or base} requires multiple categories")
            continue
        if positives.intersection(negatives):
            contradictions.append(f"{details['display'] or base} is both included and excluded")
            continue
        for category in sorted(positives):
            simplified.append({
                "feature": base,
                "operator": "=",
                "threshold": category,
                "kind": "categorical",
                "display_feature": details["display"] or humanize_label(base),
            })
        for category in sorted(negatives):
            simplified.append({
                "feature": base,
                "operator": "!=",
                "threshold": category,
                "kind": "categorical",
                "display_feature": details["display"] or humanize_label(base),
            })

    simplified.extend(passthrough)
    return simplified, contradictions


def render_condition(condition):
    if condition.get("kind") == "categorical":
        feature = condition.get("display_feature") or humanize_label(condition.get("feature"))
        category = humanize_label(condition.get("threshold"))
        return f"{feature} = {category}" if condition.get("operator") == "=" else f"{feature} is not {category}"

    feature = humanize_label(condition.get("feature"))
    threshold = _format_number(condition.get("threshold"))
    if condition.get("operator") == "<=":
        return f"{feature} <= {threshold}"
    return f"{feature} > {threshold}"


def render_business_condition(condition):
    text = render_condition(condition)
    text = text.replace(" is not ", " != ")
    return text


def _path_mask(X, simplified_conditions):
    if X is None or not isinstance(X, pd.DataFrame) or X.empty:
        return None
    mask = pd.Series(True, index=X.index)
    for condition in simplified_conditions:
        feature = condition.get("feature")
        if condition.get("kind") == "categorical":
            continue
        if feature not in X.columns:
            continue
        values = pd.to_numeric(X[feature], errors="coerce")
        threshold = _safe_float(condition.get("threshold"))
        if threshold is None:
            continue
        if condition.get("operator") == "<=":
            mask &= values <= threshold
        else:
            mask &= values > threshold
    return mask


def _region_signature(region):
    return {
        f"{cond.get('feature')}:{cond.get('operator')}:{_format_number(cond.get('threshold'))}:{cond.get('direction')}"
        for cond in (region.get("display_conditions") or region.get("raw_conditions") or region.get("conditions", []))
    }


def _similarity(left, right):
    left_sig = _region_signature(left)
    right_sig = _region_signature(right)
    if not left_sig or not right_sig:
        return 0.0
    return len(left_sig & right_sig) / float(len(left_sig | right_sig))


def _is_meaningful_condition(condition):
    feature = str(condition.get("feature") or "").strip().lower()
    if not feature:
        return False
    if feature in {
        "leaf",
        "value",
        "samples",
        "sample",
        "squared_error",
        "mse",
        "mae",
        "impurity",
        "prediction",
        "node",
    }:
        return False
    if feature.startswith(("value", "samples", "squared_error", "impurity")):
        return False
    return condition.get("operator") in {"<=", ">", "<", ">=", "=", "!="}


def _ordered_meaningful_conditions(raw_conditions, one_hot_metadata=None, max_conditions=3):
    one_hot_metadata = one_hot_metadata or {}
    selected = []

    for condition in raw_conditions or []:
        if not _is_meaningful_condition(condition):
            continue

        feature = str(condition.get("feature") or "").strip()
        meta = one_hot_metadata.get(feature)
        threshold = _safe_float(condition.get("threshold"))
        if meta and threshold is not None and 0.0 <= threshold <= 1.0:
            base = str(meta.get("base_feature") or feature)
            category = str(meta.get("category") or "").strip()
            selected.append({
                "feature": base,
                "operator": "=" if condition.get("operator") == ">" else "!=",
                "threshold": category,
                "kind": "categorical",
                "direction": condition.get("direction"),
                "display_feature": meta.get("base_feature_display") or humanize_label(base),
            })
        else:
            selected.append({
                "feature": feature,
                "operator": condition.get("operator"),
                "threshold": condition.get("threshold"),
                "kind": "numeric",
                "direction": condition.get("direction"),
            })

        if len(selected) == max_conditions:
            break

    return selected


def _importance_lookup(feature_importance_df):
    if not isinstance(feature_importance_df, pd.DataFrame) or feature_importance_df.empty:
        return {}, []
    feature_col = "Feature" if "Feature" in feature_importance_df.columns else feature_importance_df.columns[0]
    score_col = next(
        (col for col in ["Importance", "RF_Importance", "Tree Importance", "Tree_Importance"] if col in feature_importance_df.columns),
        None,
    )
    if score_col is None:
        numeric_cols = [col for col in feature_importance_df.columns if col != feature_col and pd.api.types.is_numeric_dtype(feature_importance_df[col])]
        score_col = numeric_cols[0] if numeric_cols else None
    if score_col is None:
        return {}, []

    scores = {}
    for _, row in feature_importance_df.iterrows():
        feature = str(row.get(feature_col) or "").strip()
        score = _safe_float(row.get(score_col), default=0.0) or 0.0
        if feature:
            scores[feature] = score
    top_features = [feature for feature, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]]
    return scores, top_features


def _leaf_regions(model, feature_names, X=None, y=None, feature_importance_df=None, encoded_feature_mapping=None, feature_scales=None):
    tree_ = model.tree_
    one_hot_metadata = infer_one_hot_metadata(feature_names, X=X, encoded_feature_mapping=encoded_feature_mapping)
    importance_scores, top_features = _importance_lookup(feature_importance_df)
    root_samples = max(int(tree_.n_node_samples[0]), 1)
    regions = []

    def recurse(node, raw_conditions):
        feature_idx = tree_.feature[node]
        if feature_idx != _tree.TREE_UNDEFINED:
            feature = str(feature_names[feature_idx])
            threshold = float(tree_.threshold[node])
            recurse(tree_.children_left[node], raw_conditions + [{
                "feature": feature,
                "operator": "<=",
                "threshold": threshold,
                "direction": "LEFT",
                "node": node,
                "child": int(tree_.children_left[node]),
            }])
            recurse(tree_.children_right[node], raw_conditions + [{
                "feature": feature,
                "operator": ">",
                "threshold": threshold,
                "direction": "RIGHT",
                "node": node,
                "child": int(tree_.children_right[node]),
            }])
            return

        simplified, contradictions = simplify_conditions(raw_conditions, one_hot_metadata)
        ordered_conditions = _ordered_meaningful_conditions(raw_conditions, one_hot_metadata, max_conditions=3)
        leaf_value = float(tree_.value[node][0][0])
        sample_count = int(tree_.n_node_samples[node])
        coverage = sample_count / float(root_samples)
        impurity = max(float(tree_.impurity[node]), 0.0)
        prediction_min = leaf_value
        prediction_max = leaf_value

        mask = _path_mask(X, simplified)
        if mask is not None and y is not None:
            y_series = pd.to_numeric(pd.Series(y), errors="coerce")
            covered_y = y_series.loc[mask.reindex(y_series.index, fill_value=False)].dropna()
            if not covered_y.empty:
                prediction_min = float(covered_y.quantile(0.10))
                prediction_max = float(covered_y.quantile(0.90))
        elif impurity > 0:
            spread = math.sqrt(impurity)
            prediction_min = leaf_value - spread
            prediction_max = leaf_value + spread

        condition_features = {cond.get("feature") for cond in simplified}
        top_overlap = len(condition_features.intersection(top_features[:3]))
        repeated_feature_count = sum(count - 1 for count in Counter(c["feature"] for c in raw_conditions).values() if count > 1)
        dominant_ratio = 0.0
        if raw_conditions:
            dominant_ratio = max(Counter(c["feature"] for c in raw_conditions).values()) / float(len(raw_conditions))

        regions.append({
            "node": node,
            "conditions": simplified,
            "display_conditions": ordered_conditions or simplified[:3],
            "raw_conditions": raw_conditions,
            "contradictions": contradictions,
            "prediction": leaf_value,
            "prediction_min": prediction_min,
            "prediction_max": prediction_max,
            "samples": sample_count,
            "coverage": coverage,
            "mean_target": leaf_value,
            "impurity": impurity,
            "top_overlap": top_overlap,
            "repeated_feature_count": repeated_feature_count,
            "dominant_ratio": dominant_ratio,
            "importance_scores": importance_scores,
            "top_features": top_features,
        })

    recurse(0, [])
    return regions, top_features


def _score_region(region, high=True):
    direction_score = region["prediction"] if high else -region["prediction"]
    coverage_bonus = min(region["coverage"], 0.25) * 4.0
    importance_bonus = region["top_overlap"] * 0.35
    quality_penalty = (region["repeated_feature_count"] * 0.15) + max(region["dominant_ratio"] - 0.60, 0.0)
    if region["contradictions"]:
        quality_penalty += 2.0
    if len(region["conditions"]) < 2:
        quality_penalty += 0.5
    return direction_score + coverage_bonus + importance_bonus - quality_penalty


def _select_distinct_regions(regions):
    valid_regions = [r for r in regions if not r["contradictions"] and r["conditions"]]
    if not valid_regions:
        valid_regions = regions
    high_candidates = sorted(valid_regions, key=lambda r: r.get("prediction", float("-inf")), reverse=True)
    low_candidates = sorted(valid_regions, key=lambda r: r.get("prediction", float("inf")))
    high_region = high_candidates[0] if high_candidates else None
    low_region = low_candidates[0] if low_candidates else None

    if high_region and low_region and high_region["node"] == low_region["node"] and len(valid_regions) > 1:
        low_region = next((candidate for candidate in low_candidates if candidate["node"] != high_region["node"]), low_region)
        if low_region and high_region["node"] == low_region["node"]:
            high_region = next((candidate for candidate in high_candidates if candidate["node"] != low_region["node"]), high_region)
    if high_region and low_region and _region_signature(high_region) == _region_signature(low_region) and len(valid_regions) > 1:
        low_region = next(
            (candidate for candidate in low_candidates if _region_signature(candidate) != _region_signature(high_region)),
            low_region,
        )
        if low_region and _region_signature(high_region) == _region_signature(low_region):
            high_region = next(
                (candidate for candidate in high_candidates if _region_signature(candidate) != _region_signature(low_region)),
                high_region,
            )
    return low_region, high_region


def validate_regression_regions(low_region, high_region, top_features):
    warnings = []
    if not low_region or not high_region:
        return ["Could not extract both high and low regression regions."]

    if _region_signature(low_region) == _region_signature(high_region):
        warnings.append("Low and high regression paths are identical.")

    similarity = _similarity(low_region, high_region)
    if similarity >= 0.60:
        warnings.append(f"High/low rule similarity is high ({similarity:.0%}).")

    for label, region in [("low", low_region), ("high", high_region)]:
        if len(region.get("conditions", [])) < 2:
            warnings.append(f"{label} region is shallow.")
        if region.get("dominant_ratio", 0.0) > 0.67:
            warnings.append(f"{label} region depends heavily on one feature.")
        if region.get("repeated_feature_count", 0) > 0:
            warnings.append(f"{label} region had repeated feature splits before simplification.")
        if region.get("contradictions"):
            warnings.extend(region["contradictions"])

    used_features = {
        cond.get("feature")
        for region in [low_region, high_region]
        for cond in region.get("conditions", [])
    }
    if top_features:
        missing = [feature for feature in top_features[:3] if feature not in used_features]
        if len(missing) == len(top_features[:3]):
            warnings.append("Extracted rules do not include the strongest feature-importance drivers.")
    return warnings


def region_to_path_string(region, label):
    header = f"Best Path for Class {'1' if str(label).lower().startswith('above') else '0'}: {label}"
    lines = [header]
    for condition in (region.get("display_conditions") or region.get("raw_conditions") or region.get("conditions", []))[:3]:
        if condition.get("kind") == "categorical":
            operator = "=" if condition.get("operator") == "=" else "!="
            lines.append(f"- {condition.get('display_feature') or humanize_label(condition.get('feature'))} {operator} {humanize_label(condition.get('threshold'))}")
        else:
            lines.append(f"- {condition.get('feature')} {condition.get('operator')} {_format_number(condition.get('threshold'))}")
    lines.append(
        "Prediction Range: "
        f"{_format_number(region.get('prediction_min'))} - {_format_number(region.get('prediction_max'))}"
    )
    lines.append(f"Coverage: {region.get('coverage', 0.0) * 100:.1f}% of samples")
    lines.append(f"Node Samples: {int(region.get('samples', 0))}")
    lines.append(f"Mean Target Value: {_format_number(region.get('mean_target'))}")
    return "\n".join(lines)


def extract_best_tree_paths(
    tree_model,
    feature_names,
    target_mean=None,
    mode="regression",
    X=None,
    y=None,
    feature_importance_df=None,
    encoded_feature_mapping=None,
    feature_scales=None,
):
    """
    Shared decision-tree extraction pipeline.

    Traverses the full fitted tree once, collects every root-to-leaf path,
    selects distinct LOW/HIGH leaves, and returns validated path strings.
    Current DT and Target DT should both call this function so threshold,
    path, and sentence rendering inputs are produced by identical logic.
    """
    if mode != "regression":
        raise ValueError("extract_best_tree_paths currently supports regression trees.")

    return explain_regression_tree_regions(
        tree_model,
        feature_names,
        X=X,
        y=y,
        feature_importance_df=feature_importance_df,
        encoded_feature_mapping=encoded_feature_mapping,
        feature_scales=feature_scales,
    )


def explain_regression_tree_regions(model, feature_names, X=None, y=None, feature_importance_df=None, encoded_feature_mapping=None, feature_scales=None):
    regions, top_features = _leaf_regions(
        model,
        feature_names,
        X=X,
        y=y,
        feature_importance_df=feature_importance_df,
        encoded_feature_mapping=encoded_feature_mapping,
        feature_scales=feature_scales,
    )
    low_region, high_region = _select_distinct_regions(regions)
    warnings = validate_regression_regions(low_region, high_region, top_features)
    high_regions = [
        region for region in sorted(regions, key=lambda item: item.get("prediction", float("-inf")), reverse=True)
        if not region.get("contradictions") and region.get("conditions")
    ][:3]
    low_regions = [
        region for region in sorted(regions, key=lambda item: item.get("prediction", float("inf")))
        if not region.get("contradictions") and region.get("conditions")
    ][:3]

    result = {
        "high_value_path": region_to_path_string(high_region, "Above Mean").splitlines()[1:] if high_region else [],
        "low_value_path": region_to_path_string(low_region, "Below Mean").splitlines()[1:] if low_region else [],
        "max_value": high_region["prediction"] if high_region else None,
        "min_value": low_region["prediction"] if low_region else None,
        "high_region": high_region,
        "low_region": low_region,
        "top_high_regions": high_regions,
        "top_low_regions": low_regions,
        "validation_warnings": warnings,
        "top_features": top_features,
        "all_regions": regions,
    }
    return result


def _format_currency_like(value):
    value = _safe_float(value)
    if value is None:
        return "N/A"
    if abs(value - round(value)) < 0.01:
        return f"${int(round(value))}"
    return f"${value:.2f}"


def _business_interpretation_for_region(region, high=True):
    conditions = region.get("display_conditions") or region.get("conditions") or []
    readable = [render_business_condition(condition).lower() for condition in conditions]
    if not readable:
        return "This path marks an important business segment worth reviewing."

    if high:
        return (
            "Records matching this combination tend to produce stronger target values. "
            "This path matters because it highlights the conditions most associated with the best business outcomes."
        )
    return (
        "Records matching this combination tend to produce weaker target values. "
        "This path matters because it highlights conditions that may require monitoring, cleanup, or process improvement."
    )


SEMANTIC_FEATURE_MAP = {
    "positive_metrics": {
        "annualincome",
        "purchasefrequency",
        "customersatisfaction",
        "engagementscore",
        "loyaltyprogram",
        "discountsavailed",
        "revenue",
        "numberofpurchases",
        "sessioncount",
        "loyaltyscore",
    },
    "reverse_metrics": {
        "lastpurchasedaysago",
        "dayssincelastpurchase",
        "churnrisk",
        "complaintcount",
        "returncount",
        "abandonmentscore",
    },
}


def _semantic_feature_key(feature):
    return re.sub(r"[^a-z0-9]+", "", str(feature or "").strip().lower())


def _semantic_feature_label(feature):
    key = _semantic_feature_key(feature)
    labels = {
        "annualincome": "annual income",
        "purchasefrequency": "purchase frequency",
        "customersatisfaction": "customer satisfaction",
        "engagementscore": "engagement score",
        "loyaltyprogram": "loyalty program participation",
        "discountsavailed": "discount utilization",
        "revenue": "revenue",
        "numberofpurchases": "number of purchases",
        "sessioncount": "session count",
        "loyaltyscore": "loyalty score",
        "lastpurchasedaysago": "time since the last purchase",
        "dayssincelastpurchase": "time since the last purchase",
        "churnrisk": "churn risk",
        "complaintcount": "complaint count",
        "returncount": "return count",
        "abandonmentscore": "abandonment score",
    }
    return labels.get(key, humanize_label(feature).lower())


def _semantic_metric_type(feature):
    key = _semantic_feature_key(feature)
    if key in SEMANTIC_FEATURE_MAP["reverse_metrics"]:
        return "reverse"
    if key in SEMANTIC_FEATURE_MAP["positive_metrics"]:
        return "positive"
    return "neutral"


def _business_class_headings(target_variable):
    target_text = humanize_label(target_variable) or "Target"
    normalized = _semantic_feature_key(target_text)
    if "purchase" in normalized or "buyer" in normalized:
        return "Not Purchased", "Purchased"
    if "churn" in normalized:
        return "Not Churned", "Churned"
    if "fraud" in normalized:
        return "Not Fraud", "Fraud"
    return f"Lower {target_text}", f"Higher {target_text}"


def _one_hot_business_phrase(condition):
    feature = humanize_label(condition.get("display_feature") or condition.get("feature"))
    category = humanize_label(condition.get("threshold"))
    if not feature or not category:
        return ""
    if condition.get("operator") == "=":
        return f"{feature} is {category}"
    return f"{feature} is not {category}"


def _fallback_one_hot_condition(condition):
    if condition.get("kind") == "categorical":
        return None
    feature = str(condition.get("feature") or "").strip()
    threshold = _safe_float(condition.get("threshold"))
    if "_" not in feature or threshold is None or not 0.0 <= threshold <= 1.0:
        return None
    base, category = feature.rsplit("_", 1)
    if not base or not category or re.fullmatch(r"[-+]?\d+(\.\d+)?", category):
        return None
    return {
        "feature": base,
        "operator": "=" if condition.get("operator") in {">", ">="} else "!=",
        "threshold": category,
        "kind": "categorical",
        "display_feature": humanize_label(base),
    }


def _numeric_business_phrase(feature, operator, high_class=True):
    metric_type = _semantic_metric_type(feature)
    label = _semantic_feature_label(feature)

    if metric_type == "reverse":
        if _semantic_feature_key(feature) in {"lastpurchasedaysago", "dayssincelastpurchase"}:
            return "More recent purchases" if high_class else "Longer time since the last purchase"
        return f"Lower {label}" if high_class else f"Higher {label}"

    if metric_type == "positive":
        if "frequency" in _semantic_feature_key(feature):
            return "More frequent purchase activity" if high_class else "Less frequent purchase activity"
        return f"Higher {label}" if high_class else f"Lower {label}"

    if operator in {"<=", "<"}:
        return f"Lower {label}"
    if operator in {">", ">="}:
        return f"Higher {label}"
    return label.capitalize()


def _label_is_positive(class_label):
    label = _semantic_feature_key(class_label)
    negative_markers = {
        "not",
        "no",
        "non",
        "bad",
        "negative",
        "fail",
        "failure",
        "fraud",
        "risk",
        "churn",
        "reject",
        "denied",
        "cancel",
        "lost",
        "lower",
        "below",
    }
    positive_markers = {
        "yes",
        "good",
        "positive",
        "success",
        "approve",
        "purchase",
        "purchased",
        "buyer",
        "converted",
        "retain",
        "higher",
        "above",
    }
    if any(marker in label for marker in negative_markers):
        return False
    if any(marker in label for marker in positive_markers):
        return True
    return None


def _class_polarity_map(class_0_label, class_1_label):
    class_0_positive = _label_is_positive(class_0_label)
    class_1_positive = _label_is_positive(class_1_label)

    if class_0_positive is None and class_1_positive is None:
        class_0_positive, class_1_positive = False, True
    elif class_0_positive is None:
        class_0_positive = not class_1_positive
    elif class_1_positive is None:
        class_1_positive = not class_0_positive
    elif class_0_positive == class_1_positive:
        class_0_positive, class_1_positive = False, True

    return {
        str(class_0_label): bool(class_0_positive),
        str(class_1_label): bool(class_1_positive),
    }


def _parse_tree_step(step):
    clean = re.sub(r"^(?:Ã¢â‚¬Â¢|â€¢|•|\?|-|\s)+", "", str(step or "")).strip()
    clean = re.sub(r"\[.*?\]", "", clean).strip()
    clean = re.sub(r",\s*value\s*=\s*[-+]?\d*\.?\d+", "", clean, flags=re.IGNORECASE).strip()
    clean = clean.strip("()")
    if not clean or "predict class" in clean.lower():
        return None

    numeric_match = re.match(r"(.+?)\s*(<=|>=|<|>)\s*([-+]?\d*\.?\d+)", clean)
    if numeric_match:
        feature, operator, threshold = numeric_match.groups()
        return {
            "feature": feature.strip(),
            "operator": operator.strip(),
            "threshold": threshold.strip(),
            "kind": "numeric",
        }

    category_match = re.match(r"(.+?)\s*(=|!=)\s*(.+)", clean)
    if category_match:
        feature, operator, threshold = category_match.groups()
        return {
            "feature": feature.strip(),
            "operator": operator.strip(),
            "threshold": threshold.strip(),
            "kind": "categorical",
            "display_feature": humanize_label(feature.strip()),
        }

    return None


def _conditions_from_class_path(path_record, one_hot_metadata=None):
    one_hot_metadata = one_hot_metadata or {}
    conditions = []
    for step in (path_record or {}).get("path") or []:
        condition = _parse_tree_step(step)
        if not condition:
            continue
        meta = one_hot_metadata.get(condition["feature"])
        threshold = _safe_float(condition.get("threshold"))
        if meta and threshold is not None and 0.0 <= threshold <= 1.0:
            condition = {
                "feature": meta.get("base_feature") or condition["feature"],
                "operator": "=" if condition.get("operator") in {">", ">="} else "!=",
                "threshold": meta.get("category") or "",
                "kind": "categorical",
                "display_feature": meta.get("base_feature_display") or humanize_label(meta.get("base_feature")),
            }
        elif _fallback_one_hot_condition(condition):
            condition = _fallback_one_hot_condition(condition)
        conditions.append(condition)
    return conditions


def _classification_path_records(path_group):
    records = list((path_group or {}).get("top_paths") or [])
    if (path_group or {}).get("path"):
        records.append(path_group)
    return records


def _classification_rule_direction(condition):
    operator = str((condition or {}).get("operator") or "").strip()
    if (condition or {}).get("kind") == "categorical" or operator in {"=", "!="}:
        return "is" if operator == "=" else "is_not"
    if operator in {"<=", "<"}:
        return "lower"
    if operator in {">", ">="}:
        return "higher"
    return ""


def _classification_rule_feature_key(condition):
    feature = (condition or {}).get("display_feature") or (condition or {}).get("feature")
    return _semantic_feature_key(feature)


def _classification_rule_text(rule):
    condition = rule.get("condition") or {}
    class_label = str(rule.get("class") or "").strip() or "this class"
    feature = condition.get("display_feature") or humanize_label(condition.get("feature"))
    operator = str(condition.get("operator") or "").strip()
    threshold = condition.get("threshold")
    high_class = bool(rule.get("semantic_high_class", True))

    if condition.get("kind") == "categorical" or operator in {"=", "!="}:
        verb = "is" if operator == "=" else "is not"
        return f"Rule #{rule.get('rank')}: {feature} {verb} {humanize_label(threshold)} is associated with {class_label}."

    phrase = _numeric_business_phrase(condition.get("feature"), operator, high_class=high_class)
    return f"Rule #{rule.get('rank')}: {phrase} is associated with {class_label}."


def _classification_rule_score(path_record, condition, depth):
    samples = int((path_record or {}).get("samples") or (path_record or {}).get("samples_sum") or 0)
    purity = _safe_float((path_record or {}).get("purity"), 0.0) or 0.0
    if purity > 1.0:
        purity = purity / 100.0
    impurity_reduction = _safe_float((path_record or {}).get("impurity_reduction"), 0.0) or 0.0
    score = (samples * 0.6) + (depth * 0.2) + (purity * 0.2)
    if impurity_reduction:
        score += impurity_reduction * 0.2
    return score


def _extract_classification_rule_candidates(path_group, class_label, one_hot_metadata=None):
    candidates = []
    for path_record in _classification_path_records(path_group):
        conditions = _conditions_from_class_path(path_record, one_hot_metadata)
        depth = len(conditions)
        if not depth:
            continue
        for condition_index, condition in enumerate(conditions):
            direction = _classification_rule_direction(condition)
            feature_key = _classification_rule_feature_key(condition)
            if not direction or not feature_key:
                continue
            candidates.append({
                "feature": condition.get("display_feature") or condition.get("feature"),
                "feature_key": feature_key,
                "direction": direction,
                "class": class_label,
                "samples": int(path_record.get("samples") or path_record.get("samples_sum") or 0),
                "support": int(path_record.get("samples") or 0),
                "purity": _safe_float(path_record.get("purity"), 0.0) or 0.0,
                "impurity": _safe_float(path_record.get("impurity"), None),
                "impurity_reduction": _safe_float(path_record.get("impurity_reduction"), 0.0) or 0.0,
                "depth": depth,
                "condition_depth": condition_index + 1,
                "condition": condition,
                "score": _classification_rule_score(path_record, condition, depth),
            })
    return candidates


def _rank_classification_rules(rule_candidates, max_rules=3):
    ranked = sorted(
        rule_candidates or [],
        key=lambda rule: (
            rule.get("score", 0.0),
            rule.get("samples", 0),
            rule.get("support", 0),
            rule.get("impurity_reduction", 0.0),
            rule.get("depth", 0),
            -rule.get("condition_depth", 0),
        ),
        reverse=True,
    )

    kept_by_conflict = {}
    for rule in ranked:
        conflict_key = (rule.get("class"), rule.get("feature_key"))
        current = kept_by_conflict.get(conflict_key)
        if current is None:
            kept_by_conflict[conflict_key] = rule
            continue
        current_direction = current.get("direction")
        direction = rule.get("direction")
        opposite = {current_direction, direction} == {"higher", "lower"} or {current_direction, direction} == {"is", "is_not"}
        if opposite and rule.get("score", 0.0) > current.get("score", 0.0):
            kept_by_conflict[conflict_key] = rule

    unique_rules = sorted(
        kept_by_conflict.values(),
        key=lambda rule: (
            rule.get("score", 0.0),
            rule.get("samples", 0),
            rule.get("support", 0),
            rule.get("impurity_reduction", 0.0),
        ),
        reverse=True,
    )

    selected = []
    seen = set()
    for rule in unique_rules:
        key = (
            rule.get("class"),
            rule.get("feature_key"),
            rule.get("direction"),
            str((rule.get("condition") or {}).get("threshold") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(rule)
        if len(selected) >= max_rules:
            break

    for index, rule in enumerate(selected, start=1):
        rule["rank"] = index
    return selected


def _remove_cross_class_direction_conflicts(rules_0, rules_1):
    best_by_feature_direction = {}
    for rule in list(rules_0 or []) + list(rules_1 or []):
        key = (rule.get("feature_key"), rule.get("direction"))
        current = best_by_feature_direction.get(key)
        if current is None or rule.get("score", 0.0) > current.get("score", 0.0):
            best_by_feature_direction[key] = rule

    keep_ids = {id(rule) for rule in best_by_feature_direction.values()}
    cleaned_0 = [rule for rule in rules_0 or [] if id(rule) in keep_ids]
    cleaned_1 = [rule for rule in rules_1 or [] if id(rule) in keep_ids]

    for rules in [cleaned_0, cleaned_1]:
        for index, rule in enumerate(rules, start=1):
            rule["rank"] = index
    return cleaned_0, cleaned_1


def _classification_root_feature(path_groups, one_hot_metadata=None):
    for path_group in path_groups:
        for path_record in _classification_path_records(path_group):
            conditions = _conditions_from_class_path(path_record, one_hot_metadata)
            if conditions:
                root_condition = conditions[0]
                return root_condition.get("display_feature") or humanize_label(root_condition.get("feature"))
    return ""


def format_classification_business_paths(
    class_0_paths,
    class_1_paths,
    class_names,
    target_variable=None,
    X=None,
    encoded_feature_mapping=None,
):
    class_0_label = str(class_names[0]) if len(class_names or []) > 0 else "Class 0"
    class_1_label = str(class_names[1]) if len(class_names or []) > 1 else "Class 1"
    if all(re.fullmatch(r"(?:class\s*)?\d+", label.strip(), flags=re.IGNORECASE) for label in [class_0_label, class_1_label]):
        class_0_label, class_1_label = _business_class_headings(target_variable)

    feature_names = []
    for path_group in [class_0_paths, class_1_paths]:
        for path_record in list((path_group or {}).get("top_paths") or []) + [path_group or {}]:
            for step in (path_record or {}).get("path") or []:
                condition = _parse_tree_step(step)
                if condition:
                    feature_names.append(condition["feature"])
    one_hot_metadata = infer_one_hot_metadata(feature_names, X=X, encoded_feature_mapping=encoded_feature_mapping)

    root_feature = _classification_root_feature([class_0_paths, class_1_paths], one_hot_metadata)
    class_polarities = _class_polarity_map(class_0_label, class_1_label)
    rules_0 = _rank_classification_rules(
        _extract_classification_rule_candidates(class_0_paths, class_0_label, one_hot_metadata),
        max_rules=3,
    )
    rules_1 = _rank_classification_rules(
        _extract_classification_rule_candidates(class_1_paths, class_1_label, one_hot_metadata),
        max_rules=3,
    )
    for rule in rules_0:
        rule["semantic_high_class"] = class_polarities.get(str(class_0_label), False)
    for rule in rules_1:
        rule["semantic_high_class"] = class_polarities.get(str(class_1_label), True)
    rules_0, rules_1 = _remove_cross_class_direction_conflicts(rules_0, rules_1)

    lines = []
    if root_feature:
        lines.append(f"Top Driver: {root_feature} is the primary factor influencing the target outcome.")
        lines.append("")
    if rules_0:
        lines.append(class_0_label)
        lines.extend(_classification_rule_text(rule) for rule in rules_0)
    if rules_1:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(class_1_label)
        lines.extend(_classification_rule_text(rule) for rule in rules_1)
    return "\n".join(_validate_business_lines(lines)).strip()


def _format_regression_path(region, index, high=True):
    lines = [
        f"Path {index}:",
        f"Predicted Value: {_format_currency_like(region.get('prediction'))}",
        "",
        "Conditions:",
    ]
    conditions = list(region.get("display_conditions") or region.get("conditions") or [])
    # Business users asked for leaf-to-root explanations, so reverse the route.
    rendered_conditions = [render_business_condition(condition) for condition in reversed(conditions)]
    for condition_index, condition_text in enumerate(rendered_conditions):
        prefix = "IF" if condition_index == 0 else "AND"
        lines.append(f"{prefix} {condition_text}")
    lines.extend([
        "",
        "Business Interpretation:",
        _business_interpretation_for_region(region, high=high),
        "",
        "--------------------------------",
    ])
    return "\n".join(lines)


def _plain_condition_text(condition):
    return _semantic_condition_text(condition, high_class=True)


def _semantic_condition_text(condition, high_class=True):
    fallback_one_hot = _fallback_one_hot_condition(condition)
    if fallback_one_hot:
        return _one_hot_business_phrase(fallback_one_hot)
    if condition.get("kind") == "categorical":
        return _one_hot_business_phrase(condition)
    return _numeric_business_phrase(
        condition.get("feature"),
        condition.get("operator"),
        high_class=high_class,
    )


def _condition_business_key(condition):
    if condition.get("kind") == "categorical":
        return f"cat:{_semantic_feature_key(condition.get('feature'))}:{_semantic_feature_key(condition.get('threshold'))}"
    return f"num:{_semantic_feature_key(condition.get('feature'))}"


def _raw_condition_signature(condition):
    return (
        _condition_business_key(condition),
        str(condition.get("operator") or "").strip(),
        round(_safe_float(condition.get("threshold"), default=0.0) or 0.0, 6),
    )


def _region_business_conditions(regions, max_conditions=8):
    selected = []
    seen = set()
    for region in regions:
        conditions = region.get("display_conditions") or region.get("conditions") or []
        for condition in conditions:
            if not _is_meaningful_condition(condition):
                continue
            key = _condition_business_key(condition)
            if not key or key in seen:
                continue
            seen.add(key)
            selected.append(condition)
            if len(selected) == max_conditions:
                return selected
    return selected


def _build_business_bullets(conditions, class_label, high_class, skip_signatures=None, max_conditions=3):
    skip_signatures = skip_signatures or set()
    bullets = []
    seen = set()
    for condition in conditions:
        if _raw_condition_signature(condition) in skip_signatures:
            continue
        text = _semantic_condition_text(condition, high_class=high_class).strip()
        if not text:
            continue
        key = re.sub(r"\s+", " ", text.lower())
        if key in seen:
            continue
        seen.add(key)
        suffix = "customers" if "purchase" in _semantic_feature_key(class_label) else "records"
        bullets.append(f"* {text} is associated with {class_label} {suffix}.")
        if len(bullets) == max_conditions:
            break
    return bullets


def _mirror_numeric_condition(condition):
    opposite = dict(condition)
    opposite["operator"] = ">" if condition.get("operator") in {"<=", "<"} else "<="
    return opposite


def _complete_paired_bullets(low_conditions, high_conditions, low_label, high_label, max_conditions=3):
    low_bullets = _build_business_bullets(low_conditions, low_label, high_class=False, max_conditions=max_conditions)
    high_bullets = _build_business_bullets(high_conditions, high_label, high_class=True, max_conditions=max_conditions)

    low_features = {_condition_business_key(condition) for condition in low_conditions}
    high_features = {_condition_business_key(condition) for condition in high_conditions}

    for condition in high_conditions:
        if len(low_bullets) >= max_conditions:
            break
        if condition.get("kind") == "categorical" or _condition_business_key(condition) in low_features:
            continue
        low_bullets.extend(_build_business_bullets([_mirror_numeric_condition(condition)], low_label, high_class=False, max_conditions=1))

    for condition in low_conditions:
        if len(high_bullets) >= max_conditions:
            break
        if condition.get("kind") == "categorical" or _condition_business_key(condition) in high_features:
            continue
        high_bullets.extend(_build_business_bullets([_mirror_numeric_condition(condition)], high_label, high_class=True, max_conditions=1))

    return low_bullets[:max_conditions], high_bullets[:max_conditions]


def _validate_business_lines(lines):
    forbidden_patterns = [
        r"\b(?:less than|greater than)\b",
        r"<=|>=|<|>",
        r"\b\d+(?:\.\d+)?\b",
        r"[a-z]+_[a-z0-9_]+",
    ]
    cleaned = []
    for line in lines:
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in forbidden_patterns):
            continue
        cleaned.append(line)
    return cleaned


def format_regression_business_paths(result, target_variable):
    high_regions = list((result or {}).get("top_high_regions") or [])
    low_regions = list((result or {}).get("top_low_regions") or [])
    if not high_regions and (result or {}).get("high_region"):
        high_regions = [result["high_region"]]
    if not low_regions and (result or {}).get("low_region"):
        low_regions = [result["low_region"]]

    low_label, high_label = _business_class_headings(target_variable)
    low_conditions = _region_business_conditions(low_regions[:1], max_conditions=8)
    high_conditions = _region_business_conditions(high_regions[:1], max_conditions=8)

    low_signatures = {_raw_condition_signature(condition) for condition in low_conditions}
    high_signatures = {_raw_condition_signature(condition) for condition in high_conditions}
    shared_signatures = low_signatures.intersection(high_signatures)
    if shared_signatures:
        low_conditions = [condition for condition in low_conditions if _raw_condition_signature(condition) not in shared_signatures]
        high_conditions = [condition for condition in high_conditions if _raw_condition_signature(condition) not in shared_signatures]

    low_bullets, high_bullets = _complete_paired_bullets(
        low_conditions,
        high_conditions,
        low_label,
        high_label,
        max_conditions=3,
    )

    lines = []
    if low_bullets:
        lines.append(low_label)
        lines.extend(low_bullets)
    if high_bullets:
        if lines:
            lines.append("")
        lines.append(high_label)
        lines.extend(high_bullets)
    return "\n".join(_validate_business_lines(lines)).strip()

