"""
Locked Build contracts for the ecommerce SXI problems (1–6).

Each problem is selected and run individually ($500). PROBLEM_GROUPS remains for
reference / packaging docs only — the chat Build picker no longer locks groups.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


PROBLEMS: dict[int, dict[str, Any]] = {
    1: {
        "id": 1,
        "slug": "margin_shielding",
        "name": "Full-Price Buyers — Don’t Waste Discounts",
        "simple_meaning": (
            "Among buyers, find who pays full price — do not give them a coupon. "
            "Discounts only for buyers who need them."
        ),
        "customer_lens": "Find who already pays full price — do not give them a discount.",
        "target": "is_full_price_buyer",
        "task_type": "classification",
        "good": 1,
        "bad": 0,
        "improvement": "increasing",
        "features": [
            "first_item_category_enc",
            "total_sessions",
            "engagement_time",
            "scroll_depth",
            "cart_abandonment",
            "search_clicks",
            "churn_risk",
            "add_to_carts",
            "acq_channel_enc",
            "age_group_enc",
            "gender_enc",
            "device_enc",
        ],
        "forbidden_features": [
            "purchases",
            "purchaseRevenue",
            "totalRevenue",
            "itemsPurchased",
            "previous_purchases",
            "is_buyer",
            "promo_clicks",
            "sessionCampaign",
            "campaign",
            "coupon",
            "coupon_used",
            "couponCode",
            "discount",
            "discount_amount",
            "promo_discount_amount",
            "bought_with_discount",
        ],
        "optional_extras": [
            "engagementRate",
            "checkouts",
            "itemsViewed",
            "viewsPerUser",
        ],
        "metrics_priority": ["auc", "recall", "precision"],
        "dashboard_widgets": [
            "intent_kpis",
            "offer_segments",
            "threshold_slider",
            "top_features",
        ],
    },
    2: {
        "id": 2,
        "slug": "product_evaluation",
        "name": "Product Evaluation Beyond First Purchase",
        "simple_meaning": (
            "Rank products by long-term customer value, not just the first sale."
        ),
        "customer_lens": "Stop ranking only on day-1 sales.",
        "target": "is_repeat_buyer",
        "task_type": "classification",
        "good": 1,
        "bad": 0,
        "improvement": "increasing",
        "features": [
            "first_item_category_enc",
            "total_sessions",
            "views_per_user",
            "engagement_time",
            "session_engaged_rate",
            "bounce_rate",
            "scroll_depth",
            "cart_abandonment",
            "add_to_carts",
            "checkouts_per_user",
            "items_viewed",
            "items_added_to_cart",
            "promo_clicks",
            "search_clicks",
            "acq_channel_enc",
            "device_enc",
            "item_category2_enc",
            "order_month",
            "order_quarter",
            "order_dayofweek",
            "is_weekend",
        ],
        "forbidden_features": [
            "returningUsers",
            "returningUserRate",
            "previous_purchases",
            "purchase_frequency",
            "purchaseRevenue",
            "totalRevenue",
            "purchases",
            "transactions",
            "next_purchase_date",
            "days_until_next_purchase",
            "n_orders",
        ],
        "optional_extras": [
            "itemBrand",
            "itemCategory",
            "event_timestamp",
            "date",
        ],
        "metrics_priority": ["auc", "recall", "precision"],
        "dashboard_widgets": [
            "ltv_kpis",
            "feature_importance",
            "four_tier_strategy",
            "ltv_leaderboard",
        ],
        "requires_longitudinal": True,
        "min_repeat_rate": 0.05,
    },
    3: {
        "id": 3,
        "slug": "ltv_gateway",
        "name": "LTV Gateway SKU Discovery",
        "simple_meaning": (
            "Find first-purchase products that turn new buyers into high-value "
            "long-term customers."
        ),
        "customer_lens": "Which first SKU creates high-LTV customers.",
        "target": "is_high_ltv",
        "task_type": "classification",
        "good": 1,
        "bad": 0,
        "improvement": "increasing",
        "features": [
            "first_item_category_enc",
            "first_product_type_enc",
            "total_sessions",
            "engagement_time",
            "scroll_depth",
            "cart_abandonment",
            "promo_clicks",
            "search_clicks",
            "churn_risk",
            "acq_channel_enc",
            "age_group_enc",
            "gender_enc",
            "device_enc",
            "add_to_carts",
        ],
        "forbidden_features": [],
        "optional_extras": [
            "itemCategory",
            "itemBrand",
            "quantity",
        ],
        "metrics_priority": ["auc", "recall", "precision"],
        "dashboard_widgets": [
            "gateway_kpis",
            "catalogue_leaderboard",
            "rank_divergence",
            "hero_tiers",
        ],
    },
    4: {
        "id": 4,
        "slug": "repurchase",
        "name": "Product-Specific Repurchase Prediction",
        "simple_meaning": (
            "Predict the right day to remind someone to buy again for each product "
            "type (not a fixed 30 days)."
        ),
        "customer_lens": "30-day emails are wrong timing.",
        "target": "days_until_next_purchase",
        "task_type": "regression",
        "good": None,
        "bad": None,
        "improvement": "decreasing",
        "features": [
            "sessionsPerUser",
            "viewsPerUser",
            "engagement_time_msec",
            "session_engaged",
            "purchaseRevenue",
            "cart_abandonment_rate",
            "previous_purchases",
            "tenure_days",
            "purchase_frequency",
            "avg_order_value",
            "unique_categories",
            "cat_mean_target",
            "cat_std_target",
        ],
        "stage1_features": [
            "sessionsPerUser",
            "viewsPerUser",
            "engagement_time_msec",
            "session_engaged",
            "purchaseRevenue",
            "cart_abandonment_rate",
        ],
        "stage2_features": [
            "previous_purchases",
            "tenure_days",
            "purchase_frequency",
            "avg_order_value",
            "unique_categories",
            "cat_mean_target",
            "cat_std_target",
        ],
        "forbidden_features": [
            "next_purchase_date",
            "is_repurchase_ready",
            "time_to_event",
            "event_observed",
        ],
        "optional_extras": [
            "itemCategory",
        ],
        "allow_partial_features": True,
        "min_features": 6,
        "metrics_priority": ["r2", "mae", "rmse"],
        "dashboard_widgets": [
            "km_curves",
            "t50_actions",
            "category_windows",
        ],
        "requires_longitudinal": True,
        "min_repeat_rate": 0.05,
    },
    5: {
        "id": 5,
        "slug": "churn",  # kept for dashboard/path compatibility; story is site activity
        "name": "Keep Users Active on Site",
        "simple_meaning": (
            "Some users have not returned recently — traffic goes quiet. "
            "This finds who is fading by last-visit recency so you can lift engagement "
            "and keep them browsing longer before interest drops away."
        ),
        "customer_lens": (
            "Find quiet traffic early (days since last visit) — re-engage with longer "
            "sessions, not blanket discounts."
        ),
        "target": "is_low_activity",
        "task_type": "classification",
        "good": 0,
        "bad": 1,
        "improvement": "decreasing",
        "features": [
            # Engagement / session quality predict quiet traffic; they are NOT in the label
            "session_volume_score",
            "frequency_score",
            "monetary_score",
            "engagement_depth",
            "behavioral_degradation",
            "promo_sensitivity",
            "cart_abandon_severity",
            "conversion_efficiency",
            "bounce_rate",
            "session_engaged_rate",
            "engagement_rate",
            "avg_scroll_depth",
            "total_purchases",
            "previous_purchases",
            "is_buyer",
            "returning_user",
            "cart_abandonment_total",
            "promo_clicks",
            "search_clicks",
            "total_sessions",
            "events_per_session",
            "event_count",
            "engagement_time_msec",
            "avg_session_duration",
            "total_revenue",
        ],
        # Label is recency-only; never train on recency or target aliases
        "forbidden_features": [
            "recency_days",
            "last_activity_date",
            "activity_label_threshold_days",
            "ground_truth_churn",
            "days_inactive",
            "last_active_date",
        ],
        "optional_extras": [
            "DAU_MAU",
            "returningUserRate",
            "purchase_frequency",
            "promo_exposure",
        ],
        "metrics_priority": ["auc", "f1", "recall", "precision"],
        "dashboard_widgets": [
            "suppress_vs_winback",
            "lifecycle_bars",
            "channel_lift",
            "intervention_timing",
        ],
    },
    6: {
        "id": 6,
        "slug": "supply_chain",
        "name": "Supply-Chain Aware Product Ranking",
        "simple_meaning": (
            "Predict next-7-day SKU demand from a GA4 itemId×day panel with rolling "
            "history features, then rank products. Optional ERP inventory unlocks "
            "OOS shielding and profit-aware supply-chain ranking."
        ),
        "customer_lens": "Which SKUs will sell next — and which get views but not buys.",
        "target": "itemsPurchased_future",
        "task_type": "regression",
        "good": None,
        "bad": None,
        "improvement": "increasing",
        "features": [
            "dxi_demand_score",
            "itemsViewed",
            "itemsAddedToCart",
            "cartToViewRate",
            "purchaseToViewRate",
            "unit_price",
            "item_category_enc",
            "item_id_enc",
            "purchases_roll_3",
            "purchases_roll_7",
            "purchases_roll_14",
            "purchases_roll_28",
            "views_roll_7",
            "views_roll_14",
            "views_roll_28",
            "atc_roll_14",
            "sku_demand_baseline",
            "purchase_momentum",
            "purchases_lag_1",
            "day_of_week",
        ],
        "forbidden_features": [],
        "optional_extras": [
            "itemCategory",
            "stock_units",
            "margin",
            "shippingAmount",
            "atc_roll_7",
            "atc_roll_28",
            "purchases_lag_7",
        ],
        "allow_partial_features": True,
        "min_features": 8,
        "metrics_priority": ["r2", "mae", "rmse"],
        "dashboard_widgets": [
            "sku_leaderboard",
            "demand_traps",
            "erp_banner",
            "driver_weights",
        ],
    },
    7: {
        "id": 7,
        "slug": "churn_prediction",
        "name": "Churn",
        "simple_meaning": (
            "Label customers as churned when their latest activity is more than "
            "60 days before the dataset's latest date, then predict who to save."
        ),
        "customer_lens": "Intervene on inactive customers before they are gone for good.",
        "target": "is_churned",
        "task_type": "classification",
        "good": 0,
        "bad": 1,
        "improvement": "decreasing",
        "allow_partial_features": True,
        "min_features": 6,
        "features": [
            "total_sessions",
            "engagement_time",
            "session_engaged_rate",
            "bounce_rate",
            "scroll_depth",
            "cart_abandonment",
            "promo_clicks",
            "search_clicks",
            "add_to_carts",
            "acq_channel_enc",
            "device_enc",
            "total_purchases",
            "previous_purchases",
            "recency_days",
            "purchase_frequency",
            "avg_order_value",
            "events_per_session",
            "engagement_rate",
        ],
        "forbidden_features": [
            "ground_truth_churn",
            "customer_churn",
        ],
        "optional_extras": [
            "DAU_MAU",
            "returningUserRate",
            "tenure_days",
            "promo_exposure",
        ],
        "metrics_priority": ["auc", "f1", "recall", "precision"],
        "dashboard_widgets": [
            "churn_kpis",
            "risk_segments",
            "feature_importance",
            "intervention_timing",
        ],
    },
    8: {
        "id": 8,
        "slug": "promotion_sensitivity",
        "name": "Who Actually Needs a Promo — Discount Only When It Works",
        "simple_meaning": (
            "Who actually responds to promotions? Not every customer needs a discount. "
            "This model scores each user's promo response rate (promo-used ÷ promo-eligible "
            "activity) and targets discounts only where they are likely to change behavior."
        ),
        "customer_lens": "Spend promo budget only on people it actually moves (≥70% response rate).",
        "target": "promotion_sensitivity",
        "task_type": "regression",
        "good": None,
        "bad": None,
        "improvement": "increasing",
        "features": [
            # Behavioural / acquisition predictors only — never promotion_sensitivity
            # or its promo÷total components (those leak → inflated fit).
            "engagement_rate",
            "bounce_rate",
            "engagement_time",
            "scroll_depth",
            "total_sessions",
            "views_per_session",
            "events_per_session",
            "device_enc",
            "acq_channel_enc",
            "age_enc",
            "gender_enc",
            "city_enc",
        ],
        "forbidden_features": [],
        # Leakage note: promo÷total rates / commerce totals that define the label must
        # not enter locked features (see features list + NON_FEATURE_COLUMNS / pass-through).
        "optional_extras": [
            "promo_discount_amount",
            "coupon_used",
            "tenure_days",
            "search_clicks",
        ],
        "allow_partial_features": True,
        "min_features": 6,
        "metrics_priority": ["r2", "mae", "within_10pct"],
        "dashboard_widgets": [
            "sensitivity_kpis",
            "promo_segments",
            "feature_importance",
            "targeting_decision",
        ],
    },
}

# P8 business targeting: users with promotion_sensitivity ≥ this get promo budget.
P8_PROMO_TARGET_THRESHOLD = 0.70


def list_problems() -> list[dict[str, Any]]:
    return [deepcopy(PROBLEMS[i]) for i in sorted(PROBLEMS)]


# Three modular $1000 packages — 2 related targets × $500 each (SXI Platform v1.2)
PROBLEM_GROUPS: dict[int, dict[str, Any]] = {
    1: {
        "id": 1,
        "slug": "immediate_margin_conversion",
        "name": "Immediate Margin & Conversion",
        "price_label": "$1000",
        "price_per_problem": "$500",
        "simple_meaning": (
            "Protect margin on first purchase and keep site visitors engaged — "
            "discount only when needed; re-engage quiet traffic with longer sessions."
        ),
        "problem_ids": [1, 5],
        "problem_roles": {
            1: "Full-price buyer / don’t waste discounts (is_full_price_buyer)",
            5: "Keep users active on site (is_low_activity)",
        },
    },
    2: {
        "id": 2,
        "slug": "post_purchase_reengagement",
        "name": "Post-Purchase & Re-Engagement",
        "price_label": "$1000",
        "price_per_problem": "$500",
        "simple_meaning": (
            "Grow repeat value after the first sale and time replenishment emails "
            "by product — not a fixed 30-day blast."
        ),
        "problem_ids": [2, 4],
        "problem_roles": {
            2: "Beyond first purchase (is_repeat_buyer)",
            4: "Product-specific repurchase timing (days_until_next_purchase)",
        },
    },
    3: {
        "id": 3,
        "slug": "ltv_supply_chain",
        "name": "Long-Term Value & Supply-Chain",
        "price_label": "$1000",
        "price_per_problem": "$500",
        "simple_meaning": (
            "Find gateway SKUs that create high-LTV customers and rank products "
            "by demand × margin with stockout shielding."
        ),
        "problem_ids": [3, 6],
        "problem_roles": {
            3: "LTV gateway SKU discovery (is_high_ltv)",
            6: "Supply-chain aware ranking (itemsPurchased_future)",
        },
    },
}


def list_groups() -> list[dict[str, Any]]:
    return [deepcopy(PROBLEM_GROUPS[i]) for i in sorted(PROBLEM_GROUPS)]


def get_problem(problem_id: int | str) -> dict[str, Any] | None:
    try:
        pid = int(problem_id)
    except (TypeError, ValueError):
        return None
    problem = PROBLEMS.get(pid)
    return deepcopy(problem) if problem else None


def get_group(group_id: int | str) -> dict[str, Any] | None:
    try:
        gid = int(group_id)
    except (TypeError, ValueError):
        return None
    group = PROBLEM_GROUPS.get(gid)
    if not group:
        return None
    out = deepcopy(group)
    out["problems"] = [p for p in (get_problem(pid) for pid in out["problem_ids"]) if p]
    return out


def group_picker_html() -> str:
    """Legacy group picker (kept for docs / older call sites). Prefer problem_picker_html."""
    lines = [
        "<b>Pick ONE add-on module ($1000)</b> — each runs <b>2 related problems</b> "
        "($500 each) and opens <b>2 independent Executive Dashboards</b>.<br>"
        "Full data · no balancing · Top Features + Target DT / correlation / 3-level improvement.<br><br>"
    ]
    for g in list_groups():
        lines.append(
            f"<b>Group {g['id']}. {g['name']}</b> "
            f"<span style='opacity:0.75'>({g.get('price_label') or '$1000'} · $500 × 2 problems)</span><br>"
            f"<span style='opacity:0.85'>{g['simple_meaning']}</span><br>"
        )
        for pid in g["problem_ids"]:
            p = get_problem(pid)
            role = (g.get("problem_roles") or {}).get(pid) or ""
            if p:
                lines.append(
                    f"&nbsp;&nbsp;• <b>P{pid}</b> {p['name']} "
                    f"<span style='opacity:0.8'>— target <code>{p['target']}</code></span>"
                )
                if role:
                    lines.append(f"<br>&nbsp;&nbsp;&nbsp;&nbsp;<span style='opacity:0.7'>{role}</span>")
                lines.append("<br>")
        lines.append("<br>")
    lines.append("➡ Type <b>1</b>, <b>2</b>, or <b>3</b> to lock the group and run both problems.")
    return "".join(lines)


def problem_picker_html() -> str:
    """Chat picker fallback — Unified mode redirects users to Marketplace."""
    lines = [
        "<b>Select ONE solution</b> (1 credit each) — locks that solution only, "
        "runs DXI once, and opens its <b>Executive Dashboard</b> when available.<br>"
        "Prefer the <a href='/portal/marketplace/'>Solution Marketplace</a> "
        "for all 10 solutions and credit packs.<br><br>"
    ]
    for p in list_problems():
        lens = p.get("customer_lens") or ""
        lines.append(
            f"<b>{p['id']}. {p['name']}</b> "
            f"<span style='opacity:0.75'>(1 credit)</span><br>"
            f"<span style='opacity:0.85'>{p.get('simple_meaning') or ''}</span>"
        )
        if lens:
            lines.append(f"<br><span style='opacity:0.7'>{lens}</span>")
        lines.append(
            f"<br><span style='opacity:0.8'>target <code>{p['target']}</code> · "
            f"{p.get('task_type') or 'classification'}</span><br><br>"
        )
    lines.append("➡ Type <b>1</b>–<b>8</b> to lock that solution and continue.")
    return "".join(lines)


def target_info_from_problem(problem: dict[str, Any]) -> dict[str, Any]:
    """Agent-2 / SXI target_info payload from a locked contract."""
    target = problem["target"]
    improvement = str(problem.get("improvement") or "increasing").strip().lower()
    good = problem.get("good")
    bad = problem.get("bad")
    task_type = problem.get("task_type") or "classification"
    is_regression = task_type == "regression"

    # SXI expects signed % improvement (e.g. +20 / -20), not only a direction word
    if improvement == "decreasing":
        target_change = "decreasing"
        target_improvement = -20
    else:
        target_change = "increasing"
        target_improvement = 20

    if is_regression:
        if improvement == "decreasing":
            # Lower numeric values are better (e.g. fewer days to repurchase)
            selected, other = "Below_mean", "Above_mean"
            selected_meaning = "shorter / better-timed"
            other_meaning = "longer / worse-timed"
        else:
            selected, other = "Above_mean", "Below_mean"
            selected_meaning = "higher / better"
            other_meaning = "lower / worse"
        return {
            "Target Outcome": target,
            "Target Outcome Type": "Numeric",
            "Task Type": "regression",
            "Outcome Labels": [selected, other],
            "Selected Outcome": selected,
            "Selected Outcome Meaning": "Good",
            "Good Outcome": selected,
            "Bad Outcome": other,
            "Good Outcome Label": selected_meaning,
            "Bad Outcome Label": other_meaning,
            "Good Outcome Value": selected,
            "Bad Outcome Value": other,
            "Target Outcome Improvement": target_improvement,
            "Target Outcome change": target_change,
            "Improvement Direction": improvement,
            "Use Case": problem["name"],
            "Problem Id": problem["id"],
            "Problem Slug": problem["slug"],
            "Simple Meaning": problem["simple_meaning"],
            "Locked Features": list(problem["features"]),
            "Forbidden Features": list(problem.get("forbidden_features") or []),
            "Build Mode": "problem_contract",
            "No Balancing": True,
            "No Downsample": True,
            "Show Decision Tree": True,
            "Show Top Features": True,
            "Show Target Features": True,
            "Show Correlation": True,
            "Show Improvement Levels": True,
            "Target Improvement Pct": target_improvement,
            "Requires Longitudinal": bool(problem.get("requires_longitudinal")),
            "Min Repeat Rate": problem.get("min_repeat_rate"),
        }

    good_val = 1 if good is None else good
    bad_val = 0 if bad is None else bad
    # Churn-style: good=0 retained, bad=1 churn — still categorical ints
    try:
        good_val = int(good_val)
    except (TypeError, ValueError):
        good_val = good_val
    try:
        bad_val = int(bad_val)
    except (TypeError, ValueError):
        bad_val = bad_val

    # Correlation / SXI "Current" rate must match the metric we report on the dashboard:
    # - increasing problems → track the good class rate (e.g. buyers)
    # - decreasing problems → track the bad class rate (e.g. churn)
    if improvement == "decreasing":
        selected_val = bad_val
        selected_meaning = "Bad"
    else:
        selected_val = good_val
        selected_meaning = "Yes" if good_val == 1 else "Good"

    return {
        "Target Outcome": target,
        "Target Outcome Type": "Categorical",
        "Task Type": "classification",
        "Outcome Labels": [str(good_val), str(bad_val)],
        "Selected Outcome": str(selected_val),
        "Selected Outcome Meaning": selected_meaning,
        "Good Outcome Label": str(good_val),
        "Bad Outcome Label": str(bad_val),
        "Good Outcome Value": good_val,
        "Bad Outcome Value": bad_val,
        "Good Outcome": str(good_val),
        "Bad Outcome": str(bad_val),
        "Target Outcome Improvement": target_improvement,
        "Target Outcome change": target_change,
        "Improvement Direction": improvement,
        "Use Case": problem["name"],
        "Problem Id": problem["id"],
        "Problem Slug": problem["slug"],
        "Simple Meaning": problem["simple_meaning"],
        "Locked Features": list(problem["features"]),
        "Forbidden Features": list(problem.get("forbidden_features") or []),
        "Build Mode": "problem_contract",
        "No Balancing": True,
        "No Downsample": True,
        "Show Decision Tree": True,
        "Show Top Features": True,
        "Show Target Features": True,
        "Show Correlation": True,
        "Show Improvement Levels": True,
        "Target Improvement Pct": target_improvement,
        "Requires Longitudinal": bool(problem.get("requires_longitudinal")),
        "Min Repeat Rate": problem.get("min_repeat_rate"),
    }


# Target definitions for dashboards + TARGET_REPORT.md (raw vs derived + formula)
TARGET_DEFINITIONS: dict[int, dict[str, Any]] = {
    1: {
        "target": "is_full_price_buyer",
        "task_type": "classification",
        "improvement_pct": 20,
        "improvement_label": "+20% full-price buyers",
        "source": "RAW if present, else DERIVED among buyers",
        "formula": (
            "Keep buyers only (is_buyer = 1); drop non-buyers. "
            "If is_full_price_buyer exists → coerce to 0/1 on that buyer set. "
            "Else: is_full_price_buyer = 1 when the buyer purchased without a "
            "discount/promo; = 0 when the buyer purchased with a discount. "
            "Discount used when coupon/coupon_used is present, or discount "
            "amount > 0, else GA4 proxy = discount-like sessionCampaign "
            "(sale/offer/clearance/coupon/…). "
            "Coupon / campaign / purchase-total columns are dropped after the "
            "label is built. Train/test/predict this buyer-only label."
        ),
        "notes": (
            "This is not Conversion (is_buyer on all traffic). "
            "Conversion = will they buy. Problem 1 = among buyers, who pays full price."
        ),
    },
    2: {
        "target": "is_repeat_buyer",
        "task_type": "classification",
        "improvement_pct": 20,
        "improvement_label": "+20% repeat buyers",
        "source": "LONGITUDINAL if order history has repeats, else RAW / PROXY",
        "formula": (
            "If orders can be sequenced (same customer, 2+ distinct purchase dates) → "
            "is_repeat_buyer = 1 when the customer has a later order after first purchase. "
            "Else if is_repeat_buyer exists in the upload → 0/1. Else proxy: "
            "(returningUsers > 0 AND is_buyer = 1) OR (previous_purchases > 0 AND is_buyer = 1). "
            "Last-resort fallback: is_repeat_buyer = is_buyer (disclosed as proxy)."
        ),
        "notes": (
            "Proxy is not true repeat behavior. Dashboard must show target_source. "
            "Forbidden: previous_purchases, purchase_frequency, next_purchase_date, ltv_proxy, "
            "repeat_purchase_* (post-outcome leakage at first-purchase snapshot)."
        ),
    },
    3: {
        "target": "is_high_ltv",
        "task_type": "classification",
        "improvement_pct": 20,
        "improvement_label": "+20% high-LTV buyers",
        "source": "RAW if present, else DERIVED / PROXY",
        "formula": (
            "Among buyers (is_buyer = 1): is_high_ltv = 1 if revenue ≥ 75th percentile "
            "of revenue among buyers (revenue from totalRevenue / purchaseRevenue / ltv). "
            "Fallback: is_high_ltv = is_buyer when no revenue column."
        ),
        "notes": "first_product_type_enc may be encoded from itemCategory2 / itemName / brand.",
    },
    4: {
        "target": "days_until_next_purchase",
        "task_type": "regression",
        "improvement_pct": -20,
        "improvement_label": "−20% days to next purchase (shorter / better-timed)",
        "source": "LONGITUDINAL if uncensored next-order dates exist, else RAW / PROXY",
        "formula": (
            "If order sequences exist → days_until_next_purchase = (next_purchase_date − snapshot_date).days "
            "for uncensored customers. Else if the column (or alias) exists → use it. Else proxy: "
            "intensity = 3·purchase_frequency + 1.5·previous_purchases + 0.8·engagement + 1/sessions; "
            "days = 150 / (1 + intensity), clipped (~7–120d active, ~120–180d inactive)."
        ),
        "notes": (
            "Proxy is not a true inter-purchase interval — report must disclose target_source=proxy. "
            "cat_mean_target / cat_std_target use leave-one-out. "
            "Forbidden: next_purchase_date, is_repurchase_ready, time_to_event, event_observed."
        ),
    },
    5: {
        "target": "is_low_activity",
        "task_type": "classification",
        "improvement_pct": -20,
        "improvement_label": "−20% low-activity users (more engaged traffic)",
        "source": "RAW if present, else DERIVED from recency_days only (leakage-safe)",
        "formula": (
            "If is_low_activity exists → 0/1. "
            "Else quiet-traffic rule: is_low_activity = 1 when recency_days > median. "
            "Requires usable date / event_timestamp (no engagement/bounce in the label). "
            "Engagement, bounce, and session-quality columns are training features only. "
            "Forbidden from training: recency_days, last_activity_date, "
            "activity_label_threshold_days, ground_truth_churn. "
            "ground_truth_churn is kept as a pass-through alias for older dashboards."
        ),
        "notes": (
            "Good = 0 (recently active); bad = 1 (quiet traffic / high recency). "
            "Goal: decrease quiet users by lifting engagement and session length. "
            "Distinct from P7 60-day inactivity (is_churned)."
        ),
    },
    6: {
        "target": "itemsPurchased_future",
        "task_type": "regression",
        "improvement_pct": 20,
        "improvement_label": "+20% future items purchased",
        "source": "RAW / MAPPED (required)",
        "formula": (
            "Use itemsPurchased_future or mapped itemsPurchased as continuous target. "
            "FE also builds cartToViewRate, purchaseToViewRate, dxi_demand_score as features — not the target."
        ),
        "notes": "Build gaps if future/units label is missing.",
    },
    8: {
        "target": "promotion_sensitivity",
        "task_type": "regression",
        "improvement_pct": 20,
        "improvement_label": "+20% promo-sensitive reach (≥70% response — smarter promo spend)",
        "source": "DERIVED from sessionCampaign × purchases/ATC",
        "formula": (
            "Keep all upload rows for SXI. "
            "promotion_sensitivity = (campaign purchases + campaign ATCs) / "
            "(total purchases + total ATCs) at user level (0–1, display as %). "
            "Interpretation: promo-used ÷ total eligible commerce acts "
            "(e.g. 3 of 6 → 50%). "
            "Any valid sessionCampaign counts — no fixed promo list; only empty / "
            "(not set) / null-like values are excluded from the campaign numerator. "
            "Campaign columns and intermediate calc columns are removed after the score "
            "is created. "
            "SXI regresses promotion_sensitivity directly. "
            "Business flag is_promo_sensitive = 1 when promotion_sensitivity ≥ 0.70 "
            "(target discounts only above 70%). "
            "Training features are behavioural/demographic only."
        ),
        "notes": (
            "Shared logic: pipeline.promotion_sensitivity.create_promotion_sensitivity_feature. "
            "Do not train on promotion_sensitivity pass-through columns or promo÷total rates."
        ),
    },
}


def get_target_definition(problem_id: int | None) -> dict[str, Any]:
    """Return target raw/derived definition for dashboards and reports."""
    pid = int(problem_id or 0)
    base = deepcopy(TARGET_DEFINITIONS.get(pid) or {})
    problem = get_problem(pid)
    if problem:
        base.setdefault("target", problem.get("target"))
        base.setdefault("task_type", problem.get("task_type"))
        base["problem_id"] = pid
        base["problem_name"] = problem.get("name")
        base["simple_meaning"] = problem.get("simple_meaning")
    return base
