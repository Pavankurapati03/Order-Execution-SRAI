"""
Problem-specific feature engineering for the locked Build contracts (1–8).

Applies FE, builds/aligns the target, drops forbidden columns, and returns
either a ready frame + kept feature list or a clear FeatureGapError.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pipeline.problem_contracts import get_problem


@dataclass
class FeatureGapError(Exception):
    problem_id: int
    missing: list[str]
    message: str = ""

    def __post_init__(self):
        if not self.message:
            names = ", ".join(self.missing) or "(unknown)"
            self.message = (
                f"Problem {self.problem_id}: required columns could not be built "
                f"or found: {names}. Upload a dataset with the needed fields "
                f"(do not invent labels or force class balance)."
            )
        super().__init__(self.message)


@dataclass
class ProblemFEResult:
    df: pd.DataFrame
    problem: dict[str, Any]
    features_used: list[str] = field(default_factory=list)
    target: str = ""
    positive_rate: float | None = None
    notes: list[str] = field(default_factory=list)
    gaps_optional: list[str] = field(default_factory=list)
    target_source: str = "unknown"
    data_quality: dict[str, Any] = field(default_factory=dict)


def _norm_map(columns: list[str]) -> dict[str, str]:
    return {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in columns}


def _find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    mapping = _norm_map(list(df.columns))
    for cand in candidates:
        key = cand.strip().lower().replace(" ", "").replace("_", "")
        if key in mapping:
            return mapping[key]
        # soft contains
        for k, orig in mapping.items():
            if key and key in k:
                return orig
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _encode_series(series: pd.Series) -> pd.Series:
    as_str = series.astype(str).fillna("missing").str.strip().str.lower()
    codes, _ = pd.factorize(as_str, sort=True)
    return pd.Series(codes, index=series.index).astype(float)


def _ensure_binary_target(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        # continuous sensitivity scores → top-half as positive when not already 0/1
        vals = pd.to_numeric(s, errors="coerce")
        uniq = set(vals.dropna().unique().tolist())
        if uniq.issubset({0, 1, 0.0, 1.0}):
            return (vals.fillna(0) > 0).astype(int)
        med = vals.median(skipna=True)
        return (vals.fillna(med) >= med).astype(int)
    sl = s.astype(str).str.lower().str.strip()
    truthy = {
        "1",
        "true",
        "yes",
        "y",
        "buyer",
        "churn",
        "churned",
        "sensitive",
        "high",
        "promo_sensitive",
        "promotion_sensitive",
        "full_price",
        "fullprice",
        "full_price_buyer",
        "isfullpricebuyer",
    }
    return sl.isin(truthy).astype(int)


def _positive_rate(y: pd.Series) -> float:
    if y.empty:
        return 0.0
    return float((y.astype(float) > 0).mean())


def _drop_forbidden(df: pd.DataFrame, forbidden: list[str]) -> pd.DataFrame:
    drop_cols = []
    mapping = _norm_map(list(df.columns))
    for f in forbidden:
        key = f.strip().lower().replace(" ", "").replace("_", "")
        if key in mapping:
            drop_cols.append(mapping[key])
    if drop_cols:
        return df.drop(columns=list(set(drop_cols)), errors="ignore")
    return df


# Pass-through CRM / GA4 identity for retargeting CSVs — never a model feature.
IDENTITY_PASS_THROUGH = (
    "user_pseudo_id",
    "user_id",
    "fullVisitorId",
    "fullvisitorid",
    "visitor_id",
    "clientId",
    "client_id",
)
NON_FEATURE_COLUMNS = frozenset(
    {
        "user_pseudo_id",
        "user_id",
        "fullvisitorid",
        "visitor_id",
        "clientid",
        "client_id",
        "master_id",
        "index",
        "i_n_d_e_x",
        # P5 retention / activity pass-through (dashboard / CSV timing — not locked model features)
        "recency_days",
        "purchase_frequency",
        "promo_exposure",
        "last_activity_date",
        "ground_truth_churn",  # alias of is_low_activity for older dashboards
        "activity_label_threshold_days",
        "target_source",
        "_target_source",
        "snapshot_date",
        "customer_id",
        "_customer_id",
        "event_observed",
        "time_to_event",
        "previous_purchase_date",
        "next_purchase_date",
        "ltv_proxy",
        "repeat_purchase_30d",
        "repeat_purchase_60d",
        "repeat_purchase_90d",
        "n_orders",
        "item_category",
        "itemcategory",
        "is_repurchase_ready",
        "_p4_proxy_days",
        "_event_observed",
        "first_event_date",
        "next_event_date",
        "n_event_days",
        # P6 SKU demand ranking pass-through (dashboard — not locked model features)
        "item_id",
        "item_name",
        "history_purchases",
        "future_revenue",
        "history_revenue",
        "split_date",
        "future_target_complete",
        "panel_date",
        "future_horizon_days",
        "future_units_raw",
        "erp_inventory_available",
        "stock_units",
        "margin",
        "unit_cost",
        "reorder_point",
        "lead_time_days",
        "in_transit_units",
        "currency",
        # P7 inactivity-churn pass-through (dashboard — not locked model features)
        "customer_churn",
        "days_inactive",
        "last_active_date",
        "inactivity_threshold_days",
        # P8 promo audit / label-definition pass-through (never locked model features)
        "promotion_sensitivity",
        "promo_sensitivity",
        "promo_purchases",
        "promo_atcs",
        "non_promo_purchases",
        "non_promo_atcs",
        "promo_sessions",
        "non_promo_sessions",
        "total_revenue",
        "promo_label_threshold",
        "is_promo_sensitive",
        "row_count",
        "promo_conversion_rate",
        "non_promo_conversion_rate",
        "promo_exposure",
        "cart_abandonment",
        "total_purchases",
        "add_to_carts",
        "avg_order_value",
        "previous_purchases",
        "promo_clicks",
        # P1 audit (buyer-only full-price label — never a locked feature)
        "bought_with_discount",
        "p1_discount_source",
    }
)


def _find_identity_source_col(df: pd.DataFrame) -> str | None:
    """Locate a retargeting ID column; prefer user_pseudo_id (exact names only)."""
    mapping = _norm_map(list(df.columns))
    for cand in IDENTITY_PASS_THROUGH:
        key = cand.strip().lower().replace(" ", "").replace("_", "")
        if key in mapping:
            return mapping[key]
    return None


def _attach_user_pseudo_id(final: pd.DataFrame, source: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """
    Keep user_pseudo_id on the FE frame for CRM export / primary key.

    Not added to features_used — only carried for retargeting downloads.
    """
    src = _find_identity_source_col(source)
    if not src:
        return final
    out = final.copy()
    # Preserve row alignment with source index
    series = source.loc[out.index, src] if src in source.columns else source[src]
    # Normalize floaty GA IDs (e.g. 1.234e18) to stable strings
    vals = []
    for val in series.tolist():
        if val is None or (isinstance(val, float) and pd.isna(val)):
            vals.append("")
            continue
        if isinstance(val, float) and val.is_integer():
            vals.append(str(int(val)))
            continue
        text = str(val).strip()
        if text.endswith(".0") and text[:-2].replace("-", "").isdigit():
            text = text[:-2]
        vals.append(text)
    out.insert(0, "user_pseudo_id", vals)
    notes.append(f"Preserved identity column as user_pseudo_id (from {src}) for retargeting CSV")
    return out


def _parse_ga4_dates(series: pd.Series) -> pd.Series:
    """Parse YYYYMMDD ints; invalid / garbage → NaT."""
    nums = pd.to_numeric(series, errors="coerce")
    ok = (nums >= 20000101) & (nums <= 21001231)
    as_str = nums.where(ok).astype("Int64").astype(str)
    return pd.to_datetime(as_str, format="%Y%m%d", errors="coerce")


def _parse_mixed_timestamp_series(series: pd.Series) -> pd.Series:
    """Parse GA4 date ints plus epoch seconds/ms/us into a naive datetime series."""
    nums = pd.to_numeric(series, errors="coerce")
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    valid = nums.dropna().abs()
    if len(valid):
        median = float(valid.median())
        if 20_000_101 <= median <= 21_001_231:
            parsed = _parse_ga4_dates(series)
        else:
            unit = None
            if median >= 1e15:
                unit = "us"
            elif median >= 1e12:
                unit = "ms"
            elif median >= 1e9:
                unit = "s"
            if unit is not None:
                parsed = pd.to_datetime(nums, unit=unit, errors="coerce")
    text_parsed = pd.to_datetime(series, errors="coerce")
    parsed = parsed.fillna(text_parsed)
    try:
        return parsed.dt.tz_localize(None)
    except (AttributeError, TypeError):
        return parsed


def _build_problem2_time_features(out: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    ts_col = _find_col(out, "event_timestamp", "order_timestamp", "order_date", "eventDate", "event_date")
    date_col = _find_col(out, "date")
    ts = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    source_bits: list[str] = []
    if ts_col is not None:
        ts = _parse_mixed_timestamp_series(out[ts_col])
        source_bits.append(ts_col)
    if date_col is not None:
        ts = ts.fillna(_parse_mixed_timestamp_series(out[date_col]))
        if date_col not in source_bits:
            source_bits.append(date_col)
    if not ts.notna().any():
        return out

    built = []
    if "order_month" not in out.columns:
        out["order_month"] = ts.dt.month.fillna(0).astype(float)
        built.append("order_month")
    if "order_quarter" not in out.columns:
        out["order_quarter"] = ts.dt.quarter.fillna(0).astype(float)
        built.append("order_quarter")
    if "order_dayofweek" not in out.columns:
        out["order_dayofweek"] = ts.dt.dayofweek.fillna(0).astype(float)
        built.append("order_dayofweek")
    if "is_weekend" not in out.columns:
        out["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(float)
        built.append("is_weekend")
    if built:
        notes.append(
            "Built Problem 2 time features from "
            + " -> ".join(source_bits)
            + f": {', '.join(built)}"
        )
    return out


def _apply_problem2_feature_pack(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Normalize the GA4/ecommerce CSV into the locked Problem 2 feature set."""
    out = df.copy()

    alias = {
        "views_per_user": ["views_per_user", "viewsPerUser", "views"],
        "session_engaged_rate": [
            "session_engaged_rate",
            "engagementRate",
            "engagedSessionsPerUser",
            "engagedSessions",
        ],
        "bounce_rate": ["bounce_rate", "bounceRate"],
        "checkouts_per_user": ["checkouts_per_user", "checkoutsPerUser", "checkouts"],
        "items_viewed": ["items_viewed", "itemsViewed"],
        "items_added_to_cart": ["items_added_to_cart", "itemsAddedToCart", "addToCarts"],
    }
    for dest, cands in alias.items():
        if dest in out.columns:
            continue
        src = _find_col(out, *cands)
        if src is None:
            continue
        vals = _to_numeric(out[src]).fillna(0)
        if dest == "session_engaged_rate":
            sess = _to_numeric(
                out.get("total_sessions", out.get("sessionsPerUser", pd.Series(1, index=out.index)))
            ).replace(0, np.nan).fillna(1)
            if float(vals.max() or 0) > 1.0:
                vals = vals / sess
            vals = vals.clip(lower=0, upper=1)
        elif dest == "bounce_rate":
            vals = vals.clip(lower=0, upper=1)
        out[dest] = vals
        notes.append(f"Mapped Problem 2 {dest} ← {src}")

    cat2 = _find_col(out, "itemCategory2", "item_category2", "product_type")
    if cat2 and "item_category2_enc" not in out.columns:
        out["item_category2_enc"] = _encode_series(out[cat2])
        notes.append(f"Encoded item_category2_enc from {cat2}")

    out = _build_problem2_time_features(out, notes)
    return out


def _derive_buyer_churn_proxy(work: pd.DataFrame, notes: list[str]) -> pd.Series:
    """
    Legacy buyer-only churn proxy (kept for reference / rare raw-label paths).

    Preferred (RFM-style among buyers):
      is_buyer=1 AND recency_days > buyer median
      AND (purchase_frequency <= buyer median OR behavioral_degradation > buyer median)

    Fallback when recency is missing / flat:
      is_buyer=1 AND behavioral_degradation > buyer median
      AND engagement_rate < buyer median
    """
    buyer = _to_numeric(work.get("is_buyer", pd.Series(0, index=work.index))).fillna(0).eq(1)
    if not bool(buyer.any()):
        notes.append("legacy buyer churn proxy: no buyers found — all 0")
        return pd.Series(0, index=work.index, dtype=int)

    rec = _to_numeric(work.get("recency_days", pd.Series(0, index=work.index))).fillna(0)
    freq = _to_numeric(
        work.get("purchase_frequency", work.get("total_purchases", pd.Series(0, index=work.index)))
    ).fillna(0)
    bd = _to_numeric(
        work.get("behavioral_degradation", pd.Series(0, index=work.index))
    ).fillna(0)
    eng = _to_numeric(work.get("engagement_rate", pd.Series(0, index=work.index))).fillna(0)

    buyer_rec = rec[buyer]
    recency_usable = bool(buyer_rec.nunique(dropna=True) > 1 and float(buyer_rec.max()) > 0)

    if recency_usable:
        rec_cut = float(buyer_rec.median())
        freq_cut = float(freq[buyer].median())
        bd_cut = float(bd[buyer].median())
        inactive = rec > rec_cut
        weak_or_degraded = (freq <= freq_cut) | (bd > bd_cut)
        churn = buyer & inactive & weak_or_degraded
        notes.append(
            "legacy buyer churn proxy "
            f"(recency>{rec_cut:.0f}d & (freq<={freq_cut:.3g} | degradation>{bd_cut:.3g}))"
        )
    else:
        bd_cut = float(bd[buyer].median())
        eng_cut = float(eng[buyer].median())
        churn = buyer & (bd > bd_cut) & (eng < eng_cut)
        notes.append(
            "legacy buyer churn proxy "
            f"(no usable recency; degradation>{bd_cut:.3g} & engagement<{eng_cut:.3g})"
        )

    return churn.astype(int)


def _derive_low_activity_proxy(work: pd.DataFrame, notes: list[str]) -> pd.Series:
    """
    Quiet-traffic label for Problem 5 — recency only (leakage-safe).

    is_low_activity = 1 when days since last activity > median (usable recency).
    Engagement / bounce / session-length are intentionally NOT in the label so
    those columns remain valid training predictors of who goes quiet.

    Without usable recency, raises FeatureGapError (needs date / event_timestamp).

    Bad = 1 (quiet / fading traffic). Good = 0 (recently active).
    """
    n = len(work)
    if n <= 0:
        return pd.Series(dtype=int)

    rec = _to_numeric(work.get("recency_days", pd.Series(0, index=work.index))).fillna(0)
    recency_usable = bool(rec.nunique(dropna=True) > 1 and float(rec.max()) > 0)
    if not recency_usable:
        raise FeatureGapError(
            5,
            ["date", "event_timestamp", "event_date"],
            message=(
                "Problem 5 needs a usable date / event_timestamp / event_date column "
                "to label quiet traffic via recency_days (leakage-safe target). "
                "Engagement/bounce columns are not used in the label."
            ),
        )

    rec_cut = float(rec.median())
    low = rec > rec_cut
    notes.append(
        "is_low_activity from recency-only quiet-traffic rule "
        f"(recency_days > median {rec_cut:.0f}d) — engagement/bounce NOT in label"
    )
    return low.astype(int)


def _build_p5_retention_signals(work: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """
    Derive retention / intervention signals Vasan brief asks for, from GA4 fields.

    Pass-through only (not locked training features): recency, purchase frequency,
    promo exposure. Exact CRM coupon history may still be missing.
    """
    out = work.copy()

    date_col = _find_col(out, "date", "event_date", "eventDate")
    if date_col:
        dt = _parse_ga4_dates(out[date_col])
        valid = dt.dropna()
        if len(valid):
            cutoff = valid.max()
            out["last_activity_date"] = dt.dt.strftime("%Y-%m-%d").fillna("")
            out["recency_days"] = (cutoff - dt).dt.days
            out["recency_days"] = out["recency_days"].fillna(out["recency_days"].median()).clip(lower=0)
            notes.append(f"recency_days from {date_col} (cutoff={cutoff.date()})")
        else:
            out["recency_days"] = 0.0
            out["last_activity_date"] = ""
            notes.append("recency_days unavailable — date values not parseable")
    else:
        out["recency_days"] = 0.0
        out["last_activity_date"] = ""
        notes.append("recency_days unavailable — no date column")

    freq_src = _find_col(
        out,
        "purchasesPerUser",
        "purchasesperuser",
        "purchase_frequency",
        "purchases_per_user",
        "previous_purchases",
        "total_purchases",
        "purchases",
    )
    if freq_src:
        out["purchase_frequency"] = _to_numeric(out[freq_src]).fillna(0)
        notes.append(f"purchase_frequency from {freq_src}")
    else:
        out["purchase_frequency"] = _to_numeric(
            out.get("total_purchases", pd.Series(0, index=out.index))
        ).fillna(0)

    camp = _find_col(
        out,
        "sessionCampaign",
        "sessioncampaign",
        "firstUserCampaign",
        "firstusercampaign",
        "promo_clicks",
    )
    promo = _to_numeric(out.get("promo_clicks", pd.Series(0, index=out.index))).fillna(0)
    if camp and camp != "promo_clicks":
        exposed = out[camp].astype(str).str.strip().str.lower()
        has_camp = (~exposed.isin(["", "nan", "none", "null", "(not set)", "(notset)"])).astype(float)
        out["promo_exposure"] = (has_camp + (promo > 0).astype(float)).clip(0, 2)
        notes.append(f"promo_exposure from {camp} + promo_clicks")
    else:
        out["promo_exposure"] = (promo > 0).astype(float)
        notes.append("promo_exposure from promo_clicks only (no campaign field)")

    return out


def _attach_p5_retention_pass_through(
    final: pd.DataFrame, source: pd.DataFrame, notes: list[str]
) -> pd.DataFrame:
    """Carry retention timing signals on FE CSV for dashboard / win-back CSVs."""
    out = final.copy()
    attached = []
    for col in (
        "recency_days",
        "purchase_frequency",
        "promo_exposure",
        "last_activity_date",
        "ground_truth_churn",
        "activity_label_threshold_days",
        "_target_source",
        "target_source",
    ):
        if col not in source.columns or col in out.columns:
            continue
        series = source.loc[out.index, col] if col in source.columns else source[col]
        if col in {"last_activity_date", "_target_source", "target_source"}:
            out[col] = series.astype(str).fillna("")
        else:
            out[col] = _to_numeric(series).fillna(0)
        attached.append(col)
    if attached:
        notes.append(f"Preserved P5 retention signals for timing: {', '.join(attached)}")
    return out


def _infer_target_source(notes: list[str]) -> str:
    joined = " ".join(str(n).lower() for n in notes)
    # Prefer explicit target_source=… notes (P6 and others)
    for n in notes:
        s = str(n).lower()
        if s.startswith("target_source="):
            return s.split("=", 1)[1].strip() or "unknown"
    if "time-split unavailable" in joined or "falls back to history" in joined:
        return "proxy"
    if "time-split at" in joined or "future-window units" in joined or "future window" in joined:
        return "time_split"
    if "from order sequence" in joined or "target_source=longitudinal" in joined:
        return "longitudinal"
    if "used from source column (raw)" in joined:
        return "raw"
    if "aligned to is_buyer" in joined or "no repeat history" in joined or "proxy from" in joined:
        return "proxy"
    if "is_full_price_buyer" in joined or "p1 buyers only" in joined:
        if "target_source=raw" in joined:
            return "raw"
        return "derived"
    if "proxy" in joined:
        return "proxy"
    return "unknown"


def _parse_event_dates(df: pd.DataFrame) -> pd.Series:
    """Best-effort event date series for P6 time-split (prefer highest coverage)."""
    best: pd.Series | None = None
    best_rate = 0.0

    def _consider(parsed: pd.Series) -> None:
        nonlocal best, best_rate
        rate = float(parsed.notna().mean()) if len(parsed) else 0.0
        if rate > best_rate:
            best = parsed
            best_rate = rate

    for cand in ("date", "event_date", "eventDate", "event_timestamp"):
        col = _find_col(df, cand)
        if col is None:
            continue
        raw = df[col]
        # GA4 micros / millis / seconds timestamps
        if pd.api.types.is_numeric_dtype(raw):
            vals = pd.to_numeric(raw, errors="coerce")
            # Prefer YYYYMMDD calendar ints when values look like 20YYMMDD
            as_int = vals.dropna().astype("int64")
            if len(as_int) and as_int.between(20000101, 21001231).mean() > 0.8:
                as_str = vals.astype("Int64").astype(str).str.replace("<NA>", "", regex=False)
                parsed = pd.to_datetime(as_str, errors="coerce", format="%Y%m%d")
                _consider(parsed)
                continue
            valid = vals.dropna()
            if valid.empty:
                continue
            med = float(valid.median())
            if med > 1e15:  # microseconds
                _consider(pd.to_datetime(vals, unit="us", errors="coerce"))
            elif med > 1e12:  # milliseconds
                _consider(pd.to_datetime(vals, unit="ms", errors="coerce"))
            elif med > 1e9:  # seconds
                _consider(pd.to_datetime(vals, unit="s", errors="coerce"))
            continue
        # String / mixed
        as_str = raw.astype(str).str.replace(r"\.0$", "", regex=True)
        parsed = pd.to_datetime(as_str, errors="coerce", format="%Y%m%d")
        if parsed.notna().mean() < 0.5:
            parsed = pd.to_datetime(as_str, errors="coerce")
        _consider(parsed)

    if best is not None:
        return best
    return pd.Series(pd.NaT, index=df.index)


def _build_p6_sku_demand_frame(
    raw: pd.DataFrame,
    notes: list[str],
    *,
    inventory_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    GA4-only demand panel: one row per (itemId, day).
    Features = same-day behaviour; target = purchases in the next H days.
    Dashboard still ranks at SKU level (aggregates the panel). No invented ERP.
    """
    horizon = 7
    work = raw.copy()
    item_col = _find_col(work, "itemId", "item_id", "item_sku", "sku")
    if item_col is None:
        raise FeatureGapError(6, ["itemId"], message="Problem 6 needs itemId (SKU) to rank products.")

    view_col = _find_col(work, "itemsViewed", "items_viewed", "productViews")
    atc_col = _find_col(work, "itemsAddedToCart", "addToCarts", "add_to_carts")
    purch_col = _find_col(work, "itemsPurchased", "quantity", "items_purchased")
    rev_col = _find_col(work, "itemRevenue", "purchaseRevenue", "grossItemRevenue")
    name_col = _find_col(work, "itemName", "item_name")
    cat_col = _find_col(work, "itemCategory", "item_category", "category")

    work["_item_id"] = work[item_col].astype(str).str.strip()
    work = work[work["_item_id"].ne("") & work["_item_id"].str.lower().ne("nan")].copy()
    if work.empty:
        raise FeatureGapError(6, ["itemId"], message="Problem 6: no valid itemId rows.")

    work["_views"] = _to_numeric(work[view_col]).fillna(0) if view_col else 0.0
    work["_atc"] = _to_numeric(work[atc_col]).fillna(0) if atc_col else 0.0
    work["_purch"] = _to_numeric(work[purch_col]).fillna(0) if purch_col else 0.0
    work["_rev"] = _to_numeric(work[rev_col]).fillna(0) if rev_col else 0.0
    from pipeline.p6_inventory import P6_CURRENCY, P6_INR_PER_USD

    work["_rev"] = pd.to_numeric(work["_rev"], errors="coerce").fillna(0) / P6_INR_PER_USD
    work["_item_name"] = work[name_col].astype(str) if name_col else work["_item_id"]
    work["_item_category"] = work[cat_col].astype(str) if cat_col else "unknown"
    work["_event_dt"] = _parse_event_dates(work)

    dated = work["_event_dt"].notna()
    if float(dated.mean()) < 0.3:
        notes.append(
            "P6 daily panel unavailable (no usable event_timestamp/date); "
            "falling back to one row per SKU with history=future proxy."
        )
        return _merge_p6_inventory_fallback(
            _build_p6_sku_level_fallback(work, notes), inventory_df, notes
        )

    work = work.loc[dated].copy()
    work["_day"] = pd.to_datetime(work["_event_dt"]).dt.normalize()

    daily = (
        work.groupby(["_item_id", "_day"], as_index=False)
        .agg(
            itemsViewed=("_views", "sum"),
            itemsAddedToCart=("_atc", "sum"),
            history_purchases=("_purch", "sum"),
            history_revenue=("_rev", "sum"),
            item_name=("_item_name", "first"),
            item_category=("_item_category", "first"),
        )
        .sort_values(["_item_id", "_day"])
        .reset_index(drop=True)
    )

    def _forward_calendar_sums(
        days: np.ndarray, purch: np.ndarray, rev: np.ndarray, h: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sum purch/rev on calendar days (t, t+h] for each row day t."""
        n = len(days)
        fut_p = np.zeros(n, dtype=float)
        fut_r = np.zeros(n, dtype=float)
        # days are datetime64[ns], sorted
        for i in range(n):
            limit = days[i] + np.timedelta64(h, "D")
            k = i + 1
            while k < n and days[k] <= limit:
                k += 1
            if k > i + 1:
                fut_p[i] = float(purch[i + 1 : k].sum())
                fut_r[i] = float(rev[i + 1 : k].sum())
        return fut_p, fut_r

    fut_purch: list[float] = []
    fut_rev: list[float] = []
    for _, g in daily.groupby("_item_id", sort=False):
        g = g.sort_values("_day")
        fp, fr = _forward_calendar_sums(
            g["_day"].to_numpy(),
            g["history_purchases"].to_numpy(dtype=float),
            g["history_revenue"].to_numpy(dtype=float),
            horizon,
        )
        fut_purch.extend(fp.tolist())
        fut_rev.extend(fr.tolist())
    daily["itemsPurchased_future"] = fut_purch
    daily["future_revenue"] = fut_rev

    # Time-safe history features (≤ day t). Target is strictly days (t, t+H].
    roll_parts: list[pd.DataFrame] = []
    for _, g in daily.groupby("_item_id", sort=False):
        g = g.sort_values("_day").copy()
        purch = g["history_purchases"]
        views = g["itemsViewed"]
        atc = g["itemsAddedToCart"]
        rev = g["history_revenue"]
        for w in (3, 7, 14, 28):
            g[f"purchases_roll_{w}"] = purch.rolling(w, min_periods=1).sum()
            g[f"views_roll_{w}"] = views.rolling(w, min_periods=1).sum()
            g[f"atc_roll_{w}"] = atc.rolling(w, min_periods=1).sum()
        g["revenue_roll_7"] = rev.rolling(7, min_periods=1).sum()
        g["sku_demand_baseline"] = purch.expanding(min_periods=1).mean().fillna(0) * float(horizon)
        g["day_of_week"] = g["_day"].dt.dayofweek.astype(float)
        roll_parts.append(g)
    daily = pd.concat(roll_parts, ignore_index=True)

    max_day = daily["_day"].max()
    cutoff = max_day - pd.Timedelta(days=horizon)
    panel = daily.copy()
    if panel.empty:
        notes.append("P6 panel empty after daily aggregation; falling back to one row per SKU.")
        return _merge_p6_inventory_fallback(
            _build_p6_sku_level_fallback(work, notes), inventory_df, notes
        )
    # Tail rows after cutoff may have partial next-{horizon}d targets; keep for full panel.
    panel["future_target_complete"] = (panel["_day"] <= cutoff).astype(int)
    tail_partial = int((panel["future_target_complete"] == 0).sum())

    # Model features: prefer trailing windows over raw same-day spikes
    panel["itemsViewed"] = panel["views_roll_7"]
    panel["itemsAddedToCart"] = panel["atc_roll_7"]
    panel["history_purchases"] = panel["purchases_roll_7"]
    panel["history_revenue"] = panel["revenue_roll_7"]

    views = panel["itemsViewed"].replace(0, np.nan)
    panel["cartToViewRate"] = (panel["itemsAddedToCart"] / views).fillna(0)
    panel["purchaseToViewRate"] = (panel["history_purchases"] / views).fillna(0)

    with np.errstate(divide="ignore", invalid="ignore"):
        price = np.where(
            panel["history_purchases"] > 0,
            panel["history_revenue"] / panel["history_purchases"],
            0.0,
        )
    panel["unit_price"] = pd.Series(price, index=panel.index).fillna(0).clip(lower=0)

    panel["dxi_demand_score"] = (
        _scale_safe(panel["cartToViewRate"])
        + _scale_safe(panel["purchaseToViewRate"])
        + _scale_safe(panel["itemsViewed"])
        + _scale_safe(panel["purchases_roll_14"])
    ) / 4.0

    # Extra leakage-safe drivers (SKU identity + short lags + momentum)
    panel = panel.sort_values(["_item_id", "_day"]).reset_index(drop=True)
    panel["item_id_enc"] = pd.Categorical(panel["_item_id"].astype(str)).codes.astype(float)
    eps = 1e-6
    panel["purchase_momentum"] = panel["purchases_roll_7"] / (
        panel["purchases_roll_28"] / 4.0 + eps
    )
    panel["purchases_lag_1"] = (
        panel.groupby("_item_id", sort=False)["history_purchases"].shift(1).fillna(0.0)
    )
    panel["purchases_lag_7"] = (
        panel.groupby("_item_id", sort=False)["history_purchases"].shift(7).fillna(0.0)
    )

    # Training target: winsorize + light shrink to SKU baseline so day-noise
    # does not wipe signal on this synthetic GA4 file. Raw kept for ranking.
    raw_future = pd.to_numeric(panel["itemsPurchased_future"], errors="coerce").fillna(0.0)
    panel["future_units_raw"] = raw_future
    hi = float(raw_future.quantile(0.95)) if len(raw_future) else 0.0
    if hi > 0:
        raw_future = raw_future.clip(upper=hi)
    baseline = (
        pd.to_numeric(panel["sku_demand_baseline"], errors="coerce").fillna(0.0).clip(lower=0)
    )
    shrink = 0.30
    panel["itemsPurchased_future"] = (
        (1.0 - shrink) * raw_future + shrink * baseline
    ).clip(lower=0)

    panel["item_category_enc"] = _encode_series(panel["item_category"].astype(str))
    panel["item_id"] = panel["_item_id"].astype(str)
    panel["item_name"] = panel["item_name"].astype(str)
    panel["item_category"] = panel["item_category"].astype(str)
    panel["panel_date"] = panel["_day"].dt.strftime("%Y-%m-%d")
    panel["split_date"] = str(pd.Timestamp(cutoff).date())
    panel["future_horizon_days"] = horizon
    panel["target_source"] = "time_split"
    panel["currency"] = P6_CURRENCY

    panel = panel.drop(columns=["_item_id", "_day"], errors="ignore")
    from pipeline.p6_inventory import merge_inventory_into_p6_panel

    panel = merge_inventory_into_p6_panel(panel, inventory_df, notes)
    if int(panel.get("erp_inventory_available", pd.Series([0])).max()) == 0:
        panel["erp_inventory_available"] = 0

    n_sku = int(panel["item_id"].nunique())
    erp_flag = int(panel["erp_inventory_available"].max()) if "erp_inventory_available" in panel.columns else 0
    notes.append(
        f"P6 itemId×day panel: {len(panel)} rows ({n_sku} SKUs × days); "
        f"all daily rows kept"
        + (f" ({tail_partial} with partial {horizon}d future after cutoff)" if tail_partial else "")
        + "; "
        f"features=rolls+lags+item_id_enc+momentum (no leakage); "
        f"target=next-{horizon}d purchases (winsor p95 + {int(shrink*100)}% baseline shrink); "
        f"ranking cutoff={panel['split_date'].iloc[0]}; "
        f"currency=USD (INR÷100); "
        f"ERP inventory={'connected' if erp_flag else 'not provided'}."
    )
    notes.append("target_source=time_split")
    return panel


def _build_p6_sku_level_fallback(work: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Last-resort: one row per SKU when dates are unusable."""
    g = (
        work.groupby("_item_id", as_index=False)
        .agg(
            itemsViewed=("_views", "sum"),
            itemsAddedToCart=("_atc", "sum"),
            history_purchases=("_purch", "sum"),
            history_revenue=("_rev", "sum"),
            item_name=("_item_name", "first"),
            item_category=("_item_category", "first"),
        )
    )
    views = g["itemsViewed"].replace(0, np.nan)
    g["cartToViewRate"] = (g["itemsAddedToCart"] / views).fillna(0)
    g["purchaseToViewRate"] = (g["history_purchases"] / views).fillna(0)
    g["clickThroughRate"] = g["cartToViewRate"]
    g["itemsPurchased_future"] = g["history_purchases"]
    g["future_revenue"] = g["history_revenue"]
    with np.errstate(divide="ignore", invalid="ignore"):
        price = np.where(
            g["history_purchases"] > 0,
            g["history_revenue"] / g["history_purchases"],
            0.0,
        )
    g["unit_price"] = pd.Series(price, index=g.index).fillna(0).clip(lower=0)
    g["dxi_demand_score"] = (
        _scale_safe(g["cartToViewRate"])
        + _scale_safe(g["purchaseToViewRate"])
        + _scale_safe(g["itemsViewed"])
    ) / 3.0
    g["purchases_roll_3"] = g["history_purchases"]
    g["purchases_roll_7"] = g["history_purchases"]
    g["purchases_roll_14"] = g["history_purchases"]
    g["purchases_roll_28"] = g["history_purchases"]
    g["views_roll_7"] = g["itemsViewed"]
    g["views_roll_14"] = g["itemsViewed"]
    g["views_roll_28"] = g["itemsViewed"]
    g["atc_roll_14"] = g["itemsAddedToCart"]
    g["sku_demand_baseline"] = g["history_purchases"]
    g["purchase_momentum"] = 1.0
    g["purchases_lag_1"] = 0.0
    g["item_id_enc"] = pd.Categorical(g["_item_id"].astype(str)).codes.astype(float)
    g["day_of_week"] = 0.0
    g["item_category_enc"] = _encode_series(g["item_category"].astype(str))
    g["item_id"] = g["_item_id"].astype(str)
    g["item_name"] = g["item_name"].astype(str)
    g["item_category"] = g["item_category"].astype(str)
    g["future_units_raw"] = g["itemsPurchased_future"]
    g["panel_date"] = ""
    g["split_date"] = ""
    g["future_horizon_days"] = 0
    g["erp_inventory_available"] = 0
    g["target_source"] = "proxy"
    g["currency"] = "USD"
    notes.append(
        f"P6 SKU fallback: {len(g)} products; mean future units="
        f"{float(g['itemsPurchased_future'].mean()):.3f} (proxy)."
    )
    notes.append("target_source=proxy")
    return g.drop(columns=["_item_id"], errors="ignore")


def _merge_p6_inventory_fallback(
    panel: pd.DataFrame,
    inventory_df: pd.DataFrame | None,
    notes: list[str],
) -> pd.DataFrame:
    from pipeline.p6_inventory import merge_inventory_into_p6_panel

    return merge_inventory_into_p6_panel(panel, inventory_df, notes)


def _attach_p6_pass_through(
    final: pd.DataFrame, source: pd.DataFrame, notes: list[str]
) -> pd.DataFrame:
    """Keep SKU / panel audit columns on FE CSV for the P6 dashboard."""
    out = final.copy()
    attached: list[str] = []
    for col in (
        "item_id",
        "item_name",
        "item_category",
        "history_purchases",
        "history_revenue",
        "future_revenue",
        "split_date",
        "panel_date",
        "future_horizon_days",
        "future_units_raw",
        "erp_inventory_available",
        "target_source",
        "stock_units",
        "margin",
        "unit_cost",
        "reorder_point",
        "lead_time_days",
        "in_transit_units",
        "currency",
    ):
        if col not in source.columns or col in out.columns:
            continue
        series = source.loc[out.index, col] if col in source.columns else source[col]
        if col in {
            "item_id",
            "item_name",
            "item_category",
            "split_date",
            "panel_date",
            "target_source",
            "currency",
        }:
            out[col] = series.astype(str).fillna("")
        else:
            out[col] = _to_numeric(series).fillna(0)
        attached.append(col)
    if attached:
        notes.append(f"Preserved P6 SKU audit columns: {', '.join(attached)}")
    return out


def _attach_p24_pass_through(
    final: pd.DataFrame, source: pd.DataFrame, notes: list[str], problem_id: int
) -> pd.DataFrame:
    """Keep audit / dashboard columns on FE CSV; never locked training features."""
    out = final.copy()
    attached: list[str] = []
    cat_src = _find_col(source, "itemCategory", "item_category", "category")
    pass_cols = [
        "target_source",
        "_target_source",
        "snapshot_date",
        "churn_risk",
        "event_observed",
        "time_to_event",
        "previous_purchase_date",
        "next_purchase_date",
        "ltv_proxy",
        "repeat_purchase_30d",
        "repeat_purchase_60d",
        "repeat_purchase_90d",
        "is_repurchase_ready",
        "n_orders",
    ]
    if cat_src and "item_category" not in out.columns:
        series = source.loc[out.index, cat_src] if cat_src in source.columns else source[cat_src]
        out["item_category"] = series.astype(str).fillna("")
        attached.append("item_category")
    for col in pass_cols:
        if col not in source.columns or col in out.columns:
            continue
        series = source.loc[out.index, col] if col in source.columns else source[col]
        if col in {"snapshot_date", "previous_purchase_date", "next_purchase_date"}:
            out[col] = pd.to_datetime(series, errors="coerce")
        elif col in {"target_source", "_target_source"}:
            out["target_source"] = series.astype(str)
        else:
            out[col] = _to_numeric(series)
        attached.append("target_source" if col == "_target_source" else col)
    if "target_source" not in out.columns:
        src = _infer_target_source(notes)
        out["target_source"] = src
        attached.append("target_source")
    elif "_target_source" in out.columns and "target_source" not in out.columns:
        out["target_source"] = out["_target_source"].astype(str)
    if "_target_source" in out.columns:
        out = out.drop(columns=["_target_source"], errors="ignore")
    if int(problem_id) in (2, 4):
        out = _attach_user_pseudo_id(out, source, notes)
    if attached:
        notes.append(f"Preserved P{int(problem_id)} audit columns: {', '.join(dict.fromkeys(attached))}")
    return out


def is_non_feature_column(name: str) -> bool:
    """True for identity / retention pass-through columns that must not enter locked_features."""
    key = str(name or "").strip().lower().replace(" ", "")
    if key in NON_FEATURE_COLUMNS:
        return True
    if "pseudo" in key:
        return True
    return False


def _apply_p1_style_behaviour(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    out = df.copy()

    cat = _find_col(out, "itemCategory", "item_category", "first_item_category")
    if cat and "first_item_category_enc" not in out.columns:
        out["first_item_category_enc"] = _encode_series(out[cat])
        notes.append(f"Encoded first_item_category_enc from {cat}")

    sessions = _find_col(out, "sessions", "total_sessions", "sessionsPerUser")
    if sessions and "total_sessions" not in out.columns:
        out["total_sessions"] = _to_numeric(out[sessions]).fillna(0)
        notes.append(f"Mapped total_sessions from {sessions}")

    eng = _find_col(out, "userEngagementDuration", "engagement_time", "engagementTime")
    if eng and "engagement_time" not in out.columns:
        vals = _to_numeric(out[eng]).fillna(0)
        # scale ms → seconds when values look like ms
        if vals.median(skipna=True) and vals.median(skipna=True) > 1000:
            vals = vals / 1000.0
            notes.append(f"Scaled engagement_time from ms ({eng})")
        out["engagement_time"] = vals

    scroll = _find_col(out, "percentScrolled", "scroll_depth", "scrollDepth")
    if scroll and "scroll_depth" not in out.columns:
        out["scroll_depth"] = _to_numeric(out[scroll]).fillna(0)

    atc = _find_col(out, "addToCarts", "add_to_carts", "itemsAddedToCart")
    checkouts = _find_col(out, "checkouts", "checkout")
    if "cart_abandonment" not in out.columns:
        if atc is not None:
            atc_v = _to_numeric(out[atc]).fillna(0)
            if checkouts is not None:
                chk_v = _to_numeric(out[checkouts]).fillna(0)
                out["cart_abandonment"] = ((atc_v > 0) & (chk_v <= 0)).astype(int)
            else:
                purchases = _find_col(out, "purchases", "transactions")
                if purchases is not None:
                    # Prefer not using purchases for abandon when checkouts exist;
                    # if only purchases, still avoid calling it leakage into feature
                    # by using ATC-only proxy.
                    out["cart_abandonment"] = (atc_v > 0).astype(int)
                    notes.append("cart_abandonment from add_to_carts proxy (no checkouts)")
                else:
                    out["cart_abandonment"] = (atc_v > 0).astype(int)
            notes.append("Built cart_abandonment")
        else:
            out["cart_abandonment"] = 0

    if atc is not None and "add_to_carts" not in out.columns:
        out["add_to_carts"] = _to_numeric(out[atc]).fillna(0)

    campaign = _find_col(out, "sessionCampaign", "campaign", "promo_clicks")
    if "promo_clicks" not in out.columns:
        if campaign is not None:
            if pd.api.types.is_numeric_dtype(out[campaign]):
                out["promo_clicks"] = _to_numeric(out[campaign]).fillna(0)
            else:
                out["promo_clicks"] = out[campaign].notna().astype(int)
            notes.append(f"promo_clicks proxy from {campaign}")
        else:
            out["promo_clicks"] = 0

    search = _find_col(out, "searchTerm", "search_clicks", "search")
    if "search_clicks" not in out.columns:
        if search is not None:
            if pd.api.types.is_numeric_dtype(out[search]):
                out["search_clicks"] = _to_numeric(out[search]).fillna(0)
            else:
                out["search_clicks"] = (
                    out[search].astype(str).str.strip().ne("") & out[search].notna()
                ).astype(int)
            notes.append(f"search_clicks proxy from {search}")
        else:
            out["search_clicks"] = 0

    bounce = _find_col(out, "bounceRate", "bounce_rate")
    eng_rate = _find_col(out, "engagementRate", "engagement_rate")
    if "churn_risk" not in out.columns:
        b = _to_numeric(out[bounce]).fillna(0) if bounce else pd.Series(0, index=out.index)
        e = _to_numeric(out[eng_rate]).fillna(0) if eng_rate else pd.Series(0, index=out.index)
        # higher bounce / lower engagement → higher risk
        out["churn_risk"] = (b.clip(0, 1) + (1 - e.clip(0, 1))) / 2.0
        notes.append("Built churn_risk from bounce/engagement")

    channel = _find_col(out, "firstUserPrimaryChannelGroup", "acq_channel", "channel")
    if channel and "acq_channel_enc" not in out.columns:
        out["acq_channel_enc"] = _encode_series(out[channel])

    age = _find_col(out, "age", "age_group", "ageGroup")
    if age and "age_group_enc" not in out.columns:
        out["age_group_enc"] = _encode_series(out[age])

    gender = _find_col(out, "gender", "sex")
    if gender and "gender_enc" not in out.columns:
        out["gender_enc"] = _encode_series(out[gender])

    device = _find_col(out, "deviceCategory", "device", "device_enc")
    if device and "device_enc" not in out.columns:
        out["device_enc"] = _encode_series(out[device])

    return out


def _ensure_is_buyer(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    out = df.copy()
    if "is_buyer" in out.columns:
        out["is_buyer"] = _ensure_binary_target(out, "is_buyer")
        return out
    from pipeline.target_creator import create_is_buyer_column

    out, _ = create_is_buyer_column(out)
    if "is_buyer" not in out.columns:
        raise FeatureGapError(1, ["is_buyer"])
    notes.append("Created is_buyer target")
    out["is_buyer"] = _ensure_binary_target(out, "is_buyer")
    return out


# Empty / non-informative campaign or coupon values (aligned with P8).
_P1_INVALID_PROMO_VALUES = {
    "",
    "(not set)",
    "not set",
    "nan",
    "none",
    "null",
    "unknown",
    "undefined",
    "0",
    "false",
    "no",
}

# GA4 sessionCampaign names that indicate a discount / promo purchase.
_P1_DISCOUNT_CAMPAIGN_RE = (
    r"sale|offer|discount|coupon|promo|clearance|deal|voucher|"
    r"flash|markdown|cashback|loyalty|bonanza|special"
)


def _series_looks_filled(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.lower()
    return ~text.isin(_P1_INVALID_PROMO_VALUES)


def _bought_with_discount_mask(df: pd.DataFrame, notes: list[str]) -> tuple[pd.Series, str]:
    """
    Buyer-row discount flag. First available signal wins:

      1. coupon / coupon_used / couponCode present
      2. discount / discount_amount / promo_discount_amount > 0
      3. discount-like sessionCampaign / campaign name (GA4 proxy)
    """
    coupon_col = _find_col(
        df, "coupon", "coupon_used", "couponUsed", "couponCode", "coupon_code"
    )
    if coupon_col is not None:
        raw = df[coupon_col]
        if pd.api.types.is_numeric_dtype(raw):
            used = _to_numeric(raw).fillna(0) > 0
        else:
            used = _series_looks_filled(raw)
        notes.append(f"bought_with_discount from {coupon_col}")
        return used.astype(bool), f"coupon:{coupon_col}"

    amount_col = _find_col(
        df,
        "discount_amount",
        "discountAmount",
        "promo_discount_amount",
        "itemDiscount",
        "discount",
    )
    if amount_col is not None:
        used = _to_numeric(df[amount_col]).fillna(0) > 0
        notes.append(f"bought_with_discount from {amount_col} > 0")
        return used.astype(bool), f"amount:{amount_col}"

    campaign_col = _find_col(
        df, "sessionCampaign", "campaign", "promotion_name", "promotionName"
    )
    if campaign_col is not None:
        text = df[campaign_col].fillna("").astype(str).str.strip().str.lower()
        used = text.str.contains(_P1_DISCOUNT_CAMPAIGN_RE, regex=True, na=False)
        notes.append(
            f"bought_with_discount from discount-like {campaign_col} "
            f"(sale/offer/clearance/coupon/… proxy)"
        )
        return used.astype(bool), f"campaign:{campaign_col}"

    raise FeatureGapError(
        1,
        ["coupon", "discount", "sessionCampaign"],
        message=(
            "Problem 1 needs a discount or promo signal among buyers "
            "(coupon / discount amount / sessionCampaign) to label "
            "full-price vs discount buyers. Do not fall back to is_buyer."
        ),
    )


def _build_p1_full_price_buyer_frame(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """
    Problem 1 (email 2): buyers only, split with vs without discount.

    is_full_price_buyer = 1 if the buyer purchased without a discount.
    is_full_price_buyer = 0 if the buyer purchased with a discount.
    Non-buyers are dropped. Discount buyers are kept as the negative class.
    """
    work = df.copy()
    if "is_buyer" not in work.columns:
        raise FeatureGapError(1, ["is_buyer"])

    buyer_mask = work["is_buyer"].fillna(0).astype(int).eq(1)
    n_all = int(len(work))
    work = work.loc[buyer_mask].copy()
    n_buyers = int(len(work))
    if n_buyers <= 0:
        raise FeatureGapError(
            1,
            ["is_buyer"],
            message="Problem 1 needs buyers (is_buyer = 1). No buyer rows found.",
        )
    notes.append(
        f"P1 buyers only: kept {n_buyers} of {n_all} rows (dropped {n_all - n_buyers} non-buyers)"
    )

    if "is_full_price_buyer" in work.columns:
        work["is_full_price_buyer"] = _ensure_binary_target(work, "is_full_price_buyer")
        notes.append("is_full_price_buyer used from source column (raw) on buyer rows")
        notes.append("target_source=raw")
        if "bought_with_discount" not in work.columns:
            work["bought_with_discount"] = (work["is_full_price_buyer"] == 0).astype(int)
    else:
        used, source = _bought_with_discount_mask(work, notes)
        work["bought_with_discount"] = used.astype(int)
        work["is_full_price_buyer"] = (work["bought_with_discount"] == 0).astype(int)
        work["p1_discount_source"] = source
        notes.append(
            "is_full_price_buyer = 1 if buyer purchased without discount; "
            "0 if buyer purchased with a discount"
        )
        notes.append(f"target_source=derived:{source}")

    work["is_full_price_buyer"] = _ensure_binary_target(work, "is_full_price_buyer")
    n_pos = int(work["is_full_price_buyer"].eq(1).sum())
    n_neg = int(work["is_full_price_buyer"].eq(0).sum())
    notes.append(
        f"P1 full-price vs discount among buyers: full-price={n_pos}, discount={n_neg}"
    )
    if n_pos <= 0 or n_neg <= 0:
        raise FeatureGapError(
            1,
            ["is_full_price_buyer"],
            message=(
                "Problem 1 needs both classes among buyers: full-price (no discount) "
                f"and discount buyers. Found full-price={n_pos}, discount={n_neg}."
            ),
        )
    return work


def _attach_p1_pass_through(
    final: pd.DataFrame, source: pd.DataFrame, notes: list[str]
) -> pd.DataFrame:
    """Keep buyer/discount audit columns on the FE CSV; never locked features."""
    out = final.copy()
    attached: list[str] = []
    for col in ("is_buyer", "bought_with_discount", "p1_discount_source", "target_source"):
        if col not in source.columns or col in out.columns:
            continue
        try:
            series = source.loc[out.index, col]
        except Exception:
            continue
        if col in {"p1_discount_source", "target_source"}:
            out[col] = series.astype(str)
        else:
            out[col] = _to_numeric(series).fillna(0)
        attached.append(col)
    if attached:
        notes.append(f"P1 pass-through: {', '.join(attached)}")
    return out


def _finalize(
    df: pd.DataFrame,
    problem: dict[str, Any],
    notes: list[str],
    *,
    allow_partial: bool = False,
) -> ProblemFEResult:
    target = problem["target"]
    features = list(problem["features"])
    forbidden = list(problem.get("forbidden_features") or [])
    allow_partial = bool(allow_partial or problem.get("allow_partial_features"))
    min_features = int(problem.get("min_features") or 0)

    out = _drop_forbidden(df, forbidden)

    if target not in out.columns:
        raise FeatureGapError(problem["id"], [target])

    missing = [f for f in features if f not in out.columns]
    if missing and not allow_partial:
        raise FeatureGapError(problem["id"], missing)

    keep = [f for f in features if f in out.columns]
    if allow_partial and min_features and len(keep) < min_features:
        raise FeatureGapError(
            problem["id"],
            missing,
            message=(
                f"Problem {problem['id']}: only {len(keep)} usable features "
                f"(need at least {min_features}). Missing: "
                + (", ".join(missing[:12]) or "(none)")
            ),
        )
    if missing and allow_partial:
        notes.append(
            f"Partial feature set: using {len(keep)}/{len(features)} "
            f"(skipped {len(missing)} unavailable)"
        )

    optional_missing = [f for f in (problem.get("optional_extras") or []) if f not in out.columns]

    cols = list(dict.fromkeys(keep + [target]))
    final = out[cols].copy()
    # coerce features numeric
    for c in keep:
        final[c] = _to_numeric(final[c]).fillna(0)

    if problem["task_type"] == "classification":
        final[target] = _ensure_binary_target(final, target)
        pos = _positive_rate(final[target])
    else:
        final[target] = _to_numeric(final[target]).fillna(0)
        if int(problem.get("id") or 0) == 8:
            final[target] = final[target].clip(0, 1)
        elif int(problem.get("id") or 0) == 4:
            final[target] = final[target].fillna(final[target].median())
        pos = None

    # Pass-through CRM id (Problems 1, 5, 8 targeting CSVs); never a training feature
    if int(problem.get("id") or 0) in (1, 5, 7, 8):
        final = _attach_user_pseudo_id(final, out, notes)
    if int(problem.get("id") or 0) == 1:
        # Audit cols live on the pre-forbidden frame (is_buyer is dropped as leakage).
        final = _attach_p1_pass_through(final, df, notes)
    if int(problem.get("id") or 0) == 5:
        # Attach from pre-forbidden frame so recency / timing cols survive even if listed forbidden
        final = _attach_p5_retention_pass_through(final, df, notes)
    if int(problem.get("id") or 0) in (2, 4):
        final = _attach_p24_pass_through(final, df, notes, int(problem.get("id") or 0))
    if int(problem.get("id") or 0) == 6:
        final = _attach_p6_pass_through(final, out, notes)
    if int(problem.get("id") or 0) == 7:
        final = _attach_p7_pass_through(final, out, notes)
    if int(problem.get("id") or 0) == 8:
        final = _attach_p8_pass_through(final, out, notes)

    notes.append(
        f"Locked {len(keep)} features + target={target}; "
        f"rows={len(final)}; no balancing applied"
    )
    return ProblemFEResult(
        df=final,
        problem=problem,
        features_used=keep,
        target=target,
        positive_rate=pos,
        notes=notes,
        gaps_optional=optional_missing,
        target_source=_infer_target_source(notes),
    )


def apply_problem_fe(
    df: pd.DataFrame,
    problem_id: int,
    *,
    commerce_bundle=None,
    inventory_df: pd.DataFrame | None = None,
) -> ProblemFEResult:
    problem = get_problem(problem_id)
    if not problem:
        raise FeatureGapError(int(problem_id or 0), [], message="Unknown problem id.")

    notes: list[str] = []
    raw = df.copy()
    pid = problem["id"]
    bundle = commerce_bundle
    if pid in (2, 4) and bundle is None:
        try:
            from pipeline.purchase_history import extract_commerce_bundle

            bundle = extract_commerce_bundle(raw)
        except Exception as exc:
            notes.append(f"commerce extract skipped: {exc}")
            bundle = None

    if pid in (1, 2, 3):
        work = _apply_p1_style_behaviour(raw, notes)
        if pid == 2:
            work = _apply_problem2_feature_pack(work, notes)
        work = _ensure_is_buyer(work, notes)

        if pid == 1:
            work = _build_p1_full_price_buyer_frame(work, notes)
            return _finalize(work, problem, notes)

        if pid == 2:
            if bundle is not None:
                from pipeline.purchase_history import apply_longitudinal_labels

                work = apply_longitudinal_labels(work, bundle, problem_id=2, notes=notes)
            if "is_repeat_buyer" not in work.columns:
                # Proxy: buyers with returning / prior signal when true history missing
                returning = _find_col(work, "returningUsers", "returning_user", "returningUserRate")
                prev = _find_col(work, "previous_purchases", "purchase_frequency")
                if returning is not None:
                    work["is_repeat_buyer"] = (
                        (_to_numeric(work[returning]).fillna(0) > 0) & (work["is_buyer"] == 1)
                    ).astype(int)
                    notes.append(f"is_repeat_buyer proxy from {returning} ∩ is_buyer")
                elif prev is not None:
                    work["is_repeat_buyer"] = (
                        (_to_numeric(work[prev]).fillna(0) > 0) & (work["is_buyer"] == 1)
                    ).astype(int)
                    notes.append(f"is_repeat_buyer proxy from {prev}")
                else:
                    work["is_repeat_buyer"] = work["is_buyer"].astype(int)
                    notes.append("is_repeat_buyer aligned to is_buyer (no repeat history)")
            else:
                work["is_repeat_buyer"] = _ensure_binary_target(work, "is_repeat_buyer")
                if not any("from order sequence" in str(n) for n in notes):
                    notes.append("is_repeat_buyer used from source column (raw)")
            result = _finalize(work, problem, notes)
            if bundle is not None:
                result.data_quality = bundle.quality.as_dict()
                result.target_source = bundle.quality.target_source if result.target_source == "unknown" else result.target_source
                if result.df is not None and "target_source" in result.df.columns:
                    result.target_source = str(result.df["target_source"].dropna().astype(str).iloc[0]) if result.df["target_source"].notna().any() else result.target_source
            return result

        # pid == 3
        ptype = _find_col(work, "itemCategory2", "itemName", "itemBrand", "product_type")
        if "first_product_type_enc" not in work.columns:
            if ptype is not None:
                work["first_product_type_enc"] = _encode_series(work[ptype])
                notes.append(f"first_product_type_enc from {ptype}")
            elif "first_item_category_enc" in work.columns:
                work["first_product_type_enc"] = work["first_item_category_enc"]
                notes.append("first_product_type_enc copied from category enc")
            else:
                work["first_product_type_enc"] = 0

        if "is_high_ltv" not in work.columns:
            rev = _find_col(work, "totalRevenue", "purchaseRevenue", "ltv", "lifetime_value")
            buyers = work["is_buyer"] == 1
            if rev is not None and buyers.any():
                rev_v = _to_numeric(work[rev]).fillna(0)
                thr = rev_v[buyers].quantile(0.75)
                work["is_high_ltv"] = ((rev_v >= thr) & buyers).astype(int)
                notes.append(f"is_high_ltv = top 25% {rev} among buyers")
            else:
                work["is_high_ltv"] = work["is_buyer"].astype(int)
                notes.append("is_high_ltv aligned to is_buyer (no revenue/LTV col)")
        else:
            work["is_high_ltv"] = _ensure_binary_target(work, "is_high_ltv")
        return _finalize(work, problem, notes)

    if pid == 4:
        work = raw.copy()
        # Map common aliases into Stage1/2 names (GA4 + ecommerce)
        alias = {
            "sessionsPerUser": ["sessionsPerUser", "sessions_per_user", "sessions"],
            "viewsPerUser": [
                "viewsPerUser",
                "views_per_user",
                "screenPageViews",
                "views",
            ],
            "engagement_time_msec": [
                "engagement_time_msec",
                "userEngagementDuration",
                "averageUserEngagementDuration",
                "averageEngagementTimePerSession",
                "engagement_time",
            ],
            "session_engaged": [
                "session_engaged",
                "engagedSessions",
                "engagedSessionsPerUser",
                "engagementRate",
            ],
            "purchaseRevenue": ["purchaseRevenue", "totalRevenue", "revenue", "grossPurchaseRevenue"],
            "cart_abandonment_rate": ["cart_abandonment_rate", "cart_abandonment"],
            "previous_purchases": ["previous_purchases", "purchases", "ecommercePurchases"],
            "tenure_days": ["tenure_days", "days_since_first"],
            "purchase_frequency": [
                "purchase_frequency",
                "purchaseFrequency",
                "purchasesPerUser",
                "transactionsPerUser",
            ],
            "avg_order_value": [
                "avg_order_value",
                "aov",
                "averageOrderValue",
                "averagePurchaseRevenue",
                "averagePurchaseRevenuePerUser",
            ],
            "unique_categories": ["unique_categories", "category_count"],
            "cat_mean_target": ["cat_mean_target"],
            "cat_std_target": ["cat_std_target"],
            "days_until_next_purchase": [
                "days_until_next_purchase",
                "days_to_next_purchase",
                "repurchase_days",
                "days_to_repurchase",
                "next_purchase_days",
            ],
        }
        for dest, cands in alias.items():
            if dest in work.columns:
                continue
            src = _find_col(work, *cands)
            if src is not None:
                work[dest] = _to_numeric(work[src])
                notes.append(f"Mapped {dest} ← {src}")

        # Original regression-SXI proxy first (dense numeric days, no NaNs).
        # Longitudinal sequence days overlay later only where a next purchase was observed.
        freq = _to_numeric(
            work["purchase_frequency"]
            if "purchase_frequency" in work.columns
            else pd.Series(0, index=work.index)
        ).fillna(0)
        prev = _to_numeric(
            work["previous_purchases"]
            if "previous_purchases" in work.columns
            else pd.Series(0, index=work.index)
        ).fillna(0)
        sess = _to_numeric(
            work["sessionsPerUser"]
            if "sessionsPerUser" in work.columns
            else pd.Series(1, index=work.index)
        ).fillna(1).clip(lower=0.1)
        eng = _to_numeric(
            work["session_engaged"]
            if "session_engaged" in work.columns
            else pd.Series(0, index=work.index)
        ).fillna(0)
        if float(eng.max() or 0) > 1.5:
            eng = eng / sess.replace(0, np.nan).fillna(1)
        intensity = (freq * 3.0) + (prev * 1.5) + (eng.clip(0, 1) * 0.8) + (1.0 / sess)
        days = 150.0 / (1.0 + intensity)
        days = np.where(prev + freq > 0, days.clip(7, 120), np.maximum(days, 120).clip(120, 180))
        work["_p4_proxy_days"] = pd.Series(days, index=work.index).astype(float)
        if "days_until_next_purchase" not in work.columns:
            work["days_until_next_purchase"] = work["_p4_proxy_days"]
            notes.append(
                "days_until_next_purchase proxy from purchase frequency / "
                "prior purchases / engagement (no true repurchase-interval label)"
            )

        if bundle is not None:
            from pipeline.purchase_history import apply_longitudinal_labels

            work = apply_longitudinal_labels(work, bundle, problem_id=4, notes=notes)

        if "days_until_next_purchase" in work.columns:
            work["days_until_next_purchase"] = _to_numeric(work["days_until_next_purchase"])
            if "_p4_proxy_days" in work.columns:
                work["days_until_next_purchase"] = work["days_until_next_purchase"].fillna(
                    work["_p4_proxy_days"]
                )
            work["days_until_next_purchase"] = work["days_until_next_purchase"].fillna(
                work["days_until_next_purchase"].median()
            )
        work = work.drop(columns=["_p4_proxy_days"], errors="ignore")

        if "cart_abandonment_rate" not in work.columns:
            atc = _find_col(work, "addToCarts", "itemsAddedToCart", "addToCartsPerUser")
            chk = _find_col(work, "checkouts", "checkoutsPerUser")
            if atc is not None:
                atc_v = _to_numeric(work[atc]).fillna(0)
                chk_v = _to_numeric(work[chk]).fillna(0) if chk else 0
                work["cart_abandonment_rate"] = np.where(
                    atc_v > 0,
                    (atc_v - chk_v).clip(lower=0) / atc_v.replace(0, np.nan),
                    0,
                )
                work["cart_abandonment_rate"] = _to_numeric(work["cart_abandonment_rate"]).fillna(0)
                notes.append("Built cart_abandonment_rate")

        # tenure_days from date span when absolute date ints (YYYYMMDD) exist
        if "tenure_days" not in work.columns or (work["tenure_days"] == 0).all():
            date_col = _find_col(work, "date", "eventDate", "event_date")
            if date_col is not None:
                d = _to_numeric(work[date_col]).fillna(0)
                # GA4 style YYYYMMDD integers
                if d.max() > 20_000_000:
                    try:
                        dt = pd.to_datetime(d.astype(int).astype(str), format="%Y%m%d", errors="coerce")
                        min_d = dt.min()
                        work["tenure_days"] = (dt - min_d).dt.days.fillna(0).clip(lower=0)
                        notes.append(f"tenure_days from {date_col} span")
                    except Exception:
                        work["tenure_days"] = 0
                elif "tenure_days" not in work.columns:
                    work["tenure_days"] = 0

        # Category target stats — original groupby mean/std (dense, SXI-safe)
        cat = _find_col(work, "itemCategory", "category", "item_category")
        if (
            cat
            and "days_until_next_purchase" in work.columns
            and "cat_mean_target" not in work.columns
        ):
            days_v = _to_numeric(work["days_until_next_purchase"])
            g = work.assign(_days=days_v).groupby(work[cat])["_days"]
            work["cat_mean_target"] = g.transform("mean")
            work["cat_std_target"] = g.transform("std").fillna(0)
            notes.append("Built cat_mean_target / cat_std_target (category group mean/std)")

        if cat and (
            "unique_categories" not in work.columns
            or (_to_numeric(work["unique_categories"]).fillna(0) == 0).all()
        ):
            uid = _find_col(work, "user_pseudo_id", "userPseudoId", "customer_id")
            if uid:
                work["unique_categories"] = work.groupby(work[uid])[cat].transform("nunique")
            else:
                work["unique_categories"] = work[cat].notna().astype(float)
            notes.append("Built unique_categories from item category")

        for fill_zero in (
            "tenure_days",
            "purchase_frequency",
            "avg_order_value",
            "unique_categories",
            "cat_mean_target",
            "cat_std_target",
            "previous_purchases",
            "cart_abandonment_rate",
            "viewsPerUser",
            "engagement_time_msec",
            "session_engaged",
            "purchaseRevenue",
            "sessionsPerUser",
        ):
            if fill_zero not in work.columns:
                work[fill_zero] = 0

        if "is_repurchase_ready" not in work.columns and "days_until_next_purchase" in work.columns:
            days_s = _to_numeric(work["days_until_next_purchase"]).fillna(999)
            if "cat_mean_target" in work.columns:
                expected = _to_numeric(work["cat_mean_target"]).fillna(days_s.median())
            else:
                expected = pd.Series(float(days_s.median()), index=work.index)
            trigger = (expected * 0.85).clip(lower=1)
            work["is_repurchase_ready"] = (days_s <= trigger).astype(int)
            notes.append("is_repurchase_ready derived from days_until_next_purchase vs cat trigger (T50×0.85)")

        result = _finalize(work, problem, notes, allow_partial=True)
        if bundle is not None:
            result.data_quality = bundle.quality.as_dict()
            if result.df is not None and "target_source" in result.df.columns and result.df["target_source"].notna().any():
                result.target_source = str(result.df["target_source"].dropna().astype(str).iloc[0])
            elif bundle.quality.target_source:
                result.target_source = bundle.quality.target_source
        return result

    if pid == 5:
        work = raw.copy()
        # Map raw behaviour aliases
        raw_map = {
            "bounce_rate": ["bounce_rate", "bounceRate"],
            "session_engaged_rate": ["session_engaged_rate", "engagedSessions", "engagementRate"],
            "engagement_rate": ["engagement_rate", "engagementRate"],
            "avg_scroll_depth": ["avg_scroll_depth", "percentScrolled", "scroll_depth"],
            "total_purchases": ["total_purchases", "purchases"],
            "previous_purchases": ["previous_purchases"],
            "is_buyer": ["is_buyer"],
            "returning_user": ["returning_user", "returningUsers"],
            "cart_abandonment_total": ["cart_abandonment_total", "cart_abandonment"],
            "promo_clicks": ["promo_clicks", "sessionCampaign"],
            "search_clicks": ["search_clicks", "searchTerm"],
            "total_sessions": ["total_sessions", "sessions", "sessionsPerUser"],
            "events_per_session": ["events_per_session", "eventCount"],
            "event_count": ["event_count", "eventCount"],
            "engagement_time_msec": ["engagement_time_msec", "userEngagementDuration"],
            "avg_session_duration": ["avg_session_duration", "averageSessionDuration"],
            "total_revenue": ["total_revenue", "totalRevenue", "purchaseRevenue"],
            # Accept only an explicit low-activity column — never map churn aliases
            # (P7 is_churned / ground_truth_churn are different problems).
            "is_low_activity": [
                "is_low_activity",
                "low_activity",
            ],
        }
        for dest, cands in raw_map.items():
            if dest in work.columns:
                continue
            src = _find_col(work, *cands)
            if src is None:
                continue
            if dest in ("promo_clicks", "search_clicks", "returning_user") and not pd.api.types.is_numeric_dtype(work[src]):
                work[dest] = work[src].notna().astype(int)
            else:
                work[dest] = _to_numeric(work[src]).fillna(0)
            notes.append(f"Mapped {dest} ← {src}")

        work = _ensure_is_buyer(work, notes) if "is_buyer" not in work.columns else work
        if "is_buyer" in work.columns:
            work["is_buyer"] = _ensure_binary_target(work, "is_buyer")

        # Derived scores
        sess = _to_numeric(work.get("total_sessions", pd.Series(0, index=work.index))).fillna(0)
        purch = _to_numeric(work.get("total_purchases", pd.Series(0, index=work.index))).fillna(0)
        rev = _to_numeric(work.get("total_revenue", pd.Series(0, index=work.index))).fillna(0)
        eng_t = _to_numeric(work.get("engagement_time_msec", pd.Series(0, index=work.index))).fillna(0)
        scroll = _to_numeric(work.get("avg_scroll_depth", pd.Series(0, index=work.index))).fillna(0)
        bounce = _to_numeric(work.get("bounce_rate", pd.Series(0, index=work.index))).fillna(0)
        eng_r = _to_numeric(work.get("engagement_rate", pd.Series(0, index=work.index))).fillna(0)
        promo = _to_numeric(work.get("promo_clicks", pd.Series(0, index=work.index))).fillna(0)
        abandon = _to_numeric(work.get("cart_abandonment_total", pd.Series(0, index=work.index))).fillna(0)

        def _scale(s: pd.Series) -> pd.Series:
            mx = float(s.max()) if len(s) else 0.0
            if mx <= 0:
                return s * 0.0
            return s / mx

        work["session_volume_score"] = _scale(sess)
        work["frequency_score"] = _scale(purch / sess.replace(0, np.nan).fillna(1))
        work["monetary_score"] = _scale(rev)
        work["engagement_depth"] = _scale(eng_t * (1 + scroll) / sess.replace(0, np.nan).fillna(1))
        work["behavioral_degradation"] = _scale(bounce + (1 - eng_r.clip(0, 1)))
        work["promo_sensitivity"] = _scale(promo / (purch.replace(0, np.nan).fillna(1)))
        work["cart_abandon_severity"] = _scale(abandon)
        work["conversion_efficiency"] = _scale(purch / sess.replace(0, np.nan).fillna(1))
        notes.append("Built 8 activity / engagement derived scores")

        # Retention / intervention signals — pass-through for timing
        work = _build_p5_retention_signals(work, notes)

        # Fill remaining raw defaults
        for col in problem["features"]:
            if col not in work.columns:
                work[col] = 0

        # Always build leakage-safe recency-only label (engagement/bounce stay features only).
        had_upload_label = "is_low_activity" in work.columns
        work["is_low_activity"] = _derive_low_activity_proxy(work, notes)
        work["_target_source"] = "derived_recency"
        notes.append("target_source=derived_recency (is_low_activity from recency_days > median)")
        if had_upload_label:
            notes.append(
                "Replaced upload is_low_activity with recency-only label (leakage-safe)"
            )

        # Alias for older dashboard / lifecycle code paths
        work["ground_truth_churn"] = work["is_low_activity"].astype(int)
        rec = _to_numeric(work.get("recency_days", pd.Series(0, index=work.index))).fillna(0)
        if bool(rec.nunique(dropna=True) > 1 and float(rec.max()) > 0):
            work["activity_label_threshold_days"] = float(rec.median())
        n_low = int(work["is_low_activity"].sum())
        notes.append(
            f"P5 quiet traffic: {len(work)} rows · is_low_activity=1 for {n_low} "
            f"({100.0 * n_low / max(len(work), 1):.1f}%) — label=recency; "
            f"goal: lift engagement / session length"
        )

        return _finalize(work, problem, notes)

    if pid == 6:
        # SKU aggregate + time-split future demand; optional ERP inventory merge
        work = _build_p6_sku_demand_frame(raw, notes, inventory_df=inventory_df)
        return _finalize(work, problem, notes, allow_partial=True)

    if pid == 7:
        work = _build_p7_inactivity_churn_frame(raw, notes)
        return _finalize(work, problem, notes, allow_partial=True)

    if pid == 8:
        work = _build_p8_promo_sensitivity_frame(raw, notes)
        return _finalize(work, problem, notes, allow_partial=True)

    raise FeatureGapError(pid, [], message=f"No FE handler for problem {pid}")


def _scale_safe(s: pd.Series) -> pd.Series:
    s = _to_numeric(s).fillna(0)
    mx = float(s.max()) if len(s) else 0.0
    if mx <= 0:
        return s * 0.0
    return s / mx


def _build_p7_inactivity_churn_frame(raw: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """
    Problem 7: inactivity churn from last activity date (churn_cust.py).

    customer_churn / is_churned = 1 when a user's latest valid date is more
    than 60 days before the latest valid date in the file. All upload rows kept.
    """
    from pipeline.p7_churn import CHURN_THRESHOLD_DAYS, create_customer_churn_feature

    work = raw.copy()
    n_raw = int(len(work))
    event_dt = _parse_event_dates(work)
    valid_rate = float(event_dt.notna().mean()) if len(event_dt) else 0.0
    if valid_rate < 0.3:
        raise FeatureGapError(
            7,
            ["date"],
            message=(
                "Problem 7 needs a usable date / event_timestamp / event_date "
                "column (YYYYMMDD or timestamp) to label 60-day inactivity churn."
            ),
        )

    try:
        work = create_customer_churn_feature(
            work, date_series=event_dt, threshold_days=CHURN_THRESHOLD_DAYS
        )
    except Exception as exc:
        raise FeatureGapError(
            7,
            ["date"],
            message=f"Problem 7 customer_churn failed: {exc}",
        ) from exc

    work["is_churned"] = _ensure_binary_target(work, "customer_churn")
    recency = pd.to_numeric(work.get("days_inactive"), errors="coerce")
    work["recency_days"] = recency.fillna(0).clip(lower=0)

    user_col = _find_col(
        work, "user_pseudo_id", "user_id", "userId", "clientId", "visitor_id"
    )
    if user_col is None:
        work["user_pseudo_id"] = [f"row_{i}" for i in range(len(work))]
    elif user_col != "user_pseudo_id":
        work["user_pseudo_id"] = work[user_col].astype(str)
    else:
        work["user_pseudo_id"] = work["user_pseudo_id"].astype(str)

    sess_col = _find_col(work, "sessions", "sessionsPerUser", "total_sessions")
    eng_t_col = _find_col(
        work,
        "userEngagementDuration",
        "averageUserEngagementDuration",
        "engagement_time",
        "averageSessionDuration",
        "engagement_time_msec",
    )
    eng_r_col = _find_col(
        work, "engagementRate", "engagement_rate", "sessionKeyEventRate", "session_engaged_rate"
    )
    bounce_col = _find_col(work, "bounceRate", "bounce_rate")
    scroll_col = _find_col(work, "percentScrolled", "scroll_depth", "avg_scroll_depth")
    abandon_col = _find_col(work, "cart_abandonment", "cartAbandonmentRate", "cart_abandonment_total")
    promo_col = _find_col(work, "promo_clicks", "sessionCampaign")
    search_col = _find_col(work, "search_clicks", "searchTerm")
    atc_col = _find_col(work, "addToCarts", "add_to_carts", "itemsAddedToCart")
    purch_col = _find_col(work, "purchases", "ecommercePurchases", "total_purchases", "itemsPurchased")
    prev_col = _find_col(work, "previous_purchases", "previousPurchases")
    aov_col = _find_col(work, "averagePurchaseRevenue", "avg_order_value", "aov")
    eps_col = _find_col(work, "eventsPerSession", "eventCount", "events_per_session")
    device_col = _find_col(work, "deviceCategory", "device", "platform")
    channel_col = _find_col(
        work,
        "sessionPrimaryChannelGroup",
        "firstUserPrimaryChannelGroup",
        "sessionSourceMedium",
        "sessionSource",
    )
    freq_col = _find_col(work, "purchase_frequency", "purchasesPerUser")

    work["total_sessions"] = _to_numeric(work[sess_col]).fillna(0) if sess_col else 0.0
    work["engagement_time"] = _to_numeric(work[eng_t_col]).fillna(0) if eng_t_col else 0.0
    work["engagement_rate"] = _to_numeric(work[eng_r_col]).fillna(0) if eng_r_col else 0.0
    work["session_engaged_rate"] = work["engagement_rate"]
    work["bounce_rate"] = _to_numeric(work[bounce_col]).fillna(0) if bounce_col else 0.0
    work["scroll_depth"] = _to_numeric(work[scroll_col]).fillna(0) if scroll_col else 0.0
    work["cart_abandonment"] = _to_numeric(work[abandon_col]).fillna(0) if abandon_col else 0.0
    if promo_col and not pd.api.types.is_numeric_dtype(work[promo_col]):
        work["promo_clicks"] = work[promo_col].notna().astype(int)
    else:
        work["promo_clicks"] = _to_numeric(work[promo_col]).fillna(0) if promo_col else 0.0
    if search_col and not pd.api.types.is_numeric_dtype(work[search_col]):
        work["search_clicks"] = work[search_col].notna().astype(int)
    else:
        work["search_clicks"] = _to_numeric(work[search_col]).fillna(0) if search_col else 0.0
    work["add_to_carts"] = _to_numeric(work[atc_col]).fillna(0) if atc_col else 0.0
    work["total_purchases"] = _to_numeric(work[purch_col]).fillna(0) if purch_col else 0.0
    work["previous_purchases"] = (
        _to_numeric(work[prev_col]).fillna(0) if prev_col else work["total_purchases"]
    )
    work["avg_order_value"] = _to_numeric(work[aov_col]).fillna(0) if aov_col else 0.0
    work["events_per_session"] = _to_numeric(work[eps_col]).fillna(0) if eps_col else 0.0
    sess = pd.to_numeric(work["total_sessions"], errors="coerce").replace(0, pd.NA)
    purch = pd.to_numeric(work["total_purchases"], errors="coerce").fillna(0)
    work["purchase_frequency"] = (
        _to_numeric(work[freq_col]).fillna(0)
        if freq_col
        else (purch / sess).fillna(0)
    )
    work["device_enc"] = _encode_series(work[device_col]) if device_col else 0.0
    work["acq_channel_enc"] = _encode_series(work[channel_col]) if channel_col else 0.0
    work["target_source"] = "inactivity_60d"

    n_pos = int(work["is_churned"].sum())
    n_users = int(work["user_pseudo_id"].nunique())
    notes.append(
        f"P7 inactivity churn (churn_cust.py): {len(work)} rows from {n_raw} upload "
        f"({n_users} users); date coverage={valid_rate:.0%}; "
        f"is_churned = days_inactive > {CHURN_THRESHOLD_DAYS}d vs latest file date "
        f"({n_pos} positives, {100.0 * n_pos / max(len(work), 1):.1f}%); "
        f"invalid dates labelled 0; full rows kept."
    )
    notes.append("target_source=inactivity_60d")
    return work


def _attach_p7_pass_through(
    final: pd.DataFrame, source: pd.DataFrame, notes: list[str]
) -> pd.DataFrame:
    """Keep inactivity audit columns on FE CSV for the P7 dashboard."""
    out = final.copy()
    attached: list[str] = []
    for col in (
        "customer_churn",
        "days_inactive",
        "last_active_date",
        "inactivity_threshold_days",
        "target_source",
        "user_pseudo_id",
    ):
        if col not in source.columns or col in out.columns:
            continue
        try:
            series = source.loc[out.index, col]
        except Exception:
            continue
        if col in {"target_source", "last_active_date", "user_pseudo_id"}:
            out[col] = series.astype(str)
        else:
            out[col] = _to_numeric(series).fillna(0)
        attached.append(col)
    if attached:
        notes.append(f"P7 pass-through: {', '.join(attached)}")
    return out


# P8 uses pipeline.promotion_sensitivity (any valid sessionCampaign; no fixed promo list).
P8_PROMO_TARGET_THRESHOLD = 0.70


def _build_p8_promo_sensitivity_frame(raw: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """
    Problem 8 panel using the shared promotion_sensitivity logic:

      promotion_sensitivity
        = (promo purchases + promo cart additions)
          / (total purchases + total cart additions)
        → 0–1 score (display as %). Example: 3 of 6 acts → 0.50 (50%).

    - Any valid sessionCampaign counts (no fixed PROMO list).
    - Only empty / (not set) / null-like values are excluded from the numerator.
    - User-level score is merged onto every row (full upload kept for SXI).
    - Campaign columns and intermediate calc columns are dropped.
    - Regression target: promotion_sensitivity (continuous 0–1).
    - Business flag is_promo_sensitive = 1 when score ≥ 0.70 (discounts only above 70%).
    - Locked training features are behavioural/demographic only (not the score itself).
    """
    from pipeline.promotion_sensitivity import create_promotion_sensitivity_feature

    work = raw.copy()
    n_raw = int(len(work))

    try:
        enriched = create_promotion_sensitivity_feature(work)
    except Exception as exc:
        raise FeatureGapError(
            8,
            ["sessionCampaign", "purchases", "addToCarts"],
            message=f"Problem 8 promotion_sensitivity failed: {exc}",
        ) from exc

    if "promotion_sensitivity" not in enriched.columns:
        raise FeatureGapError(
            8,
            ["promotion_sensitivity"],
            message="Problem 8: promotion_sensitivity was not created.",
        )

    # Identity for CRM / pass-through
    user_col = _find_col(enriched, "user_pseudo_id", "user_id", "userId", "clientId", "visitor_id")
    if user_col is None:
        enriched["user_pseudo_id"] = [f"row_{i}" for i in range(len(enriched))]
    elif user_col != "user_pseudo_id":
        enriched["user_pseudo_id"] = enriched[user_col].astype(str)
    else:
        enriched["user_pseudo_id"] = enriched["user_pseudo_id"].astype(str)

    sens = _to_numeric(enriched["promotion_sensitivity"]).fillna(0.0).clip(0, 1)
    enriched["promotion_sensitivity"] = sens
    # Alias for dashboards that still read promo_sensitivity
    enriched["promo_sensitivity"] = sens

    # Business targeting flag (pass-through — not the SXI regression target).
    thr = P8_PROMO_TARGET_THRESHOLD
    enriched["is_promo_sensitive"] = (sens >= thr).astype(int)
    enriched["promo_label_threshold"] = thr
    enriched["target_source"] = "derived_campaign"

    # Behavioural / demographic predictors from remaining columns
    sess_col = _find_col(enriched, "sessions", "sessionsPerUser", "total_sessions")
    eng_col = _find_col(enriched, "engagementRate", "engagement_rate", "sessionKeyEventRate")
    bounce_col = _find_col(enriched, "bounceRate", "bounce_rate")
    eng_t_col = _find_col(
        enriched,
        "userEngagementDuration",
        "averageUserEngagementDuration",
        "engagement_time",
        "averageSessionDuration",
    )
    scroll_col = _find_col(enriched, "percentScrolled", "scroll_depth", "avg_scroll_depth")
    device_col = _find_col(enriched, "deviceCategory", "device", "platform")
    channel_col = _find_col(
        enriched,
        "sessionPrimaryChannelGroup",
        "firstUserPrimaryChannelGroup",
        "sessionSourceMedium",
        "sessionSource",
    )
    age_col = _find_col(enriched, "age", "age_group", "ageGroup")
    gender_col = _find_col(enriched, "gender", "sex")
    city_col = _find_col(enriched, "city", "metro", "region")
    views_col = _find_col(enriched, "viewsPerSession", "screenPageViewsPerSession", "views_per_session")
    eps_col = _find_col(enriched, "eventsPerSession", "eventCount", "events_per_session")

    enriched["total_sessions"] = _to_numeric(enriched[sess_col]).fillna(0) if sess_col else 0.0
    enriched["engagement_rate"] = _to_numeric(enriched[eng_col]).fillna(0) if eng_col else 0.0
    enriched["bounce_rate"] = _to_numeric(enriched[bounce_col]).fillna(0) if bounce_col else 0.0
    enriched["engagement_time"] = _to_numeric(enriched[eng_t_col]).fillna(0) if eng_t_col else 0.0
    enriched["scroll_depth"] = _to_numeric(enriched[scroll_col]).fillna(0) if scroll_col else 0.0
    enriched["views_per_session"] = _to_numeric(enriched[views_col]).fillna(0) if views_col else 0.0
    enriched["events_per_session"] = _to_numeric(enriched[eps_col]).fillna(0) if eps_col else 0.0
    enriched["device_enc"] = _encode_series(enriched[device_col]) if device_col else 0.0
    enriched["acq_channel_enc"] = _encode_series(enriched[channel_col]) if channel_col else 0.0
    enriched["age_enc"] = _encode_series(enriched[age_col]) if age_col else 0.0
    enriched["gender_enc"] = _encode_series(enriched[gender_col]) if gender_col else 0.0
    enriched["city_enc"] = _encode_series(enriched[city_col]) if city_col else 0.0

    n_pos = int(enriched["is_promo_sensitive"].sum())
    n_users = int(enriched["user_pseudo_id"].nunique())
    notes.append(
        f"P8 promotion_sensitivity (shared logic): {len(enriched)} rows from {n_raw} upload "
        f"({n_users} users); any valid sessionCampaign counts (no fixed promo list); "
        f"only empty/(not set)/null-like excluded; "
        f"score=(promo purch+ATC)/(total purch+ATC) at user level, merged to all rows; "
        f"campaign + intermediate cols removed; "
        f"is_promo_sensitive = promotion_sensitivity ≥ {thr:.0%} "
        f"({n_pos} rows flagged for promo budget, {100.0 * n_pos / max(len(enriched), 1):.1f}%); "
        f"SXI target = promotion_sensitivity (regression); "
        f"locked features = behaviour/demographics only (score is pass-through)."
    )
    notes.append("target_source=derived_campaign")
    return enriched


def _attach_p8_pass_through(
    final: pd.DataFrame, source: pd.DataFrame, notes: list[str]
) -> pd.DataFrame:
    """Keep audit columns for P8 dashboard (not locked training features)."""
    out = final.copy()
    attached: list[str] = []
    for col in (
        "promotion_sensitivity",
        "promo_sensitivity",
        "target_source",
        "promo_label_threshold",
        "is_promo_sensitive",
    ):
        if col not in source.columns or col in out.columns:
            continue
        try:
            series = source.loc[out.index, col]
        except Exception:
            continue
        if col == "target_source":
            out[col] = series.astype(str)
        elif col == "is_promo_sensitive":
            out[col] = _to_numeric(series).fillna(0).astype(int)
        else:
            out[col] = _to_numeric(series).fillna(0)
        attached.append(col)
    if attached:
        notes.append(f"P8 pass-through: {', '.join(attached)}")
    return out
