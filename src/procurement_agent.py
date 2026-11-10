"""
Stage 3: Advanced Procurement & Multi-Vendor Replenishment Engine.

Implements the 4 Core Stage 3 Target Variables:
  1. PRICE:
     RandomForestRegressor on purchase_cost -> predicted_procurement_price per supplier
  2. ORDER_VALIDATION:
     Binary gate: 1 (ORDER REQUIRED) if inventory_position <= ROP and Q > 0 else 0
  3. MULTIPLE VENDORS:
     MCDA Vendor Scoring (40% Price + 30% LeadTime + 20% Reliability + 10% Margin)
     -> Rank -> Filter eligible -> Allocate proportionally across top vendors
  4. PROCUREMENT_VALIDATION:
     5-point audit check: Order, Quantity, Vendor, Price, LeadTime -> 1 (VALID) or 0 (INVALID)
"""

import math
import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def predict_procurement_prices_ml(
    raw_df: pd.DataFrame,
    sku_id: str = "SKU0001",
    candidate_suppliers: Optional[List[str]] = None,
    order_quantity: float = 13150.0,
    month: int = 9,
    weekday: int = 5
) -> Dict[str, float]:
    """
    ML Price Predictor:
    Trains a RandomForestRegressor on historical transaction purchase_cost to dynamically
    predict expected unit procurement cost per supplier based on order size, timing, and vendor ID.
    """
    sku_df = raw_df[raw_df["sku_id"] == sku_id].copy()
    if len(sku_df) < 50:
        sku_df = raw_df.copy()

    sku_df["date"] = pd.to_datetime(sku_df["date"])
    sku_df["month"] = sku_df["date"].dt.month
    sku_df["weekday"] = sku_df["date"].dt.weekday

    le = LabelEncoder()
    sku_df["sup_enc"] = le.fit_transform(sku_df["supplier_id"].astype(str))

    features = ["units_sold", "month", "weekday", "sup_enc"]
    for col in features:
        if col not in sku_df.columns:
            sku_df[col] = 0.0

    X = sku_df[features].fillna(0)
    y = sku_df["purchase_cost"].fillna(3.75)

    rf = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    if candidate_suppliers is None:
        candidate_suppliers = sku_df["supplier_id"].value_counts().head(5).index.tolist()

    predicted_prices = {}
    for sup in candidate_suppliers:
        try:
            enc = le.transform([str(sup)])[0]
        except Exception:
            enc = 0
        test_in = pd.DataFrame([{
            "units_sold": order_quantity,
            "month": month,
            "weekday": weekday,
            "sup_enc": enc
        }])
        pred_cost = float(rf.predict(test_in)[0])
        predicted_prices[sup] = round(pred_cost, 2)

    return predicted_prices


def evaluate_order_validation(
    inventory_position: float,
    reorder_point: float,
    procurement_qty: float
) -> Dict[str, Any]:
    """
    Target 2: ORDER_VALIDATION
    IF inventory_position <= ROP AND procurement_qty > 0:
      -> 1 (ORDER REQUIRED)
    ELSE:
      -> 0 (NO ORDER)
    """
    order_required = bool(inventory_position <= reorder_point and procurement_qty > 0)
    flag = 1 if order_required else 0
    status_text = "ORDER REQUIRED" if flag == 1 else "NO ORDER"
    return {
        "order_validation_flag": flag,
        "order_validation_status": status_text,
        "inventory_position": inventory_position,
        "reorder_point": reorder_point,
        "procurement_qty": procurement_qty
    }


def score_and_allocate_vendors(
    raw_df: pd.DataFrame,
    sku_id: str = "SKU0001",
    total_quantity: int = 13150,
    moq: int = 500,
    batch_size: int = 50,
    candidate_suppliers: Optional[List[str]] = None,
    predicted_prices: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Target 3: MULTIPLE VENDORS
    Discovers all suppliers for SKU, computes MCDA composite score:
      Score = 40% Price + 30% LeadTime + 20% Reliability + 10% Margin
    Ranks suppliers, selects top-2 eligible, and allocates volume (e.g. 70% / 30%).
    """
    sku_df = raw_df[raw_df["sku_id"] == sku_id].copy()
    if candidate_suppliers is None:
        candidate_suppliers = sku_df["supplier_id"].value_counts().head(5).index.tolist()

    if predicted_prices is None:
        predicted_prices = {sup: 3.75 for sup in candidate_suppliers}

    vendor_stats = []
    for sup in candidate_suppliers:
        sub = sku_df[sku_df["supplier_id"] == sup]
        if len(sub) > 0:
            lt = float(sub["lead_time_days"].mean())
            rel = float(1.0 - sub["stock_out_flag"].mean())
            mrg = float(sub["margin_pct"].mean()) if "margin_pct" in sub.columns else 0.35
        else:
            lt = 6.5
            rel = 0.97
            mrg = 0.35

        pred_p = predicted_prices.get(sup, 3.75)
        vendor_stats.append({
            "supplier_id": sup,
            "predicted_price": pred_p,
            "lead_time": round(lt, 2),
            "reliability": round(rel, 4),
            "margin_pct": round(mrg, 4)
        })

    v_df = pd.DataFrame(vendor_stats)

    # Normalization (0 to 1 scale)
    min_p, max_p = v_df["predicted_price"].min(), v_df["predicted_price"].max()
    p_denom = max_p - min_p if max_p != min_p else 1.0
    v_df["score_price"] = 1.0 - (v_df["predicted_price"] - min_p) / p_denom

    min_lt, max_lt = v_df["lead_time"].min(), v_df["lead_time"].max()
    lt_denom = max_lt - min_lt if max_lt != min_lt else 1.0
    v_df["score_lead_time"] = 1.0 - (v_df["lead_time"] - min_lt) / lt_denom

    min_rel, max_rel = v_df["reliability"].min(), v_df["reliability"].max()
    rel_denom = max_rel - min_rel if max_rel != min_rel else 1.0
    v_df["score_reliability"] = (v_df["reliability"] - min_rel) / rel_denom

    min_m, max_m = v_df["margin_pct"].min(), v_df["margin_pct"].max()
    m_denom = max_m - min_m if max_m != min_m else 1.0
    v_df["score_margin"] = (v_df["margin_pct"] - min_m) / m_denom

    # Weighted Composite Score = 40% Price + 30% LeadTime + 20% Reliability + 10% Margin
    v_df["composite_score"] = (
        0.40 * v_df["score_price"] +
        0.30 * v_df["score_lead_time"] +
        0.20 * v_df["score_reliability"] +
        0.10 * v_df["score_margin"]
    )

    ranked_df = v_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    ranked_df["rank"] = range(1, len(ranked_df) + 1)

    # Select top-2 vendors for split allocation (70% primary, 30% secondary)
    top_primary = ranked_df.iloc[0]
    top_secondary = ranked_df.iloc[1] if len(ranked_df) > 1 else top_primary

    primary_raw_qty = total_quantity * 0.70
    primary_qty = int(math.ceil(primary_raw_qty / batch_size) * batch_size)
    primary_qty = max(primary_qty, moq)

    secondary_qty = total_quantity - primary_qty
    if secondary_qty < moq and total_quantity > moq:
        secondary_qty = int(math.ceil(secondary_qty / batch_size) * batch_size)
        if secondary_qty < moq:
            secondary_qty = moq
            primary_qty = max(moq, total_quantity - secondary_qty)

    allocations = [
        {
            "supplier_id": top_primary["supplier_id"],
            "role": "Primary Supplier (Rank 1)",
            "composite_score": round(float(top_primary["composite_score"]), 4),
            "predicted_unit_cost": float(top_primary["predicted_price"]),
            "lead_time_days": float(top_primary["lead_time"]),
            "reliability": float(top_primary["reliability"]),
            "allocated_quantity": primary_qty,
            "allocation_share_percent": round(primary_qty / total_quantity * 100, 1),
            "order_cost_eur": round(primary_qty * float(top_primary["predicted_price"]), 2)
        },
        {
            "supplier_id": top_secondary["supplier_id"],
            "role": "Secondary Supplier (Rank 2)",
            "composite_score": round(float(top_secondary["composite_score"]), 4),
            "predicted_unit_cost": float(top_secondary["predicted_price"]),
            "lead_time_days": float(top_secondary["lead_time"]),
            "reliability": float(top_secondary["reliability"]),
            "allocated_quantity": secondary_qty,
            "allocation_share_percent": round(secondary_qty / total_quantity * 100, 1),
            "order_cost_eur": round(secondary_qty * float(top_secondary["predicted_price"]), 2)
        }
    ]

    return {
        "ranked_suppliers": ranked_df.to_dict(orient="records"),
        "allocations": allocations,
        "total_allocated_quantity": primary_qty + secondary_qty
    }


def audit_procurement_validation(
    order_validation_flag: int,
    total_quantity: int,
    allocations: List[Dict[str, Any]],
    list_price: float,
    moq: int = 500,
    max_tolerable_lead_time: float = 12.0
) -> Dict[str, Any]:
    """
    Target 4: PROCUREMENT_VALIDATION
    5 checks:
      1. Order Trigger Check: order_validation == 1
      2. Quantity Check: allocated_quantity >= MOQ for each vendor
      3. Vendor Check: selected vendors are active and scored > 0
      4. Price Check: unit cost < list_price (positive margin)
      5. Lead Time Check: lead_time <= max_tolerable_lead_time
    Returns 1 (VALID) or 0 (INVALID - REPLAN).
    """
    check_order = bool(order_validation_flag == 1)
    check_qty = bool(all(a["allocated_quantity"] >= moq for a in allocations))
    check_vendor = bool(len(allocations) > 0 and all(a["composite_score"] > 0 for a in allocations))
    check_price = bool(all(a["predicted_unit_cost"] < list_price for a in allocations))
    check_lead_time = bool(all(a["lead_time_days"] <= max_tolerable_lead_time for a in allocations))

    all_passed = bool(check_order and check_qty and check_vendor and check_price and check_lead_time)
    flag = 1 if all_passed else 0
    status_text = "VALID" if flag == 1 else "INVALID — REPLAN"

    return {
        "procurement_validation_flag": flag,
        "procurement_validation_status": status_text,
        "checks": {
            "check_order_trigger": "PASS" if check_order else "FAIL",
            "check_quantity_moq": "PASS" if check_qty else "FAIL",
            "check_vendor_eligibility": "PASS" if check_vendor else "FAIL",
            "check_price_profitability": "PASS" if check_price else "FAIL",
            "check_lead_time_compliance": "PASS" if check_lead_time else "FAIL"
        }
    }


def generate_purchase_order(
    demand_plan: Dict[str, Any],
    raw_df: Optional[pd.DataFrame] = None,
    sku_id: str = "SKU0001",
    sku_name: str = "BrandA Soda",
    candidate_suppliers: Optional[List[str]] = None,
    moq: int = 500,
    batch_size: int = 50,
    list_price: float = 6.24
) -> Dict[str, Any]:
    """
    Execute Master Stage 3 Procurement & Replenishment Decision Logic.
    
    Coordinates the 4 Target Variables:
      1. PRICE: ML predicted procurement cost per supplier
      2. ORDER_VALIDATION: Binary order trigger flag (1/0)
      3. MULTIPLE VENDORS: Ranking & Multi-vendor proportional split
      4. PROCUREMENT_VALIDATION: 5-point audit compliance gate (1/0)
    """
    nrq = demand_plan.get("quality_adjusted_nrq", demand_plan.get("net_replenishment_quantity_NRQ", 0.0))
    inventory_pos = demand_plan.get("inventory_position", 248.0)
    rop = demand_plan.get("reorder_point_ROP", 13163.0)
    effective_list_price = demand_plan.get("weightage_list_price", list_price)

    # 1. Base required quantity rounded to batch multiple
    if nrq <= 0:
        base_total_qty = 0
    else:
        base_quantity = max(nrq, float(moq))
        base_total_qty = int(math.ceil(base_quantity / batch_size) * batch_size)

    # 2. TARGET 2: ORDER_VALIDATION
    order_val = evaluate_order_validation(inventory_pos, rop, base_total_qty)

    # 3. TARGET 1: PRICE (ML RandomForestRegressor)
    if raw_df is not None:
        predicted_prices = predict_procurement_prices_ml(
            raw_df=raw_df,
            sku_id=sku_id,
            candidate_suppliers=candidate_suppliers,
            order_quantity=base_total_qty
        )
    else:
        predicted_prices = {"S014": 3.71, "S015": 3.73, "S028": 3.77, "S055": 3.87, "S057": 3.83}

    # 4. TARGET 3: MULTIPLE VENDORS (Ranking & Proportional Allocation)
    if raw_df is not None and base_total_qty > 0:
        multi_vendor_data = score_and_allocate_vendors(
            raw_df=raw_df,
            sku_id=sku_id,
            total_quantity=base_total_qty,
            moq=moq,
            batch_size=batch_size,
            candidate_suppliers=candidate_suppliers,
            predicted_prices=predicted_prices
        )
    else:
        multi_vendor_data = {
            "ranked_suppliers": [],
            "allocations": [
                {
                    "supplier_id": "S014",
                    "role": "Primary Supplier",
                    "composite_score": 0.654,
                    "predicted_unit_cost": 3.71,
                    "lead_time_days": 6.52,
                    "reliability": 0.969,
                    "allocated_quantity": base_total_qty,
                    "allocation_share_percent": 100.0,
                    "order_cost_eur": round(base_total_qty * 3.71, 2)
                }
            ],
            "total_allocated_quantity": base_total_qty
        }

    allocations = multi_vendor_data["allocations"]
    total_allocated = sum(a["allocated_quantity"] for a in allocations)
    total_order_cost = round(sum(a["order_cost_eur"] for a in allocations), 2)
    blended_unit_cost = round(total_order_cost / total_allocated, 2) if total_allocated > 0 else 0.0
    projected_gross_profit = round((total_allocated * effective_list_price) - total_order_cost, 2)

    # 5. TARGET 4: PROCUREMENT_VALIDATION (5-Point Audit Gate)
    audit_data = audit_procurement_validation(
        order_validation_flag=order_val["order_validation_flag"],
        total_quantity=total_allocated,
        allocations=allocations,
        list_price=effective_list_price,
        moq=moq
    )

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    primary_sup = allocations[0]["supplier_id"] if len(allocations) > 0 else "S014"
    primary_lt = int(round(allocations[0]["lead_time_days"])) if len(allocations) > 0 else 7
    expected_delivery = (datetime.date.today() + datetime.timedelta(days=primary_lt)).strftime("%Y-%m-%d")

    return {
        "po_id": f"PO-MULTI-{sku_id}-{today_str.replace('-', '')}",
        "po_date": today_str,
        "expected_delivery_date": expected_delivery,
        "sku_id": sku_id,
        "sku_name": sku_name,
        "net_replenishment_requirement_NRQ": nrq,
        "recommended_po_order_quantity": total_allocated,
        "effective_selling_price": effective_list_price,
        "blended_unit_purchase_cost": blended_unit_cost,
        "total_order_cost_eur": total_order_cost,
        "total_projected_gross_margin_eur": projected_gross_profit,
        "po_status": "PURCHASE_ORDER_ISSUED" if audit_data["procurement_validation_flag"] == 1 else "HOLD_FOR_REVIEW",

        # --- The 4 Core Stage 3 Targets ---
        "target_1_predicted_prices": predicted_prices,
        "target_2_order_validation_flag": order_val["order_validation_flag"],
        "target_2_order_validation_status": order_val["order_validation_status"],
        "target_3_multi_vendor_allocations": allocations,
        "target_3_ranked_suppliers": multi_vendor_data["ranked_suppliers"],
        "target_4_procurement_validation_flag": audit_data["procurement_validation_flag"],
        "target_4_procurement_validation_status": audit_data["procurement_validation_status"],
        "target_4_audit_checks": audit_data["checks"]
    }
