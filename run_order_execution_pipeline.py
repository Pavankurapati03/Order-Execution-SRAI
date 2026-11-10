"""
Master Order Execution Pipeline Runner.
Connects:
  - Stage 1: Sales Demand Forecasting (Meta Prophet)
  - Stage 2: ML-Powered Demand Planning (5 Core Targets + S&OP Inventory Plan)
  - Stage 3: Procurement Decision Engine (Purchase Order Generation with MOQ & Batching)

Outputs:
  - artifacts/demand_planning_results.csv
  - artifacts/purchase_order_recommendations.csv
  - artifacts/order_execution_summary.json
  - artifacts/plots/order_execution_inventory_flow.png
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.demand_planner import calculate_demand_plan
from src.procurement_agent import generate_purchase_order

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def main():
    print("================================================================================")
    print("       STAGE 1 -> STAGE 2 -> STAGE 3 ORDER EXECUTION PIPELINE                   ")
    print("       ML-Powered Demand Planning with 5 Core Target Variables                  ")
    print("================================================================================")
    
    artifacts_dir = "artifacts"
    plots_dir = os.path.join(artifacts_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # STAGE 1 OUTPUT INGESTION
    # -------------------------------------------------------------------------
    forecast_csv = os.path.join(artifacts_dir, "prophet_forecast_results.csv")
    if not os.path.exists(forecast_csv):
        print(f"Error: {forecast_csv} not found. Run Stage 1 forecasting first!")
        return
        
    forecast_df = pd.read_csv(forecast_csv)
    print(f"\n[STAGE 1 Output Loaded]: {len(forecast_df)} forecast rows from {forecast_csv}")
    
    # Fast load relevant columns from raw dataset for Stage 2 ML models & aggregations
    raw_csv_path = "fmcg_sales_3years_1M_rows.csv"
    raw_cols = [
        'date', 'sku_id', 'store_id', 'channel', 'units_sold', 'list_price', 
        'discount_pct', 'promo_flag', 'stock_on_hand', 'stock_out_flag', 
        'lead_time_days', 'supplier_id', 'purchase_cost', 'margin_pct'
    ]
    if os.path.exists(raw_csv_path):
        print(f"[Loading Raw SKU Telemetry]: Reading {raw_csv_path} for Stage 2 features...")
        raw_df = pd.read_csv(raw_csv_path, usecols=raw_cols)
    else:
        print("[Notice]: Raw CSV not found locally, using default historical parameters.")
        raw_df = None
    
    # -------------------------------------------------------------------------
    # STAGE 2: ML-POWERED DEMAND PLANNING ENGINE
    # -------------------------------------------------------------------------
    print("\n[STAGE 2]: Running ML-Powered Demand Planning Engine...")
    demand_plan = calculate_demand_plan(
        forecast_df=forecast_df,
        raw_df=raw_df,
        sku_id="SKU0001",
        target_supplier_id="S008",
        planning_horizon_days=30,      # Standard monthly S&OP planning cycle
        stock_on_hand=248,             # Current warehouse inventory snapshot
        in_transit=0,
        backorders=0,
        rmse_error=260.10,             # Stage 1 Prophet holdout error
        service_level_z=1.65           # 95% service level
    )
    
    print("\n" + "="*70)
    print("  STAGE 2: DEMAND PLANNING - 5 CORE TARGET VARIABLES")
    print("="*70)
    print(f"  1. global_plan_demand_quantity:  {demand_plan['global_plan_demand_quantity']:,.0f} units  (30-day S&OP horizon)")
    print(f"     [Raw Forecast: {demand_plan['raw_forecast_demand_quantity']:,.0f} | Trend Adj: {demand_plan['trend_adjustment']:.2f} (Ratio: {demand_plan['trend_ratio_30_90']:.2f}) | Buffer: +{demand_plan['planning_buffer_percent']}%\\]")
    print(f"  2. lead_time:                    {demand_plan['lead_time']} days  [{demand_plan['lead_time_source']}\\]")
    print(f"     [ML Regressor: {demand_plan['lead_time_ml_exact']}d | Median: {demand_plan['lead_time_median']:.0f}d | P90 SLA: {demand_plan['lead_time_p90_sla']:.0f}d | Mean: {demand_plan['lead_time_mean']:.2f}d\\]")
    print(f"  3. weightage_list_price:         EUR {demand_plan['weightage_list_price']:.2f} / unit  (Volume-weighted)")
    print(f"  4. vendor_count:                 {demand_plan['vendor_count']} approved active suppliers")
    print(f"  5. vendor_defect_rate:           {demand_plan['vendor_defect_rate_percent']:.2f}%  (Calibrated supplier quality)")
    print("-" * 70)
    print("  --- S&OP Inventory Positions & Financial Valuation ---")
    print(f"   * Global Plan Financial Valuation:  EUR {demand_plan['global_plan_valuation_eur']:,.2f}")
    print(f"   * Mean Daily Forecast Demand:       {demand_plan['mean_daily_forecast_demand']:,.2f} units/day")
    print(f"   * Lead-Time Demand (DLT):           {demand_plan['lead_time_demand_DLT']:,.2f} units")
    print(f"   * Safety Stock (SS @ 95% SL):       {demand_plan['safety_stock_SS']:,.0f} units")
    print(f"   * Reorder Point (ROP):              {demand_plan['reorder_point_ROP']:,.0f} units")
    print(f"   * Current Stock On Hand:            {demand_plan['stock_on_hand']} units (DoS: {demand_plan['days_of_supply_DoS']} days)")
    print(f"   * Net Replenishment (NRQ):          {demand_plan['net_replenishment_quantity_NRQ']:,.0f} units")
    print(f"   * Quality-Adjusted NRQ:             {demand_plan['quality_adjusted_nrq']:,.0f} units  (Compensates for defect rate)")
    print(f"   * Replenishment Triggered:          {demand_plan['reorder_triggered']}")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # STAGE 3: PROCUREMENT & MULTI-VENDOR REPLENISHMENT ENGINE
    # -------------------------------------------------------------------------
    print("\n[STAGE 3]: Running Advanced Procurement & Multi-Vendor Replenishment Engine...")
    purchase_order = generate_purchase_order(
        demand_plan=demand_plan,
        raw_df=raw_df,
        sku_id="SKU0001",
        sku_name="BrandA Soda",
        moq=500,
        batch_size=50
    )
    
    print("\n" + "="*70)
    print("  STAGE 3: PROCUREMENT & REPLENISHMENT - 4 CORE TARGET VARIABLES")
    print("="*70)
    print("  1. TARGET 1: PRICE (ML RandomForestRegressor Predicted Procurement Costs)")
    for sup, price in purchase_order["target_1_predicted_prices"].items():
        print(f"     * Supplier {sup}: EUR {price:.2f} / unit")
        
    print(f"\n  2. TARGET 2: ORDER_VALIDATION: {purchase_order['target_2_order_validation_flag']} ({purchase_order['target_2_order_validation_status']})")
    print(f"     [Condition: Inventory Position ({demand_plan['inventory_position']:.0f}) <= ROP ({demand_plan['reorder_point_ROP']:.0f}) AND Q > 0]")
    
    print("\n  3. TARGET 3: MULTIPLE VENDORS (MCDA Scoring & Volume Allocation)")
    print("     [Score = 40% Price + 30% LeadTime + 20% Reliability + 10% Margin]")
    for alloc in purchase_order["target_3_multi_vendor_allocations"]:
        print(f"     * {alloc['role']}: {alloc['supplier_id']}")
        print(f"       Score: {alloc['composite_score']:.4f} | Unit Cost: EUR {alloc['predicted_unit_cost']:.2f} | LT: {alloc['lead_time_days']:.1f}d | Rel: {alloc['reliability']*100:.1f}%")
        print(f"       Allocated Quantity: {alloc['allocated_quantity']:,} units ({alloc['allocation_share_percent']}%) | Cost: EUR {alloc['order_cost_eur']:,.2f}")
        
    print(f"\n  4. TARGET 4: PROCUREMENT_VALIDATION: {purchase_order['target_4_procurement_validation_flag']} ({purchase_order['target_4_procurement_validation_status']})")
    print("     [5-Point Audit Checks]:")
    for check_name, check_res in purchase_order["target_4_audit_checks"].items():
        print(f"     * {check_name}: {check_res}")
    print("-" * 70)
    print("  --- Master Purchase Order Summary ---")
    print(f"   * PO Number:                    {purchase_order['po_id']}")
    print(f"   * Total Replenishment Quantity: {purchase_order['recommended_po_order_quantity']:,} units")
    print(f"   * Blended Unit Cost:            EUR {purchase_order['blended_unit_purchase_cost']:.2f} / unit")
    print(f"   * Effective Selling Price:      EUR {purchase_order['effective_selling_price']:.2f} / unit")
    print(f"   * Total Order Value:            EUR {purchase_order['total_order_cost_eur']:,.2f}")
    print(f"   * Total Projected Gross Margin: EUR {purchase_order['total_projected_gross_margin_eur']:,.2f}")
    print(f"   * Overall PO Status:            {purchase_order['po_status']}")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # SAVE ARTIFACTS
    # -------------------------------------------------------------------------
    # Save Demand Planning CSV
    dp_df = pd.DataFrame([demand_plan])
    dp_csv_path = os.path.join(artifacts_dir, "demand_planning_results.csv")
    dp_df.to_csv(dp_csv_path, index=False)
    print(f"\n[Saved]: Stage 2 results -> {dp_csv_path}")
    
    # Save PO CSV (Multi-Vendor Allocations)
    po_df = pd.DataFrame(purchase_order["target_3_multi_vendor_allocations"])
    po_csv_path = os.path.join(artifacts_dir, "purchase_order_recommendations.csv")
    po_df.to_csv(po_csv_path, index=False)
    print(f"[Saved]: Stage 3 results -> {po_csv_path}")
    
    # Save Pipeline Summary JSON
    summary_json_path = os.path.join(artifacts_dir, "order_execution_summary.json")
    pipeline_summary = {
        "stage_1_sales_forecasting": {
            "model_used": "Meta Prophet (Adaptive)",
            "target_sku": "SKU0001",
            "holdout_wmape_percent": 11.18,
            "holdout_r2": 0.8693
        },
        "stage_2_demand_planning": demand_plan,
        "stage_3_procurement": purchase_order
    }
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(pipeline_summary, f, indent=2)
    print(f"[Saved]: Master pipeline summary -> {summary_json_path}")
        
    # -------------------------------------------------------------------------
    # GENERATE VISUAL INVENTORY FLOW CHART
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    
    categories = [
        'Current Stock\nOn-Hand', 
        'Lead-Time\nDemand (DLT)', 
        'Safety Stock\n(SS @ 95%)', 
        'Reorder Point\n(ROP)', 
        'Net Req.\n(NRQ)', 
        'Defect-Adjusted\nReq. (Adj NRQ)', 
        'Final PO Order\nQuantity (Q)'
    ]
    values = [
        demand_plan['stock_on_hand'],
        demand_plan['lead_time_demand_DLT'],
        demand_plan['safety_stock_SS'],
        demand_plan['reorder_point_ROP'],
        demand_plan['net_replenishment_quantity_NRQ'],
        demand_plan['quality_adjusted_nrq'],
        purchase_order['recommended_po_order_quantity']
    ]
    colors = ['#3182ce', '#d69e2e', '#e53e3e', '#dd6b20', '#38a169', '#805ad5', '#2b6cb0']
    
    bars = ax.bar(categories, values, color=colors, width=0.55, edgecolor='#2d3748', linewidth=1.2)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold')
                    
    subtitle = (
        f"Global Plan: {demand_plan['global_plan_demand_quantity']:,.0f} units | "
        f"ML Lead Time: {demand_plan['lead_time']}d | "
        f"Defect Rate: {demand_plan['vendor_defect_rate_percent']:.1f}% | "
        f"Vendors: {demand_plan['vendor_count']}"
    )
    ax.set_title("Order Execution Flow: Demand Planning & Procurement Waterfall (SKU0001)", fontsize=13, fontweight='bold', pad=18)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center', fontsize=9.5, color='#4a5568')
    ax.set_ylabel("Quantity (Units)", fontsize=11, labelpad=8)
    ax.set_ylim(0, max(values) * 1.18)
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "order_execution_inventory_flow.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[Saved]: Inventory flow plot -> {plot_path}")
    
    print("\n================================================================================")
    print("           ORDER EXECUTION PIPELINE EXECUTED SUCCESSFULLY!                      ")
    print("================================================================================")

if __name__ == "__main__":
    main()
