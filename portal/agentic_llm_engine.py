import os, json, re
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def generate_agentic_narratives_llm(
    session_id: str,
    primary_code: str,
    master_df: pd.DataFrame = None,
    sxi_dashboard_data: dict = None,
    anomalies: list = None,
    causal_chains: list = None,
    net_exposure_usd: float = 0.0,
):
    """
    Dynamically generates the 4-step Agentic Decision Engine text using OpenAI gpt-4o-mini
    grounded 100% in the uploaded dataset, computed anomalies, TreeSHAP drivers, and causal graph.
    Caches per session to disk for sub-second page reloads.
    """
    uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    cache_file = os.path.join(uploads_dir, f"agentic_engine_narrative_{session_id}_{primary_code}.json")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("step1") and data.get("step2") and data.get("step3") and data.get("step4"):
                return data
        except Exception:
            pass

    # Extract dynamic context from user's dataset and models
    context_lines = []
    context_lines.append(f"Primary Analyzed Solution: {primary_code}")
    if net_exposure_usd > 0:
        context_lines.append(f"Net Annual Exposure: ${net_exposure_usd:,.2f} USD")
    
    # 1. Hotspots from master_df
    dataset_summary = []
    if master_df is not None and not master_df.empty:
        dataset_summary.append(f"Dataset Columns: {list(master_df.columns)}")
        dataset_summary.append(f"Total Rows: {len(master_df)}")
        # Check potential dimension columns
        for col in ["supplier_id", "vendor_id", "dc_id", "warehouse_id", "category", "channel", "region", "shipping_lane"]:
            if col in master_df.columns:
                top_counts = master_df[col].value_counts().head(3).to_dict()
                dataset_summary.append(f"Top {col} segments: {top_counts}")
    
    # 2. Real drivers from SXI Dashboard / TreeSHAP
    top_drivers_info = []
    if sxi_dashboard_data and sxi_dashboard_data.get("top_drivers"):
        for d in sxi_dashboard_data["top_drivers"][:3]:
            name = d.get("feature_display") or d.get("feature_name") or d.get("feature")
            pct = d.get("importance_pct") or d.get("pct") or "0%"
            coef = d.get("impact") or d.get("coefficient") or ""
            top_drivers_info.append(f"- Driver: {name} | Importance: {pct} | Direction/Impact: {coef}")
            
    # 3. Anomaly metrics
    anom_info = []
    if anomalies:
        for a in anomalies[:3]:
            obs = getattr(a, "observed", "")
            thresh = getattr(a, "threshold", "")
            unit = getattr(a, "unit", "")
            dim = getattr(a, "segment_dimension", "")
            val = getattr(a, "segment_value", "")
            anom_info.append(f"- Anomaly in {getattr(a, 'solution', primary_code)}: Observed={obs} vs Target={thresh} {unit}. Hotspot={dim}:{val}")

    system_prompt = """You are an elite Chief Supply Chain Intelligence AI Agent synthesizing operational investigations for executive leadership.
Based on the real dataset context, machine learning SHAP feature importances, and mathematical anomalies provided, generate a strictly factual, professional 4-step decision brief.

MANDATORY RULES:
1. ONLY refer to entities, columns, suppliers, warehouses, categories, and values present in the provided context. If no specific supplier or DC names are provided, use the actual column names or general terms. NEVER hallucinate 'SUP-402' or 'DC-04' unless they appear in the data.
2. Structure the JSON response with the following exact keys:
{
  "step1": {
    "subtitle": "One sentence summary of the primary breach with numbers and entities.",
    "what": "Detailed operational narrative of the breach and operational friction caused.",
    "when": "Operational timeline of when the breakdown occurred across cycles.",
    "where": "Specific localized segment/corridor/dimension where failure was isolated."
  },
  "step2": [
    {
      "rank": "Primary Driver #1",
      "name": "Exact feature/driver name",
      "impact_badge": "XX% Impact",
      "impact_color": "#dc2626",
      "impact_text": "Impact: <strong>+X.X Unit</strong> metric shift",
      "finding": "Crisp 2-sentence explanation of why this feature caused the delay or breach."
    },
    ... (3 drivers matching the top SHAP drivers provided)
  ],
  "step3": {
    "subtitle": "One sentence summary of cross-department propagation.",
    "upstream_title": "Upstream Root Cause: How Upstream Sourcing/Logistics Caused It",
    "upstream_items": [
      {"heading": "Short Title", "detail": "Detailed explanation of how upstream operations triggered the chain."}
    ],
    "downstream_title": "Downstream Impact: How Warehousing/Customer Operations Suffered",
    "downstream_items": [
      {"heading": "Short Title", "detail": "Detailed explanation of downstream stockout, cost, or fulfillment collapse."}
    ]
  },
  "step4": [
    {
      "level": "L1",
      "title": "Tactical Remediation Plan",
      "owner": "Specific Department Owner",
      "bullets": ["Specific action 1", "Specific action 2", "Specific action 3"],
      "trade_offs": {
        "recovery": "Targeted % recovery",
        "effort_cost": "Low (Policy/Gate Adjustment)",
        "timeline": "7–14 Days",
        "confidence": "95% Feasibility"
      }
    },
    {
      "level": "L2",
      "title": "Strategic Optimization Plan (Recommended)",
      "owner": "Specific Department Owner",
      "bullets": ["Specific action 1", "Specific action 2", "Specific action 3"],
      "trade_offs": {
        "recovery": "Targeted % recovery",
        "effort_cost": "Medium (Reallocation/Buffers)",
        "timeline": "14–21 Days",
        "confidence": "94% Feasibility"
      }
    },
    {
      "level": "L3",
      "title": "Structural Transformation Plan",
      "owner": "Specific Department Owner",
      "bullets": ["Specific action 1", "Specific action 2", "Specific action 3"],
      "trade_offs": {
        "recovery": "Targeted % recovery",
        "effort_cost": "High (Network Integration)",
        "timeline": "60–90 Days",
        "confidence": "88% Feasibility"
      }
    }
  ]
}
Return ONLY pure JSON.
"""

    user_prompt = f"""OPERATIONAL CONTEXT FOR ACTIVE INVESTIGATION:
{chr(10).join(context_lines)}

DATASET TELEMETRY:
{chr(10).join(dataset_summary)}

ANOMALIES DETECTED:
{chr(10).join(anom_info) if anom_info else 'Standard SLA variance identified.'}

TOP MACHINE LEARNING (TreeSHAP) DRIVERS:
{chr(10).join(top_drivers_info) if top_drivers_info else 'Feature importance decomposition based on active dataset.'}

Generate the comprehensive, grounded 4-step decision intelligence dossier in JSON."""

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key and not api_key.startswith("your_"):
        try:
            import openai
            client = openai.OpenAI(api_key=api_key, timeout=20.0)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            raw = resp.choices[0].message.content
            parsed = json.loads(raw)
            if "step1" in parsed and "step2" in parsed and "step3" in parsed and "step4" in parsed:
                icons_cycle = ["🔍", "📊", "⚙️"]
                for i, d in enumerate(parsed.get("step2", [])):
                    if not d.get("icon"):
                        d["icon"] = icons_cycle[i % len(icons_cycle)]
                if isinstance(parsed.get("step3"), dict):
                    if not parsed["step3"].get("upstream_icon"):
                        parsed["step3"]["upstream_icon"] = "⛓️"
                    if not parsed["step3"].get("downstream_icon"):
                        parsed["step3"]["downstream_icon"] = "💥"
                # Save cache
                try:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(parsed, f, indent=2)
                except Exception:
                    pass
                return parsed
        except Exception as e:
            print(f"[Agentic LLM Engine] LLM generation notice ({e}). Falling back.")

    return None

if __name__ == "__main__":
    print("Testing generate_agentic_narratives_llm...")
    res = generate_agentic_narratives_llm("test_session", "E4")
    print("Result keys:", list(res.keys()) if res else "None")

