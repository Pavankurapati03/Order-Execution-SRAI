"""
Dashboard 4 — Friction vs Traction presentation layer.

Uses shared row-level overlap analysis from actual DXI `composite_dxi`
and `is_buyer` outputs. Does not invent DXI scores or synthetic visitor counts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent2.models import DxiRunResult
from agent2.utils.subindex_overlap import build_overlap_analyses
from agent2.utils.subindex_dxi import DASHBOARD_KEY, SUBINDEX_CATEGORIES


SECTION_DISPLAY = {
    "marketing": "Marketing",
    "demographic": "Demographic",
    "engagement": "Engagement",
    "kpi_behaviour": "KPI/Behaviour",
    "transaction": "Transaction",
}

SECTION_ACCENT = {
    "marketing": "#2563eb",
    "demographic": "#7c3aed",
    "engagement": "#d97706",
    "kpi_behaviour": "#059669",
    "transaction": "#0891b2",
}


def _f(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: Optional[float], digits: int = 1) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _status_from_net(net: float) -> str:
    if net >= 20:
        return "Strong Traction"
    if net >= 0:
        return "Balanced"
    if net >= -20:
        return "Friction Risk"
    return "High Friction"


def _factors_from_row(row: DxiRunResult) -> List[dict]:
    """Prefer stored top_features; fall back to linked execution feature_importance."""
    feats: List[dict] = []
    top_feats = row.top_features if isinstance(row.top_features, list) else []
    for feat in top_feats:
        if not isinstance(feat, dict):
            continue
        name = str(feat.get("feature") or feat.get("Feature") or "").strip()
        if not name:
            continue
        importance = _f(feat.get("importance") if "importance" in feat else feat.get("Importance"))
        feats.append(
            {
                "feature": name,
                "importance": importance,
                "rank": feat.get("rank"),
            }
        )
    if feats:
        return feats

    execution = getattr(row, "execution", None)
    raw = getattr(execution, "raw_output", None) if execution is not None else None
    if not isinstance(raw, dict):
        return []
    fi = raw.get("feature_importance") or {}
    current = fi.get("current") if isinstance(fi, dict) else None
    parsed: List[dict] = []
    if isinstance(current, list):
        for idx, item in enumerate(current, start=1):
            if not isinstance(item, dict):
                continue
            name = str(
                item.get("feature")
                or item.get("Feature")
                or item.get("name")
                or item.get("Base_Feature")
                or ""
            ).strip()
            if not name:
                continue
            importance = _f(
                item.get("importance")
                if "importance" in item
                else item.get("Importance")
            )
            parsed.append({"feature": name, "importance": importance, "rank": idx})
    elif isinstance(current, dict):
        for idx, (name, score) in enumerate(
            sorted(current.items(), key=lambda kv: _f(kv[1]) or 0.0, reverse=True),
            start=1,
        ):
            parsed.append({"feature": str(name), "importance": _f(score), "rank": idx})
    return parsed[:8]


def build_dashboard4_data(run_id: str) -> Dict[str, Any]:
    run_id = str(run_id or "").strip()
    rows = list(
        DxiRunResult.objects.filter(run_id=run_id, status="completed")
        .exclude(category="total")
        .select_related("execution")
    )
    by_cat = {str(r.category): r for r in rows}
    overlaps = build_overlap_analyses(run_id)
    overall_overlap = overlaps.get("overall") or {}
    subindex_overlap_by_cat = overlaps.get("by_category") or {}

    sections: List[Dict[str, Any]] = []
    factors: List[Dict[str, Any]] = []

    for category in SUBINDEX_CATEGORIES:
        row = by_cat.get(category)
        display = SECTION_DISPLAY.get(category, category.replace("_", " ").title())
        dash_key = DASHBOARD_KEY.get(category, category)
        accent = SECTION_ACCENT.get(category, "#2563eb")
        overlap = subindex_overlap_by_cat.get(category) or {}

        if not row:
            section = {
                "key": dash_key,
                "category": category,
                "name": display,
                "accent": accent,
                "current_dxi": None,
                "immediate_dxi": None,
                "mid_dxi": None,
                "long_dxi": None,
                "row_count": 0,
                "feature_count": 0,
                "traction_pct": 0.0,
                "friction_pct": 0.0,
                "net_traction": 0.0,
                "traction_users": 0,
                "friction_users": 0,
                "overall_users": 0,
                "buyer_count": 0,
                "high_dxi_users": 0,
                "high_dxi_buyers": 0,
                "high_dxi_non_buyers": 0,
                "low_dxi_buyers": 0,
                "low_dxi_non_buyers": 0,
                "validation": {"valid": False, "errors": ["Subindex result missing"]},
                "status": _status_from_net(0.0),
                "available": False,
            }
            sections.append(section)
            continue

        current = _f(row.current_dxi)
        immediate = _f(row.immediate_dxi)
        mid = _f(row.mid_dxi)
        long_dxi = _f(row.long_dxi) or immediate
        metrics = {
            "traction_pct": round(float(overlap.get("traction_pct") or 0.0), 2),
            "friction_pct": round(float(overlap.get("friction_pct") or 0.0), 2),
            "net_traction": round(float(overlap.get("net_traction") or 0.0), 2),
        }

        section = {
            "key": dash_key,
            "category": category,
            "name": display,
            "accent": accent,
            "current_dxi": _round(current, 2),
            "immediate_dxi": _round(immediate, 2),
            "mid_dxi": _round(mid, 2),
            "long_dxi": _round(long_dxi, 2),
            "row_count": int(overlap.get("total_users") or row.row_count or 0),
            "feature_count": int(row.feature_count or 0),
            **metrics,
            "traction_users": int(overlap.get("high_dxi_buyers") or 0),
            "friction_users": int(overlap.get("high_dxi_non_buyers") or 0),
            "overall_users": int(overlap.get("total_users") or row.row_count or 0),
            "buyer_count": int(overlap.get("buyer_count") or 0),
            "high_dxi_users": int(overlap.get("high_dxi_users") or 0),
            "high_dxi_buyers": int(overlap.get("high_dxi_buyers") or 0),
            "high_dxi_non_buyers": int(overlap.get("high_dxi_non_buyers") or 0),
            "low_dxi_buyers": int(overlap.get("low_dxi_buyers") or 0),
            "low_dxi_non_buyers": int(overlap.get("low_dxi_non_buyers") or 0),
            "validation": overlap.get("validation") or {"valid": False, "errors": ["Overlap unavailable"]},
            "status": _status_from_net(metrics["net_traction"]),
            "available": not bool(overlap.get("error")),
        }
        sections.append(section)

        for feat in _factors_from_row(row):
            name = str(feat.get("feature") or "").strip()
            if not name:
                continue
            importance = _f(feat.get("importance"))
            impact = None
            if importance is not None:
                impact = round(importance * 100.0, 1) if importance <= 1.0 else round(importance, 1)
            factors.append(
                {
                    "factor": name,
                    "section": display,
                    "section_key": dash_key,
                    "category": category,
                    "traction": metrics["traction_pct"],
                    "friction": metrics["friction_pct"],
                    "impact": impact,
                    "rank": feat.get("rank"),
                    "status": (
                        "Top Feature — High Traction Section"
                        if metrics["traction_pct"] >= metrics["friction_pct"]
                        else "Top Feature — High Friction Section"
                    ),
                }
            )

    available = [s for s in sections if s.get("available") and s.get("high_dxi_users", 0) > 0]
    overall_traction = round(float(overall_overlap.get("traction_pct") or 0.0), 2)
    overall_friction = round(float(overall_overlap.get("friction_pct") or 0.0), 2)

    strongest = max(available, key=lambda s: s["traction_pct"]) if available else None
    highest_friction = max(available, key=lambda s: s["friction_pct"]) if available else None

    spectrum = sorted(sections, key=lambda s: s["traction_pct"], reverse=True)
    ranking = sorted(
        sections,
        key=lambda s: (s["traction_pct"], s["net_traction"]),
        reverse=True,
    )
    for idx, item in enumerate(ranking, start=1):
        item["rank"] = idx

    # Key driver: highest-impact factor from available list
    key_driver = None
    if factors:
        scored = [f for f in factors if f.get("impact") is not None]
        key_driver = max(scored, key=lambda f: f["impact"]) if scored else factors[0]

    opportunity = min(available, key=lambda s: s["net_traction"]) if available else None

    insights = {
        "strongest_traction": {
            "section": (strongest or {}).get("name") or "—",
            "traction_pct": (strongest or {}).get("traction_pct"),
            "current_dxi": (strongest or {}).get("current_dxi"),
            "text": (
                f"{strongest['name']} currently leads with {strongest['traction_pct']}% traction "
                f"from {strongest['high_dxi_buyers']} high-DXI buyers out of {strongest['high_dxi_users']} high-DXI visitors."
                if strongest
                else "Subindex DXI results are not available yet."
            ),
        },
        "biggest_friction": {
            "section": (highest_friction or {}).get("name") or "—",
            "friction_pct": (highest_friction or {}).get("friction_pct"),
            "current_dxi": (highest_friction or {}).get("current_dxi"),
            "text": (
                f"{highest_friction['name']} shows the highest friction at "
                f"{highest_friction['friction_pct']}%, with {highest_friction['high_dxi_non_buyers']} high-DXI non-buyers."
                if highest_friction
                else "Subindex DXI results are not available yet."
            ),
        },
        "key_driver": {
            "factor": (key_driver or {}).get("factor") or "—",
            "section": (key_driver or {}).get("section") or "—",
            "impact": (key_driver or {}).get("impact"),
            "text": (
                f"{key_driver['factor']} in {key_driver['section']} is the top contributing factor "
                f"(impact {key_driver['impact']}%)."
                if key_driver and key_driver.get("impact") is not None
                else (
                    f"{key_driver['factor']} in {key_driver['section']} is a leading contributing factor."
                    if key_driver
                    else "Contributing factors are not available yet."
                )
            ),
        },
        "opportunity": {
            "section": (opportunity or {}).get("name") or "—",
            "net_traction": (opportunity or {}).get("net_traction"),
            "text": (
                f"{opportunity['name']} has the lowest net traction ({opportunity['net_traction']} pts) "
                f"and {opportunity['high_dxi_non_buyers']} high-DXI non-buyers, making it the clearest improvement opportunity."
                if opportunity
                else "Opportunity ranking requires completed subindex DXI runs."
            ),
        },
        "recommended_action": {
            "text": (
                f"Prioritize reducing friction in {opportunity['name']} while protecting the traction "
                f"strength of {strongest['name']}"
                + (
                    f", focusing first on {key_driver['factor']}."
                    if key_driver and key_driver.get("factor")
                    else "."
                )
                if opportunity and strongest
                else "Complete subindex DXI runs to unlock prioritized friction/traction actions."
            ),
        },
    }

    filter_options = [
        {"key": s["key"], "name": s["name"]}
        for s in sections
    ]

    return {
        "title": "Friction vs Traction",
        "subtitle": "Identify Where Digital Experience Accelerates Customer Outcomes And Where Friction Holds Them Back.",
        "run_id": run_id,
        "executive_kpis": {
            "overall_traction": overall_traction,
            "overall_friction": overall_friction,
            "high_dxi_buyers": int(overall_overlap.get("high_dxi_buyers") or 0),
            "high_dxi_non_buyers": int(overall_overlap.get("high_dxi_non_buyers") or 0),
            "strongest_traction": {
                "section": (strongest or {}).get("name") or "—",
                "value": (strongest or {}).get("traction_pct"),
            },
            "highest_friction": {
                "section": (highest_friction or {}).get("name") or "—",
                "value": (highest_friction or {}).get("friction_pct"),
            },
        },
        "overall_overlap": overall_overlap,
        "sections": sections,
        "spectrum": spectrum,
        "factors": factors,
        "filter_options": filter_options,
        "ranking": ranking,
        "insights": insights,
        "meta": {
            "sections_available": len(available),
            "sections_total": len(SUBINDEX_CATEGORIES),
            "source": "Shared row-level overlap results from overall and subindex fulldatarl CSVs + DxiRunResult.top_features",
            "note": "Traction % = High DXI buyers / High DXI visitors. Friction % = High DXI non-buyers / High DXI visitors. Low-DXI buyers are tracked separately as organic buyers.",
            "methodology": overlaps.get("methodology") or {},
        },
    }
