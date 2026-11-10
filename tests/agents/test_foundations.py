"""
Unit Tests for Phase 1: Foundations (Schemas, Adapters, Evidence, Config, LLM Client).
"""

import pytest
from portal.agents.config import (
    is_agentic_enabled,
    get_thresholds_config,
    get_causal_graph_config,
    get_semantic_glossary,
    get_financial_assumptions,
    get_llm_settings
)
from portal.agents.schemas import (
    Severity,
    EvidenceItem,
    SolutionResult,
    FeatureWeight,
    Anomaly,
    AnomalyReport,
    Claim,
    DecisionDossier,
    ConfidenceBreakdown
)
from portal.agents.adapters import adapt_sxi_to_solution_result
from portal.agents.evidence import EvidenceBuilder, EvidenceLedger
from portal.agents.llm.budget import BudgetTracker, BudgetExceeded
from portal.agents.llm.fake import FakeLLMClient


def test_config_loaders():
    """Verify all 4 YAML config files load correctly with required structure."""
    thresh = get_thresholds_config()
    assert thresh["strategy"] == "adaptive_statistical"
    assert "statistical_fences" in thresh
    assert thresh["statistical_fences"]["percentile_upper"] == 90.0
    assert "metric_direction" in thresh
    assert "E4" in thresh["metric_direction"]

    graph = get_causal_graph_config()
    assert "nodes" in graph
    assert "edges" in graph
    assert len(graph["edges"]) >= 10

    glossary = get_semantic_glossary()
    assert "terms" in glossary
    assert "safety_stock_days" in glossary["terms"]

    fin = get_financial_assumptions()
    assert "solution_financial_map" in fin
    assert "E4" in fin["solution_financial_map"]


def test_schemas_validation():
    """Verify Pydantic v2 schemas enforce validation and constraints."""
    item = EvidenceItem(
        id="EV-E4-001",
        solution="E4",
        kind="kpi",
        key="lead_time_days.baseline",
        value=8.4,
        unit="days",
        source="sxi_engine.E4.baseline_kpi"
    )
    assert item.id == "EV-E4-001"
    assert item.value == 8.4

    # Invalid Claim with empty evidence_ids should fail validation
    with pytest.raises(Exception):
        Claim(text="Delivery lead time is 8.4 days", evidence_ids=[])


def test_adapter_sxi_to_solution_result():
    """Verify raw SXI result dictionary is safely adapted into SolutionResult."""
    mock_sxi = {
        "solution_code": "E4",
        "solution_name": "Reduce Delivery Lead Time",
        "solution_slug": "reduce-delivery-lead-time",
        "stage": "Demand Planning",
        "task_type": "regression",
        "target_name": "lead_time_days",
        "kpi_name": "Delivery Lead Time",
        "baseline_kpi": 8.4,
        "optimized_kpi": 5.1,
        "lvl1_kpi": "7.2",
        "lvl2_kpi": "6.1",
        "lvl3_kpi": "5.1",
        "current_sxi_score": 5.82,
        "target_sxi_score": 8.91,
        "model_performance": {
            "model_type": "XGBoost Regressor",
            "accuracy_pct": "98.2",
            "r2": "0.87",
            "rmse": "1.42",
            "train_samples": 24000,
            "test_samples": 6000
        },
        "weights_table": [
            {"feature": "safety_stock_days", "weight": 0.35, "direction": "decreasing", "actionable": True},
            {"feature": "carrier_rating", "weight": 0.25, "direction": "increasing", "actionable": True},
            {"feature": "leaked_feature", "weight": 0.65, "direction": "decreasing", "actionable": False}
        ]
    }

    sol_res = adapt_sxi_to_solution_result(mock_sxi)
    assert sol_res.solution == "E4"
    assert sol_res.target_column == "lead_time_days"
    assert sol_res.kpi_baseline == 8.4
    assert sol_res.kpi_optimized["L2"] == 6.1
    assert sol_res.holdout["accuracy"] == 0.982
    assert sol_res.holdout["r2"] == 0.87
    assert len(sol_res.features) == 3
    # Leaked feature > 60% should be flagged
    assert "leaked_feature" in sol_res.leakage_suspects


def test_evidence_builder_determinism():
    """Verify EvidenceBuilder generates stable, identical IDs and SHA256 hashes on repeated runs."""
    mock_sxi = {
        "solution_code": "E4",
        "target_name": "lead_time_days",
        "task_type": "regression",
        "baseline_kpi": 8.4,
        "lvl1_kpi": 7.2,
        "current_sxi_score": 5.8,
        "target_sxi_score": 8.9,
        "model_performance": {"accuracy_pct": 98.2, "train_samples": 1000, "test_samples": 200},
        "weights_table": [{"feature": "safety_stock_days", "weight": 0.35}]
    }

    sol_res_1 = adapt_sxi_to_solution_result(mock_sxi)
    sol_res_2 = adapt_sxi_to_solution_result(mock_sxi)

    ledger_1 = EvidenceBuilder.build_from_solutions([sol_res_1])
    ledger_2 = EvidenceBuilder.build_from_solutions([sol_res_2])

    assert len(ledger_1) > 0
    assert len(ledger_1) == len(ledger_2)
    # Identical IDs and content hash
    assert [i.id for i in ledger_1.all()] == [i.id for i in ledger_2.all()]
    assert ledger_1.content_hash() == ledger_2.content_hash()

    # Verify retrieval
    item = ledger_1.get("EV-E4-001")
    assert item is not None
    assert item.solution == "E4"


def test_budget_tracker_and_fake_llm():
    """Verify BudgetTracker tracks tokens, calculates cost, and FakeLLMClient operates without network."""
    budget = BudgetTracker(token_limit=1000)
    fake_client = FakeLLMClient(budget_tracker=budget)

    # Fake generation of Anomaly
    fake_anomaly = fake_client.generate_structured(
        prompt_text="Detect anomalies in E4",
        response_model=Anomaly
    )
    assert isinstance(fake_anomaly, Anomaly)
    assert budget.total_tokens > 0
    assert budget.cost_usd_estimate >= 0.0

    # Test BudgetExceeded exception
    with pytest.raises(BudgetExceeded):
        budget.record_usage(prompt_tokens=800, completion_tokens=500)
