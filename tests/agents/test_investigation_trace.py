"""
Comprehensive Unit Tests for Investigation Trace Agent & Tool Registry.

Validates:
1. AgentToolRegistry: Tool discovery, parameter validation, and atomic execution.
2. Investigation Trace Sequence: Deterministic multi-step investigation pipeline (Step Label -> Action -> Observation).
3. Step Trace Logging: Verifies full execution trace with timestamps and observations.
4. Decision Dossier Hand-off: Ensures final dossier is properly assembled for Step 5 Gate.
"""

import pytest
import pandas as pd
from portal.agents.investigation_trace import InvestigationTraceAgent
from portal.agents.schemas import FeatureWeight, SolutionResult
from portal.agents.tools import AgentToolRegistry


@pytest.fixture
def sample_dataset():
    return pd.DataFrame({
        "order_id": [f"ORD-{i:04d}" for i in range(80)],
        "lead_time_days": [5.0 + (i % 6) * 1.8 for i in range(80)],
        "is_stockout": [1 if i % 5 == 0 else 0 for i in range(80)],
        "supplier_id": [f"SUPP-{i % 4}" for i in range(80)],
        "dc_id": [f"DC-{i % 2}" for i in range(80)],
        "category": [f"CAT-{i % 3}" for i in range(80)],
        "order_date": pd.date_range("2026-01-01", periods=80, freq="D"),
    })


@pytest.fixture
def sample_solutions():
    return [
        SolutionResult(
            solution="E4",
            target_column="lead_time_days",
            target_type="continuous",
            kpi_name="Average Delivery Lead Time",
            kpi_baseline=11.4,
            sxi_baseline=5.9,
            sxi_optimized=8.2,
            primary_model="sxi_lgbm_v1",
            features=[
                FeatureWeight(name="supplier_reliability_score", weight=0.45, direction="down", actionable=True),
                FeatureWeight(name="buffer_stock_days", weight=0.35, direction="down", actionable=True),
            ],
        ),
        SolutionResult(
            solution="E8",
            target_column="is_stockout",
            target_type="binary",
            kpi_name="Stockout Probability",
            kpi_baseline=0.14,
            sxi_baseline=6.1,
            sxi_optimized=8.8,
            primary_model="sxi_xgb_v1",
            features=[
                FeatureWeight(name="safety_stock_days", weight=0.50, direction="down", actionable=True),
                FeatureWeight(name="supplier_lead_time_variance", weight=0.30, direction="up", actionable=True),
            ],
        ),
    ]


def test_tool_registry_execution(sample_dataset, sample_solutions):
    """Test 1: Verifies all tools in the registry can be called independently."""
    registry = AgentToolRegistry(dataset_df=sample_dataset, solutions=sample_solutions)
    tools = registry.list_tools()
    assert len(tools) >= 5

    # 1. Test Sentinel Scan tool
    scan_res = registry.tool_sentinel_scan()
    assert scan_res["status"] == "success"
    assert "anomalies_found" in scan_res

    # 2. Test Feature Attribution tool
    feat_res = registry.tool_explain_feature_drivers("E4")
    assert feat_res["status"] == "success"
    assert feat_res["solution"] == "E4"
    assert len(feat_res["top_drivers"]) > 0

    # 3. Test Causal Traversal tool
    causal_res = registry.tool_traverse_causal_path("E4", "E8")
    assert causal_res["status"] == "success"
    assert "path_exists" in causal_res

    # 4. Test Financial Exposure tool
    fin_res = registry.tool_calculate_financial_exposure(["E4", "E8"])
    assert fin_res["status"] == "success"
    assert "net_annual_usd" in fin_res
    assert fin_res["net_annual_usd"] >= 0
    assert "deduplication_active" in fin_res

    # 5. Test Remediation Simulation tool
    sim_res = registry.tool_simulate_remediation_action("E4", "L2")
    assert sim_res["status"] == "success"
    assert sim_res["scenario"] == "L2"
    assert sim_res["staged_for_step5_approval"] is True


def test_investigation_trace_loop(sample_dataset, sample_solutions):
    """
    Test 2: Runs the Investigation Trace sequence and validates
    dynamic multi-step trace logging, neutral step labels, and dossier compilation.
    """
    registry = AgentToolRegistry(dataset_df=sample_dataset, solutions=sample_solutions)
    agent = InvestigationTraceAgent(tool_registry=registry)

    user_goal = "Investigate supply chain fulfillment delays and provide an executive action plan."
    result = agent.run(user_goal=user_goal, session_id="test_trace_001")

    assert result.status == "completed"
    assert result.total_steps == 5
    assert len(result.trace) == 5

    # Verify each step has Thought/Label, Action, Action Input, and Observation
    expected_actions = [
        "sentinel_scan",
        "explain_feature_drivers",
        "traverse_causal_path",
        "calculate_financial_exposure",
        "simulate_remediation_action",
    ]
    for idx, step in enumerate(result.trace):
        assert step.step == idx + 1
        assert len(step.thought) > 10
        assert step.action == expected_actions[idx]
        assert isinstance(step.action_input, dict)
        assert isinstance(step.observation, dict)
        assert step.observation.get("status") == "success"
        assert step.duration_ms >= 0

    # Verify Final Answer and Decision Dossier
    assert "Investigation Trace Complete" in result.final_answer
    assert result.dossier is not None
    assert result.dossier.problem_statement is not None
    assert len(result.dossier.problem_statement.evidence_ids) >= 1
    assert len(result.dossier.actions) == 3

    # Check serialization
    res_dict = result.to_dict()
    assert isinstance(res_dict, dict)
    assert len(res_dict["trace"]) == 5
