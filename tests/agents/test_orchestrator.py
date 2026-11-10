"""
Comprehensive Unit Tests for Decision Intelligence Orchestrator.

Validates:
1. End-to-end 4-step pipeline execution (Steps 1 -> 2 -> 3 -> 4).
2. Pipeline step progress and status reporting.
3. Resilience to empty or partial inputs.
4. Correct compilation of DecisionDossier and Evidence Ledger integration.
"""

import pytest
import pandas as pd
from portal.agents.orchestrator import DecisionIntelligenceOrchestrator
from portal.agents.schemas import SolutionResult, FeatureWeight


@pytest.fixture
def sample_dataset():
    """Generates a small test dataframe with fulfillment and stockout columns."""
    return pd.DataFrame({
        "order_id": [f"ORD-{i:04d}" for i in range(100)],
        "lead_time_days": [5.0 + (i % 8) * 1.5 for i in range(100)],
        "is_stockout": [1 if i % 6 == 0 else 0 for i in range(100)],
        "supplier_id": [f"SUPP-{i % 5}" for i in range(100)],
        "dc_id": [f"DC-{i % 3}" for i in range(100)],
        "category": [f"CAT-{i % 4}" for i in range(100)],
        "order_date": pd.date_range("2026-01-01", periods=100, freq="D"),
    })


@pytest.fixture
def sample_solutions():
    return [
        SolutionResult(
            solution="E4",
            target_column="lead_time_days",
            target_type="continuous",
            kpi_name="Lead Time Days",
            kpi_baseline=11.5,
            sxi_baseline=5.8,
            sxi_optimized=8.2,
            primary_model="sxi_lgbm_v1",
            features=[
                FeatureWeight(name="supplier_reliability_score", weight=0.45, direction="down", actionable=True),
                FeatureWeight(name="order_volume_units", weight=0.30, direction="up", actionable=False),
            ],
        ),
        SolutionResult(
            solution="E8",
            target_column="is_stockout",
            target_type="binary",
            kpi_name="Stockout Rate",
            kpi_baseline=0.14,
            sxi_baseline=6.1,
            sxi_optimized=8.8,
            primary_model="sxi_xgb_v1",
            features=[
                FeatureWeight(name="safety_stock_days", weight=0.55, direction="down", actionable=True),
                FeatureWeight(name="supplier_lead_time_variance", weight=0.25, direction="up", actionable=True),
            ],
        ),
    ]


def test_orchestrator_end_to_end(sample_dataset, sample_solutions):
    """
    Test 1: Runs the complete 4-step pipeline end-to-end and verifies
    all 4 steps succeed and output a populated DecisionDossier.
    """
    orchestrator = DecisionIntelligenceOrchestrator()
    result = orchestrator.run_investigation(
        session_id="test_sess_001",
        run_id="run_test_e2e_001",
        dataset_df=sample_dataset,
        solutions=sample_solutions,
    )

    assert result.run_id == "run_test_e2e_001"
    assert result.session_id == "test_sess_001"
    assert result.duration_ms > 0

    # Step progress validation
    assert result.step_progress["step_1_sentinel"] == "completed"
    assert result.step_progress["step_2_interpreter"] == "completed"
    assert result.step_progress["step_3_reasoner"] == "completed"
    assert result.step_progress["step_4_synthesizer"] == "completed"

    # Stage outputs validation
    assert result.anomaly_report is not None
    assert result.interpreter_report is not None
    assert result.causal_analysis is not None
    assert result.dossier is not None

    # Decision Dossier content check
    assert result.dossier.problem_statement is not None
    assert len(result.dossier.problem_statement.evidence_ids) >= 1
    assert len(result.dossier.actions) == 3
    assert result.dossier.confidence.overall > 0.0

    # JSON serialization check
    res_dict = result.to_dict()
    assert isinstance(res_dict, dict)
    assert res_dict["status"] in ["completed", "clean", "warning"]
    assert "step_progress" in res_dict


def test_orchestrator_clean_slate():
    """
    Test 2: Runs investigation with empty solutions and checks clean graceful handling.
    """
    orchestrator = DecisionIntelligenceOrchestrator()
    result = orchestrator.run_investigation(
        session_id="test_sess_clean",
        solutions=[],
    )

    assert result.status == "clean"
    assert result.anomaly_report is not None
    assert len(result.anomaly_report.anomalies) == 0
    assert result.dossier is not None
    assert "within established bounds" in result.dossier.problem_statement.text
