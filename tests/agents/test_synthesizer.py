"""
Comprehensive Production Unit Tests for Stage 4 Executive Synthesis Agent.

Validates:
1. DecisionDossier Compilation: Assembles complete C-suite dossier citing verified evidence IDs.
2. Evidence Citation Invariant: Ensures every claim references >= 1 verifiable evidence IDs.
3. Multi-Tier Action Plan: Verifies L1 (Conservative, 30d), L2 (Recommended, 60d), L3 (Aggressive, 90d) actions.
4. Clean Operations Synthesis: Correctly reports operational health when no anomalies exist.
5. Mathematical Confidence Scoring: Composite score calculated across SXI reliability, causal edges, and data quality.
"""

import pytest
from portal.agents.schemas import (
    Anomaly,
    AnomalyReport,
    CausalAnalysis,
    CausalChain,
    CausalEdge,
    ExposureLedger,
    ExposureLine,
    InterpreterReport,
    Severity,
    SolutionResult,
)
from portal.agents.synthesizer import ExecutiveSynthesizer


@pytest.fixture
def sample_anomaly_report():
    return AnomalyReport(
        anomalies=[
            Anomaly(
                id="A-E4-SLA",
                solution="E4",
                test_type="target_sla",
                severity=Severity.HIGH,
                observed=12.4,
                threshold=7.0,
                unit="days",
                metric="Lead Time",
                is_primary_exposure=True,
                evidence_ids=["EV-E4-SLA-001"],
            ),
            Anomaly(
                id="A-E8-SLA",
                solution="E8",
                test_type="target_sla",
                severity=Severity.MEDIUM,
                observed=0.14,
                threshold=0.05,
                unit="fraction",
                metric="Stockout Rate",
                evidence_ids=["EV-E8-SLA-001"],
            ),
        ],
        solutions_scanned=["E4", "E8"],
    )


@pytest.fixture
def sample_causal_analysis():
    edge = CausalEdge(
        cause_solution="E4",
        effect_solution="E8",
        mechanism="Lead time delays cause downstream stockout spikes",
        status="validated",
        strength=0.88,
        test_evidence_ids=["EVID-EDGE-E4-E8"],
    )
    chain = CausalChain(
        id="chain-E4-E8",
        nodes=["E4", "E8"],
        edges=[edge],
        classification="root_cause",
        narrative="Lead time delays in Solution E4 trigger severe stockout spikes in Solution E8.",
    )
    exposure = ExposureLedger(
        lines=[
            ExposureLine(
                solution="E4",
                formula_id="sla_penalty_revenue_pct",
                inputs={"annual_revenue_usd": 10000000.0, "penalty_pct": 0.045},
                annual_usd=450000.0,
                evidence_ids=["EV-EXP-E4"],
            ),
            ExposureLine(
                solution="E8",
                formula_id="stockout_cost_usd",
                inputs={"annual_revenue_usd": 10000000.0, "stockout_rate": 0.08},
                annual_usd=800000.0,
                evidence_ids=["EV-EXP-E8"],
            ),
        ],
        overlap_adjustment_usd=160000.0,
        net_annual_usd=1090000.0,
        uses_default_assumptions=True,
    )
    return CausalAnalysis(chains=[chain], exposure=exposure)


def test_decision_dossier_compilation(sample_anomaly_report, sample_causal_analysis):
    """
    Test 1: Validates that the ExecutiveSynthesizer compiles a complete DecisionDossier
    with all required claims, actions, confidence breakdown, and operational caveats.
    """
    synthesizer = ExecutiveSynthesizer()
    interpreter_report = InterpreterReport()

    dossier = synthesizer.synthesize(
        run_id="run-test-001",
        anomaly_report=sample_anomaly_report,
        interpreter_report=interpreter_report,
        causal_analysis=sample_causal_analysis,
    )

    assert dossier.run_id == "run-test-001"
    assert dossier.problem_statement is not None
    assert "Solution E4" in dossier.problem_statement.text
    assert len(dossier.root_cause_chain) >= 1
    assert "E4 trigger" in dossier.root_cause_chain[0].text
    assert dossier.exposure_summary is not None
    assert "1,090,000.00" in dossier.exposure_summary.text
    assert "160,000.00" in dossier.exposure_summary.text  # Overlap deduction noted


def test_evidence_citation_invariant(sample_anomaly_report, sample_causal_analysis):
    """
    Test 2: Verifies that every single Claim in the DecisionDossier cites
    at least one valid Evidence ID (min_length >= 1) to prevent hallucinated claims.
    """
    synthesizer = ExecutiveSynthesizer()
    dossier = synthesizer.synthesize(
        run_id="run-test-002",
        anomaly_report=sample_anomaly_report,
        interpreter_report=InterpreterReport(),
        causal_analysis=sample_causal_analysis,
    )

    # Problem Statement Claim
    assert len(dossier.problem_statement.evidence_ids) >= 1
    assert "EV-E4-SLA-001" in dossier.problem_statement.evidence_ids

    # Root Cause Claims
    for claim in dossier.root_cause_chain:
        assert len(claim.evidence_ids) >= 1

    # Hypotheses Claims (if any)
    for claim in dossier.hypotheses:
        assert len(claim.evidence_ids) >= 1

    # Financial Exposure Claim
    assert len(dossier.exposure_summary.evidence_ids) >= 1
    assert "EVID-RSN-EXP-NET" in dossier.exposure_summary.evidence_ids


def test_multi_tier_action_plan(sample_anomaly_report, sample_causal_analysis):
    """
    Test 3: Validates the 3-scenario remediation playbook:
    L1 (Conservative, 30d), L2 (Recommended, 60d), L3 (Aggressive, 90d).
    """
    synthesizer = ExecutiveSynthesizer()
    dossier = synthesizer.synthesize(
        run_id="run-test-003",
        anomaly_report=sample_anomaly_report,
        interpreter_report=InterpreterReport(),
        causal_analysis=sample_causal_analysis,
        solutions=[
            SolutionResult(
                solution="E4",
                target_column="lead_time",
                target_type="continuous",
                kpi_name="Order Cycle Time",
                kpi_baseline=4.0,
                kpi_optimized={"L1": 3.5, "L2": 3.0, "L3": 2.5},
                sxi_baseline=65.0,
                sxi_optimized=85.0,
            )
        ],
    )

    actions = dossier.actions
    assert len(actions) == 3

    # Check scenarios
    scenarios = {a.scenario_level: a for a in actions}
    assert "L1" in scenarios
    assert "L2" in scenarios
    assert "L3" in scenarios

    assert scenarios["L1"].horizon == "30d"
    assert scenarios["L2"].horizon == "60d"
    assert scenarios["L3"].horizon == "90d"

    # Recommended action targets primary root cause E4
    assert "E4" in scenarios["L2"].target_solutions
    assert len(scenarios["L2"].expected_kpi_delta) > 0


def test_clean_operations_synthesis():
    """
    Test 4: Verifies that when zero anomalies are detected, the synthesizer
    produces a clean operational health summary without failure.
    """
    synthesizer = ExecutiveSynthesizer()
    clean_anomalies = AnomalyReport(anomalies=[], solutions_scanned=["E4", "E8"])
    empty_causal = CausalAnalysis(chains=[], exposure=ExposureLedger())

    dossier = synthesizer.synthesize(
        run_id="run-test-004",
        anomaly_report=clean_anomalies,
        interpreter_report=InterpreterReport(),
        causal_analysis=empty_causal,
    )

    assert "within established bounds" in dossier.problem_statement.text
    assert dossier.problem_statement.numbers.get("anomaly_count") == 0.0
    assert len(dossier.problem_statement.evidence_ids) >= 1


def test_confidence_breakdown_computation(sample_anomaly_report, sample_causal_analysis):
    """
    Test 5: Validates composite confidence calculation based on SXI reliability,
    validated edges, and data quality.
    """
    synthesizer = ExecutiveSynthesizer()
    solutions = [
        SolutionResult(
            solution="E4",
            target_column="lead_time_days",
            target_type="continuous",
            kpi_name="Lead Time",
            kpi_baseline=11.5,
            sxi_baseline=11.5,
            sxi_optimized=7.5,
            primary_model="sxi_lgbm_v1",
        ),
        SolutionResult(
            solution="E8",
            target_column="is_stockout",
            target_type="binary",
            kpi_name="Stockout Rate",
            kpi_baseline=0.12,
            sxi_baseline=0.12,
            sxi_optimized=0.04,
            primary_model="sxi_xgb_v1",
        ),
    ]

    dossier = synthesizer.synthesize(
        run_id="run-test-005",
        anomaly_report=sample_anomaly_report,
        interpreter_report=InterpreterReport(),
        causal_analysis=sample_causal_analysis,
        solutions=solutions,
    )

    conf = dossier.confidence
    assert 0.0 <= conf.model_reliability <= 1.0
    assert 0.0 <= conf.edge_validation <= 1.0
    assert 0.0 <= conf.data_quality <= 1.0
    assert 0.0 <= conf.overall <= 1.0
    # Both solutions have primary SXI predictors
    assert conf.model_reliability == 1.0
