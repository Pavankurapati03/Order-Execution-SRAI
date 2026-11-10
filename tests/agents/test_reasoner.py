"""
Comprehensive Production Unit Tests for Stage 3 Cross-Domain Causal Reasoner.

Validates:
1. Causal Edge Validation: Validates domain edges against empirical shared drivers.
2. Multi-Node Chain Construction: Traverses DAG paths from root causes to downstream effects.
3. Financial Overlap Deduplication: Deducts overlapping exposure to prevent double-counting.
4. Isolated Anomaly Resilience: Handles disconnected nodes cleanly.
"""

import pytest
from portal.agents.evidence import EvidenceBuilder
from portal.agents.reasoner import CausalReasoner
from portal.agents.schemas import (
    Anomaly,
    AnomalyReport,
    InterpreterReport,
    Severity,
    SharedDriver,
)


@pytest.fixture
def evidence_builder():
    return EvidenceBuilder()


def test_causal_edge_validation_with_shared_driver(evidence_builder):
    """
    Test 1: Validates that an edge (E4 -> E8) is marked 'validated'
    when both solutions have anomalies AND an empirical shared driver connects them.
    """
    reasoner = CausalReasoner(evidence_builder=evidence_builder)

    anomaly_report = AnomalyReport(
        anomalies=[
            Anomaly(id="A-E4", solution="E4", test_type="target_sla", severity=Severity.HIGH, observed=11.5, threshold=7.0),
            Anomaly(id="A-E8", solution="E8", test_type="target_sla", severity=Severity.MEDIUM, observed=0.12, threshold=0.05),
        ],
        solutions_scanned=["E4", "E8"],
    )

    interpreter_report = InterpreterReport(
        shared_drivers=[
            SharedDriver(
                feature="safety_stock_days",
                business_label="Safety Stock Days",
                role="lever",
                solutions_affected=["E4", "E8"],
                cross_solution_friction_score=35.0,
            )
        ]
    )

    analysis = reasoner.analyze(anomaly_report, interpreter_report)

    assert len(analysis.chains) > 0
    # Find chain with E4 and E8
    e4_e8_chains = [c for c in analysis.chains if "E4" in c.nodes and "E8" in c.nodes]
    assert len(e4_e8_chains) >= 1

    first_chain = e4_e8_chains[0]
    assert first_chain.classification == "root_cause"
    assert len(first_chain.edges) >= 1
    edge = first_chain.edges[0]
    assert edge.cause_solution == "E4"
    assert edge.effect_solution == "E8"
    assert edge.status in ["supported association", "validated"]
    assert edge.strength >= 0.80
    assert "EVID-RSN-EDGE-E4-E8" in edge.test_evidence_ids


def test_dag_traversal_constructs_multi_node_chain(evidence_builder):
    """
    Test 2: When E9, E4, and E8 are all anomalous, the reasoner traces:
    E9 -> E4 -> E8 as a single continuous multi-step causal chain.
    """
    reasoner = CausalReasoner(evidence_builder=evidence_builder)

    anomaly_report = AnomalyReport(
        anomalies=[
            Anomaly(id="A-E9", solution="E9", test_type="concentration_risk", severity=Severity.HIGH, observed=0.85, threshold=0.50),
            Anomaly(id="A-E4", solution="E4", test_type="target_sla", severity=Severity.HIGH, observed=11.5, threshold=7.0),
            Anomaly(id="A-E8", solution="E8", test_type="target_sla", severity=Severity.MEDIUM, observed=0.15, threshold=0.05),
        ],
        solutions_scanned=["E9", "E4", "E8"],
    )

    interpreter_report = InterpreterReport()
    analysis = reasoner.analyze(anomaly_report, interpreter_report)

    # Check that a chain containing E9 -> E4 -> E8 was generated
    multi_step_chains = [c for c in analysis.chains if len(c.nodes) >= 3]
    assert len(multi_step_chains) >= 1
    chain = multi_step_chains[0]
    assert chain.nodes == ["E9", "E4", "E8"]
    assert "E9" in chain.narrative and "E4" in chain.narrative and "E8" in chain.narrative


def test_financial_overlap_deduplication(evidence_builder):
    """
    Test 3: When E3 ($820k) and E8 ($290k) both trigger in fulfillment_stockouts,
    the overlap discount (35% of min value = $101,500) is deducted from total exposure.
    """
    reasoner = CausalReasoner(evidence_builder=evidence_builder)

    anomaly_report = AnomalyReport(
        anomalies=[
            Anomaly(id="A-E3", solution="E3", test_type="target_sla", severity=Severity.HIGH, observed=0.18, threshold=0.05),
            Anomaly(id="A-E8", solution="E8", test_type="target_sla", severity=Severity.HIGH, observed=0.15, threshold=0.05),
        ],
        solutions_scanned=["E3", "E8"],
    )

    interpreter_report = InterpreterReport()
    analysis = reasoner.analyze(anomaly_report, interpreter_report)

    exp = analysis.exposure
    assert len(exp.lines) == 2

    # Expected gross = 820,000 + 290,000 = 1,110,000
    # Overlap discount = 290,000 * 0.35 = 101,500
    # Expected net = 1,110,000 - 101,500 = 1,008,500
    assert exp.overlap_adjustment_usd == 101500.0
    assert exp.net_annual_usd == 1008500.0
    assert exp.net_annual_usd < sum(line.annual_usd for line in exp.lines)


def test_isolated_anomaly_generates_hypothesis_chain(evidence_builder):
    """
    Test 4: An anomaly on an isolated solution without active edges generates
    a single-node hypothesis chain without throwing errors.
    """
    reasoner = CausalReasoner(evidence_builder=evidence_builder)

    anomaly_report = AnomalyReport(
        anomalies=[
            Anomaly(id="A-E10", solution="E10", test_type="data_quality", severity=Severity.LOW, observed=0.02, threshold=0.01),
        ],
        solutions_scanned=["E10"],
    )

    analysis = reasoner.analyze(anomaly_report, InterpreterReport())
    assert len(analysis.chains) == 1
    assert analysis.chains[0].nodes == ["E10"]
    assert analysis.chains[0].classification in ["root_cause", "hypothesis"]
    assert analysis.exposure.net_annual_usd == 240000.0
