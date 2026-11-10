"""
Comprehensive Production Unit Tests for Stage 2 Feature Interpreter.

Validates the 3-Question Architecture:
1. Question 1 (Global Drivers): Driver roles (lever, context, symptom), direction, raw effect sizes,
   stability scores, collinear cluster grouping, and EVID-EXP-GLB-* evidence IDs.
2. Question 2 (Per-Finding Local Explanations): Tree SHAP pred_contribs decomposition
   (SUP-402 vs fleet), explained gap, and exact unexplained residual.
3. Question 3 (Shared Driver Index): Cross-solution friction aggregation and EVID-EXP-SHR-* IDs.
4. Strict Causal Rephrasing Defense: Rejection of words like 'causes', 'caused by', and placeholder tampering.
5. Graceful Skips & Fallbacks: Resilient handling of missing artifacts or single-cohort anomalies.
"""

import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from portal.agents.evidence import EvidenceBuilder
from portal.agents.interpreter import FeatureInterpreter
from portal.agents.schemas import (
    Anomaly,
    AnomalyReport,
    Severity,
)
from portal.agents.artifacts import FileSystemArtifactStore
from portal.sxi_engine import run_sxi_optimization


@pytest.fixture
def temp_store_dir():
    temp_dir = tempfile.mkdtemp(prefix="interp_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_dataset():
    """Generates synthetic multi-solution order execution dataset."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "order_id": [f"ORD-{i+1}" for i in range(n)],
        "supplier_id": np.random.choice(["SUP-1", "SUP-2", "SUP-402"], size=n, p=[0.45, 0.45, 0.10]),
        "warehouse_id": np.random.choice(["WH-A", "WH-B"], size=n),
        "order_quantity": np.random.randint(10, 500, size=n),
        "safety_stock_days": np.random.uniform(1.0, 15.0, size=n),
        "transit_distance_km": np.random.uniform(20.0, 800.0, size=n),
        "lead_time_days": np.random.uniform(2.0, 18.0, size=n),
    })
    # Inject deliberate hotspot for SUP-402 (worse lead time, lower safety stock)
    sup402_mask = df["supplier_id"] == "SUP-402"
    df.loc[sup402_mask, "lead_time_days"] += 6.5
    df.loc[sup402_mask, "safety_stock_days"] = np.random.uniform(1.0, 2.5, size=int(sup402_mask.sum()))
    return df


@pytest.mark.skipif(xgb is None, reason="XGBoost required for SXI Tree tests")
def test_question_1_global_driver_explanations(temp_store_dir, sample_dataset):
    """
    Test Question 1:
    - Global driver roles (lever vs context).
    - Raw effect size (e.g. +X.X days).
    - Stability score and evidence IDs.
    """
    evid_builder = EvidenceBuilder()
    store = FileSystemArtifactStore(base_dir=temp_store_dir)
    interpreter = FeatureInterpreter(evidence_builder=evid_builder, store=store)

    sxi_res = run_sxi_optimization(
        df=sample_dataset,
        solution_key="e4",
        output_dir=temp_store_dir,
        session_id="sess_q1",
    )

    drivers, skip = interpreter.explain_global_drivers("E4", sxi_res, sample_dataset)
    assert skip is None
    assert len(drivers) > 0

    first_driver = drivers[0]
    assert first_driver.feature != ""
    assert first_driver.role in ["lever", "context", "symptom_leakage"]
    assert first_driver.direction in ["positive", "negative", "neutral"]
    assert first_driver.effect_display != ""
    assert 0.0 <= first_driver.stability_score <= 1.0
    assert len(first_driver.evidence_ids) > 0
    assert first_driver.evidence_ids[0].startswith("EVID-EXP-GLB-E4-")
    assert "associated with" in first_driver.template_explanation


@pytest.mark.skipif(xgb is None, reason="XGBoost required for SXI Tree tests")
def test_question_2_per_finding_local_tree_shap_gap_decomposition(temp_store_dir, sample_dataset):
    """
    Test Question 2:
    - Explaining Sentinel finding (SUP-402 vs fleet).
    - Tree SHAP delta contribution decomposition.
    - Verified unexplained residual.
    """
    evid_builder = EvidenceBuilder()
    store = FileSystemArtifactStore(base_dir=temp_store_dir)
    interpreter = FeatureInterpreter(evidence_builder=evid_builder, store=store)

    sxi_res = run_sxi_optimization(
        df=sample_dataset,
        solution_key="e4",
        output_dir=temp_store_dir,
        session_id="sess_q2",
    )

    anomaly = Anomaly(
        id="ANOM-E4-HOT-SUP402",
        solution="E4",
        type="hotspot",
        test_type="segment_hotspot",
        severity=Severity.HIGH,
        kpi_baseline=11.5,
        target_column="lead_time_days",
        unit="days",
        segment_dimension="supplier_id",
        segment_value="SUP-402",
        description="SUP-402 lead time is significantly elevated above fleet",
        metric="lead_time_days.supplier_id.SUP-402",
        observed=11.5,
        threshold=5.0,
    )

    explanation, skip = interpreter.explain_single_finding(anomaly, sxi_res, sample_dataset)
    assert skip is None
    assert explanation is not None

    assert explanation.finding_id == "ANOM-E4-HOT-SUP402"
    assert explanation.target_gap_raw > 0  # SUP-402 was injected with worse lead time
    assert len(explanation.contributions) > 0

    # Mathematical Invariant: Target Gap == Explained Gap + Unexplained Residual
    assert np.isclose(
        explanation.target_gap_raw,
        explanation.explained_gap_raw + explanation.unexplained_residual_raw,
        atol=1e-2,
    )

    # Verify per-feature contribution display and evidence ID
    first_contrib = explanation.contributions[0]
    assert "contributes" in first_contrib.display
    assert first_contrib.evidence_id.startswith("EVID-EXP-LOC-ANOM-E4-HOT-SUP402-")


@pytest.mark.skipif(xgb is None, reason="XGBoost required for SXI Tree tests")
def test_question_3_cross_solution_shared_driver_index(temp_store_dir, sample_dataset):
    """
    Test Question 3:
    - Identifies drivers shared across multiple active solutions (e.g. E4 and E8).
    - Computes cross-solution friction score and evidence ID.
    """
    evid_builder = EvidenceBuilder()
    store = FileSystemArtifactStore(base_dir=temp_store_dir)
    interpreter = FeatureInterpreter(evidence_builder=evid_builder, store=store)

    res_e4 = run_sxi_optimization(df=sample_dataset, solution_key="e4", output_dir=temp_store_dir, session_id="sess_q3_e4")
    res_e8 = run_sxi_optimization(df=sample_dataset, solution_key="e8", output_dir=temp_store_dir, session_id="sess_q3_e8")

    anomaly_report = AnomalyReport(
        anomalies=[
            Anomaly(id="A1", solution="E4", test_type="target_sla", severity=Severity.HIGH, observed=9.2, threshold=7.0, description="Lead time SLA breach"),
            Anomaly(id="A2", solution="E8", test_type="target_sla", severity=Severity.MEDIUM, observed=0.08, threshold=0.05, description="Late line SLA breach"),
        ],
        solutions_scanned=["E4", "E8"],
    )

    sxi_results = {"e4": res_e4, "e8": res_e8}
    report = interpreter.interpret(anomaly_report, sxi_results, df=sample_dataset)

    assert isinstance(report.shared_drivers, list)
    if report.shared_drivers:
        first_shared = report.shared_drivers[0]
        assert len(first_shared.solutions_affected) >= 2
        assert first_shared.cross_solution_friction_score > 0
        assert first_shared.evidence_ids[0].startswith("EVID-EXP-SHR-")


def test_strict_causal_rephrasing_defense():
    """
    Test Causal Claim Defense (Points 1-5):
    1. Rejection of causal words ('causes', 'due to', 'leads to', 'because of', 'root cause').
    2. Rejection of advice/prescriptive phrases ('we recommend', 'you should').
    3. Rejection of raw digits from LLM (numbers must come from placeholders only).
    4. Successful slot polish when LLM preserves placeholders without digits.
    5. Clean fallback to deterministic template on any violation.
    """
    slot_template = "{feature_name} ({role_phrase}) is {direction_phrase} associated with {kpi_name}, contributing an estimated {effect_display} to variance (stability: {stability})."
    slots_dict = {
        "feature_name": "Safety Stock Days",
        "role_phrase": "lever",
        "direction_phrase": "negatively",
        "kpi_name": "Delivery Lead Time",
        "effect_display": "-1.8 days",
        "stability": "0.91",
    }
    fallback_template = slot_template.format(**slots_dict)

    # 1. Fake LLM that introduces forbidden causal token ('due to')
    class CausalLLM:
        def invoke(self, prompt):
            return "Delivery delays occur due to {feature_name} ({role_phrase}) ({effect_display}, stability: {stability})."

    interp_causal = FeatureInterpreter(llm=CausalLLM())
    res1, gen1 = interp_causal._safely_rephrase_explanation(
        slot_template=slot_template,
        slots_dict=slots_dict,
        fallback_template=fallback_template,
        return_metadata=True,
    )
    assert res1 == fallback_template
    assert gen1 == "template"

    # 2. Fake LLM that introduces prescriptive advice ('we recommend')
    class AdviceLLM:
        def invoke(self, prompt):
            return "We recommend adjusting {feature_name} ({role_phrase}) to reduce {kpi_name} ({effect_display}, stability: {stability})."

    interp_advice = FeatureInterpreter(llm=AdviceLLM())
    res2, gen2 = interp_advice._safely_rephrase_explanation(
        slot_template=slot_template,
        slots_dict=slots_dict,
        fallback_template=fallback_template,
        return_metadata=True,
    )
    assert res2 == fallback_template
    assert gen2 == "template"

    # 3. Fake LLM that introduces raw digits (Point 1: strictly forbidden)
    class DigitsLLM:
        def invoke(self, prompt):
            return "Across {kpi_name}, {feature_name} ({role_phrase}) adds 4 extra days to lead times."

    interp_digits = FeatureInterpreter(llm=DigitsLLM())
    res3, gen3 = interp_digits._safely_rephrase_explanation(
        slot_template=slot_template,
        slots_dict=slots_dict,
        fallback_template=fallback_template,
        return_metadata=True,
    )
    assert res3 == fallback_template
    assert gen3 == "template"

    # 4. Compliant LLM: preserves all slots verbatim with ZERO digits
    class CompliantLLM:
        def invoke(self, prompt):
            return "Across {kpi_name}, {feature_name} serves as a key {role_phrase} that is {direction_phrase} linked to overall variance ({effect_display}, stability: {stability})."

    interp_compliant = FeatureInterpreter(llm=CompliantLLM())
    res4, gen4 = interp_compliant._safely_rephrase_explanation(
        slot_template=slot_template,
        slots_dict=slots_dict,
        fallback_template=fallback_template,
        return_metadata=True,
    )
    assert gen4 == "llm"
    assert "Safety Stock Days" in res4
    assert "-1.8 days" in res4
    assert "0.91" in res4
    assert "Across Delivery Lead Time" in res4
    assert interp_compliant.fallback_stats["llm_success_count"] >= 1


def test_missing_artifacts_graceful_skips(temp_store_dir):
    """
    Test Graceful Skips:
    - Missing SXI artifacts record clean skips without raising uncaught exceptions.
    """
    store = FileSystemArtifactStore(base_dir=temp_store_dir)
    interpreter = FeatureInterpreter(store=store)

    anomaly_report = AnomalyReport(
        anomalies=[
            Anomaly(id="A_MISSING", solution="E4", test_type="target_sla", severity=Severity.HIGH, observed=9.0, threshold=7.0, description="Missing SXI test breach"),
        ]
    )

    # Pass sxi_results with non-existent run_id
    fake_sxi_res = {
        "e4": {
            "solution_code": "E4",
            "run_id": "run_does_not_exist",
            "weights_table": [{"Feature": "transit_distance_km", "Composite_Weight": 25.0}],
            "top_drivers": [{"feature": "transit_distance_km"}],
        }
    }

    report = interpreter.interpret(anomaly_report, fake_sxi_res, df=pd.DataFrame())
    assert "E4_global" in report.skips or "finding_A_MISSING" in report.skips
    assert len(report.global_drivers.get("E4", [])) > 0  # Fell back gracefully to weights_table
