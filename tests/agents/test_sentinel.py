"""
Comprehensive Production Unit Tests for Stage 1 Sentinel.
Validates all Claude review criteria:
1. Window Consistency Invariant: Overall mean must mathematically lie between partitioned window means.
2. Large-n Tiny-Effect Resilience: 50,000 rows with 0.2-day difference (p < 0.05, Cohen's d < 0.20) must NOT fire.
3. Multiple Testing Protection: Pure noise across 200 suppliers yields zero false hotspots after Benjamini-Hochberg FDR.
4. Financial Deduplication: Multiple findings on E4 share FAM-E4 and allocate exposure ONCE (no double/triple counting).
5. Right-Censoring: Incomplete pending orders are filtered out.
6. Graceful Skips: Datasets lacking date/supplier columns record clean skip notes in tests_skipped.
"""

import pytest
import pandas as pd
import numpy as np

from portal.agents.schemas import (
    Severity,
    SolutionResult,
    FeatureWeight,
    AnomalyReport
)
from portal.agents.evidence import EvidenceBuilder
from portal.agents.sentinel import Sentinel, compute_cohens_d


@pytest.fixture
def clean_solution():
    """A high-performing, reliable solution meeting all SLAs."""
    return SolutionResult(
        solution="E4",
        target_column="lead_time_days",
        target_type="continuous",
        kpi_name="Average Delivery Lead Time",
        kpi_baseline=5.2,  # Well below the 7.0 day SLA target
        kpi_optimized={"L1": 4.8, "L2": 4.2, "L3": 3.8},
        sxi_baseline=7.5,
        sxi_optimized=8.9,
        models_run=["XGBoost", "CatBoost"],
        primary_model="XGBoost Ensemble",
        holdout={"r2": 0.85, "rmse": 0.4},
        n_train=1000,
        n_test=250,
        features=[
            FeatureWeight(name="safety_stock_days", weight=0.35, direction="down"),
            FeatureWeight(name="carrier_rating", weight=0.25, direction="up")
        ],
        leakage_suspects=[],
        data_quality={"missing_pct": 0.0, "outlier_pct": 1.2}
    )


# -----------------------------------------------------------------------------
# Test 1: Window Consistency Invariant
# -----------------------------------------------------------------------------
def test_window_consistency_invariant(clean_solution):
    """When partitioned into recent and reference windows, overall mean must equal weighted sum."""
    np.random.seed(42)
    dates = pd.date_range(end="2026-10-01", periods=120, freq="D")
    vals = np.random.normal(loc=6.0, scale=1.0, size=120)
    df = pd.DataFrame({"date": dates, "lead_time_days": vals})

    sentinel = Sentinel()
    recent_df, ref_df, _ = sentinel._split_temporal_windows(df)

    assert recent_df is not None and ref_df is not None
    recent_mean = float(recent_df["lead_time_days"].mean())
    ref_mean = float(ref_df["lead_time_days"].mean())
    overall_mean = float(df["lead_time_days"].mean())

    weighted_mean = (len(recent_df) * recent_mean + len(ref_df) * ref_mean) / len(df)
    assert abs(overall_mean - weighted_mean) < 1e-6
    # Overall mean must sit between the two window means
    min_mean = min(recent_mean, ref_mean)
    max_mean = max(recent_mean, ref_mean)
    assert min_mean <= overall_mean <= max_mean


# -----------------------------------------------------------------------------
# Test 2: Large-n Tiny-Effect Resilience (50,000 Rows, 0.2 Day Shift)
# -----------------------------------------------------------------------------
def test_large_n_tiny_effect_no_alert(clean_solution):
    """At 50,000 rows, a tiny 0.2-day difference has p < 0.05 but Cohen's d < 0.20 -> NO Test B alert."""
    np.random.seed(42)
    n_ref = 37500
    n_recent = 12500

    ref_dates = pd.date_range(end="2026-09-01", periods=n_ref, freq="h")
    recent_dates = pd.date_range(start="2026-09-02", periods=n_recent, freq="h")

    # Only a 0.2-day difference (5.0 vs 5.2 days, SD = 2.0 -> Cohen's d = 0.10)
    ref_vals = np.random.normal(loc=5.0, scale=2.0, size=n_ref)
    recent_vals = np.random.normal(loc=5.2, scale=2.0, size=n_recent)

    d = compute_cohens_d(recent_vals, ref_vals)
    assert d < 0.20  # Negligible effect size

    dataset_df = pd.DataFrame({
        "date": list(ref_dates) + list(recent_dates),
        "lead_time_days": list(ref_vals) + list(recent_vals)
    })

    ledger = EvidenceBuilder.build_from_solutions([clean_solution])
    sentinel = Sentinel()
    report = sentinel.analyze([clean_solution], ledger, dataset_df)

    # Test B must NOT fire because Cohen's d is below 0.20
    temporal_anoms = [a for a in report.anomalies if a.test_type == "temporal_drift"]
    assert len(temporal_anoms) == 0


# -----------------------------------------------------------------------------
# Test 3: Multiple Testing Protection (Pure Noise Across 200 Suppliers)
# -----------------------------------------------------------------------------
def test_pure_noise_across_200_suppliers_no_hotspots_after_bh(clean_solution):
    """Generating 200 suppliers from the identical distribution must produce ZERO false hotspots after Benjamini-Hochberg."""
    np.random.seed(42)
    n_suppliers = 200
    rows_per_sup = 40
    total_n = n_suppliers * rows_per_sup

    # All 200 suppliers drawn from the EXACT same normal distribution
    supplier_ids = [f"SUP-{i:03d}" for i in range(n_suppliers) for _ in range(rows_per_sup)]
    lead_times = np.random.normal(loc=5.0, scale=1.0, size=total_n)

    dataset_df = pd.DataFrame({
        "supplier_id": supplier_ids,
        "lead_time_days": lead_times
    })

    ledger = EvidenceBuilder.build_from_solutions([clean_solution])
    sentinel = Sentinel()
    report = sentinel.analyze([clean_solution], ledger, dataset_df)

    # After Benjamini-Hochberg FDR correction, pure noise should yield zero hotspots
    hotspots = [a for a in report.anomalies if a.test_type == "segment_hotspot"]
    assert len(hotspots) == 0


# -----------------------------------------------------------------------------
# Test 4: Financial Exposure Deduplication (Family ID Grouping)
# -----------------------------------------------------------------------------
def test_two_findings_one_solution_no_double_counting(clean_solution):
    """When Solution E4 triggers both an SLA breach (Test A) and Temporal Drift (Test B),
    the $210,000 exposure is counted ONCE, not twice!"""
    # Create severe breach and severe temporal drift
    np.random.seed(42)
    ref_dates = pd.date_range(end="2026-09-01", periods=90, freq="D")
    ref_vals = np.random.normal(loc=5.0, scale=0.8, size=90)

    recent_dates = pd.date_range(start="2026-09-02", periods=30, freq="D")
    recent_vals = np.random.normal(loc=11.5, scale=1.0, size=30)  # Breaches 7.0d SLA AND regressed from 5.0d

    dataset_df = pd.DataFrame({
        "date": list(ref_dates) + list(recent_dates),
        "lead_time_days": list(ref_vals) + list(recent_vals)
    })

    ledger = EvidenceBuilder.build_from_solutions([clean_solution])
    sentinel = Sentinel()
    report = sentinel.analyze([clean_solution], ledger, dataset_df)

    # Both Test A and Test B fire
    e4_anoms = [a for a in report.anomalies if a.solution == "E4"]
    assert len(e4_anoms) >= 2

    # Verify family_id grouping
    for a in e4_anoms:
        assert a.family_id == "FAM-E4"
        assert a.solution_exposure_usd == 210000.0

    # Primary finding carries the allocated exposure; secondary carries 0.0
    primary_findings = [a for a in e4_anoms if a.is_primary_exposure]
    secondary_findings = [a for a in e4_anoms if not a.is_primary_exposure]

    assert len(primary_findings) == 1
    assert primary_findings[0].allocated_exposure_usd == 210000.0
    for sec in secondary_findings:
        assert sec.allocated_exposure_usd == 0.0

    # Report total exposure counts it exactly ONCE
    assert report.total_exposure_usd == 210000.0


# -----------------------------------------------------------------------------
# Test 5: Right-Censoring (Incomplete Orders Filtered Out)
# -----------------------------------------------------------------------------
def test_incomplete_orders_censoring_handling(clean_solution):
    """Pending/incomplete orders with artificial low lead time must be filtered out."""
    delivered_leads = [12.0, 14.0, 11.0, 13.0, 12.5, 13.5, 14.2, 11.8, 12.1, 13.0]  # 10 delivered orders
    pending_leads = [1.0] * 5  # 5 pending orders with fake low lead time
    dataset_df = pd.DataFrame({
        "lead_time_days": delivered_leads + pending_leads,
        "is_delivered": [1] * 10 + [0] * 5
    })

    ledger = EvidenceBuilder.build_from_solutions([clean_solution])
    sentinel = Sentinel()
    report = sentinel.analyze([clean_solution], ledger, dataset_df)

    sla_breaches = [a for a in report.anomalies if a.test_type == "target_sla"]
    assert len(sla_breaches) >= 1
    # Observed must reflect only delivered orders (mean around 12.7), not diluted by the pending 1.0s
    assert sla_breaches[0].observed > 10.0


# -----------------------------------------------------------------------------
# Test 6: Graceful Skips for Missing Dimensions
# -----------------------------------------------------------------------------
def test_missing_columns_graceful_skips(clean_solution):
    """When dataset has 60 rows but lacks date and supplier columns, Sentinel records clear skip notes in tests_skipped."""
    dataset_df = pd.DataFrame({
        "lead_time_days": [5.0] * 60
    })

    ledger = EvidenceBuilder.build_from_solutions([clean_solution])
    sentinel = Sentinel()
    report = sentinel.analyze([clean_solution], ledger, dataset_df)

    assert "test_b_temporal" in report.tests_skipped
    assert "lacks 'date' column" in report.tests_skipped["test_b_temporal"]
    assert "test_c_hotspots" in report.tests_skipped
