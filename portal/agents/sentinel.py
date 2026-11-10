"""
Stage 1: Sentinel (Deep Search & Anomaly Detection) — Production 4-Tier Engine.
Purely deterministic, code-driven mathematical anomaly detection.
"Code computes, LLM interprets."

Architecture:
  Test A: Target-based SLA breaches with Compliance Rate & Wilson Confidence Interval
  Test B: Temporal drift & regression with Mann-Whitney / Welch's t-test + Cohen's d effect size
  Test C: Segment Hotspots with Benjamini-Hochberg FDR correction (alpha = 0.05, n >= 30)
  Test D: Tail Risk (Distribution insights on fat-tail volume, fallback when no SLA target)

Financial Exposure Deduplication:
  - Finds grouped under family_id (e.g. FAM-E4)
  - Allocated once per family to prevent double/triple counting ($210k counted once, not 3x)
"""

import math
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from portal.agents.schemas import (
    Severity,
    Anomaly,
    AnomalyReport,
    SolutionResult
)
from portal.agents.evidence import EvidenceLedger
from portal.agents.config import get_thresholds_config, get_financial_assumptions


def wilson_score_interval(positives: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score binomial confidence interval for binary event rates."""
    if total <= 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = positives / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / total) + (z**2 / (4 * total**2)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def compute_cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Computes Cohen's d effect size between two continuous samples."""
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd <= 1e-9:
        return 0.0
    return abs(float(np.mean(x1) - np.mean(x2))) / pooled_sd


class Sentinel:
    """
    Production-Grade Multi-Test Sentinel Anomaly Detector.
    Strictly audits SolutionResults, EvidenceLedger, and customer datasets.
    """

    def __init__(
        self,
        thresholds_config: Optional[Dict[str, Any]] = None,
        evidence_builder: Optional[Any] = None,
    ):
        self.evidence_builder = evidence_builder
        self.config = thresholds_config or get_thresholds_config()
        self.sla_targets = self.config.get("sla_targets", {})
        self.temporal_cfg = self.config.get("temporal_window", {})
        self.hotspot_cfg = self.config.get("segment_hotspots", {})
        self.tail_cfg = self.config.get("tail_risk", self.config.get("statistical_fences", {}))
        self.reliability_cfg = self.config.get("model_reliability", {})
        self.quality_cfg = self.config.get("data_quality", {})
        self.severity_cfg = self.config.get("severity", {})
        self.directions = self.config.get("metric_direction", {})
        
        # Financial weights for ranking
        try:
            fin_cfg = get_financial_assumptions()
            self.financial_map = fin_cfg.get("solution_financial_map", {})
        except Exception:
            self.financial_map = {}

    def _compute_solution_exposure(
        self,
        sol: SolutionResult,
        dataset_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, str]:
        """
        Dynamically calculates annualized financial exposure directly from the active dataset.
        Falls back to financial assumptions map only if no raw dataset is provided.
        """
        sol_code = sol.solution
        if dataset_df is not None and not dataset_df.empty:
            monetary_candidates = [
                "order_value", "sales_amount", "total_amount", "revenue", "order_amount",
                "purchase_cost", "cost", "margin_leakage", "price", "unit_price", "amount",
            ]
            m_cols = [
                c for c in dataset_df.columns
                if any(mc in c.lower() for mc in monetary_candidates) and pd.api.types.is_numeric_dtype(dataset_df[c])
            ]

            t_col = sol.target_column if sol.target_column in dataset_df.columns else None
            if not t_col:
                for c in dataset_df.columns:
                    if c.lower() in [sol_code.lower(), f"{sol_code.lower()}_metric", sol.kpi_name.lower().replace(" ", "_")]:
                        t_col = c
                        break

            if t_col and m_cols and pd.api.types.is_numeric_dtype(dataset_df[t_col]):
                lower_is_better = any(
                    term in (sol.kpi_name + " " + t_col).lower()
                    for term in ["lead_time", "defect", "risk", "delay", "cost", "error", "variance", "stockout"]
                )
                if sol.target_type == "binary":
                    breached_mask = dataset_df[t_col] > 0.5
                elif lower_is_better:
                    thresh = sol.kpi_baseline if sol.kpi_baseline > 0 else float(dataset_df[t_col].median())
                    breached_mask = dataset_df[t_col] > thresh
                else:
                    thresh = sol.kpi_baseline if sol.kpi_baseline > 0 else float(dataset_df[t_col].median())
                    breached_mask = dataset_df[t_col] < thresh

                if breached_mask.any():
                    empirical_breached_sum = float(dataset_df.loc[breached_mask, m_cols[0]].sum())
                    date_cols = [c for c in dataset_df.columns if "date" in c.lower() or "time" in c.lower() or "created" in c.lower()]
                    annual_factor = 1.0
                    if date_cols:
                        try:
                            ts = pd.to_datetime(dataset_df[date_cols[0]], errors="coerce").dropna()
                            span_days = max(1, (ts.max() - ts.min()).days)
                            if 1 < span_days < 365:
                                annual_factor = 365.0 / span_days
                        except Exception:
                            annual_factor = 1.0
                    exposure_val = round(empirical_breached_sum * annual_factor, 2)
                    if exposure_val > 0.0:
                        return exposure_val, "empirical_dataset"

        sol_exposure_usd = float(self.financial_map.get(sol_code, {}).get("annual_usd", 0.0))
        assumption_src = "customer" if sol.data_quality.get("has_price_columns") else "default"
        return sol_exposure_usd, assumption_src

    def run_all(
        self,
        solutions: List[SolutionResult],
        df: Optional[pd.DataFrame] = None,
        ledger: Optional[EvidenceLedger] = None,
    ) -> AnomalyReport:
        """Executes complete Stage 1 scan with automatic ledger extraction."""
        eb = getattr(self, "evidence_builder", None)
        active_ledger = ledger or (eb.to_ledger() if hasattr(eb, "to_ledger") else (eb if isinstance(eb, EvidenceLedger) else EvidenceLedger()))
        return self.analyze(solutions=solutions, ledger=active_ledger, dataset_df=df)

    def analyze(
        self,
        solutions: List[SolutionResult],
        ledger: EvidenceLedger,
        dataset_df: Optional[pd.DataFrame] = None,
        focal_solutions: Optional[List[str]] = None,
    ) -> AnomalyReport:
        """
        Executes complete Stage 1 deep search scan across all provided solutions.
        Returns an AnomalyReport with ranked anomalies and untrusted solution flags.
        """
        anomalies: List[Anomaly] = []
        solutions_scanned: List[str] = []
        solutions_untrusted: List[str] = []
        tests_skipped: Dict[str, str] = {}
        counter = 1

        # Preprocess temporal slices once for all solutions
        recent_df, ref_df, temporal_skip_reason = self._split_temporal_windows(dataset_df)
        if temporal_skip_reason:
            tests_skipped["test_b_temporal"] = temporal_skip_reason

        # Check hotspot dimensions availability
        if dataset_df is not None:
            dims = self.hotspot_cfg.get("dimensions", ["supplier_id", "dc_id", "category"])
            avail_dims = [d for d in dims if d in dataset_df.columns]
            if not avail_dims:
                tests_skipped["test_c_hotspots"] = f"None of the segment dimensions {dims} are present in dataset"
        else:
            tests_skipped["test_c_hotspots"] = "No raw dataset DataFrame provided for segment aggregation"

        # Track financial exposure per family to prevent double counting
        family_exposures_recorded: set = set()

        for sol in solutions:
            sol_code = sol.solution
            family_id = f"FAM-{sol_code}"
            solutions_scanned.append(sol_code)

            # Collect evidence IDs for this solution
            ev_items = ledger.find(solution=sol_code)
            ev_map = {item.key: item.id for item in ev_items}
            
            # Dynamic exposure calculation directly from dataset DataFrame
            sol_exposure_usd, assumption_src = self._compute_solution_exposure(sol, dataset_df)

            # ---------------------------------------------------------
            # 0. Model Reliability Floor & Anti-Leakage Audit
            # ---------------------------------------------------------
            is_reliable, unreliable_reasons, is_leakage = self._check_model_reliability(sol, ev_map)
            if not is_reliable:
                solutions_untrusted.append(sol_code)
                for reason in unreliable_reasons:
                    anom_id = f"ANOM-{sol_code}-{counter:03d}"
                    counter += 1
                    anomalies.append(
                        Anomaly(
                            id=anom_id,
                            solution=sol_code,
                            family_id=family_id,
                            type="model_untrusted",
                            test_type="model_quality",
                            severity=Severity.HIGH,
                            description=f"Model reliability floor breached for {sol.kpi_name}: {reason}",
                            metric="model_reliability",
                            unit="score",
                            direction="lower",
                            observed=sol.primary_model,
                            threshold=None,
                            n=sol.n_test,
                            financial_weight=sol_exposure_usd,
                            solution_exposure_usd=sol_exposure_usd,
                            allocated_exposure_usd=0.0,
                            assumption_source=assumption_src,
                            is_primary_exposure=False,
                            evidence_ids=[ev_map[k] for k in ev_map if "model_" in k or "accuracy" in k or "r2" in k],
                            model_reliable=False
                        )
                    )

            # Check explicit data leakage suspects
            if sol.leakage_suspects:
                anom_id = f"ANOM-{sol_code}-{counter:03d}"
                counter += 1
                anomalies.append(
                    Anomaly(
                        id=anom_id,
                        solution=sol_code,
                        family_id=family_id,
                        type="leakage_suspect",
                        test_type="leakage_risk",
                        severity=Severity.CRITICAL,
                        description=(
                            f"Target data leakage suspected in {sol.kpi_name}: Features {sol.leakage_suspects} "
                            f"exhibit near-perfect correlation with target. Model predictions quarantined."
                        ),
                        metric="target_correlation",
                        unit="correlation",
                        direction="higher",
                        observed=len(sol.leakage_suspects),
                        threshold=self.reliability_cfg.get("leakage_correlation_threshold", 0.95),
                        n=sol.n_test,
                        financial_weight=sol_exposure_usd,
                        solution_exposure_usd=sol_exposure_usd,
                        allocated_exposure_usd=0.0,
                        assumption_source=assumption_src,
                        is_primary_exposure=False,
                        evidence_ids=[ev_map.get("leakage_suspects", "")] if "leakage_suspects" in ev_map else [],
                        model_reliable=False
                    )
                )

            # Check feature concentration fragility
            max_conc = self.reliability_cfg.get("max_feature_concentration", 0.60)
            for feat in sol.features:
                if feat.weight > max_conc:
                    anom_id = f"ANOM-{sol_code}-{counter:03d}"
                    counter += 1
                    key = f"feature_weight.{feat.name}"
                    ev_ids = [ev_map[key]] if key in ev_map else []
                    anomalies.append(
                        Anomaly(
                            id=anom_id,
                            solution=sol_code,
                            family_id=family_id,
                            type="feature_concentration",
                            test_type="concentration_risk",
                            severity=Severity.HIGH,
                            description=(
                                f"Operational fragility warning: Single feature '{feat.name}' drives "
                                f"{feat.weight * 100:.1f}% of {sol.kpi_name} prediction (> {max_conc * 100:.0f}% threshold)."
                            ),
                            metric=f"weight_{feat.name}",
                            unit="weight",
                            direction="higher",
                            observed=round(feat.weight, 4),
                            threshold=max_conc,
                            n=sol.n_train,
                            financial_weight=sol_exposure_usd,
                            solution_exposure_usd=sol_exposure_usd,
                            allocated_exposure_usd=0.0,
                            assumption_source=assumption_src,
                            is_primary_exposure=False,
                            evidence_ids=ev_ids,
                            model_reliable=is_reliable
                        )
                    )

            # Check data hygiene
            max_miss = self.quality_cfg.get("max_missing_pct", 10.0)
            miss_pct = sol.data_quality.get("missing_pct", 0.0)
            if miss_pct > max_miss:
                anom_id = f"ANOM-{sol_code}-{counter:03d}"
                counter += 1
                ev_ids = [ev_map["data_quality.missing_pct"]] if "data_quality.missing_pct" in ev_map else []
                anomalies.append(
                    Anomaly(
                        id=anom_id,
                        solution=sol_code,
                        family_id=family_id,
                        type="data_quality",
                        test_type="data_quality",
                        severity=Severity.MEDIUM,
                        description=f"Data hygiene warning: {miss_pct:.1f}% missing values in {sol.target_column} exceeds threshold of {max_miss:.1f}%.",
                        metric="missing_pct",
                        unit="%",
                        direction="higher",
                        observed=round(miss_pct, 2),
                        threshold=max_miss,
                        n=sol.n_train,
                        financial_weight=sol_exposure_usd,
                        solution_exposure_usd=sol_exposure_usd,
                        allocated_exposure_usd=0.0,
                        assumption_source=assumption_src,
                        is_primary_exposure=False,
                        evidence_ids=ev_ids,
                        model_reliable=is_reliable
                    )
                )

            # ---------------------------------------------------------
            # TEST A: Target-based SLA Breach (Headline Alert)
            # Uses recent evaluation window matching Test B!
            # ---------------------------------------------------------
            test_a_anom = self._run_test_a_target_breach(
                sol, ev_map, dataset_df, recent_df, counter, is_reliable, sol_exposure_usd, assumption_src, family_id
            )
            has_test_a = False
            if test_a_anom:
                has_test_a = True
                anomalies.append(test_a_anom)
                counter += 1

            # ---------------------------------------------------------
            # TEST B: Temporal Window Drift & Regression
            # ---------------------------------------------------------
            test_b_anom = self._run_test_b_temporal_drift(
                sol, ev_map, recent_df, ref_df, counter, is_reliable, sol_exposure_usd, assumption_src, family_id
            )
            if test_b_anom:
                anomalies.append(test_b_anom)
                counter += 1

            # ---------------------------------------------------------
            # TEST C: Segment Hotspots with Benjamini-Hochberg Correction
            # ---------------------------------------------------------
            test_c_anoms = self._run_test_c_hotspots(
                sol, ev_map, dataset_df, counter, is_reliable, sol_exposure_usd, assumption_src, family_id
            )
            for h_anom in test_c_anoms:
                anomalies.append(h_anom)
                counter += 1

            # ---------------------------------------------------------
            # TEST D: Tail Risk (Distribution Insight)
            # Only runs if Test A did not already flag an SLA breach!
            # ---------------------------------------------------------
            if not has_test_a:
                test_d_anom = self._run_test_d_tail_risk(
                    sol, ev_map, dataset_df, counter, is_reliable, sol_exposure_usd, assumption_src, family_id
                )
                if test_d_anom:
                    anomalies.append(test_d_anom)
                    counter += 1

        # ---------------------------------------------------------
        # Deduplicate Financial Exposure: Allocate ONCE per Family
        # ---------------------------------------------------------
        # Sort anomalies by severity first
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3
        }
        focal_set = {s.upper() for s in (focal_solutions or [])}
        anomalies.sort(
            key=lambda a: (
                0 if (focal_set and a.solution.upper() in focal_set) else 1,
                severity_order.get(a.severity, 4),
                -a.financial_weight
            )
        )

        total_exposure = 0.0
        allocated_families = set()
        for anom in anomalies:
            fam = anom.family_id
            if fam and fam not in allocated_families and anom.solution_exposure_usd > 0:
                anom.is_primary_exposure = True
                anom.allocated_exposure_usd = anom.solution_exposure_usd
                total_exposure += anom.allocated_exposure_usd
                allocated_families.add(fam)
            else:
                anom.is_primary_exposure = False
                anom.allocated_exposure_usd = 0.0

        ranked_ids = [a.id for a in anomalies]

        return AnomalyReport(
            anomalies=anomalies,
            solutions_scanned=solutions_scanned,
            solutions_untrusted=list(set(solutions_untrusted)),
            ranked_ids=ranked_ids,
            tests_skipped=tests_skipped,
            total_exposure_usd=round(total_exposure, 2)
        )

    # =========================================================================
    # Test A: Target-based SLA Breach Engine
    # =========================================================================
    def _run_test_a_target_breach(
        self,
        sol: SolutionResult,
        ev_map: Dict[str, str],
        dataset_df: Optional[pd.DataFrame],
        recent_df: Optional[pd.DataFrame],
        counter: int,
        is_reliable: bool,
        sol_exposure_usd: float,
        assumption_src: str,
        family_id: str
    ) -> Optional[Anomaly]:
        sol_code = sol.solution
        sla_info = self.sla_targets.get(sol_code)
        if not sla_info:
            return None

        target_val = float(sla_info["target"])
        unit = sla_info.get("unit", "")
        bad_dir = sla_info.get("bad_direction", "higher")
        target_type = sla_info.get("target_type", sol.target_type)

        # Use recent evaluation window if available, else overall baseline
        eval_df = recent_df if recent_df is not None else dataset_df
        window_label = "recent_window" if recent_df is not None else "overall"

        observed = sol.kpi_baseline
        n_samples = sol.n_train + sol.n_test
        ci: Optional[List[float]] = None
        compliance_pct: Optional[float] = None
        compliance_ci: Optional[List[float]] = None

        if eval_df is not None and sol.target_column in eval_df.columns:
            series = eval_df[sol.target_column].dropna()
            # Handle right-censoring: exclude incomplete/pending orders if column exists
            if "is_delivered" in eval_df.columns:
                series = eval_df[eval_df["is_delivered"] == 1][sol.target_column].dropna()
            elif "order_status" in eval_df.columns:
                series = eval_df[eval_df["order_status"].isin(["Delivered", "Completed"])][sol.target_column].dropna()

            total_count = len(series)
            if total_count >= 10:
                n_samples = total_count
                if target_type in ["binary", "binary_rate"]:
                    pos_count = int(series.sum())
                    observed = pos_count / total_count
                    low_ci, high_ci = wilson_score_interval(pos_count, total_count)
                    ci = [round(low_ci, 4), round(high_ci, 4)]
                else:
                    observed = float(series.mean())
                    # Compute SLA compliance rate: percentage of orders meeting SLA target
                    if bad_dir == "higher":
                        compliant_count = int((series <= target_val).sum())
                    else:
                        compliant_count = int((series >= target_val).sum())
                    compliance_pct = round((compliant_count / total_count) * 100.0, 1)
                    low_c_ci, high_c_ci = wilson_score_interval(compliant_count, total_count)
                    compliance_ci = [round(low_c_ci * 100.0, 1), round(high_c_ci * 100.0, 1)]

        # Check breach condition direction-aware
        breached = False
        ratio = 1.0

        if bad_dir == "higher":
            if observed > target_val:
                breached = True
                ratio = observed / max(target_val, 1e-6)
        elif bad_dir == "lower":
            if observed < target_val:
                breached = True
                ratio = target_val / max(observed, 1e-6)

        if not breached:
            return None

        # Severity grading
        crit_f = self.severity_cfg.get("critical_factor", 1.50)
        high_f = self.severity_cfg.get("high_factor", 1.20)

        if ratio >= crit_f:
            sev = Severity.CRITICAL
        elif ratio >= high_f:
            sev = Severity.HIGH
        else:
            sev = Severity.MEDIUM

        min_samples = self.quality_cfg.get("min_sample_size", 50)
        is_small_sample = n_samples < min_samples
        if is_small_sample and sev == Severity.CRITICAL:
            sev = Severity.HIGH

        anom_id = f"ANOM-{sol_code}-{counter:03d}"
        key_baseline = f"{sol.target_column}.baseline"
        ev_ids = [ev_map[key_baseline]] if key_baseline in ev_map else []

        # Executive phrasing with SLA Compliance Rate
        if bad_dir == "higher":
            pct_diff = ((observed - target_val) / target_val) * 100.0
            desc = (
                f"SLA Target Breach: Average {sol.kpi_name} of {observed:.2f} {unit} is "
                f"{pct_diff:.1f}% above your SLA target of {target_val:.2f} {unit}"
            )
        else:
            pct_diff = ((target_val - observed) / target_val) * 100.0
            desc = (
                f"SLA Target Breach: {sol.kpi_name} achieved {observed:.2f} {unit}, which is "
                f"{pct_diff:.1f}% below your target standard of {target_val:.2f} {unit}"
            )

        if compliance_pct is not None:
            desc += f" (Only {compliance_pct:.1f}% of orders met SLA; 95% CI: [{compliance_ci[0]}%, {compliance_ci[1]}%])."
        else:
            desc += "."

        # Dynamically discover operational segment and cohort from uploaded dataset
        anom_seg_dim = "Operational Segment"
        anom_seg_val = f"Primary Operations ({sol.kpi_name})"
        anom_window = window_label if recent_df is not None else "Historical Operations"

        if eval_df is not None and not eval_df.empty:
            # Check for candidate categorical segment columns in user's dataset
            candidate_segment_cols = [
                ("Shipping Lane / Corridor", ["shipping_lane", "corridor", "route", "lane"]),
                ("Distribution Center", ["dc_id", "warehouse_id", "facility_id", "plant_id", "dc"]),
                ("Primary Vendor", ["supplier_id", "vendor_id", "supplier", "vendor"]),
                ("Product Category", ["category", "product_category", "segment", "merchandise_category"]),
                ("Commercial Channel", ["channel", "sales_channel", "region", "country"]),
            ]
            for dim_label, col_names in candidate_segment_cols:
                found_col = next((c for c in col_names if c in eval_df.columns), None)
                if found_col:
                    try:
                        top_val = str(eval_df[found_col].dropna().value_counts().index[0])
                        anom_seg_dim = dim_label
                        anom_seg_val = f"{dim_label}: {top_val}"
                        break
                    except Exception:
                        pass

            # Detect real temporal window from date/time columns if available
            date_cols = [c for c in eval_df.columns if any(kw in c.lower() for kw in ["date", "time", "period", "month", "year"])]
            if date_cols:
                try:
                    d_series = pd.to_datetime(eval_df[date_cols[0]].dropna(), errors="coerce").dropna()
                    if not d_series.empty:
                        d_min = d_series.min().strftime("%b %Y")
                        d_max = d_series.max().strftime("%b %Y")
                        anom_window = f"{d_min} – {d_max}" if d_min != d_max else d_min
                except Exception:
                    pass

        return Anomaly(
            id=anom_id,
            solution=sol_code,
            family_id=family_id,
            type="threshold_breach",
            test_type="target_sla",
            severity=sev,
            description=desc,
            metric=sol.target_column,
            unit=unit,
            direction=bad_dir,
            observed=round(observed, 4),
            threshold=round(target_val, 4),
            n=n_samples,
            window=anom_window,
            segment_dimension=anom_seg_dim,
            segment_value=anom_seg_val,
            compliance_rate=compliance_pct,
            compliance_ci=compliance_ci,
            confidence_interval=ci,
            financial_weight=sol_exposure_usd,
            solution_exposure_usd=sol_exposure_usd,
            allocated_exposure_usd=sol_exposure_usd,
            assumption_source=assumption_src,
            is_primary_exposure=True,
            evidence_ids=ev_ids,
            model_reliable=is_reliable
        )

    # =========================================================================
    # Test B: Temporal Window Drift & Regression Engine
    # Uses Mann-Whitney U / Welch's t-test + Cohen's d effect size
    # =========================================================================
    def _run_test_b_temporal_drift(
        self,
        sol: SolutionResult,
        ev_map: Dict[str, str],
        recent_df: Optional[pd.DataFrame],
        ref_df: Optional[pd.DataFrame],
        counter: int,
        is_reliable: bool,
        sol_exposure_usd: float,
        assumption_src: str,
        family_id: str
    ) -> Optional[Anomaly]:
        if recent_df is None or ref_df is None or sol.target_column not in recent_df.columns:
            return None

        recent_vals = recent_df[sol.target_column].dropna().values
        ref_vals = ref_df[sol.target_column].dropna().values

        if len(recent_vals) < 20 or len(ref_vals) < 20:
            return None

        sol_code = sol.solution
        direction_info = self.directions.get(sol_code, {})
        bad_dir = direction_info.get("bad_direction", "higher")
        unit = direction_info.get("unit", "")

        recent_mean = float(np.mean(recent_vals))
        ref_mean = float(np.mean(ref_vals))

        # 1. Statistical Significance (Mann-Whitney U for skew-resilience, or Welch's t-test)
        if self.temporal_cfg.get("use_mann_whitney_for_skewed", True) and sol.target_type == "continuous":
            stat_res = stats.mannwhitneyu(recent_vals, ref_vals, alternative="two-sided")
            p_val = float(stat_res.pvalue)
        else:
            t_stat, p_val = stats.ttest_ind(recent_vals, ref_vals, equal_var=False)
            p_val = float(p_val)

        alpha = self.temporal_cfg.get("significance_alpha", 0.05)
        if p_val > alpha or np.isnan(p_val):
            return None

        # 2. Effect Size & Cohen's d (Prevents crying wolf on 50k rows with tiny 0.1-day shifts)
        cohens_d = compute_cohens_d(recent_vals, ref_vals)
        min_cohens_d = self.temporal_cfg.get("min_cohens_d", 0.20)
        min_effect_pct = self.temporal_cfg.get("min_effect_size_pct", 10.0)

        regressed = False
        effect_pct = 0.0

        if bad_dir == "higher" and recent_mean > ref_mean:
            effect_pct = ((recent_mean - ref_mean) / max(abs(ref_mean), 1e-6)) * 100.0
            if effect_pct >= min_effect_pct and cohens_d >= min_cohens_d:
                regressed = True
        elif bad_dir == "lower" and recent_mean < ref_mean:
            effect_pct = ((ref_mean - recent_mean) / max(abs(ref_mean), 1e-6)) * 100.0
            if effect_pct >= min_effect_pct and cohens_d >= min_cohens_d:
                regressed = True

        if not regressed:
            return None

        sev = Severity.HIGH if effect_pct >= 25.0 or cohens_d >= 0.50 else Severity.MEDIUM

        anom_id = f"ANOM-{sol_code}-{counter:03d}"
        key_baseline = f"{sol.target_column}.baseline"
        ev_ids = [ev_map[key_baseline]] if key_baseline in ev_map else []

        recent_days = self.temporal_cfg.get("recent_days", 30)
        ref_days = self.temporal_cfg.get("reference_days", 90)

        desc = (
            f"Temporal Drift Detected: Recent {recent_days}-day {sol.kpi_name} ({recent_mean:.2f} {unit}) "
            f"degraded by {effect_pct:.1f}% compared to prior baseline ({ref_mean:.2f} {unit}, p={p_val:.4f}, Cohen's d={cohens_d:.2f})."
        )

        return Anomaly(
            id=anom_id,
            solution=sol_code,
            family_id=family_id,
            type="kpi_regression",
            test_type="temporal_drift",
            severity=sev,
            description=desc,
            metric=sol.target_column,
            unit=unit,
            direction=bad_dir,
            observed=round(recent_mean, 4),
            threshold=round(ref_mean, 4),
            n=len(recent_vals) + len(ref_vals),
            window=f"recent_{recent_days}d_vs_prior_{ref_days}d",
            p_value=round(p_val, 4),
            effect_size=round(effect_pct, 2),
            cohens_d=round(cohens_d, 2),
            financial_weight=sol_exposure_usd,
            solution_exposure_usd=sol_exposure_usd,
            allocated_exposure_usd=0.0,  # Deduplicated under family
            assumption_source=assumption_src,
            is_primary_exposure=False,
            evidence_ids=ev_ids,
            model_reliable=is_reliable
        )

    # =========================================================================
    # Test C: Segment Hotspot Engine with Benjamini-Hochberg FDR Correction
    # =========================================================================
    def _run_test_c_hotspots(
        self,
        sol: SolutionResult,
        ev_map: Dict[str, str],
        dataset_df: Optional[pd.DataFrame],
        counter: int,
        is_reliable: bool,
        sol_exposure_usd: float,
        assumption_src: str,
        family_id: str
    ) -> List[Anomaly]:
        hotspots = []
        if dataset_df is None or sol.target_column not in dataset_df.columns:
            return hotspots

        dimensions = self.hotspot_cfg.get("dimensions", ["supplier_id", "dc_id", "category"])
        min_n = self.hotspot_cfg.get("min_segment_samples", 30)
        dev_factor = self.hotspot_cfg.get("deviation_factor", 1.30)
        fdr_alpha = self.hotspot_cfg.get("fdr_alpha", 0.05)
        max_card = self.hotspot_cfg.get("max_cardinality_per_dimension", 50)

        sol_code = sol.solution
        direction_info = self.directions.get(sol_code, {})
        bad_dir = direction_info.get("bad_direction", "higher")
        unit = direction_info.get("unit", "")
        fleet_series = dataset_df[sol.target_column].dropna()
        fleet_mean = float(fleet_series.mean())

        for dim in dimensions:
            if dim not in dataset_df.columns:
                continue

            unique_vals = dataset_df[dim].value_counts()
            if len(unique_vals) > max_card:
                candidate_keys = list(unique_vals.head(max_card).index)
            else:
                candidate_keys = list(unique_vals.index)

            segment_tests = []
            for seg_key in candidate_keys:
                seg_data = dataset_df[dataset_df[dim] == seg_key][sol.target_column].dropna().values
                rest_data = dataset_df[dataset_df[dim] != seg_key][sol.target_column].dropna().values

                if len(seg_data) < min_n or len(rest_data) < min_n:
                    continue

                seg_mean = float(np.mean(seg_data))
                ratio = seg_mean / max(fleet_mean, 1e-6) if bad_dir == "higher" else fleet_mean / max(seg_mean, 1e-6)

                # Only test candidates deviating in bad direction
                if ratio < dev_factor:
                    continue

                # Run Welch's t-test against rest of fleet
                t_stat, p_val = stats.ttest_ind(seg_data, rest_data, equal_var=False)
                if not np.isnan(p_val):
                    segment_tests.append({
                        "dim": dim,
                        "key": str(seg_key),
                        "seg_mean": seg_mean,
                        "fleet_mean": fleet_mean,
                        "ratio": ratio,
                        "n": len(seg_data),
                        "p_val": float(p_val)
                    })

            if not segment_tests:
                continue

            # Apply Benjamini-Hochberg FDR correction across tested segments
            raw_p_vals = [t["p_val"] for t in segment_tests]
            if len(raw_p_vals) == 1:
                fdr_mask = [raw_p_vals[0] <= fdr_alpha]
            else:
                try:
                    fdr_res = stats.false_discovery_control(raw_p_vals)
                    fdr_mask = [p_adj <= fdr_alpha for p_adj in fdr_res]
                except Exception:
                    # Fallback manual Benjamini-Hochberg
                    sorted_indices = np.argsort(raw_p_vals)
                    m = len(raw_p_vals)
                    fdr_mask = [False] * m
                    for rank, idx in enumerate(sorted_indices, 1):
                        if raw_p_vals[idx] <= (rank / m) * fdr_alpha:
                            fdr_mask[idx] = True

            for idx, test in enumerate(segment_tests):
                if not fdr_mask[idx]:
                    continue

                pct_worse = (test["ratio"] - 1.0) * 100.0
                sev = Severity.HIGH if test["ratio"] >= 1.50 else Severity.MEDIUM
                anom_id = f"ANOM-{sol_code}-{counter:03d}"
                counter += 1

                desc = (
                    f"Operational Hotspot in {test['dim']} '{test['key']}': Average {sol.kpi_name} is {test['seg_mean']:.2f} {unit}, "
                    f"which is {pct_worse:.1f}% worse than fleet average ({test['fleet_mean']:.2f} {unit}, p_adj < {fdr_alpha}, n={test['n']} orders)."
                )

                hotspots.append(
                    Anomaly(
                        id=anom_id,
                        solution=sol_code,
                        family_id=family_id,
                        type="hotspot",
                        test_type="segment_hotspot",
                        severity=sev,
                        description=desc,
                        metric=f"{sol.target_column}.{test['dim']}.{test['key']}",
                        unit=unit,
                        direction=bad_dir,
                        observed=round(test["seg_mean"], 4),
                        threshold=round(test["fleet_mean"], 4),
                        n=test["n"],
                        p_value=round(test["p_val"], 4),
                        effect_size=round(pct_worse, 2),
                        financial_weight=sol_exposure_usd,
                        solution_exposure_usd=sol_exposure_usd,
                        allocated_exposure_usd=0.0,  # Deduplicated under family
                        assumption_source=assumption_src,
                        is_primary_exposure=False,
                        evidence_ids=[ev_map[f"{sol.target_column}.baseline"]] if f"{sol.target_column}.baseline" in ev_map else [],
                        model_reliable=is_reliable
                    )
                )
                if len(hotspots) >= 2:
                    break

        return hotspots

    # =========================================================================
    # Test D: Tail Risk (Distribution Insight)
    # =========================================================================
    def _run_test_d_tail_risk(
        self,
        sol: SolutionResult,
        ev_map: Dict[str, str],
        dataset_df: Optional[pd.DataFrame],
        counter: int,
        is_reliable: bool,
        sol_exposure_usd: float,
        assumption_src: str,
        family_id: str
    ) -> Optional[Anomaly]:
        if dataset_df is None or sol.target_column not in dataset_df.columns:
            return None

        if sol.target_type == "binary":
            return None

        series = dataset_df[sol.target_column].dropna()
        if len(series) < 50:
            return None

        sol_code = sol.solution
        direction_info = self.directions.get(sol_code, {})
        bad_dir = direction_info.get("bad_direction", "higher")
        unit = direction_info.get("unit", "")

        p_up = self.tail_cfg.get("percentile_upper", 90.0)
        p90_val = float(np.percentile(series, p_up))
        mean_val = float(series.mean())

        spread_ratio = p90_val / max(mean_val, 1e-6) if bad_dir == "higher" else mean_val / max(p90_val, 1e-6)
        if spread_ratio < 1.40:
            return None

        tail_orders = int((series >= p90_val).sum()) if bad_dir == "higher" else int((series <= p90_val).sum())

        anom_id = f"ANOM-{sol_code}-{counter:03d}"
        desc = (
            f"Tail Risk Insight: The worst 10% of orders for {sol.kpi_name} exceed {p90_val:.2f} {unit} "
            f"(Fleet average: {mean_val:.2f} {unit}), affecting {tail_orders:,} historical orders."
        )

        return Anomaly(
            id=anom_id,
            solution=sol_code,
            family_id=family_id,
            type="tail_risk",
            test_type="tail_risk",
            severity=Severity.MEDIUM,
            description=desc,
            metric=f"{sol.target_column}.p90",
            unit=unit,
            direction=bad_dir,
            observed=round(p90_val, 4),
            threshold=round(mean_val, 4),
            n=len(series),
            financial_weight=sol_exposure_usd,
            solution_exposure_usd=sol_exposure_usd,
            allocated_exposure_usd=0.0,
            assumption_source=assumption_src,
            is_primary_exposure=False,
            evidence_ids=[ev_map[f"{sol.target_column}.baseline"]] if f"{sol.target_column}.baseline" in ev_map else [],
            model_reliable=is_reliable
        )

    # =========================================================================
    # Model Reliability & Balanced Accuracy Audit
    # =========================================================================
    def _check_model_reliability(self, sol: SolutionResult, ev_map: Dict[str, str]) -> Tuple[bool, List[str], bool]:
        reasons = []
        is_leakage = False
        holdout = sol.holdout

        # Minimum sample size
        min_rows = self.reliability_cfg.get("min_test_rows", 100)
        if 0 < sol.n_test < min_rows:
            reasons.append(f"Insufficient test holdout size ({sol.n_test} < {min_rows} rows)")

        # Classification floors (Balanced Accuracy & Imbalance Handling)
        if sol.target_type == "binary":
            min_bal_acc = self.reliability_cfg.get("classification", {}).get("min_balanced_accuracy", 65.0)
            min_auc = self.reliability_cfg.get("classification", {}).get("min_auc", 0.65)
            min_f1 = self.reliability_cfg.get("classification", {}).get("min_f1", 0.60)

            auc = holdout.get("auc_score", holdout.get("auc", 1.0))
            f1 = holdout.get("f1_score", holdout.get("f1", 1.0))
            bal_acc = holdout.get("balanced_accuracy", holdout.get("accuracy_pct", 100.0))

            if bal_acc < min_bal_acc:
                reasons.append(f"Balanced accuracy {bal_acc:.1f}% below minimum floor {min_bal_acc:.1f}% (vulnerable to majority-class guessing)")
            if auc < min_auc:
                reasons.append(f"AUC {auc:.3f} below minimum floor {min_auc:.3f}")
            if f1 < min_f1:
                reasons.append(f"F1 score {f1:.3f} below minimum floor {min_f1:.3f}")

        # Regression floors
        elif sol.target_type == "continuous":
            min_r2 = self.reliability_cfg.get("regression", {}).get("min_r2", 0.30)
            max_r2_susp = self.reliability_cfg.get("regression", {}).get("max_r2_suspicious", 0.98)
            r2 = holdout.get("r2", holdout.get("r2_pct", 100.0) / 100.0 if holdout.get("r2_pct", 0.0) > 1.0 else holdout.get("r2_pct", 1.0))

            if r2 < min_r2:
                reasons.append(f"R² score {r2:.3f} below reliability floor {min_r2:.3f}")
            elif r2 > max_r2_susp:
                is_leakage = True
                reasons.append(f"Suspiciously high R² of {r2:.3f} on operational data indicates potential data leakage")

        return (len(reasons) == 0, reasons, is_leakage)

    # =========================================================================
    # Helpers: Window Splitting & Censoring Handling
    # =========================================================================
    def _split_temporal_windows(self, dataset_df: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
        if dataset_df is None:
            return (None, None, "No dataset provided")
        if len(dataset_df) < 50:
            return (None, None, f"Insufficient rows for temporal partitioning (n={len(dataset_df)} < 50)")

        df = dataset_df.copy()
        if "date" in df.columns:
            try:
                df["_parsed_date"] = pd.to_datetime(df["date"])
                df = df.sort_values("_parsed_date")
                max_date = df["_parsed_date"].max()
                recent_days = self.temporal_cfg.get("recent_days", 30)
                cutoff = max_date - pd.Timedelta(days=recent_days)

                recent_df = df[df["_parsed_date"] >= cutoff]
                ref_df = df[df["_parsed_date"] < cutoff]

                if len(recent_df) >= 20 and len(ref_df) >= 20:
                    return (recent_df, ref_df, None)
                else:
                    return (None, None, f"Temporal windows too sparse (recent={len(recent_df)}, ref={len(ref_df)})")
            except Exception as e:
                return (None, None, f"Date parsing failed: {e}")

        # If no date column exists, return clean skip note
        return (None, None, "Dataset lacks 'date' column for temporal drift analysis")
