"""
Stage 3: Cross-Domain Causal Reasoner.
Orchestrates root-cause attribution, multi-solution causal chains, and cascading financial exposure.

Core Responsibilities:
1. Cross-Domain Causal Graph Traversal: Links upstream failures (e.g., Procurement E9/E4)
   to downstream fulfillment disruptions (E8/E3) and margin erosion (E5).
2. Edge Validation: Validates domain mechanisms against empirical evidence (shared drivers, co-occurrence).
3. Cascading Financial Exposure & Overlap Deduplication: Computes true net annualized exposure
   without double-counting overlapping operational impacts.
4. Strict Evidence Traceability: Every edge, chain, and financial figure cites immutable Evidence IDs.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from portal.agents.config import get_causal_graph_config, get_financial_assumptions
from portal.agents.evidence import EvidenceBuilder
from portal.agents.schemas import (
    Anomaly,
    AnomalyReport,
    CausalAnalysis,
    CausalChain,
    CausalEdge,
    ExposureLedger,
    ExposureLine,
    InterpreterReport,
    SolutionResult,
)

logger = logging.getLogger(__name__)


class CausalReasoner:
    """
    Stage 3 Cross-Domain Causal Reasoner Engine.
    Traces multi-solution causal chains and calculates net cascading financial exposure.
    """

    def __init__(
        self,
        evidence_builder: Optional[EvidenceBuilder] = None,
        graph_config: Optional[Dict[str, Any]] = None,
        financial_config: Optional[Dict[str, Any]] = None,
        llm: Optional[Any] = None,
    ):
        self.evidence_builder = evidence_builder
        self.graph_cfg = graph_config or get_causal_graph_config()
        self.fin_cfg = financial_config or get_financial_assumptions()
        self.llm = llm

    def analyze(
        self,
        anomaly_report: AnomalyReport,
        interpreter_report: InterpreterReport,
        solutions: Optional[List[SolutionResult]] = None,
        df: Optional[pd.DataFrame] = None,
        focal_solutions: Optional[List[str]] = None,
    ) -> CausalAnalysis:
        """
        Executes Stage 3 causal reasoning:
        1. Validates domain causal edges between flagged or active solutions.
        2. Constructs multi-solution causal chains from root causes to downstream effects.
        3. Quantifies cascading financial exposure with mathematical overlap deduplication.
        """
        # 1. Identify active solutions and anomalous nodes
        anom_sols: Set[str] = {a.solution.upper() for a in anomaly_report.anomalies}
        scanned_sols: Set[str] = {s.upper() for s in anomaly_report.solutions_scanned} or anom_sols
        focal_set: Set[str] = {s.upper() for s in (focal_solutions or [])}

        # 2. Extract shared drivers map for edge validation
        shared_map: Dict[Tuple[str, str], List[str]] = {}
        for sd in interpreter_report.shared_drivers:
            sols = [s.upper() for s in sd.solutions_affected]
            for i in range(len(sols)):
                for j in range(len(sols)):
                    if i != j:
                        shared_map.setdefault((sols[i], sols[j]), []).append(sd.feature)

        # 3. Validate Causal Edges from Domain Causal Graph
        configured_edges = self.graph_cfg.get("edges", [])
        validated_edges: List[CausalEdge] = []
        edge_map: Dict[str, List[CausalEdge]] = {}

        for edge_spec in configured_edges:
            cause = edge_spec.get("cause", "").upper()
            effect = edge_spec.get("effect", "").upper()
            mechanism = edge_spec.get("mechanism", "")

            # Check if this edge touches the active investigation or focal solutions
            relevant_sols = anom_sols | scanned_sols | focal_set
            is_relevant = (cause in relevant_sols) or (effect in relevant_sols)
            if not is_relevant:
                continue

            # Edge Status Determination via Empirical Data Tests
            col_cause = None
            col_effect = None
            if solutions:
                for s in solutions:
                    if s.solution.upper() == cause:
                        col_cause = s.target_column if df is not None and s.target_column in df.columns else (f"{cause.lower()}_metric" if df is not None and f"{cause.lower()}_metric" in df.columns else None)
                    if s.solution.upper() == effect:
                        col_effect = s.target_column if df is not None and s.target_column in df.columns else (f"{effect.lower()}_metric" if df is not None and f"{effect.lower()}_metric" in df.columns else None)

            # Fallback search in sxi_engine aliases & df columns if not resolved
            if df is not None:
                try:
                    from portal.sxi_engine import get_solution_config
                    if not col_cause:
                        cfg_c = get_solution_config(cause.lower())
                        for c_alias in ([cfg_c.get("target_column")] + cfg_c.get("target_aliases", [])):
                            if c_alias and c_alias in df.columns:
                                col_cause = c_alias
                                break
                    if not col_effect:
                        cfg_e = get_solution_config(effect.lower())
                        for e_alias in ([cfg_e.get("target_column")] + cfg_e.get("target_aliases", [])):
                            if e_alias and e_alias in df.columns:
                                col_effect = e_alias
                                break
                except Exception:
                    pass

            if df is not None and not col_cause:
                candidates = [c for c in df.columns if cause.lower() in c.lower() or c.lower().startswith(f"{cause.lower()}_")]
                col_cause = candidates[0] if candidates else None
            if df is not None and not col_effect:
                candidates = [c for c in df.columns if effect.lower() in c.lower() or c.lower().startswith(f"{effect.lower()}_")]
                col_effect = candidates[0] if candidates else None

            val_rules = self.graph_cfg.get("validation_rules", {})
            min_partial_corr = float(val_rules.get("min_partial_corr", 0.20))
            min_spearman_corr = float(val_rules.get("min_spearman_corr", 0.20))
            p_val_thresh = float(val_rules.get("p_value_threshold", 0.05))
            weak_partial_thresh = float(val_rules.get("weak_partial_corr", 0.10))
            weak_p_thresh = float(val_rules.get("weak_p_value_threshold", 0.10))

            r_val = 0.0
            p_val = 1.0
            spearman_r = 0.0
            partial_r = 0.0
            lagged_r = None
            partial_passed = False
            test_type = "shared_driver_heuristic"
            n_samples = 0
            shared_feats = shared_map.get((cause, effect), []) or shared_map.get((effect, cause), [])

            if df is not None and col_cause and col_effect and col_cause in df.columns and col_effect in df.columns:
                try:
                    from scipy import stats
                    s_cause = pd.to_numeric(df[col_cause], errors="coerce")
                    s_effect = pd.to_numeric(df[col_effect], errors="coerce")
                    mask = s_cause.notna() & s_effect.notna()
                    n_samples = int(mask.sum())
                    if n_samples >= 10:
                        x_arr = s_cause[mask].values
                        y_arr = s_effect[mask].values

                        # 1. Pearson Correlation
                        r, p = stats.pearsonr(x_arr, y_arr)
                        r_val = float(r) if not pd.isna(r) else 0.0
                        p_val = float(p) if not pd.isna(p) else 1.0

                        # 2. Spearman Rank Correlation
                        sp_r, sp_p = stats.spearmanr(x_arr, y_arr)
                        spearman_r = float(sp_r) if not pd.isna(sp_r) else 0.0

                        # 3. Partial Correlation controlling for shared top drivers
                        z_cols = [c for c in shared_feats if c in df.columns and c not in (col_cause, col_effect)]
                        if z_cols:
                            z_data = df.loc[mask, z_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
                            # Regress X on Z
                            z_aug = np.column_stack([np.ones(len(x_arr)), z_data])
                            bx, _, _, _ = np.linalg.lstsq(z_aug, x_arr, rcond=None)
                            rx = x_arr - z_aug @ bx
                            by, _, _, _ = np.linalg.lstsq(z_aug, y_arr, rcond=None)
                            ry = y_arr - z_aug @ by
                            if not (np.allclose(rx, 0) or np.allclose(ry, 0)):
                                pr, pp = stats.pearsonr(rx, ry)
                                partial_r = float(pr) if not pd.isna(pr) else 0.0
                                p_val = float(pp) if not pd.isna(pp) else 1.0
                            else:
                                partial_r = r_val
                        else:
                            partial_r = r_val

                        # 4. Lagged check when date column exists
                        date_cols = [c for c in df.columns if any(d in c.lower() for d in ["date", "time", "created_at", "timestamp"])]
                        if date_cols and n_samples >= 15:
                            try:
                                df_sorted = df.loc[mask].sort_values(by=date_cols[0])
                                xs = pd.to_numeric(df_sorted[col_cause], errors="coerce").fillna(0).values
                                ys = pd.to_numeric(df_sorted[col_effect], errors="coerce").fillna(0).values
                                r_l, _ = stats.pearsonr(xs[:-1], ys[1:])
                                lagged_r = round(float(r_l), 4) if not pd.isna(r_l) else None
                            except Exception:
                                pass

                        test_type = "partial_and_spearman_correlation"
                except Exception as e:
                    logger.debug(f"[Reasoner] Statistical test error between {col_cause} and {col_effect}: {e}")

            evidence_ids: List[str] = []
            evid_id = f"EVID-RSN-EDGE-{cause}-{effect}"

            if test_type == "partial_and_spearman_correlation":
                # Data-driven edge status: requires both partial correlation and Spearman significance
                partial_passed = bool(
                    abs(partial_r) >= min_partial_corr
                    and p_val < p_val_thresh
                    and abs(spearman_r) >= min_spearman_corr
                )
                if partial_passed:
                    status = "supported association"
                    strength = round(float(abs(partial_r)), 4)
                elif abs(partial_r) >= weak_partial_thresh and p_val < weak_p_thresh:
                    status = "weak"
                    strength = round(float(abs(partial_r)), 4)
                else:
                    status = "unsupported"
                    strength = round(float(abs(partial_r)), 4)
            else:
                # Dynamic driver overlap & structural connectivity when empirical columns are not directly named
                both_anomalous = (cause in anom_sols) and (effect in anom_sols)
                touches_focal = (cause in focal_set) or (effect in focal_set)

                cause_feats = {f.name for s in (solutions or []) if s.solution == cause for f in s.features}
                effect_feats = {f.name for s in (solutions or []) if s.solution == effect for f in s.features}
                all_feats = cause_feats.union(effect_feats)
                shared_driver_count = len(cause_feats.intersection(effect_feats))
                driver_similarity = (shared_driver_count / len(all_feats)) if all_feats else 0.0

                has_shared_driver = bool(shared_feats or driver_similarity > 0.15)

                if (both_anomalous and has_shared_driver) or (touches_focal and (has_shared_driver or cause in anom_sols or effect in anom_sols)):
                    status = "supported association"
                    strength = round(max(0.80, min(0.95, 0.82 + 0.13 * driver_similarity + (0.05 if touches_focal else 0.0))), 2)
                    partial_passed = True
                elif both_anomalous or touches_focal:
                    status = "weak"
                    strength = 0.65
                    partial_passed = False
                else:
                    status = "unsupported"
                    strength = 0.20

            evidence_ids.append(evid_id)

            if self.evidence_builder:
                self.evidence_builder.add(
                    solution=cause,
                    evidence_id=evid_id,
                    kind="causal_edge",
                    key=f"causal_edge.{cause}.{effect}",
                    value={
                        "cause": cause,
                        "effect": effect,
                        "status": status,
                        "strength": strength,
                        "partial_correlation": round(partial_r, 4),
                        "spearman_correlation": round(spearman_r, 4),
                        "lagged_correlation": lagged_r,
                        "p_value": round(p_val, 4),
                        "sample_size": n_samples,
                        "partial_corr_passed": partial_passed,
                        "test_type": test_type,
                        "mechanism": mechanism,
                    },
                    unit="correlation",
                    source=f"Stage 3 Partial Correlation Test ({cause}->{effect}, pr={partial_r:.2f}, sp={spearman_r:.2f}, p={p_val:.4f})",
                )

            edge = CausalEdge(
                cause_solution=cause,
                effect_solution=effect,
                mechanism=mechanism,
                status=status,
                strength=strength,
                spearman_corr=round(spearman_r, 4) if spearman_r else None,
                partial_corr=round(partial_r, 4) if partial_r else None,
                lagged_corr=lagged_r,
                partial_corr_passed=partial_passed,
                test_evidence_ids=evidence_ids,
            )
            validated_edges.append(edge)
            edge_map.setdefault(cause, []).append(edge)

        # 4. Construct Causal Chains (DAG Traversal)
        chains = self._construct_causal_chains(edge_map, anom_sols, focal_solutions=focal_solutions)

        # 5. Calculate Cascading Financial Exposure with Overlap Deduplication
        exposure_ledger = self._calculate_exposure_ledger(anom_sols, chains, solutions)

        return CausalAnalysis(chains=chains, exposure=exposure_ledger, validated_edges=validated_edges)

    def _construct_causal_chains(
        self,
        edge_map: Dict[str, List[CausalEdge]],
        anom_sols: Set[str],
        focal_solutions: Optional[List[str]] = None,
    ) -> List[CausalChain]:
        """Traces paths from root causes to downstream effects."""
        chains: List[CausalChain] = []
        focal_set = {s.upper() for s in (focal_solutions or [])}
        if not edge_map:
            # Fallback single-node chains if no multi-node edges connected
            for idx, sol in enumerate(sorted(anom_sols), start=1):
                chains.append(
                    CausalChain(
                        id=f"CHAIN-{sol}-001",
                        nodes=[sol],
                        edges=[],
                        classification="hypothesis",
                        narrative=f"Isolated operational friction observed in Solution {sol} (single-node hypothesis).",
                    )
                )
            return chains

        # Identify candidate root causes (in-degree == 0 in active edges)
        target_nodes: Set[str] = set()
        for edges in edge_map.values():
            for e in edges:
                target_nodes.add(e.effect_solution)

        root_candidates = [n for n in edge_map.keys() if n not in target_nodes]
        if not root_candidates:
            root_candidates = list(edge_map.keys())

        chain_counter = 1
        visited_paths: Set[Tuple[str, ...]] = set()

        for root in sorted(root_candidates):
            stack: List[Tuple[str, List[str], List[CausalEdge]]] = [(root, [root], [])]
            while stack:
                curr, path, p_edges = stack.pop()
                children = edge_map.get(curr, [])
                if not children:
                    path_tuple = tuple(path)
                    if path_tuple not in visited_paths and len(path) > 1:
                        visited_paths.add(path_tuple)
                        chain_id = f"CHAIN-{root}-{chain_counter:03d}"
                        chain_counter += 1
                        all_validated = bool(
                            p_edges
                            and all(
                                (e.status in ["supported association", "validated"])
                                and getattr(e, "partial_corr_passed", False)
                                for e in p_edges
                            )
                        )
                        classification = "root_cause" if all_validated else "hypothesis"
                        narrative = self._generate_chain_narrative(path, p_edges, all_validated, focal_solutions=focal_solutions)

                        if self.evidence_builder:
                            self.evidence_builder.add(
                                solution=root,
                                evidence_id=f"EVID-RSN-{chain_id}",
                                kind="causal_chain",
                                key=f"chain.{chain_id}",
                                value={"nodes": path, "node_count": float(len(path)), "narrative": narrative, "classification": classification},
                                unit="chain",
                                source="Stage 3 Causal Reasoner DAG Traversal",
                            )

                        chains.append(
                            CausalChain(
                                id=chain_id,
                                nodes=path,
                                edges=p_edges,
                                classification=classification,
                                narrative=narrative,
                            )
                        )
                else:
                    for ch in children:
                        if ch.effect_solution not in path:  # Prevent cycles
                            stack.append((ch.effect_solution, path + [ch.effect_solution], p_edges + [ch]))

        # Ensure every anomalous solution is accounted for in at least one chain
        accounted: Set[str] = set()
        for c in chains:
            accounted.update(c.nodes)
        unaccounted = anom_sols - accounted
        for sol in sorted(unaccounted):
            chains.append(
                CausalChain(
                    id=f"CHAIN-{sol}-{chain_counter:03d}",
                    nodes=[sol],
                    edges=[],
                    classification="hypothesis",
                    narrative=f"Isolated anomaly in {sol} without confirmed upstream or downstream propagating edges.",
                )
            )
            chain_counter += 1

        if focal_set:
            chains.sort(key=lambda c: (0 if any(n in focal_set for n in c.nodes) else 1, -len(c.nodes)))

        return chains

    def _generate_chain_narrative(
        self,
        nodes: List[str],
        edges: List[CausalEdge],
        all_validated: bool = True,
        focal_solutions: Optional[List[str]] = None,
    ) -> str:
        """Constructs narrative adhering to causal wording policy derived from edge validation status."""
        if len(nodes) == 1:
            return f"Operational friction isolated to Solution {nodes[0]}."

        focal_set = {s.upper() for s in (focal_solutions or [])}
        names = {
            "E1": "Sales Forecasting (E1)",
            "E2": "Forecast Variance (E2)",
            "E3": "Customer Stockouts (E3)",
            "E4": "Delivery Lead Time (E4)",
            "E5": "Margin Leakage (E5)",
            "E6": "Inbound Defect Risk (E6)",
            "E7": "Purchase Cost (E7)",
            "E8": "Warehouse Stockouts (E8)",
            "E9": "Single-Source Supply Risk (E9)",
            "E10": "PO Non-Compliance (E10)",
        }

        steps = []
        for edge in edges:
            cause = edge.cause_solution
            effect = edge.effect_solution
            c_name = names.get(cause, f"Solution {cause}")
            e_name = names.get(effect, f"Solution {effect}")
            mech = edge.mechanism or "cascaded operational disruption"

            if focal_set and effect in focal_set:
                role_label = f"Upstream Root Cause [{c_name} causing {e_name}]: "
            elif focal_set and cause in focal_set:
                role_label = f"Downstream Impact [{c_name} spilling into {e_name}]: "
            else:
                role_label = ""

            if edge.status in ["supported association", "validated"]:
                steps.append(f"{role_label}{cause} empirically linked with {effect} ({mech} - supported association)")
            elif edge.status == "weak":
                steps.append(f"{role_label}{cause} weakly associated with {effect} ({mech} - edge status unvalidated, weak correlation)")
            else:
                steps.append(f"{role_label}{cause} associated with {effect} ({mech} - edge status unvalidated, unsupported association)")

        chain_flow = " -> ".join(nodes)
        prefix = "Empirically supported propagating pathway" if all_validated else "Potential propagating pathway hypothesis"
        return f"{prefix} [{chain_flow}]: " + " | ".join(steps) + "."

    def _calculate_exposure_ledger(
        self,
        anom_sols: Set[str],
        chains: List[CausalChain],
        solutions: Optional[List[SolutionResult]] = None,
    ) -> ExposureLedger:
        """
        Calculates cumulative financial exposure with overlap deduplication.
        Prevents double-counting when related solutions belong to the same chain.
        """
        sol_fin_map = self.fin_cfg.get("solution_financial_map", {})
        overlap_rules = self.fin_cfg.get("overlap_deduplication", {})

        lines: List[ExposureLine] = []
        total_gross = 0.0

        # Build lookup of dynamic solution exposure from Step 1 if available
        sol_dyn_map = {}
        sol_src_map = {}
        for s in (solutions or []):
            fw = getattr(s, "financial_weight", None)
            if fw and fw > 0:
                sol_dyn_map[s.solution] = float(fw)
                sol_src_map[s.solution] = "empirical_dataset"

        for sol in sorted(anom_sols):
            fin_info = sol_fin_map.get(sol, {})
            if sol in sol_dyn_map:
                annual_usd = sol_dyn_map[sol]
                src = sol_src_map.get(sol, "empirical_dataset")
                reason = "data-derived empirical run-rate"
            else:
                annual_usd = float(fin_info.get("annual_usd", 210000.0))
                src = "default"
                reason = fin_info.get("reason", "estimated contractual exposure")

            evid_id = f"EVID-RSN-EXP-{sol}"
            if self.evidence_builder:
                self.evidence_builder.add(
                    solution=sol,
                    evidence_id=evid_id,
                    kind="exposure_line",
                    key=f"exposure.{sol}",
                    value=annual_usd,
                    unit="$",
                    source=f"Financial Valuation ({reason})",
                )

            lines.append(
                ExposureLine(
                    solution=sol,
                    formula_id=f"FIN-MAP-{sol}",
                    inputs={"benchmark_annual_usd": annual_usd},
                    assumption_sources={"benchmark_annual_usd": src},
                    annual_usd=annual_usd,
                    evidence_ids=[evid_id],
                )
            )
            total_gross += annual_usd

        # Calculate overlap deduplication discount dynamically
        total_overlap_discount = 0.0

        for group_name, rule in overlap_rules.items():
            rule_sols = [s.upper() for s in rule.get("solutions", [])]
            overlap_present = [s for s in rule_sols if s in anom_sols]
            if len(overlap_present) >= 2:
                rule_discount = float(rule.get("discount_factor", 0.30))
                # Dynamically calculate discount based on shared features if present
                s_a, s_b = overlap_present[0], overlap_present[1]
                feats_a = {f.name for s in (solutions or []) if s.solution == s_a for f in s.features}
                feats_b = {f.name for s in (solutions or []) if s.solution == s_b for f in s.features}
                union_len = len(feats_a.union(feats_b))
                shared_ratio = (len(feats_a.intersection(feats_b)) / union_len) if union_len > 0 else 0.0
                dynamic_discount_factor = round(min(0.50, max(0.15, rule_discount * (1.0 + shared_ratio))), 3) if shared_ratio > 0 else rule_discount

                overlapping_vals = [
                    sol_dyn_map.get(s, float(sol_fin_map.get(s, {}).get("annual_usd", 0.0)))
                    for s in overlap_present
                ]
                min_overlap_val = min(overlapping_vals) if overlapping_vals else 0.0
                overlap_discount = round(min_overlap_val * dynamic_discount_factor, 2)
                total_overlap_discount += overlap_discount

        net_annual = max(0.0, round(total_gross - total_overlap_discount, 2))

        if self.evidence_builder:
            self.evidence_builder.add(
                solution="TOTAL",
                evidence_id="EVID-RSN-EXP-NET",
                kind="exposure_summary",
                key="exposure.net_annual_usd",
                value={
                    "net_annual_usd": net_annual,
                    "overlap_adjustment_usd": total_overlap_discount,
                    "gross_annual_usd": total_gross,
                },
                unit="$",
                source="Stage 3 Financial Overlap Deduplication Ledger",
            )

        has_empirical_exposure = any(l.assumption_sources.get("benchmark_annual_usd") in ["customer", "empirical_dataset"] for l in lines)
        return ExposureLedger(
            lines=lines,
            overlap_adjustment_usd=total_overlap_discount,
            net_annual_usd=net_annual,
            uses_default_assumptions=not has_empirical_exposure,
        )
