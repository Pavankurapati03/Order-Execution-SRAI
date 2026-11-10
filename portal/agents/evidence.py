"""
Evidence Builder & Evidence Ledger:
Builds immutable, deterministic EvidenceItem records from SXI solution outputs.
Enforces the core rule: Every claim generated in downstream dossiers must cite an evidence ID.
"""

import hashlib
import json
from typing import List, Dict, Optional, Any
from portal.agents.schemas import EvidenceItem, SolutionResult


class EvidenceLedger:
    """
    Searchable, hashable in-memory repository of all EvidenceItems for a specific run.
    """
    def __init__(self, items: List[EvidenceItem]):
        self._items: Dict[str, EvidenceItem] = {item.id: item for item in items}
        # Compute deterministic content hash across all items
        canonical_str = json.dumps([item.model_dump() for item in sorted(items, key=lambda x: x.id)], sort_keys=True, default=str)
        self._hash: str = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def get(self, evidence_id: str) -> Optional[EvidenceItem]:
        """Retrieve an item by exact ID."""
        return self._items.get(evidence_id)

    def find(self, solution: Optional[str] = None, key: Optional[str] = None, kind: Optional[str] = None) -> List[EvidenceItem]:
        """Search items by solution code, key prefix, or evidence kind."""
        results = []
        for item in self._items.values():
            if solution and item.solution != solution:
                continue
            if key and not item.key.startswith(key):
                continue
            if kind and item.kind != kind:
                continue
            results.append(item)
        return results

    def all(self) -> List[EvidenceItem]:
        """Return all evidence items sorted by ID."""
        return sorted(list(self._items.values()), key=lambda x: x.id)

    def content_hash(self) -> str:
        """Returns stable SHA256 hash representing all ground truth evidence."""
        return self._hash

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, evidence_id: str) -> bool:
        return evidence_id in self._items


class EvidenceBuilder:
    """
    Constructs an EvidenceLedger from normalized SolutionResult models,
    or dynamically records atomic evidence items with strict deterministic citation IDs.
    """
    def __init__(self, initial_items: Optional[List[EvidenceItem]] = None):
        self._items: List[EvidenceItem] = list(initial_items or [])

    def add(
        self,
        solution: str,
        evidence_id: str,
        kind: str,
        key: str,
        value: Any,
        unit: Optional[str] = None,
        source: str = "",
    ) -> EvidenceItem:
        """Adds a single ground-truth evidence item to the builder."""
        item = EvidenceItem(
            id=evidence_id,
            solution=solution,
            kind=kind,
            key=key,
            value=value,
            unit=unit,
            source=source,
        )
        self._items.append(item)
        return item

    @property
    def items(self) -> List[EvidenceItem]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return True

    def to_ledger(self) -> EvidenceLedger:
        """Converts accumulated evidence items into an immutable EvidenceLedger."""
        return EvidenceLedger(self._items)

    @staticmethod
    def build_from_solutions(solutions: List[SolutionResult]) -> EvidenceLedger:
        all_items: List[EvidenceItem] = []

        # Sort solutions by solution code for strict determinism
        sorted_solutions = sorted(solutions, key=lambda s: s.solution)

        for sol in sorted_solutions:
            idx = 1
            sol_code = sol.solution

            def make_id() -> str:
                nonlocal idx
                res = f"EV-{sol_code}-{idx:03d}"
                idx += 1
                return res

            # 1. Baseline and Optimized KPIs
            all_items.append(EvidenceItem(
                id=make_id(),
                solution=sol_code,
                kind="kpi",
                key=f"{sol.target_column}.baseline",
                value=sol.kpi_baseline,
                source=f"sxi_engine.{sol.solution}.baseline_kpi"
            ))

            for lvl_key in ["L1", "L2", "L3"]:
                if lvl_key in sol.kpi_optimized:
                    all_items.append(EvidenceItem(
                        id=make_id(),
                        solution=sol_code,
                        kind="scenario",
                        key=f"{sol.target_column}.optimized.{lvl_key}",
                        value=sol.kpi_optimized[lvl_key],
                        source=f"sxi_engine.{sol.solution}.scenario_{lvl_key}"
                    ))

            # 2. SXI Scores
            all_items.append(EvidenceItem(
                id=make_id(),
                solution=sol_code,
                kind="sxi_score",
                key="sxi.baseline",
                value=sol.sxi_baseline,
                unit="/10",
                source=f"sxi_engine.{sol.solution}.current_sxi_score"
            ))
            all_items.append(EvidenceItem(
                id=make_id(),
                solution=sol_code,
                kind="sxi_score",
                key="sxi.optimized",
                value=sol.sxi_optimized,
                unit="/10",
                source=f"sxi_engine.{sol.solution}.target_sxi_score"
            ))

            # 3. Model Holdout Metrics & Sample Sizes
            for m_key, m_val in sorted(sol.holdout.items()):
                all_items.append(EvidenceItem(
                    id=make_id(),
                    solution=sol_code,
                    kind="model_metric",
                    key=f"holdout.{m_key}",
                    value=m_val,
                    source=f"sxi_engine.{sol.solution}.holdout.{m_key}"
                ))

            all_items.append(EvidenceItem(
                id=make_id(),
                solution=sol_code,
                kind="model_metric",
                key="samples.n_train",
                value=sol.n_train,
                unit="rows",
                source=f"sxi_engine.{sol.solution}.train_samples"
            ))
            all_items.append(EvidenceItem(
                id=make_id(),
                solution=sol_code,
                kind="model_metric",
                key="samples.n_test",
                value=sol.n_test,
                unit="rows",
                source=f"sxi_engine.{sol.solution}.test_samples"
            ))

            # 4. Top Feature Weights (up to top 10)
            for f_idx, feat in enumerate(sol.features[:10]):
                all_items.append(EvidenceItem(
                    id=make_id(),
                    solution=sol_code,
                    kind="feature_importance",
                    key=f"feature.{feat.name}.weight",
                    value=round(feat.weight, 4),
                    unit="importance",
                    source=f"sxi_engine.{sol.solution}.weights_table[rank={f_idx+1}]"
                ))

            # 5. Data Quality & Leakage Flags
            for dq_key, dq_val in sorted(sol.data_quality.items()):
                all_items.append(EvidenceItem(
                    id=make_id(),
                    solution=sol_code,
                    kind="data_quality",
                    key=f"data_quality.{dq_key}",
                    value=dq_val,
                    unit="%",
                    source=f"sxi_engine.{sol.solution}.data_quality.{dq_key}"
                ))

            for leak in sol.leakage_suspects:
                all_items.append(EvidenceItem(
                    id=make_id(),
                    solution=sol_code,
                    kind="model_metric",
                    key=f"leakage_suspect.{leak}",
                    value=True,
                    source=f"sxi_engine.{sol.solution}.leakage_guard"
                ))

        return EvidenceLedger(all_items)
