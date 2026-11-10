"""
Stage Orchestration Layer: Decision Intelligence Orchestrator.
Executes the complete 4-step UNDERSTAND Stage of the Quantellix Order Execution
Agentic AI framework:

Step 1: Sentinel (Deep Search & Anomaly Detection)
Step 2: Feature Interpreter (Root-Cause Driver & Tree SHAP Explanation)
Step 3: Cross-Domain Causal Reasoner (Dependency Graph & Financial Overlap Deduplication)
Step 4: Executive Synthesis Agent (Decision Dossier Compilation for Step 5 Approval Gate)

Core Invariants:
1. Zero-pickle pure native SXI artifact storage.
2. Complete mathematical accounting invariance: Target Gap = Explained Gap + Unexplained Residual.
3. Financial overlap deduplication across cascading failure modes.
4. Ground-truth Evidence ledger: Every executive claim cites verifiable Evidence IDs.
5. Strict terminology governance: Never use the word "model"; exclusively use SXI predictor / artifact.
"""

import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from portal.agents.adapters import adapt_sxi_to_solution_result
from portal.agents.artifacts import ArtifactStore, get_artifact_store
from portal.agents.config import (
    get_causal_graph_config,
    get_financial_assumptions,
    get_thresholds_config,
)
from portal.agents.evidence import EvidenceBuilder, EvidenceLedger
from portal.agents.interpreter import FeatureInterpreter
from portal.agents.reasoner import CausalReasoner
from portal.agents.schemas import (
    AnomalyReport,
    CausalAnalysis,
    DecisionDossier,
    InterpreterReport,
    SolutionResult,
)
from portal.agents.sentinel import Sentinel
from portal.agents.synthesizer import ExecutiveSynthesizer

logger = logging.getLogger(__name__)


@dataclass
class InvestigationResult:
    """Consolidated outcome of the 4-step UNDERSTAND stage investigation."""
    run_id: str
    session_id: str
    status: str  # "completed", "clean", "warning", "failed"
    duration_ms: float
    step_progress: Dict[str, str] = field(default_factory=dict)
    anomaly_report: Optional[AnomalyReport] = None
    interpreter_report: Optional[InterpreterReport] = None
    causal_analysis: Optional[CausalAnalysis] = None
    dossier: Optional[DecisionDossier] = None
    evidence_count: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes investigation results to a clean, JSON-friendly dictionary."""
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "step_progress": self.step_progress,
            "evidence_count": self.evidence_count,
            "anomaly_report": self.anomaly_report.model_dump() if self.anomaly_report else None,
            "interpreter_report": self.interpreter_report.model_dump() if self.interpreter_report else None,
            "causal_analysis": self.causal_analysis.model_dump() if self.causal_analysis else None,
            "dossier": self.dossier.model_dump() if self.dossier else None,
            "errors": self.errors,
        }


class DecisionIntelligenceOrchestrator:
    """
    Main Orchestrator executing the complete 4-step UNDERSTAND Stage.
    """

    def __init__(
        self,
        artifact_store: Optional[ArtifactStore] = None,
        thresholds_cfg: Optional[Dict[str, Any]] = None,
        causal_graph_cfg: Optional[Dict[str, Any]] = None,
        financial_cfg: Optional[Dict[str, Any]] = None,
        llm: Optional[Any] = None,
    ):
        self.store = artifact_store or get_artifact_store()
        self.thresholds_cfg = thresholds_cfg or get_thresholds_config()
        self.causal_graph_cfg = causal_graph_cfg or get_causal_graph_config()
        self.financial_cfg = financial_cfg or get_financial_assumptions()
        self.llm = llm

    def run_investigation(
        self,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        dataset_df: Optional[pd.DataFrame] = None,
        sxi_results: Optional[Dict[str, Any]] = None,
        solutions: Optional[List[SolutionResult]] = None,
        focal_solutions: Optional[List[str]] = None,
    ) -> InvestigationResult:
        """
        Executes Steps 1 -> 2 -> 3 -> 4 sequentially, tracking state and evidence.
        """
        start_time = time.perf_counter()
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        session_id = session_id or "default_session"

        step_progress: Dict[str, str] = {
            "step_1_sentinel": "pending",
            "step_2_interpreter": "pending",
            "step_3_reasoner": "pending",
            "step_4_synthesizer": "pending",
        }
        errors: List[str] = []
        evidence_builder = EvidenceBuilder()

        # 0. Normalize and validate inputs
        normalized_solutions: List[SolutionResult] = []
        if solutions:
            normalized_solutions = list(solutions)
        elif sxi_results:
            for k, raw in sxi_results.items():
                if isinstance(raw, dict):
                    try:
                        sol_res = adapt_sxi_to_solution_result(raw)
                        normalized_solutions.append(sol_res)
                    except Exception as e:
                        logger.warning("Failed adapting SXI result for solution %s: %s", k, e)

        # -------------------------------------------------------------
        # STEP 1: Sentinel (Deep Search & Anomaly Detection)
        # -------------------------------------------------------------
        anomaly_report: Optional[AnomalyReport] = None
        try:
            sentinel = Sentinel(thresholds_config=self.thresholds_cfg)
            anomaly_report = sentinel.analyze(
                solutions=normalized_solutions,
                ledger=evidence_builder.to_ledger(),
                dataset_df=dataset_df,
                focal_solutions=focal_solutions,
            )
            step_progress["step_1_sentinel"] = "completed"
        except Exception as e:
            msg = f"Sentinel scan failed: {str(e)}"
            logger.exception(msg)
            errors.append(msg)
            step_progress["step_1_sentinel"] = "failed"
            anomaly_report = AnomalyReport(
                anomalies=[],
                solutions_scanned=[s.solution for s in normalized_solutions],
                solutions_untrusted=[],
            )

        # -------------------------------------------------------------
        # STEP 2: Feature Interpreter (Root-Cause Driver & Tree SHAP)
        # -------------------------------------------------------------
        interpreter_report: Optional[InterpreterReport] = None
        try:
            interpreter = FeatureInterpreter(
                store=self.store,
                config=self.thresholds_cfg,
                llm=self.llm,
            )
            raw_sxi_map = sxi_results or {s.solution: s.model_dump() for s in normalized_solutions}
            interpreter_report = interpreter.interpret(
                anomaly_report=anomaly_report,
                sxi_results=raw_sxi_map,
                df=dataset_df,
            )
            step_progress["step_2_interpreter"] = "completed"
        except Exception as e:
            msg = f"Feature Interpreter failed: {str(e)}"
            logger.exception(msg)
            errors.append(msg)
            step_progress["step_2_interpreter"] = "failed"
            interpreter_report = InterpreterReport()

        # -------------------------------------------------------------
        # STEP 3: Causal Reasoner (Dependency DAG & Overlap Deduplication)
        # -------------------------------------------------------------
        causal_analysis: Optional[CausalAnalysis] = None
        try:
            reasoner = CausalReasoner(
                evidence_builder=evidence_builder,
                graph_config=self.causal_graph_cfg,
                financial_config=self.financial_cfg,
                llm=self.llm,
            )
            causal_analysis = reasoner.analyze(
                anomaly_report=anomaly_report,
                interpreter_report=interpreter_report,
                solutions=normalized_solutions,
                df=dataset_df,
                focal_solutions=focal_solutions,
            )
            step_progress["step_3_reasoner"] = "completed"
        except Exception as e:
            msg = f"Causal Reasoner failed: {str(e)}"
            logger.exception(msg)
            errors.append(msg)
            step_progress["step_3_reasoner"] = "failed"
            from portal.agents.schemas import ExposureLedger
            causal_analysis = CausalAnalysis(chains=[], exposure=ExposureLedger())

        # -------------------------------------------------------------
        # STEP 4: Executive Synthesizer (Decision Dossier Compilation)
        # -------------------------------------------------------------
        dossier: Optional[DecisionDossier] = None
        try:
            synthesizer = ExecutiveSynthesizer(llm=self.llm)
            dossier = synthesizer.synthesize(
                run_id=run_id,
                anomaly_report=anomaly_report,
                interpreter_report=interpreter_report,
                causal_analysis=causal_analysis,
                solutions=normalized_solutions,
                evidence_ledger=evidence_builder.to_ledger(),
            )
            step_progress["step_4_synthesizer"] = "completed"
        except Exception as e:
            msg = f"Executive Synthesizer failed: {str(e)}"
            logger.exception(msg)
            errors.append(msg)
            step_progress["step_4_synthesizer"] = "failed"

        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        overall_status = "completed" if not errors else ("warning" if dossier else "failed")
        if overall_status == "completed" and anomaly_report and not anomaly_report.anomalies:
            overall_status = "clean"

        return InvestigationResult(
            run_id=run_id,
            session_id=session_id,
            status=overall_status,
            duration_ms=duration_ms,
            step_progress=step_progress,
            anomaly_report=anomaly_report,
            interpreter_report=interpreter_report,
            causal_analysis=causal_analysis,
            dossier=dossier,
            evidence_count=len(evidence_builder.items),
            errors=errors,
        )
