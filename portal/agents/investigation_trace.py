"""
Investigation Trace Engine.
Executes goal-directed multi-step pipeline investigations via deterministic tool invocation
(Step -> Action -> Action Input -> Observation -> Step ... -> Final Decision Brief).

Core Invariants:
1. Deterministic step progression: The trace executes verified tool stages based on observation.
2. Complete audit trace: Every step, action, tool input, and observation is recorded.
3. Zero-hallucination math: All numerical figures derive strictly from verified tools and SXI predictors.
4. Terminology enforcement: Exclusively SXI predictor / artifact (never prohibited terms).
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from portal.agents.llm.budget import BudgetExceeded, BudgetTracker
from portal.agents.schemas import DecisionDossier, SolutionResult
from portal.agents.synthesizer import ExecutiveSynthesizer
from portal.agents.tools import AgentToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class InvestigationStep:
    """A single diagnostic investigation step."""
    step: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: Dict[str, Any]
    duration_ms: float


@dataclass
class InvestigationTraceResult:
    """Complete trace and final dossier produced by the Investigation Trace Engine."""
    run_id: str
    session_id: str
    user_goal: str
    status: str
    total_steps: int
    total_duration_ms: float
    trace: List[InvestigationStep] = field(default_factory=list)
    dossier: Optional[DecisionDossier] = None
    final_answer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "user_goal": self.user_goal,
            "status": self.status,
            "total_steps": self.total_steps,
            "total_duration_ms": self.total_duration_ms,
            "trace": [asdict(t) for t in self.trace],
            "dossier": self.dossier.model_dump() if self.dossier else None,
            "final_answer": self.final_answer,
        }


class InvestigationTraceAgent:
    """
    Investigation Trace Orchestrator for Order Execution Decision Intelligence.
    Systematically investigates anomalies, discovers root causes, and compiles decision dossiers.
    """

    def __init__(
        self,
        tool_registry: AgentToolRegistry,
        max_steps: int = 6,
        max_duration_seconds: float = 30.0,
        budget_tracker: Optional[BudgetTracker] = None,
        llm: Optional[Any] = None,
    ):
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.max_duration_seconds = max_duration_seconds
        self.budget_tracker = budget_tracker or (getattr(llm, "budget_tracker", None) if llm else None)
        self.llm = llm

    def execute_step(
        self,
        step_num: int,
        thought: str,
        action: str,
        action_input: Dict[str, Any],
        trace: List[InvestigationStep],
        start_time: float,
    ) -> Tuple[InvestigationStep, bool, Optional[str]]:
        """
        Executes a single investigation step with safety invariants and kill-switch protections:
        - Max steps enforcement
        - Time limit enforcement
        - Runaway loop detection (consecutive duplicate calls)
        - Token budget tracking & exhaustion kill-switch
        - Unknown tool and malformed argument isolation
        Returns (InvestigationStep, should_abort, abort_status).
        """
        # 1. Max steps kill-switch
        if step_num > self.max_steps:
            obs = {"status": "error", "error": f"Kill-switch activated: Max step limit ({self.max_steps}) exceeded."}
            step = InvestigationStep(step=step_num, thought=thought, action=action, action_input=action_input, observation=obs, duration_ms=0.0)
            return step, True, "aborted_max_steps"

        # 2. Max duration kill-switch
        if start_time > 0 and (time.perf_counter() - start_time) > self.max_duration_seconds:
            obs = {"status": "error", "error": f"Kill-switch activated: Execution time exceeded ({self.max_duration_seconds}s)."}
            step = InvestigationStep(step=step_num, thought=thought, action=action, action_input=action_input, observation=obs, duration_ms=0.0)
            return step, True, "aborted_timeout"

        # 3. Runaway loop detection: check if last 2 steps were identical action and identical input
        if len(trace) >= 2:
            prev1 = trace[-1]
            prev2 = trace[-2]
            if prev1.action == action and prev2.action == action and prev1.action_input == action_input and prev2.action_input == action_input:
                obs = {"status": "error", "error": f"Kill-switch activated: Runaway loop detected on action '{action}'."}
                step = InvestigationStep(step=step_num, thought=thought, action=action, action_input=action_input, observation=obs, duration_ms=0.0)
                return step, True, "aborted_loop_detected"

        # 4. Token budget tracking
        if self.budget_tracker:
            try:
                self.budget_tracker.record_usage(prompt_tokens=250, completion_tokens=100)
            except BudgetExceeded as be:
                obs = {"status": "error", "error": f"Kill-switch activated: {str(be)}"}
                step = InvestigationStep(step=step_num, thought=thought, action=action, action_input=action_input, observation=obs, duration_ms=0.0)
                return step, True, "aborted_budget_exceeded"

        # 5. Safe tool execution via registry
        t0 = time.perf_counter()
        obs = self.tool_registry.execute_tool(action, **action_input)
        d = round((time.perf_counter() - t0) * 1000.0, 2)
        step = InvestigationStep(step=step_num, thought=thought, action=action, action_input=action_input, observation=obs, duration_ms=d)
        return step, False, None

    def _run_llm_react_loop(
        self,
        user_goal: str,
        start_time: float,
    ) -> Optional[Tuple[List[InvestigationStep], str]]:
        """
        Executes an authentic Autonomous ReAct Agent Loop using Google Gemini native Tool Calling.
        The LLM autonomously decides which tool to call based on observed telemetry.
        Returns (trace, final_answer) or None if unavailable.
        """
        try:
            from portal.gemini_service import get_gemini_api_key
            api_key = get_gemini_api_key()
            if not api_key:
                return None

            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            # Build Google GenAI Tool Declarations from our AgentToolRegistry
            tool_declarations = []
            for t in self.tool_registry.list_tools():
                tool_declarations.append(types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=t["parameters"],
                ))

            if not tool_declarations:
                return None

            gemini_tool = types.Tool(function_declarations=tool_declarations)

            config = types.GenerateContentConfig(
                temperature=0.1,
                system_instruction=(
                    "You are Quantellix.AI's Autonomous Supply Chain Decision Agent. "
                    "Your mission is to systematically investigate operational telemetry, detect SLA breaches, "
                    "isolate root causes via TreeSHAP ML, trace cross-department causal propagation, "
                    "and calculate deduplicated financial exposure. "
                    "To do this, use your analytical tools step-by-step. "
                    "At each step, invoke the appropriate tool to gather empirical evidence. "
                    "Once you have gathered sufficient facts across anomaly detection, root causes, and causal propagation, "
                    "synthesize your final executive diagnostic brief."
                ),
                tools=[gemini_tool]
            )

            # Try working candidate models in order of capability and speed
            chat = None
            resp = None
            candidate_models = [
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite",
                "gemini-flash-lite-latest",
                "gemini-3.5-flash",
            ]
            for m in candidate_models:
                try:
                    test_chat = client.chats.create(model=m, config=config)
                    resp = test_chat.send_message(f"Operational Goal: {user_goal}. Begin your diagnostic investigation using your tools.")
                    chat = test_chat
                    logger.info("Autonomous ReAct Agent connected to Gemini model: %s", m)
                    break
                except Exception as e_chat:
                    logger.debug("Chat init/send failed on %s: %s", m, e_chat)
                    continue

            if not chat or not resp:
                return None

            trace: List[InvestigationStep] = []
            step_num = 1

            while step_num <= self.max_steps and resp.function_calls:
                for fc in resp.function_calls:
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}

                    # Determine descriptive operation thought
                    thought = f"Autonomous Agent Decision: Execute {tool_name} to verify empirical telemetry."
                    if tool_name == "sentinel_scan":
                        thought = "Autonomous Agent Decision: Deep search fleet telemetry to audit operational SLA breaches and identify hotspot corridors."
                    elif tool_name == "explain_feature_drivers":
                        thought = f"Autonomous Agent Decision: Execute TreeSHAP ML feature attribution for Solution {tool_args.get('solution_code', '')} to isolate primary root-cause drivers."
                    elif tool_name == "traverse_causal_path":
                        thought = f"Autonomous Agent Decision: Traverse causal DAG from {tool_args.get('cause_solution', '')} to {tool_args.get('effect_solution', '')} to verify cross-department propagation."
                    elif tool_name == "calculate_financial_exposure":
                        thought = f"Autonomous Agent Decision: Compute deduplicated net financial exposure for {tool_args.get('solution_codes', [])} to eliminate cross-stage double counting."
                    elif tool_name == "simulate_remediation_action":
                        thought = f"Autonomous Agent Decision: Simulate {tool_args.get('scenario_level', 'L2')} remediation scenario policy to quantify capital recovery."

                    t0 = time.perf_counter()
                    obs = self.tool_registry.execute_tool(tool_name, **tool_args)
                    dur = round((time.perf_counter() - t0) * 1000.0, 2)

                    trace_step = InvestigationStep(
                        step=step_num,
                        thought=thought,
                        action=tool_name,
                        action_input=tool_args,
                        observation=obs,
                        duration_ms=dur
                    )
                    trace.append(trace_step)

                    # Send tool execution result back to Gemini
                    part = types.Part.from_function_response(
                        name=tool_name,
                        response={"result": obs}
                    )
                    try:
                        resp = chat.send_message(part)
                    except Exception as e_send:
                        if "503" in str(e_send) or "UNAVAILABLE" in str(e_send):
                            time.sleep(1.0)
                            resp = chat.send_message(part)
                        else:
                            raise e_send
                    step_num += 1
                    break

            final_answer = ""
            if resp and hasattr(resp, "candidates") and resp.candidates:
                candidate = resp.candidates[0]
                if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                    text_parts = [p.text for p in candidate.content.parts if hasattr(p, "text") and p.text]
                    final_answer = " ".join(text_parts).strip()
            if not final_answer:
                final_answer = "Investigation trace completed via Autonomous ReAct Agent."
            return trace, final_answer

        except Exception as e:
            logger.warning("Notice in LLM ReAct loop: %s. Falling back to deterministic pipeline.", e)
            return None

    def run(
        self,
        user_goal: str,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> InvestigationTraceResult:
        """
        Executes the diagnostic investigation trace.
        Attempts true Gemini ReAct Tool Calling Loop first, with seamless fallback to deterministic pipeline.
        """
        start_time = time.perf_counter()
        run_id = run_id or f"auto_{int(time.time())}"
        session_id = session_id or "default_session"

        # -------------------------------------------------------------
        # 1. ATTEMPT AUTHENTIC GEMINI REACT TOOL CALLING LOOP
        # -------------------------------------------------------------
        react_res = self._run_llm_react_loop(user_goal=user_goal, start_time=start_time)
        if react_res and len(react_res[0]) > 0:
            trace, final_answer = react_res
            synthesizer = ExecutiveSynthesizer(llm=self.llm)
            dossier = synthesizer.synthesize(
                run_id=run_id,
                anomaly_report=self.tool_registry.cached_anomaly_report,
                interpreter_report=self.tool_registry.cached_interpreter_report,
                causal_analysis=self.tool_registry.cached_causal_analysis,
                solutions=self.tool_registry.solutions,
                evidence_ledger=self.tool_registry.evidence_builder.to_ledger(),
            )
            total_dur = round((time.perf_counter() - start_time) * 1000.0, 2)
            return InvestigationTraceResult(
                run_id=run_id,
                session_id=session_id,
                user_goal=user_goal,
                status="completed",
                total_steps=len(trace),
                total_duration_ms=total_dur,
                trace=trace,
                dossier=dossier,
                final_answer=final_answer,
            )

        # -------------------------------------------------------------
        # 2. DETERMINISTIC FALLBACK PIPELINE
        # -------------------------------------------------------------
        trace: List[InvestigationStep] = []
        step_num = 1

        # -------------------------------------------------------------
        # STEP 1: Audit solution fleet for SLA breaches & statistical drift
        # -------------------------------------------------------------
        step_label_1 = "Audit solution fleet for SLA breaches and statistical drift"
        action_1 = "sentinel_scan"
        action_input_1: Dict[str, Any] = {}
        st1, abort, status = self.execute_step(step_num, step_label_1, action_1, action_input_1, trace, start_time)
        trace.append(st1)
        if abort:
            return InvestigationTraceResult(run_id=run_id, session_id=session_id, user_goal=user_goal, status=status or "aborted", total_steps=len(trace), total_duration_ms=round((time.perf_counter() - start_time) * 1000.0, 2), trace=trace, final_answer=st1.observation.get("error", "Execution aborted."))
        step_num += 1
        obs_1 = st1.observation

        # Determine primary breached solution from observation
        primary_sol = "E4"
        secondary_sol = "E8"
        anomalies = obs_1.get("anomalies", [])
        if anomalies:
            primary_sol = anomalies[0].get("solution", "E4")
            for a in anomalies[1:]:
                other_sol = a.get("solution")
                if other_sol and other_sol != primary_sol:
                    secondary_sol = other_sol
                    break
            else:
                other_sols = [s.solution for s in self.tool_registry.solutions if s.solution != primary_sol]
                secondary_sol = other_sols[0] if other_sols else primary_sol

        # -------------------------------------------------------------
        # STEP 2: Feature driver attribution and statistical association
        # -------------------------------------------------------------
        step_label_2 = f"Feature driver attribution and statistical association for Solution {primary_sol}"
        action_2 = "explain_feature_drivers"
        action_input_2 = {"solution_code": primary_sol}
        st2, abort, status = self.execute_step(step_num, step_label_2, action_2, action_input_2, trace, start_time)
        trace.append(st2)
        if abort:
            return InvestigationTraceResult(run_id=run_id, session_id=session_id, user_goal=user_goal, status=status or "aborted", total_steps=len(trace), total_duration_ms=round((time.perf_counter() - start_time) * 1000.0, 2), trace=trace, final_answer=st2.observation.get("error", "Execution aborted."))
        step_num += 1
        obs_2 = st2.observation

        # -------------------------------------------------------------
        # STEP 3: Domain causal DAG path traversal
        # -------------------------------------------------------------
        step_label_3 = f"Domain causal DAG path traversal: {primary_sol} -> {secondary_sol}"
        action_3 = "traverse_causal_path"
        action_input_3 = {"cause_solution": primary_sol, "effect_solution": secondary_sol}
        st3, abort, status = self.execute_step(step_num, step_label_3, action_3, action_input_3, trace, start_time)
        trace.append(st3)
        if abort:
            return InvestigationTraceResult(run_id=run_id, session_id=session_id, user_goal=user_goal, status=status or "aborted", total_steps=len(trace), total_duration_ms=round((time.perf_counter() - start_time) * 1000.0, 2), trace=trace, final_answer=st3.observation.get("error", "Execution aborted."))
        step_num += 1
        obs_3 = st3.observation

        # -------------------------------------------------------------
        # STEP 4: Deduplicated net financial exposure computation across causal cascade
        # -------------------------------------------------------------
        anom_codes = [a.get("solution") for a in obs_1.get("anomalies", []) if a.get("solution")]
        seen_sols = set()
        cascade_sols = []
        for c in [primary_sol, secondary_sol] + anom_codes:
            if c and c not in seen_sols:
                seen_sols.add(c)
                cascade_sols.append(c)
        if not cascade_sols:
            cascade_sols = [primary_sol, secondary_sol]

        step_label_4 = f"Deduplicated cross-stage financial exposure valuation for {cascade_sols}"
        action_4 = "calculate_financial_exposure"
        action_input_4 = {"solution_codes": cascade_sols}
        st4, abort, status = self.execute_step(step_num, step_label_4, action_4, action_input_4, trace, start_time)
        trace.append(st4)
        if abort:
            return InvestigationTraceResult(run_id=run_id, session_id=session_id, user_goal=user_goal, status=status or "aborted", total_steps=len(trace), total_duration_ms=round((time.perf_counter() - start_time) * 1000.0, 2), trace=trace, final_answer=st4.observation.get("error", "Execution aborted."))
        step_num += 1
        obs_4 = st4.observation

        # -------------------------------------------------------------
        # STEP 5: Scenario remediation policy lookup (L2)
        # -------------------------------------------------------------
        step_label_5 = f"Scenario remediation policy lookup for Solution {primary_sol} (L2)"
        action_5 = "simulate_remediation_action"
        action_input_5 = {"solution_code": primary_sol, "scenario_level": "L2"}
        st5, abort, status = self.execute_step(step_num, step_label_5, action_5, action_input_5, trace, start_time)
        trace.append(st5)
        if abort:
            return InvestigationTraceResult(run_id=run_id, session_id=session_id, user_goal=user_goal, status=status or "aborted", total_steps=len(trace), total_duration_ms=round((time.perf_counter() - start_time) * 1000.0, 2), trace=trace, final_answer=st5.observation.get("error", "Execution aborted."))
        step_num += 1
        obs_5 = st5.observation

        # -------------------------------------------------------------
        # FINAL SYNTHESIS: Compile Decision Dossier & Run Verifier
        # -------------------------------------------------------------
        synthesizer = ExecutiveSynthesizer(llm=self.llm)
        dossier = synthesizer.synthesize(
            run_id=run_id,
            anomaly_report=self.tool_registry.cached_anomaly_report,
            interpreter_report=self.tool_registry.cached_interpreter_report,
            causal_analysis=self.tool_registry.cached_causal_analysis,
            solutions=self.tool_registry.solutions,
            evidence_ledger=self.tool_registry.evidence_builder.to_ledger(),
        )

        from portal.agents.verifier import verifier
        verification = verifier.verify(dossier, self.tool_registry.evidence_builder.to_ledger())

        net_exp_str = f"${obs_4.get('net_annual_usd', 0):,.2f} USD"
        final_answer = (
            f"Investigation Trace Complete (5 steps executed, verifier_passed={verification.passed}). "
            f"Cascade identified from Solution {primary_sol} to {secondary_sol}. "
            f"Total deduplicated financial exposure: {net_exp_str}. "
            f"Remediation plan '{obs_5.get('action_title')}' staged for Step 5 Human Approval Gate."
        )

        total_dur = round((time.perf_counter() - start_time) * 1000.0, 2)

        return InvestigationTraceResult(
            run_id=run_id,
            session_id=session_id,
            user_goal=user_goal,
            status="completed",
            total_steps=len(trace),
            total_duration_ms=total_dur,
            trace=trace,
            dossier=dossier,
            final_answer=final_answer,
        )


# Backward-compatible alias
TraceStep = InvestigationStep
