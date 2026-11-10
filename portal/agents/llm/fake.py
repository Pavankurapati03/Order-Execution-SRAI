"""
Deterministic Fake LLM Client for unit, integration, and safety tests.
Requires zero network connection, operates with 0ms latency, and returns typed responses.
"""

from typing import Type, TypeVar, Optional, Any, Dict
from pydantic import BaseModel
from portal.agents.llm.client import BaseLLMClient
from portal.agents.llm.budget import BudgetTracker

T = TypeVar("T", bound=BaseModel)


class FakeLLMClient(BaseLLMClient):
    """
    Mock LLM client allowing predefined or heuristic structured outputs in test environments.
    """
    def __init__(self, canned_responses: Optional[Dict[str, Any]] = None, budget_tracker: Optional[BudgetTracker] = None):
        super().__init__(budget_tracker=budget_tracker)
        self.canned_responses = canned_responses or {}
        self.call_history = []

    def set_canned(self, response_model_name: str, payload: dict):
        self.canned_responses[response_model_name] = payload

    def generate_structured(
        self,
        prompt_text: str,
        response_model: Type[T],
        model_name: Optional[str] = None,
        max_tokens: int = 1500,
        temperature: float = 0.0,
        system_instruction: str = ""
    ) -> T:
        self.call_history.append({
            "prompt": prompt_text,
            "model_name": model_name,
            "response_model": response_model.__name__
        })
        if self.budget_tracker:
            self.budget_tracker.record_usage(prompt_tokens=150, completion_tokens=80)

        # 1. Use explicit canned response if registered
        m_name = response_model.__name__
        if m_name in self.canned_responses:
            data = self.canned_responses[m_name]
            return response_model.model_validate(data)

        # 2. Otherwise generate minimal valid defaults based on field types
        fields = response_model.model_fields
        default_payload = {}
        for f_name, f_info in fields.items():
            if not f_info.is_required():
                continue
            if f_name == "severity":
                default_payload[f_name] = "medium"
            elif f_name == "type":
                default_payload[f_name] = "threshold_breach"
            elif f_name == "evidence_ids":
                default_payload[f_name] = ["EV-E4-001"]
            elif f_name == "observed":
                default_payload[f_name] = 8.4
            elif f_name == "threshold":
                default_payload[f_name] = 7.0
            else:
                default_payload[f_name] = f"test_{f_name}"

        return response_model.model_validate(default_payload)
