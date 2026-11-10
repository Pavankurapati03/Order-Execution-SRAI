"""
LLM Client Interface & Multi-Provider Dispatcher.
Enforces typed Pydantic responses, budget limits, and deterministic zero-temperature execution.
"""

import os
import json
from typing import Type, TypeVar, Dict, Any, Optional
from pydantic import BaseModel
from portal.agents.llm.budget import BudgetTracker

T = TypeVar("T", bound=BaseModel)


class BaseLLMClient:
    """Abstract base class for all LLM clients."""
    def __init__(self, budget_tracker: Optional[BudgetTracker] = None):
        self.budget_tracker = budget_tracker or BudgetTracker()

    def generate_structured(
        self,
        prompt_text: str,
        response_model: Type[T],
        model_name: Optional[str] = None,
        max_tokens: int = 1500,
        temperature: float = 0.0,
        system_instruction: str = "You are a specialized decision intelligence agent. Output only valid JSON adhering to the schema."
    ) -> T:
        raise NotImplementedError


class OpenAILLMClient(BaseLLMClient):
    """Production client using OpenAI with strict JSON mode."""
    def __init__(self, api_key: Optional[str] = None, budget_tracker: Optional[BudgetTracker] = None):
        super().__init__(budget_tracker=budget_tracker)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"[LLM] OpenAI client initialization note: {e}")

    def generate_structured(
        self,
        prompt_text: str,
        response_model: Type[T],
        model_name: Optional[str] = None,
        max_tokens: int = 1500,
        temperature: float = 0.0,
        system_instruction: str = "You are an analytical supply chain agent. Output strictly valid JSON."
    ) -> T:
        if not self.client:
            raise RuntimeError("OpenAI client not configured or missing API key.")

        model = model_name or os.getenv("AGENT_LLM_MODEL_FAST", "gpt-4o-mini")
        schema = response_model.model_json_schema()
        
        messages = [
            {"role": "system", "content": f"{system_instruction}\nRespond with JSON matching schema: {json.dumps(schema)}"},
            {"role": "user", "content": prompt_text}
        ]

        # First attempt
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                temperature=temperature
            )
            usage = resp.usage
            if usage and self.budget_tracker:
                self.budget_tracker.record_usage(usage.prompt_tokens, usage.completion_tokens)

            raw_json = resp.choices[0].message.content or "{}"
            parsed = json.loads(raw_json)
            return response_model.model_validate(parsed)
        except Exception as first_err:
            # Retry once with error feedback
            try:
                messages.append({"role": "user", "content": f"Previous JSON validation error: {first_err}. Please fix and return compliant JSON."})
                retry_resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                usage = retry_resp.usage
                if usage and self.budget_tracker:
                    self.budget_tracker.record_usage(usage.prompt_tokens, usage.completion_tokens)

                raw_json = retry_resp.choices[0].message.content or "{}"
                parsed = json.loads(raw_json)
                return response_model.model_validate(parsed)
            except Exception as retry_err:
                raise RuntimeError(f"LLM structured generation failed after retry: {retry_err}")
