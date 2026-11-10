"""
Token Budget Tracking & Execution Telemetry for Agentic Invocations.
"""

class BudgetExceeded(Exception):
    """Raised when token usage for a single investigation exceeds the configured cap."""
    pass


class BudgetTracker:
    """
    Thread-safe or run-scoped tracker for LLM token usage and estimated dollar cost.
    """
    def __init__(self, token_limit: int = 40000):
        self.token_limit = token_limit
        self.tokens_in = 0
        self.tokens_out = 0
        self.call_count = 0

    def record_usage(self, prompt_tokens: int, completion_tokens: int):
        self.tokens_in += prompt_tokens
        self.tokens_out += completion_tokens
        self.call_count += 1
        
        total = self.tokens_in + self.tokens_out
        if total > self.token_limit:
            raise BudgetExceeded(
                f"Token budget exceeded: used {total} tokens (limit: {self.token_limit})."
            )

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def cost_usd_estimate(self) -> float:
        # Based on gpt-4o-mini pricing: $0.15/1M input, $0.60/1M output
        cost_in = (self.tokens_in / 1_000_000.0) * 0.150
        cost_out = (self.tokens_out / 1_000_000.0) * 0.600
        return round(cost_in + cost_out, 6)

    def summary(self) -> dict:
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "cost_usd_est": self.cost_usd_estimate
        }
