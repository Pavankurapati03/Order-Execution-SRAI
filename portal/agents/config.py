"""
Configuration Manager for Quantellix Agentic AI.
Loads YAML defaults and merges optional customer-specific overrides from Neon PostgreSQL.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

CONFIG_DIR = Path(__file__).resolve().parent / "config"


def is_agentic_enabled() -> bool:
    """Checks whether the agentic decision layer is enabled."""
    return os.getenv("AGENTIC_ENABLED", "False").lower() in ("true", "1", "yes")


def get_llm_settings() -> Dict[str, Any]:
    """Retrieves LLM provider, models, and execution budgets from environment."""
    return {
        "provider": os.getenv("AGENT_LLM_PROVIDER", "openai"),
        "model_fast": os.getenv("AGENT_LLM_MODEL_FAST", "gpt-4o-mini"),
        "model_strong": os.getenv("AGENT_LLM_MODEL_STRONG", "gpt-4o-mini"),
        "token_budget": int(os.getenv("AGENT_TOKEN_BUDGET_PER_RUN", "40000")),
        "credit_cost": int(os.getenv("AGENT_CREDIT_COST", "0")),
        "run_sync": os.getenv("RUN_AGENTS_SYNC", "True").lower() in ("true", "1", "yes"),
        "reanalyze_rate_limit": int(os.getenv("AGENT_REANALYZE_RATE_LIMIT", "5")),
    }


def _load_yaml(filename: str) -> Dict[str, Any]:
    filepath = CONFIG_DIR / filename
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_thresholds_config(customer_id: Optional[str] = None) -> Dict[str, Any]:
    """Loads default threshold configuration and merges optional tenant overrides."""
    base = _load_yaml("thresholds.default.yaml")
    # In Phase 2/Phase 6: query agent_customer_config table if customer_id is provided
    return base


def get_causal_graph_config() -> Dict[str, Any]:
    """Loads domain causal dependency graph configuration."""
    return _load_yaml("causal_graph.yaml")


def get_semantic_glossary(customer_id: Optional[str] = None) -> Dict[str, Any]:
    """Loads semantic glossary definitions for feature translation."""
    base = _load_yaml("semantic_glossary.yaml")
    # In Phase 3: merge customer overrides from agent_semantic_mappings table
    return base


def get_financial_assumptions(customer_id: Optional[str] = None) -> Dict[str, Any]:
    """Loads default financial valuation and dollar exposure parameters."""
    return _load_yaml("financial_assumptions.default.yaml")
