"""
Data Contracts and Schema Definitions for Quantellix Agentic AI (Stage 1: UNDERSTAND).
Enforced using Pydantic v2.
"""

from __future__ import annotations
from enum import Enum
from typing import Literal, Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, ConfigDict


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceItem(BaseModel):
    """
    Atomic unit of ground truth extracted from SXI, datasets, or SXI predictors.
    Every claim generated in the dossier must cite one or more evidence IDs.
    """
    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Stable deterministic ID, e.g. EV-E4-001 or EVID-EXP-GLB-E4-001")
    solution: str = Field(..., description="Solution code, e.g. E4")
    kind: str = Field(..., description="Evidence category (e.g. kpi, sxi_score, global_driver, local_finding_contribution)")
    key: str = Field(..., description="Machine key, e.g. lead_time_days.baseline")
    value: Any = None
    unit: Optional[str] = None
    source: str = Field(default="", description="Module/function/result field of origin")


class FeatureWeight(BaseModel):
    """Normalized feature importance from ML model."""
    name: str
    weight: float = Field(..., ge=0.0, description="Normalized importance weight (0..1)")
    direction: Literal["up", "down", "unknown"] = "unknown"
    actionable: bool = True
    mean: Optional[float] = None
    p10: Optional[float] = None
    p90: Optional[float] = None


class SolutionResult(BaseModel):
    """Normalized view of a single SXI solution result."""
    solution: str = Field(..., description="Code e.g. E1..E10")
    target_column: str
    target_type: Literal["continuous", "binary"]
    kpi_name: str
    kpi_baseline: float
    kpi_optimized: Dict[str, float] = Field(default_factory=dict, description="{'L1': .., 'L2': .., 'L3': ..}")
    sxi_baseline: float
    sxi_optimized: float
    models_run: List[str] = Field(default_factory=list)
    primary_model: str = "XGBoost Ensemble"
    holdout: Dict[str, float] = Field(default_factory=dict, description="Holdout accuracy, auc, f1, rmse, r2")
    n_train: int = 0
    n_test: int = 0
    features: List[FeatureWeight] = Field(default_factory=list)
    leakage_suspects: List[str] = Field(default_factory=list)
    feature_aliases: Dict[str, str] = Field(default_factory=dict, description="dropped -> kept")
    data_quality: Dict[str, float] = Field(default_factory=dict, description="missing_pct, outlier_pct, etc.")


# =========================================================================
# Stage 1: Sentinel (Deep Search) Schemas
# =========================================================================

class Anomaly(BaseModel):
    id: str = Field(..., description="e.g. ANOM-E4-001")
    solution: str = Field(..., description="e.g. E4")
    family_id: str = Field(default="", description="e.g. FAM-E4 for deduplicated grouping")
    type: Literal[
        "threshold_breach", "kpi_regression", "hotspot", "tail_risk",
        "model_untrusted", "leakage_suspect", "feature_concentration",
        "data_quality", "small_sample", "low_sxi", "outlier_cluster"
    ] = "hotspot"
    test_type: Literal[
        "target_sla", "temporal_drift", "segment_hotspot", "tail_risk",
        "model_quality", "leakage_risk", "concentration_risk", "data_quality"
    ] = "target_sla"
    severity: Severity = Severity.MEDIUM
    description: str = ""
    metric: str = ""
    unit: Optional[str] = None
    direction: Literal["higher", "lower", "neutral"] = "higher"
    observed: Union[float, str, int] = 0.0
    threshold: Optional[Union[float, str, int]] = None
    kpi_baseline: Optional[float] = None
    sla_target: Optional[float] = None
    n: Optional[int] = None
    window: Optional[str] = None
    compliance_rate: Optional[float] = None
    compliance_ci: Optional[List[float]] = None
    confidence_interval: Optional[List[float]] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    cohens_d: Optional[float] = None
    is_primary_exposure: bool = True
    allocated_exposure_usd: float = 0.0
    solution_exposure_usd: float = 0.0
    assumption_source: Literal["customer", "default", "empirical_dataset"] = "default"
    financial_weight: float = 0.0
    evidence_ids: List[str] = Field(default_factory=list)
    model_reliable: bool = True
    # Stage 2 Interpreter cohort partitioning attributes
    target_column: Optional[str] = None
    segment_dimension: Optional[str] = None
    segment_value: Optional[Union[str, int, float]] = None


class AnomalyReport(BaseModel):
    anomalies: List[Anomaly] = Field(default_factory=list)
    solutions_scanned: List[str] = Field(default_factory=list)
    solutions_untrusted: List[str] = Field(default_factory=list, description="Excluded from causal reasoning")
    ranked_ids: List[str] = Field(default_factory=list, description="Priority order of anomaly IDs")
    tests_skipped: Dict[str, str] = Field(default_factory=dict, description="Reasons for tests not run e.g. missing date")
    total_exposure_usd: float = 0.0


# =========================================================================
# Stage 2: Feature Interpreter Schemas (The 3-Question Architecture)
# =========================================================================

DriverRole = Literal["lever", "context", "symptom_leakage"]
DriverDirection = Literal["positive", "negative", "neutral"]


class GlobalDriverExplanation(BaseModel):
    """Answers Question 1: Global driver role, direction, effect size, stability, and clusters."""
    solution: str
    feature: str
    business_label: str
    role: DriverRole
    direction: DriverDirection
    effect_size_raw: float = 0.0
    effect_unit: str = ""
    effect_display: str = ""
    stability_score: float = 0.0  # 0.0 to 1.0 based on multi-model ensemble agreement
    importance_score: float = 0.0
    importance_pct: float = 0.0
    importance_rank: int = 1
    cluster_features: List[str] = Field(default_factory=list, description="Collinear twin features grouped together")
    evidence_ids: List[str] = Field(default_factory=list)
    template_explanation: str = ""
    curated_explanation: str = ""
    generated_by: Literal["template", "llm"] = "template"
    prompt_version: str = "v2.1"


class FindingContribution(BaseModel):
    """Tree SHAP decomposition contribution for a specific Sentinel anomaly finding."""
    feature: str
    business_label: str
    finding_val: float
    baseline_val: float
    kpi_delta_contribution: float
    kpi_unit: str
    display: str = ""
    evidence_id: str = ""


class FindingExplanation(BaseModel):
    """Answers Question 2: Why did this specific Sentinel finding occur?"""
    finding_id: str
    solution: str
    target_gap_raw: float
    target_unit: str
    explained_gap_raw: float
    unexplained_residual_raw: float
    contributions: List[FindingContribution] = Field(default_factory=list)
    narrative: str = ""
    evidence_ids: List[str] = Field(default_factory=list)


class SharedDriver(BaseModel):
    """Answers Question 3: Cross-solution shared friction drivers."""
    feature: str
    business_label: str
    role: DriverRole
    solutions_affected: List[str] = Field(default_factory=list)
    cross_solution_friction_score: float = 0.0
    evidence_ids: List[str] = Field(default_factory=list)


class FeatureDiagnosis(BaseModel):
    """Backward compatibility container for single feature diagnosis."""
    solution: str
    feature: str
    business_label: str
    plain_language: str
    direction: Literal["up", "down", "unknown"] = "unknown"
    controllable: bool = True
    importance_rank: int
    evidence_ids: List[str] = Field(default_factory=list)
    mapping_status: Literal["curated", "auto_pending_review", "confirmed"] = "curated"


class FeatureDiagnosisSet(BaseModel):
    diagnoses: List[FeatureDiagnosis] = Field(default_factory=list)


class InterpreterReport(BaseModel):
    """Stage 2 complete structured output delivering answers to all 3 questions."""
    global_drivers: Dict[str, List[GlobalDriverExplanation]] = Field(default_factory=dict)
    finding_explanations: Dict[str, FindingExplanation] = Field(default_factory=dict)
    shared_drivers: List[SharedDriver] = Field(default_factory=list)
    skips: Dict[str, str] = Field(default_factory=dict)
    fallback_stats: Dict[str, int] = Field(default_factory=dict)



# =========================================================================
# Stage 3: Reasoner & Financial Exposure Schemas
# =========================================================================

class CausalEdge(BaseModel):
    cause_solution: str
    effect_solution: str
    mechanism: str
    status: Literal["supported association", "validated", "weak", "unsupported"]
    strength: Optional[float] = None
    spearman_corr: Optional[float] = None
    partial_corr: Optional[float] = None
    lagged_corr: Optional[float] = None
    partial_corr_passed: bool = False
    test_evidence_ids: List[str] = Field(default_factory=list)


class CausalChain(BaseModel):
    id: str
    nodes: List[str] = Field(..., description="Ordered list of solution codes e.g. ['E9', 'E4', 'E3', 'E5']")
    edges: List[CausalEdge] = Field(default_factory=list)
    classification: Literal["root_cause", "hypothesis"]
    narrative: str = ""


class ExposureLine(BaseModel):
    solution: str
    formula_id: str
    inputs: Dict[str, float] = Field(default_factory=dict)
    assumption_sources: Dict[str, Literal["customer", "default", "empirical_dataset"]] = Field(default_factory=dict)
    annual_usd: float
    evidence_ids: List[str] = Field(default_factory=list)


class ExposureLedger(BaseModel):
    lines: List[ExposureLine] = Field(default_factory=list)
    overlap_adjustment_usd: float = 0.0
    net_annual_usd: float = 0.0
    uses_default_assumptions: bool = True


class CausalAnalysis(BaseModel):
    chains: List[CausalChain] = Field(default_factory=list)
    exposure: ExposureLedger = Field(default_factory=ExposureLedger)
    validated_edges: List[CausalEdge] = Field(default_factory=list)



# =========================================================================
# Stage 4: Synthesizer & Verifier Schemas
# =========================================================================

class Claim(BaseModel):
    """
    Every statement in the executive dossier must be a validated Claim
    that explicitly references supporting Evidence IDs and any embedded numbers.
    """
    text: str
    evidence_ids: List[str] = Field(min_length=1, description="Citations to EvidenceItem.id")
    numbers: Dict[str, float] = Field(default_factory=dict, description="Token -> float mapping for verification")


class RecommendedAction(BaseModel):
    title: str
    scenario_level: Literal["L1", "L2", "L3"] = "L2"
    target_solutions: List[str] = Field(default_factory=list)
    owner_function: str = "Logistics"
    expected_kpi_delta: Dict[str, float] = Field(default_factory=dict)
    delta_label: Optional[str] = Field(default=None, description="Explicit human label e.g. -1.0 ppt improvement")
    delta_unit: Optional[str] = Field(default=None, description="Unit type: percentage points (ppt), relative %, or raw unit")
    delta_direction: Optional[Literal["improvement", "worsening", "neutral"]] = Field(default=None, description="Directional impact")
    horizon: Literal["30d", "60d", "90d"] = "30d"
    evidence_ids: List[str] = Field(default_factory=list)


class ConfidenceBreakdown(BaseModel):
    model_reliability: float = Field(..., ge=0.0, le=1.0)
    edge_validation: float = Field(..., ge=0.0, le=1.0)
    data_quality: float = Field(..., ge=0.0, le=1.0)
    overall: float = Field(..., ge=0.0, le=1.0)


class DecisionDossier(BaseModel):
    run_id: str
    problem_statement: Claim
    root_cause_chain: List[Claim] = Field(default_factory=list)
    hypotheses: List[Claim] = Field(default_factory=list)
    exposure_summary: Claim
    actions: List[RecommendedAction] = Field(default_factory=list)
    confidence: ConfidenceBreakdown
    caveats: List[str] = Field(default_factory=list)
    generated_by: Literal["llm", "template_fallback"] = "llm"
