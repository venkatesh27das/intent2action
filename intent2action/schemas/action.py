"""Action candidate schemas."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(StrEnum):
    """Supported risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ExecutionMode(StrEnum):
    """Recommended execution modes."""

    AUTO_POSSIBLE = "auto_possible"
    HUMAN_APPROVAL_REQUIRED = "human_approval_required"
    DRAFT_ONLY = "draft_only"
    NOT_RECOMMENDED = "not_recommended"


class ActionCandidate(BaseModel):
    """A structured action that could be taken, without executing it."""

    model_config = ConfigDict(extra="forbid")

    action_name: str = Field(min_length=1)
    action_type: str = Field(min_length=1)
    description: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    required_inputs: list[str] = Field(default_factory=list)
    available_inputs: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    suggested_tools: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    execution_mode: ExecutionMode
    ranking_score: float = Field(default=0.0, ge=0.0, le=1.0)

