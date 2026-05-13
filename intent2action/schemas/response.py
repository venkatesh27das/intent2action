"""Response schemas."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from intent2action import __version__
from intent2action.schemas.action import ActionCandidate
from intent2action.schemas.entity import ExtractedEntity
from intent2action.schemas.intent import DetectedIntent


class ActionInferenceResponse(BaseModel):
    """Validated action inference response."""

    model_config = ConfigDict(extra="forbid")

    input_summary: str = Field(min_length=1)
    input_type: Literal["text", "image"]
    schema_version: str = "1.0"
    package_version: str = __version__
    extracted_entities: list[ExtractedEntity] = Field(default_factory=list)
    detected_intents: list[DetectedIntent] = Field(default_factory=list)
    possible_actions: list[ActionCandidate] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    raw_model_output: dict[str, Any] | str | None = None
    warnings: list[str] = Field(default_factory=list)
