"""Intent detection schemas."""

from pydantic import BaseModel, ConfigDict, Field


class DetectedIntent(BaseModel):
    """A likely user or business intent."""

    model_config = ConfigDict(extra="forbid")

    intent: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)

