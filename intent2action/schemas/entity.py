"""Entity extraction schemas."""

from pydantic import BaseModel, ConfigDict, Field


class ExtractedEntity(BaseModel):
    """A key entity found in the input."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    value: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

