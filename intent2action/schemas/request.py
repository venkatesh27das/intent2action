"""Request schemas."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ActionInferenceRequest(BaseModel):
    """Request for text action inference."""

    model_config = ConfigDict(extra="forbid")

    input_type: Literal["text"]
    content: str = Field(min_length=1)
    context: dict[str, Any] | None = None

