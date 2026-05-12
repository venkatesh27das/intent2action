"""Response validation and JSON correction."""

from typing import Any

from pydantic import ValidationError

from intent2action.schemas.response import ActionInferenceResponse
from intent2action.utils.json_utils import loads_model_json


class OutputValidator:
    """Validate model output against the response schema."""

    def __init__(self, llm_client: Any | None = None, enable_repair: bool = True) -> None:
        self.llm_client = llm_client
        self.enable_repair = enable_repair

    def validate(self, raw_output: str) -> ActionInferenceResponse:
        """Parse and validate raw model JSON, asking the model once to fix invalid JSON."""

        try:
            return ActionInferenceResponse.model_validate(loads_model_json(raw_output))
        except (ValueError, ValidationError) as exc:
            if not self.enable_repair or self.llm_client is None:
                raise
            correction_prompt = (
                "The previous response failed validation. Return only corrected JSON matching "
                f"ActionInferenceResponse. Validation error: {exc}. Raw response: {raw_output}"
            )
            corrected = self.llm_client.generate_text(
                [{"role": "user", "content": correction_prompt}]
            )
            return ActionInferenceResponse.model_validate(loads_model_json(corrected))
