"""Schema tests."""

import pytest
from pydantic import ValidationError

from intent2action.schemas import ActionCandidate, ActionInferenceResponse


def test_action_candidate_validates_bounds() -> None:
    with pytest.raises(ValidationError):
        ActionCandidate(
            action_name="Bad confidence",
            action_type="classify_input",
            description="Invalid",
            rationale="Confidence is outside bounds.",
            confidence=1.2,
            risk_level="low",
            execution_mode="auto_possible",
        )


def test_response_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ActionInferenceResponse(
            input_summary="summary",
            input_type="text",
            extracted_entities=[],
            detected_intents=[],
            possible_actions=[],
            clarifying_questions=[],
            warnings=[],
            extra_field=True,
        )

