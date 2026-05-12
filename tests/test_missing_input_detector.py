"""Missing input detector tests."""

from intent2action.core.missing_input_detector import MissingInputDetector, normalize_input_name
from intent2action.schemas import ActionCandidate


def test_normalize_input_name_handles_synonyms() -> None:
    assert normalize_input_name("Dashboard URL") == "dashboard_link"


def test_missing_inputs_uses_normalized_names() -> None:
    action = ActionCandidate(
        action_name="Investigate dashboard",
        action_type="investigate_issue",
        description="Review dashboard.",
        rationale="Dashboard is blank.",
        required_inputs=["dashboard_url", "timeframe"],
        available_inputs=["dashboard_link"],
        confidence=0.8,
        risk_level="low",
        execution_mode="human_approval_required",
    )

    updated = MissingInputDetector().detect(action)

    assert updated.missing_inputs == ["timeframe"]

