"""Risk scorer tests."""

from intent2action.core.action_ranker import ActionRanker
from intent2action.core.risk_scorer import RiskScorer
from intent2action.schemas import ActionCandidate


def test_risk_scorer_overrides_to_high() -> None:
    action = ActionCandidate(
        action_name="Send external email",
        action_type="draft_email",
        description="Send external email to the customer.",
        rationale="Customer needs a response.",
        confidence=0.8,
        risk_level="low",
        execution_mode="auto_possible",
    )

    scored = RiskScorer().score(action)

    assert scored.risk_level == "high"
    assert scored.execution_mode == "not_recommended"


def test_action_ranker_uses_formula() -> None:
    action = ActionCandidate(
        action_name="Create ticket",
        action_type="create_ticket",
        description="Create an issue ticket.",
        rationale="Incident needs tracking.",
        required_inputs=["title", "owner"],
        available_inputs=["title"],
        missing_inputs=["owner"],
        confidence=0.8,
        risk_level="medium",
        execution_mode="human_approval_required",
    )

    ranked = ActionRanker().rank([action])

    assert ranked[0].ranking_score == 0.51

