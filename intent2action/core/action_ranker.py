"""Action ranking."""

from intent2action.schemas.action import ActionCandidate, RiskLevel

RISK_PENALTY = {
    RiskLevel.LOW: 0.0,
    RiskLevel.MEDIUM: 0.2,
    RiskLevel.HIGH: 0.4,
}


class ActionRanker:
    """Rank actions using confidence, completeness, and risk."""

    def rank(self, actions: list[ActionCandidate]) -> list[ActionCandidate]:
        """Compute ranking scores and return actions sorted descending."""

        ranked = [
            action.model_copy(update={"ranking_score": self.score(action)})
            for action in actions
        ]
        return sorted(ranked, key=lambda item: item.ranking_score, reverse=True)

    def score(self, action: ActionCandidate) -> float:
        """Compute bounded ranking score."""

        required_count = len(action.required_inputs)
        if required_count == 0:
            completeness = 1.0
        else:
            available = required_count - len(action.missing_inputs)
            completeness = max(0.0, min(1.0, available / required_count))
        raw_score = (
            action.confidence * 0.5
            + completeness * 0.3
            - RISK_PENALTY[action.risk_level] * 0.2
        )
        return round(max(0.0, min(1.0, raw_score)), 4)
