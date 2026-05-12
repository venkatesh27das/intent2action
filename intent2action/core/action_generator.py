"""Action generation step."""

from intent2action.schemas.action import ActionCandidate, ExecutionMode, RiskLevel


class ActionGenerator:
    """Coerce model action candidates into schema objects."""

    def generate(self, raw_actions: list[dict]) -> list[ActionCandidate]:
        """Validate action dictionaries, filling safe defaults where needed."""

        actions: list[ActionCandidate] = []
        for item in raw_actions:
            item = dict(item)
            item.setdefault("required_inputs", [])
            item.setdefault("available_inputs", [])
            item.setdefault("missing_inputs", [])
            item.setdefault("suggested_tools", [])
            item.setdefault("confidence", 0.0)
            item.setdefault("risk_level", RiskLevel.MEDIUM.value)
            item.setdefault("execution_mode", ExecutionMode.HUMAN_APPROVAL_REQUIRED.value)
            item.setdefault("ranking_score", 0.0)
            try:
                actions.append(ActionCandidate.model_validate(item))
            except Exception:
                continue
        return actions

