"""Deterministic risk scoring."""

from pathlib import Path

import yaml

from intent2action.schemas.action import ActionCandidate, ExecutionMode, RiskLevel

RISK_ORDER = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}


class RiskScorer:
    """Override model risk when local rules identify higher risk."""

    def __init__(self, rules_path: Path | None = None, enable_override: bool = True) -> None:
        self.enable_override = enable_override
        self.rules_path = (
            rules_path
            or Path(__file__).resolve().parents[1] / "registry" / "risk_rules.yaml"
        )
        self.rules = self._load_rules()

    def score(self, action: ActionCandidate) -> ActionCandidate:
        """Apply deterministic risk and execution mode rules."""

        if not self.enable_override:
            return action

        rule_risk = self._risk_from_rules(action)
        current = action.risk_level
        final_risk = rule_risk if RISK_ORDER[rule_risk] > RISK_ORDER[current] else current
        execution_mode = self._execution_mode_for_risk(final_risk, action.execution_mode)
        return action.model_copy(
            update={"risk_level": final_risk, "execution_mode": execution_mode}
        )

    def _load_rules(self) -> dict:
        if not self.rules_path.exists():
            return {}
        with self.rules_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def _risk_from_rules(self, action: ActionCandidate) -> RiskLevel:
        haystack = " ".join(
            [
                action.action_name,
                action.action_type,
                action.description,
                *action.suggested_tools,
            ]
        ).lower()
        for keyword in self.rules.get("high_risk", []):
            if keyword.lower() in haystack:
                return RiskLevel.HIGH
        for keyword in self.rules.get("medium_risk", []):
            if keyword.lower() in haystack:
                return RiskLevel.MEDIUM
        for keyword in self.rules.get("low_risk", []):
            if keyword.lower() in haystack:
                return RiskLevel.LOW
        return action.risk_level

    @staticmethod
    def _execution_mode_for_risk(
        risk_level: RiskLevel,
        current_mode: ExecutionMode,
    ) -> ExecutionMode:
        if risk_level == RiskLevel.HIGH:
            return ExecutionMode.NOT_RECOMMENDED
        if risk_level == RiskLevel.MEDIUM and current_mode == ExecutionMode.AUTO_POSSIBLE:
            return ExecutionMode.HUMAN_APPROVAL_REQUIRED
        return current_mode
