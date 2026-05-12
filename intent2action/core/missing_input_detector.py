"""Deterministic missing input detection."""

import re

from intent2action.schemas.action import ActionCandidate

SYNONYMS = {
    "url": "link",
    "uri": "link",
    "dashboard_url": "dashboard_link",
    "dashboard": "dashboard_link",
    "client": "customer",
    "email_address": "email",
    "assignee": "owner",
}


def normalize_input_name(value: str) -> str:
    """Normalize an input name for comparison."""

    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return SYNONYMS.get(normalized, normalized)


class MissingInputDetector:
    """Compute missing inputs using normalized names and simple synonyms."""

    def detect(self, action: ActionCandidate) -> ActionCandidate:
        """Return an action with deterministic missing inputs."""

        available = {normalize_input_name(item) for item in action.available_inputs}
        missing = [
            required
            for required in action.required_inputs
            if normalize_input_name(required) not in available
        ]
        return action.model_copy(update={"missing_inputs": missing})

