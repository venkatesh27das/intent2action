"""Intent detection step."""

from intent2action.schemas.intent import DetectedIntent


class IntentDetector:
    """Coerce model intents into schema objects."""

    def detect(self, raw_intents: list[dict]) -> list[DetectedIntent]:
        """Validate detected intent dictionaries."""

        intents: list[DetectedIntent] = []
        for item in raw_intents:
            try:
                intents.append(DetectedIntent.model_validate(item))
            except Exception:
                continue
        return intents

