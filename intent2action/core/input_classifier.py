"""Input classification."""

from typing import Literal


class InputClassifier:
    """Classify supported input types."""

    def classify(self, input_type: str) -> Literal["text", "image"]:
        """Return a normalized input type."""

        if input_type in {"text", "image"}:
            return input_type  # type: ignore[return-value]
        raise ValueError(f"Unsupported input_type: {input_type}")

