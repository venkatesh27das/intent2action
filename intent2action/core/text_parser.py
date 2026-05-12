"""Text parsing."""

from typing import Any


class TextParser:
    """Prepare text input for the inference pipeline."""

    def parse(self, content: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return normalized text payload."""

        cleaned = " ".join(content.strip().split())
        return {"content": cleaned, "context": context or {}}

