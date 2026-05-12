"""Utilities for extracting and repairing model JSON."""

import json
import re
from typing import Any

CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_text(model_output: str) -> str:
    """Extract the most likely JSON object from raw model output."""

    stripped = model_output.strip()
    code_match = CODE_BLOCK_RE.search(stripped)
    if code_match:
        return code_match.group(1).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def repair_common_json_issues(json_text: str) -> str:
    """Repair common LLM JSON issues without changing semantics."""

    repaired = json_text.strip()
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


def loads_model_json(model_output: str) -> dict[str, Any]:
    """Extract, repair, and parse a JSON object from model output."""

    json_text = repair_common_json_issues(extract_json_text(model_output))
    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("Model output must be a JSON object.")
    return parsed
