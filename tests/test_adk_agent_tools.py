"""Tests for ADK agent tool functions."""

from __future__ import annotations

import json
from pathlib import Path

from intent2action.agent_adk import agent
from intent2action.schemas import ActionInferenceResponse


class FakePipeline:
    """Fake pipeline for ADK tool tests."""

    def infer_from_text(self, content: str, context: dict | None = None) -> ActionInferenceResponse:
        assert content == "hello"
        assert context == {"domain": "test"}
        return _response("text")

    def infer_from_image(
        self,
        image_bytes: bytes,
        filename: str,
        context: dict | None = None,
    ) -> ActionInferenceResponse:
        assert image_bytes == b"image"
        assert filename == "chart.png"
        assert context == {"domain": "finance"}
        return _response("image")


def _response(input_type: str) -> ActionInferenceResponse:
    return ActionInferenceResponse(
        input_summary="summary",
        input_type=input_type,
        extracted_entities=[],
        detected_intents=[],
        possible_actions=[],
        clarifying_questions=[],
        raw_model_output=None,
        warnings=[],
    )


def test_infer_actions_from_text_tool(monkeypatch) -> None:
    monkeypatch.setattr(agent, "ActionInferencePipeline", lambda: FakePipeline())

    result = agent.infer_actions_from_text("hello", json.dumps({"domain": "test"}))

    assert result["input_type"] == "text"
    assert result["schema_version"] == "1.0"


def test_infer_actions_from_image_base64_tool(monkeypatch) -> None:
    monkeypatch.setattr(agent, "ActionInferencePipeline", lambda: FakePipeline())

    result = agent.infer_actions_from_image_base64(
        "aW1hZ2U=",
        filename="chart.png",
        context_json=json.dumps({"domain": "finance"}),
    )

    assert result["input_type"] == "image"


def test_agent_instruction_preserves_no_execution_boundary() -> None:
    assert "never execute actions" in agent.AGENT_INSTRUCTION
    assert "provided tools" in agent.AGENT_INSTRUCTION


def test_agent_card_declares_a2a_skills() -> None:
    card_path = Path("intent2action/agent_adk/agent_card.json")
    card = json.loads(card_path.read_text(encoding="utf-8"))

    assert card["protocolVersion"] == "0.3.0"
    assert card["preferredTransport"] == "JSONRPC"
    assert {skill["id"] for skill in card["skills"]} == {
        "infer-actions-from-text",
        "infer-actions-from-image",
    }
