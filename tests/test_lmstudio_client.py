"""LM Studio client tests."""

import json

import httpx
import pytest

from intent2action.providers.lmstudio_client import LMStudioClient, LMStudioError


def test_generate_text_uses_lmstudio_payload_shape() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    client = LMStudioClient(
        base_url="http://testserver/v1",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    result = client.generate_text([{"role": "user", "content": "hello"}])

    assert result == '{"ok": true}'
    assert captured["model"] == "test-model"
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["response_format"] == {"type": "text"}


def test_generate_multimodal_uses_data_url() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    client = LMStudioClient(
        base_url="http://testserver/v1",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    client.generate_multimodal("describe", "abc123", "image/png")

    content = captured["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["image_url"]["url"] == "data:image/png;base64,abc123"


def test_generate_text_raises_clear_error_on_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"}, request=request)

    client = LMStudioClient(
        base_url="http://testserver/v1",
        model="test-model",
        retries=0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LMStudioError, match="Unable to reach LM Studio"):
        client.generate_text([{"role": "user", "content": "hello"}])

