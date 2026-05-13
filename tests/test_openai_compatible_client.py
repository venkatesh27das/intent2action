"""OpenAI-compatible client tests."""

import json

import httpx
import pytest

from intent2action.providers.openai_compatible_client import (
    InvalidModelResponseError,
    ModelNotFoundError,
    ModelProviderAuthenticationError,
    OpenAICompatibleClient,
    VisionNotSupportedError,
)


def test_openai_compatible_client_text_request_building() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "done"}}]},
        )

    client = OpenAICompatibleClient(
        base_url="http://testserver/v1",
        api_key="not-needed",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    result = client.generate_text([{"role": "user", "content": "hello"}])

    assert result == "done"
    assert captured["url"] == "http://testserver/v1/chat/completions"
    assert captured["body"] == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.1,
    }


def test_openai_compatible_client_includes_max_tokens_when_configured() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    client = OpenAICompatibleClient(
        base_url="http://testserver/v1",
        api_key=None,
        model="test-model",
        max_tokens=512,
        transport=httpx.MockTransport(handler),
    )

    client.generate_text([{"role": "user", "content": "hello"}])

    assert captured["max_tokens"] == 512


def test_openai_compatible_client_multimodal_request_building() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "done"}}]},
        )

    client = OpenAICompatibleClient(
        base_url="http://testserver/v1",
        api_key=None,
        model="vision-model",
        transport=httpx.MockTransport(handler),
    )

    client.generate_multimodal("describe", "abc123", "image/png")

    assert captured["model"] == "vision-model"
    assert captured["temperature"] == 0.1
    assert captured["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc123"},
                },
            ],
        }
    ]


def test_no_authorization_header_when_api_key_not_needed() -> None:
    captured_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatibleClient(
        base_url="http://testserver/v1",
        api_key="not-needed",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    client.generate_text([{"role": "user", "content": "hello"}])

    assert "authorization" not in captured_headers


def test_authorization_header_when_api_key_present() -> None:
    captured_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatibleClient(
        base_url="http://testserver/v1",
        api_key="secret",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    client.generate_text([{"role": "user", "content": "hello"}])

    assert captured_headers["authorization"] == "Bearer secret"


def test_generate_multimodal_rejects_disabled_vision() -> None:
    client = OpenAICompatibleClient(
        base_url="http://testserver/v1",
        api_key=None,
        model="text-model",
        supports_vision=False,
    )

    with pytest.raises(VisionNotSupportedError, match="vision support"):
        client.generate_multimodal("describe", "abc123", "image/png")


def test_auth_failure_raises_specific_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "bad key"}},
            request=request,
        )

    client = OpenAICompatibleClient(
        base_url="http://testserver/v1",
        api_key="bad",
        model="test-model",
        max_retries=0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ModelProviderAuthenticationError) as exc_info:
        client.generate_text([{"role": "user", "content": "hello"}])

    assert exc_info.value.code == "model_provider_authentication_failed"
    assert exc_info.value.status_code == 401


def test_model_not_found_raises_specific_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "missing model"}, request=request)

    client = OpenAICompatibleClient(
        base_url="http://testserver/v1",
        api_key=None,
        model="missing-model",
        max_retries=0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ModelNotFoundError, match="missing-model"):
        client.generate_text([{"role": "user", "content": "hello"}])


def test_invalid_response_shape_raises_specific_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    client = OpenAICompatibleClient(
        base_url="http://testserver/v1",
        api_key=None,
        model="test-model",
        max_retries=0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(InvalidModelResponseError) as exc_info:
        client.generate_text([{"role": "user", "content": "hello"}])

    assert exc_info.value.code == "invalid_model_response"
