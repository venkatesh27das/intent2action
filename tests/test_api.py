"""FastAPI endpoint tests with mocked pipeline."""

from fastapi.testclient import TestClient

from intent2action.app import main
from intent2action.providers.openai_compatible_client import ModelProviderAuthenticationError
from intent2action.schemas import ActionInferenceResponse


class FakePipeline:
    """Fake pipeline for API tests."""

    def infer_from_text(self, content: str, context: dict | None = None) -> ActionInferenceResponse:
        return _response("text")

    def infer_from_image(
        self,
        image_bytes: bytes,
        filename: str,
        context: dict | None = None,
    ) -> ActionInferenceResponse:
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


def test_health_endpoint() -> None:
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "intent2action"
    assert body["version"] == "1.0.0"
    assert body["model_provider"]["type"] == "openai_compatible"
    assert "api_key" not in body["model_provider"]


def test_infer_actions_endpoint_uses_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_pipeline", lambda: FakePipeline())
    client = TestClient(main.app)

    response = client.post(
        "/infer-actions",
        json={"input_type": "text", "content": "hello", "context": {"domain": "test"}},
    )

    assert response.status_code == 200
    assert response.json()["input_type"] == "text"
    assert response.json()["schema_version"] == "1.0"


def test_infer_actions_image_endpoint_uses_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_pipeline", lambda: FakePipeline())
    client = TestClient(main.app)

    response = client.post(
        "/infer-actions/image",
        files={"image": ("test.png", b"image-bytes", "image/png")},
        data={"context": '{"domain":"test"}'},
    )

    assert response.status_code == 200
    assert response.json()["input_type"] == "image"


def test_infer_actions_image_rejects_invalid_context(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_pipeline", lambda: FakePipeline())
    client = TestClient(main.app)

    response = client.post(
        "/infer-actions/image",
        files={"image": ("test.png", b"image-bytes", "image/png")},
        data={"context": "not-json"},
    )

    assert response.status_code == 400


def test_provider_error_is_structured(monkeypatch) -> None:
    class FailingPipeline:
        def infer_from_text(
            self,
            content: str,
            context: dict | None = None,
        ) -> ActionInferenceResponse:
            raise ModelProviderAuthenticationError(
                "The model provider rejected the configured API key or credentials.",
                code="model_provider_authentication_failed",
                status_code=401,
            )

    monkeypatch.setattr(main, "get_pipeline", lambda: FailingPipeline())
    client = TestClient(main.app)

    response = client.post(
        "/infer-actions",
        json={"input_type": "text", "content": "hello"},
    )

    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "model_provider_authentication_failed"
