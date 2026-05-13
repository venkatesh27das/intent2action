"""FastAPI app for intent2action."""

import json
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from intent2action import __version__
from intent2action.app.config import get_settings
from intent2action.core.pipeline import ActionInferencePipeline
from intent2action.providers.factory import get_model_client
from intent2action.providers.openai_compatible_client import OpenAICompatibleClientError
from intent2action.schemas.request import ActionInferenceRequest
from intent2action.schemas.response import ActionInferenceResponse
from intent2action.utils.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title="intent2action",
    description="Multimodal action inference API for OpenAI-compatible model endpoints.",
    version=__version__,
)


def get_pipeline() -> ActionInferencePipeline:
    """Create a pipeline instance."""

    return ActionInferencePipeline(settings=settings)


@app.get("/health")
def health() -> dict[str, Any]:
    """Return service health."""

    return {
        "status": "ok",
        "service": "intent2action",
        "version": __version__,
        "model_provider": {
            "type": settings.model_provider_type,
            "base_url": settings.model_base_url,
            "model": settings.model_name,
            "supports_vision": settings.model_supports_vision,
        },
    }


@app.get("/health/model")
def health_model() -> dict[str, Any]:
    """Return best-effort model provider health without exposing credentials."""

    return get_model_client(settings).health_check()


@app.post("/infer-actions", response_model=ActionInferenceResponse)
def infer_actions(request: ActionInferenceRequest) -> ActionInferenceResponse:
    """Infer action candidates from text."""

    try:
        return get_pipeline().infer_from_text(request.content, request.context)
    except OpenAICompatibleClientError as exc:
        raise _provider_http_exception(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Action inference failed: {exc}") from exc


@app.post("/infer-actions/image", response_model=ActionInferenceResponse)
async def infer_actions_image(
    image: Annotated[UploadFile, File()],
    context: Annotated[str | None, Form()] = None,
) -> ActionInferenceResponse:
    """Infer action candidates from an uploaded image."""

    parsed_context: dict[str, Any] | None = None
    if context:
        try:
            parsed = json.loads(context)
            if not isinstance(parsed, dict):
                raise ValueError("context must be a JSON object")
            parsed_context = parsed
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid context JSON: {exc}") from exc

    try:
        image_bytes = await image.read()
        return get_pipeline().infer_from_image(
            image_bytes=image_bytes,
            filename=image.filename or "uploaded_image.png",
            context=parsed_context,
        )
    except OpenAICompatibleClientError as exc:
        raise _provider_http_exception(exc) from exc
    except Exception as exc:
        detail = f"Image action inference failed: {exc}"
        raise HTTPException(status_code=500, detail=detail) from exc


def _provider_http_exception(exc: OpenAICompatibleClientError) -> HTTPException:
    status_code = 503
    if exc.status_code in {401, 403}:
        status_code = 401
    elif exc.status_code in {400, 404, 415, 422}:
        status_code = 400
    return HTTPException(
        status_code=status_code,
        detail={
            "error": exc.code,
            "message": str(exc),
        },
    )
