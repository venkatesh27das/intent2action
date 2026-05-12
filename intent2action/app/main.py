"""FastAPI app for intent2action."""

import json
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from intent2action.app.config import get_settings
from intent2action.core.pipeline import ActionInferencePipeline
from intent2action.providers.lmstudio_client import LMStudioError
from intent2action.schemas.request import ActionInferenceRequest
from intent2action.schemas.response import ActionInferenceResponse
from intent2action.utils.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title="intent2action",
    description="Local-first multimodal action inference API.",
    version="0.1.0",
)


def get_pipeline() -> ActionInferencePipeline:
    """Create a pipeline instance."""

    return ActionInferencePipeline(settings=settings)


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""

    return {"status": "ok", "service": "intent2action"}


@app.post("/infer-actions", response_model=ActionInferenceResponse)
def infer_actions(request: ActionInferenceRequest) -> ActionInferenceResponse:
    """Infer action candidates from text."""

    try:
        return get_pipeline().infer_from_text(request.content, request.context)
    except LMStudioError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
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
    except LMStudioError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        detail = f"Image action inference failed: {exc}"
        raise HTTPException(status_code=500, detail=detail) from exc
