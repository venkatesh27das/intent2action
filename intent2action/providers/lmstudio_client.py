"""Backward-compatible LM Studio client import path."""

from __future__ import annotations

import httpx

from intent2action.providers.openai_compatible_client import (
    OpenAICompatibleClient,
    OpenAICompatibleClientError,
)

LMStudioError = OpenAICompatibleClientError


class LMStudioClient(OpenAICompatibleClient):
    """Compatibility wrapper for the generic OpenAI-compatible client."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str | None = "not-needed",
        model: str = "local-model",
        timeout_seconds: int | float = 120,
        max_retries: int | None = None,
        retries: int = 2,
        max_tokens: int | None = None,
        supports_vision: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries if max_retries is not None else retries,
            max_tokens=max_tokens,
            supports_vision=supports_vision,
            transport=transport,
        )
