"""LM Studio OpenAI-compatible API client."""

import logging
from collections.abc import Sequence
from typing import Any

import httpx

LOGGER = logging.getLogger(__name__)


class LMStudioError(RuntimeError):
    """Raised when LM Studio cannot return a usable response."""


class LMStudioClient:
    """Small client for LM Studio's OpenAI-compatible chat completions API."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "local-model",
        timeout_seconds: float = 120.0,
        retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = httpx.Timeout(timeout_seconds)
        self.retries = retries
        self.transport = transport

    def generate_text(self, messages: list[dict[str, Any]]) -> str:
        """Generate text from chat messages."""

        return self._chat_completion(messages)

    def generate_multimodal(self, prompt: str, image_base64: str, mime_type: str) -> str:
        """Generate text from a prompt plus base64 image."""

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                    },
                ],
            }
        ]
        return self._chat_completion(messages)

    def _chat_completion(self, messages: Sequence[dict[str, Any]]) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0.1,
            "response_format": {"type": "text"},
        }
        url = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                return str(data["choices"][0]["message"]["content"])
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = exc
                LOGGER.warning("LM Studio connection failed on attempt %s", attempt + 1)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                LOGGER.warning(
                    "LM Studio request failed on attempt %s: %s - %s",
                    attempt + 1,
                    exc,
                    exc.response.text,
                )
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
                LOGGER.warning("LM Studio request failed on attempt %s: %s", attempt + 1, exc)

        raise LMStudioError(
            "Unable to reach LM Studio or parse its response. Ensure LM Studio is running, "
            f"the local server is enabled, and LMSTUDIO_BASE_URL points to {self.base_url}."
        ) from last_error
