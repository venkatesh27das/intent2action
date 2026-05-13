"""Generic OpenAI-compatible chat completions client."""

from __future__ import annotations

import logging
from typing import Any

import httpx

LOGGER = logging.getLogger(__name__)


class OpenAICompatibleClientError(RuntimeError):
    """Raised when an OpenAI-compatible endpoint cannot return a usable response."""


class VisionNotSupportedError(OpenAICompatibleClientError):
    """Raised when multimodal inference is requested for a text-only provider."""


class OpenAICompatibleClient:
    """Client for OpenAI-compatible ``/v1/chat/completions`` endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout_seconds: int | float = 120,
        max_retries: int = 2,
        supports_vision: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = httpx.Timeout(timeout_seconds)
        self.max_retries = max_retries
        self.supports_vision = supports_vision
        self.transport = transport

    def generate_text(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.1,
    ) -> str:
        """Generate text from chat messages."""

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        return self._chat_completion(payload)

    def generate_multimodal(
        self,
        prompt: str,
        image_base64: str,
        mime_type: str,
        temperature: float = 0.1,
    ) -> str:
        """Generate text from a prompt plus base64 image."""

        if not self.supports_vision:
            raise VisionNotSupportedError(
                "The configured model provider does not have vision support enabled."
            )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}",
                            },
                        },
                    ],
                }
            ],
            "temperature": temperature,
        }
        return self._chat_completion(payload)

    def health_check(self) -> dict[str, Any]:
        """Return provider metadata and a best-effort endpoint status."""

        status: dict[str, Any] = {
            "type": "openai_compatible",
            "base_url": self.base_url,
            "model": self.model,
            "supports_vision": self.supports_vision,
        }
        try:
            with httpx.Client(
                timeout=self.timeout,
                transport=self.transport,
                headers=self._headers(),
            ) as client:
                response = client.get(f"{self.base_url}/models")
            status["reachable"] = response.status_code < 500
            status["models_endpoint_status"] = response.status_code
        except httpx.HTTPError as exc:
            status["reachable"] = False
            status["error"] = str(exc)
        return status

    def _chat_completion(self, payload: dict[str, Any]) -> str:
        url = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(
                    timeout=self.timeout,
                    transport=self.transport,
                    headers=self._headers(),
                ) as client:
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                return self._extract_content(data)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                last_error = exc
                LOGGER.warning("Model provider connection failed on attempt %s", attempt + 1)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                LOGGER.warning(
                    "Model provider request failed on attempt %s: %s - %s",
                    attempt + 1,
                    exc,
                    exc.response.text,
                )
                if exc.response.status_code in {400, 404, 422}:
                    break
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = exc
                LOGGER.warning("Model provider request failed on attempt %s: %s", attempt + 1, exc)

        raise OpenAICompatibleClientError(
            "Unable to reach the OpenAI-compatible model provider or parse its response. "
            f"Check that the endpoint is running at {self.base_url}, the model "
            f"'{self.model}' exists, and the endpoint supports /chat/completions."
        ) from last_error

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Model response content is not plain text")
        return content
