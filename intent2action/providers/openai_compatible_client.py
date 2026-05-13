"""Generic OpenAI-compatible chat completions client."""

from __future__ import annotations

import logging
from typing import Any

import httpx

LOGGER = logging.getLogger(__name__)


class OpenAICompatibleClientError(RuntimeError):
    """Raised when an OpenAI-compatible endpoint cannot return a usable response."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "model_provider_error",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class ModelProviderUnavailableError(OpenAICompatibleClientError):
    """Raised when the model provider cannot be reached."""


class ModelProviderAuthenticationError(OpenAICompatibleClientError):
    """Raised when the model provider rejects credentials."""


class ModelNotFoundError(OpenAICompatibleClientError):
    """Raised when the configured model is missing or unavailable."""


class InvalidModelResponseError(OpenAICompatibleClientError):
    """Raised when the provider returns malformed or incompatible data."""


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
        max_tokens: int | None = None,
        supports_vision: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = httpx.Timeout(timeout_seconds)
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.supports_vision = supports_vision
        self.transport = transport
        self._client = httpx.Client(
            timeout=self.timeout,
            transport=self.transport,
            headers=self._headers(),
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self._client.close()

    def __enter__(self) -> OpenAICompatibleClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

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
        self._add_max_tokens(payload)
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
                "The configured model provider does not have vision support enabled.",
                code="vision_not_supported",
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
        self._add_max_tokens(payload)
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
            response = self._client.get(f"{self.base_url}/models")
            status["reachable"] = response.status_code < 500
            status["models_endpoint_status"] = response.status_code
        except httpx.HTTPError as exc:
            status["reachable"] = False
            status["error"] = str(exc)
        return status

    def _chat_completion(self, payload: dict[str, Any]) -> str:
        url = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None
        failure: OpenAICompatibleClientError | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return self._extract_content(data)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                last_error = exc
                failure = ModelProviderUnavailableError(
                    f"Unable to reach the model provider at {self.base_url}.",
                    code="model_provider_unavailable",
                )
                LOGGER.warning("Model provider connection failed on attempt %s", attempt + 1)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                LOGGER.warning(
                    "Model provider request failed on attempt %s: %s - %s",
                    attempt + 1,
                    exc,
                    exc.response.text,
                )
                failure = self._http_error(exc)
                if exc.response.status_code in {400, 404, 422}:
                    break
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = exc
                failure = InvalidModelResponseError(
                    "The model provider returned a response that intent2action could not parse. "
                    "Expected choices[0].message.content to contain plain text JSON.",
                    code="invalid_model_response",
                )
                LOGGER.warning("Model provider request failed on attempt %s: %s", attempt + 1, exc)

        if failure is not None:
            raise failure from last_error
        raise OpenAICompatibleClientError(
            "Unable to reach the OpenAI-compatible model provider or parse its response.",
            code="model_provider_error",
        ) from last_error

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _add_max_tokens(self, payload: dict[str, Any]) -> None:
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

    def _http_error(self, exc: httpx.HTTPStatusError) -> OpenAICompatibleClientError:
        status_code = exc.response.status_code
        detail = _response_error_detail(exc.response)
        if status_code in {401, 403}:
            return ModelProviderAuthenticationError(
                "The model provider rejected the configured API key or credentials.",
                code="model_provider_authentication_failed",
                status_code=status_code,
            )
        if status_code == 404:
            return ModelNotFoundError(
                f"The configured model '{self.model}' or chat completions endpoint was not found.",
                code="model_not_found",
                status_code=status_code,
            )
        if status_code == 400 and "model" in detail.lower():
            return ModelNotFoundError(
                f"The configured model '{self.model}' was rejected by the model provider.",
                code="model_not_found",
                status_code=status_code,
            )
        if status_code in {400, 415, 422} and any(
            phrase in detail.lower() for phrase in ("image", "vision", "multimodal")
        ):
            return VisionNotSupportedError(
                "The model provider rejected the image payload. Check that the model and endpoint "
                "support OpenAI-compatible vision messages.",
                code="vision_not_supported",
                status_code=status_code,
            )
        return OpenAICompatibleClientError(
            f"The model provider returned HTTP {status_code}: {detail}",
            code="model_provider_http_error",
            status_code=status_code,
        )

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Model response content is not plain text")
        return content


def _response_error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message
        if isinstance(error, str):
            return error
    return str(data)[:500]
