"""Provider factory."""

from __future__ import annotations

from typing import Any

from intent2action.providers.openai_compatible_client import OpenAICompatibleClient


def get_model_client(config: Any) -> OpenAICompatibleClient:
    """Build the configured model client."""

    provider_type = getattr(config, "model_provider_type", "openai_compatible")
    if provider_type != "openai_compatible":
        raise ValueError(f"Unsupported model provider type: {provider_type}")

    return OpenAICompatibleClient(
        base_url=config.model_base_url,
        api_key=config.model_api_key,
        model=config.model_name,
        timeout_seconds=config.model_timeout_seconds,
        max_retries=config.model_max_retries,
        supports_vision=config.model_supports_vision,
    )
