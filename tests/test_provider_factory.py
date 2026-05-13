"""Provider factory tests."""

import pytest

from intent2action.app.config import Settings
from intent2action.providers.factory import get_model_client
from intent2action.providers.openai_compatible_client import OpenAICompatibleClient


def test_provider_factory_returns_openai_compatible_client() -> None:
    settings = Settings(
        model_provider_type="openai_compatible",
        model_base_url="http://testserver/v1",
        model_api_key="not-needed",
        model_name="test-model",
    )

    client = get_model_client(settings)

    assert isinstance(client, OpenAICompatibleClient)
    assert client.base_url == "http://testserver/v1"
    assert client.model == "test-model"


def test_provider_factory_rejects_unknown_provider() -> None:
    settings = Settings(model_provider_type="other")

    with pytest.raises(ValueError, match="Unsupported model provider type"):
        get_model_client(settings)
