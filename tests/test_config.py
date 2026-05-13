"""Configuration tests."""

from intent2action.app import config


def test_config_env_priority_new_vars_over_legacy_vars(monkeypatch) -> None:
    monkeypatch.setattr(
        config,
        "load_yaml_config",
        lambda: {
            "model_provider": {
                "base_url": "http://yaml/v1",
                "model": "yaml-model",
                "api_key": "yaml-key",
            }
        },
    )
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://legacy/v1")
    monkeypatch.setenv("LMSTUDIO_MODEL", "legacy-model")
    monkeypatch.setenv("INTENT2ACTION_BASE_URL", "http://new/v1")
    monkeypatch.setenv("INTENT2ACTION_MODEL", "new-model")
    monkeypatch.setenv("INTENT2ACTION_API_KEY", "new-key")
    monkeypatch.setenv("INTENT2ACTION_MAX_TOKENS", "512")
    monkeypatch.setenv("INTENT2ACTION_IMAGE_MAX_DIMENSION", "640")
    config.get_settings.cache_clear()

    settings = config.get_settings()

    assert settings.model_base_url == "http://new/v1"
    assert settings.model_name == "new-model"
    assert settings.model_api_key == "new-key"
    assert settings.model_max_tokens == 512
    assert settings.image_max_dimension == 640
    assert settings.lmstudio_base_url == "http://new/v1"
    assert settings.lmstudio_model == "new-model"

    config.get_settings.cache_clear()
