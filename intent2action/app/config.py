"""Application configuration."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "default.yaml"


def load_yaml_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML configuration if it exists."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    return loaded


class Settings(BaseSettings):
    """Runtime settings from defaults and environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    model_provider_type: str = Field(
        default="openai_compatible",
        alias="INTENT2ACTION_PROVIDER_TYPE",
    )
    model_base_url: str = Field(
        default="http://localhost:1234/v1",
        alias="INTENT2ACTION_BASE_URL",
    )
    model_api_key: str | None = Field(default="not-needed", alias="INTENT2ACTION_API_KEY")
    model_name: str = Field(default="local-model", alias="INTENT2ACTION_MODEL")
    model_timeout_seconds: int = Field(default=120, alias="INTENT2ACTION_TIMEOUT_SECONDS")
    model_max_retries: int = Field(default=2, alias="INTENT2ACTION_MAX_RETRIES")
    model_max_tokens: int | None = Field(default=None, alias="INTENT2ACTION_MAX_TOKENS")
    model_supports_vision: bool = Field(default=True, alias="INTENT2ACTION_SUPPORTS_VISION")
    image_max_dimension: int | None = Field(default=1280, alias="INTENT2ACTION_IMAGE_MAX_DIMENSION")

    lmstudio_base_url: str = Field(default="http://localhost:1234/v1", alias="LMSTUDIO_BASE_URL")
    lmstudio_model: str = Field(default="local-model", alias="LMSTUDIO_MODEL")
    lmstudio_timeout_seconds: int = 120
    max_actions: int = 8
    min_confidence: float = 0.2
    enable_json_repair: bool = True
    enable_risk_override: bool = True
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    """Return merged application settings."""

    config = load_yaml_config()
    model_provider = config.get("model_provider", {})
    legacy_lmstudio = config.get("lmstudio", {})
    inference = config.get("inference", {})

    provider_type = _env_or_default(
        "INTENT2ACTION_PROVIDER_TYPE",
        model_provider.get("type", "openai_compatible"),
    )
    base_url = _env_or_default(
        "INTENT2ACTION_BASE_URL",
        _env_or_default(
            "LMSTUDIO_BASE_URL",
            model_provider.get("base_url", legacy_lmstudio.get("base_url", "http://localhost:1234/v1")),
        ),
    )
    model_name = _env_or_default(
        "INTENT2ACTION_MODEL",
        _env_or_default(
            "LMSTUDIO_MODEL",
            model_provider.get("model", legacy_lmstudio.get("model", "local-model")),
        ),
    )
    timeout_seconds = _env_or_default(
        "INTENT2ACTION_TIMEOUT_SECONDS",
        model_provider.get("timeout_seconds", legacy_lmstudio.get("timeout_seconds", 120)),
    )
    max_retries = _env_or_default(
        "INTENT2ACTION_MAX_RETRIES",
        model_provider.get("max_retries", 2),
    )
    max_tokens = _env_or_default(
        "INTENT2ACTION_MAX_TOKENS",
        model_provider.get("max_tokens"),
    )
    supports_vision = _env_or_default(
        "INTENT2ACTION_SUPPORTS_VISION",
        model_provider.get("supports_vision", True),
    )
    image_max_dimension = _env_or_default(
        "INTENT2ACTION_IMAGE_MAX_DIMENSION",
        model_provider.get("image_max_dimension", 1280),
    )
    api_key = _env_or_default(
        "INTENT2ACTION_API_KEY",
        model_provider.get("api_key", "not-needed"),
    )

    return Settings(
        model_provider_type=str(provider_type),
        model_base_url=str(base_url),
        model_api_key=None if api_key is None else str(api_key),
        model_name=str(model_name),
        model_timeout_seconds=int(timeout_seconds),
        model_max_retries=int(max_retries),
        model_max_tokens=_parse_optional_int(max_tokens),
        model_supports_vision=_parse_bool(supports_vision),
        image_max_dimension=_parse_optional_int(image_max_dimension),
        lmstudio_base_url=str(base_url),
        lmstudio_model=str(model_name),
        lmstudio_timeout_seconds=int(timeout_seconds),
        max_actions=inference.get("max_actions", 8),
        min_confidence=inference.get("min_confidence", 0.2),
        enable_json_repair=inference.get("enable_json_repair", True),
        enable_risk_override=inference.get("enable_risk_override", True),
    )


def _env_or_default(env_name: str, default: Any) -> Any:
    value = os.getenv(env_name)
    return default if value is None else value


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    rendered = str(value).strip()
    if not rendered:
        return None
    parsed = int(rendered)
    return parsed if parsed > 0 else None
