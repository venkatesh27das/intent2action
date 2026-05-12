"""Application configuration."""

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

    lmstudio_base_url: str = Field(default="http://localhost:1234/v1", alias="LMSTUDIO_BASE_URL")
    lmstudio_model: str = Field(default="local-model", alias="LMSTUDIO_MODEL")
    lmstudio_timeout_seconds: float = 120.0
    max_actions: int = 8
    min_confidence: float = 0.2
    enable_json_repair: bool = True
    enable_risk_override: bool = True
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    """Return merged application settings."""

    config = load_yaml_config()
    lmstudio = config.get("lmstudio", {})
    inference = config.get("inference", {})
    return Settings(
        lmstudio_timeout_seconds=lmstudio.get("timeout_seconds", 120),
        max_actions=inference.get("max_actions", 8),
        min_confidence=inference.get("min_confidence", 0.2),
        enable_json_repair=inference.get("enable_json_repair", True),
        enable_risk_override=inference.get("enable_risk_override", True),
    )
