"""Application settings."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SUPPORTSMITH_",
        extra="ignore",
        populate_by_name=True,
    )

    service_name: str = "SupportSmith"
    app_version: str = "0.1.0"
    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"
    database_url: str = Field(
        repr=False,
        validation_alias=AliasChoices("SUPPORTSMITH_DATABASE_URL", "DATABASE_URL"),
    )
    openai_api_key: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("SUPPORTSMITH_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    chat_model: str = "gpt-5.5"
    reasoning_model: str = "gpt-5.5"
    routing_model: str = "gpt-5.5"
    fallback_chat_models: tuple[str, ...] = ("gpt-5.4", "gpt-5.1", "gpt-5")
    embedding_model: str = "text-embedding-3-small"
    max_tool_iterations: int = 6
    max_repair_attempts: int = 1
    planner_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "high"
    routing_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "low"
    verifier_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "medium"
    planner_max_completion_tokens: int = 2048
    synthesis_max_completion_tokens: int = 1024
    verifier_max_completion_tokens: int = 1024
    compliance_max_completion_tokens: int = 512


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()  # type: ignore[call-arg]
