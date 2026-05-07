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
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()  # type: ignore[call-arg]
