"""Application settings."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
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
    langsmith_tracing: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "LANGSMITH_TRACING",
            "SUPPORTSMITH_LANGSMITH_TRACING",
        ),
    )
    langsmith_api_key: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices(
            "LANGSMITH_API_KEY",
            "SUPPORTSMITH_LANGSMITH_API_KEY",
        ),
    )
    langsmith_project: str = Field(
        default="supportsmith-agent",
        validation_alias=AliasChoices(
            "LANGSMITH_PROJECT",
            "SUPPORTSMITH_LANGSMITH_PROJECT",
        ),
    )
    chat_model: str = "gpt-5.5"
    reasoning_model: str = "gpt-5.5"
    routing_model: str = "gpt-5.5"
    fallback_chat_models: tuple[str, ...] = ("gpt-5.4", "gpt-5.1", "gpt-5")
    embedding_model: str = "text-embedding-3-small"
    max_tool_iterations: int = 6
    max_repair_attempts: int = 1
    context_user_turns: int = Field(
        default=10,
        ge=0,
        validation_alias=AliasChoices(
            "SUPPORTSMITH_CONTEXT_USER_TURNS",
            "context_user_turns",
        ),
    )
    planner_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "high"
    routing_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "low"
    verifier_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "medium"
    judge_model: str = "gpt-5.5"
    judge_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "low"
    planner_max_completion_tokens: int = 2048
    synthesis_max_completion_tokens: int = 1024
    verifier_max_completion_tokens: int = 1024
    compliance_max_completion_tokens: int = 512
    judge_max_completion_tokens: int = 1024

    firecrawl_api_key: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices(
            "SUPPORTSMITH_FIRECRAWL_API_KEY",
            "FIRECRAWL_API_KEY",
        ),
    )
    api_bearer_token: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("SUPPORTSMITH_API_BEARER_TOKEN",),
    )
    admin_api_key: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("SUPPORTSMITH_ADMIN_API_KEY",),
    )
    allowed_ingestion_hosts: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices("SUPPORTSMITH_ALLOWED_INGESTION_HOSTS",),
    )
    allow_any_website_ingestion: bool = Field(
        default=False,
        validation_alias=AliasChoices("SUPPORTSMITH_ALLOW_ANY_WEBSITE_INGESTION",),
    )
    website_classifier_model: str = "gpt-5.5"
    website_classifier_reasoning_effort: Literal[
        "none", "low", "medium", "high", "xhigh"
    ] = "low"
    website_classifier_max_completion_tokens: int = 256
    website_extractor_model: str = "gpt-5.5"
    website_extractor_reasoning_effort: Literal[
        "none", "low", "medium", "high", "xhigh"
    ] = "low"
    website_extractor_max_completion_tokens: int = 512
    website_max_pages_per_job: int = 500
    website_max_chunks_per_page: int = 40
    website_max_total_chunks_per_job: int = 5000

    @field_validator("allowed_ingestion_hosts", mode="before")
    @classmethod
    def _split_hosts(cls, value: object) -> object:
        if value is None or value == "":
            return ()
        if isinstance(value, str):
            return tuple(part.strip().lower() for part in value.split(",") if part.strip())
        if isinstance(value, (list, tuple)):
            return tuple(str(item).strip().lower() for item in value if str(item).strip())
        return value


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()  # type: ignore[call-arg]
