"""Provider-neutral LLM and embedding client contracts."""

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """Single chat message passed to an LLM provider."""

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class TokenUsage(BaseModel):
    """Token usage returned by an LLM provider."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatRequest(BaseModel):
    """Provider-neutral chat completion request."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = 0.0
    response_schema: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    """Provider-neutral chat completion response."""

    model_config = ConfigDict(extra="forbid")

    content: str
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)


class EmbeddingRequest(BaseModel):
    """Provider-neutral embedding request."""

    model_config = ConfigDict(extra="forbid")

    texts: list[str]
    model: str | None = None


class EmbeddingResponse(BaseModel):
    """Provider-neutral embedding response."""

    model_config = ConfigDict(extra="forbid")

    vectors: list[list[float]]
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)


class LLMClient(Protocol):
    """Interface for chat-completion providers."""

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Return a chat completion."""


class EmbeddingClient(Protocol):
    """Interface for embedding providers."""

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Return embeddings for the supplied texts."""

