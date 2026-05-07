"""Provider-neutral LLM and embedding client contracts."""

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class ChatMessage(BaseModel):
    """Single chat message passed to an LLM provider."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class TokenUsage(BaseModel):
    """Token usage returned by an LLM provider."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponseSchema(BaseModel):
    """Structured-output schema for a Chat Completions response.

    OpenAI's Chat Completions accepts ``response_format={"type":"json_schema",
    "json_schema":{"name":..., "schema":..., "strict":...}}``. The adapter
    translates this Pydantic model into that wire shape; we keep the field name
    as ``schema_definition`` here so it does not collide with Pydantic v1's
    historical ``BaseModel.schema`` attribute.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    schema_definition: dict[str, Any]
    strict: bool = True


class ChatRequest(BaseModel):
    """Provider-neutral chat completion request."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage]
    model: str | None = None
    temperature: float | None = None
    response_schema: ChatResponseSchema | None = None
    reasoning_effort: ReasoningEffort | None = None
    max_completion_tokens: int | None = None


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

