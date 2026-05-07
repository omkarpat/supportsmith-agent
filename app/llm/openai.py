"""OpenAI Chat Completions and Embeddings adapters."""

from __future__ import annotations

from typing import Any

from openai import APIError, AsyncOpenAI

from app.llm.client import (
    ChatRequest,
    ChatResponse,
    ChatResponseSchema,
    EmbeddingClient,
    EmbeddingRequest,
    EmbeddingResponse,
    LLMClient,
    TokenUsage,
)


class LLMProviderError(RuntimeError):
    """Typed error raised when the upstream OpenAI provider call fails."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenAIChatCompletionsClient(LLMClient):
    """Wraps :class:`openai.AsyncOpenAI` to implement the project's LLMClient.

    Tests mock :class:`openai.AsyncOpenAI` (or this class's ``client`` attribute)
    so no live network call is ever made from ``tests/``. Live exercises live
    in ``evals/`` and are out of scope for the test suite.
    """

    def __init__(
        self,
        *,
        api_key: str,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.client = client or AsyncOpenAI(api_key=api_key)

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Call ``/v1/chat/completions`` and map the response into a typed shape."""
        if request.model is None:
            raise ValueError("OpenAIChatCompletionsClient requires ChatRequest.model to be set")

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [message.model_dump() for message in request.messages],
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.response_schema is not None:
            kwargs["response_format"] = _to_response_format(request.response_schema)
        if request.reasoning_effort is not None:
            kwargs["reasoning_effort"] = request.reasoning_effort
        if request.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = request.max_completion_tokens

        try:
            completion = await self.client.chat.completions.create(**kwargs)
        except APIError as exc:
            raise LLMProviderError(
                f"OpenAI Chat Completions failed: {exc}",
                status_code=getattr(exc, "status_code", None),
            ) from exc

        choice = completion.choices[0]
        usage = completion.usage
        return ChatResponse(
            content=choice.message.content or "",
            model=completion.model,
            usage=TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
            ),
        )


def _to_response_format(schema: ChatResponseSchema) -> dict[str, Any]:
    """Translate :class:`ChatResponseSchema` into the OpenAI wire payload."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.name,
            "schema": schema.schema_definition,
            "strict": schema.strict,
        },
    }


class OpenAIEmbeddingClient(EmbeddingClient):
    """Wraps :class:`openai.AsyncOpenAI` to implement the project's EmbeddingClient.

    Same SDK seam as :class:`OpenAIChatCompletionsClient`: tests mock
    ``openai.AsyncOpenAI.embeddings.create`` so no live network call is made
    from ``tests/``. Live exercises live in ``evals/``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.client = client or AsyncOpenAI(api_key=api_key)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Call ``/v1/embeddings`` and map the response into a typed shape."""
        if request.model is None:
            raise ValueError("OpenAIEmbeddingClient requires EmbeddingRequest.model to be set")
        if not request.texts:
            return EmbeddingResponse(vectors=[], model=request.model, usage=TokenUsage())

        try:
            result = await self.client.embeddings.create(
                model=request.model,
                input=request.texts,
            )
        except APIError as exc:
            raise LLMProviderError(
                f"OpenAI Embeddings failed: {exc}",
                status_code=getattr(exc, "status_code", None),
            ) from exc

        usage = result.usage
        # Embeddings responses sort by ``index``; we sort defensively to be robust.
        ordered = sorted(result.data, key=lambda item: item.index)
        return EmbeddingResponse(
            vectors=[item.embedding for item in ordered],
            model=result.model,
            usage=TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
            ),
        )
