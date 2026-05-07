"""OpenAI Chat Completions adapter tests with the SDK mocked.

No live OpenAI calls are made from this file. Live exercises live in evals.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APIError

from app.llm.client import (
    ChatMessage,
    ChatRequest,
    ChatResponseSchema,
    EmbeddingRequest,
)
from app.llm.openai import (
    LLMProviderError,
    OpenAIChatCompletionsClient,
    OpenAIEmbeddingClient,
)


def _fake_completion(
    *,
    content: str = "ok",
    model: str = "gpt-5.5-chat-latest",
    prompt_tokens: int = 7,
    completion_tokens: int = 3,
    total_tokens: int = 10,
) -> SimpleNamespace:
    """Build a minimal stand-in for openai's ChatCompletion response object."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


def _client_with(create_mock: AsyncMock) -> OpenAIChatCompletionsClient:
    sdk = MagicMock()
    sdk.chat.completions.create = create_mock
    return OpenAIChatCompletionsClient(api_key="test-key", client=sdk)


async def test_adapter_maps_response_and_token_usage() -> None:
    create = AsyncMock(return_value=_fake_completion(content="hello", total_tokens=42))
    client = _client_with(create)

    response = await client.complete(
        ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="gpt-5.5-chat-latest",
        )
    )

    assert response.content == "hello"
    assert response.model == "gpt-5.5-chat-latest"
    assert response.usage.prompt_tokens == 7
    assert response.usage.completion_tokens == 3
    assert response.usage.total_tokens == 42


async def test_adapter_passes_response_format_when_schema_set() -> None:
    create = AsyncMock(return_value=_fake_completion())
    client = _client_with(create)

    schema = ChatResponseSchema(
        name="plan",
        schema_definition={
            "type": "object",
            "properties": {"tool_name": {"type": "string"}},
            "required": ["tool_name"],
            "additionalProperties": False,
        },
    )
    await client.complete(
        ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="gpt-5.5",
            response_schema=schema,
            reasoning_effort="high",
            max_completion_tokens=512,
        )
    )

    assert create.await_args is not None
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "gpt-5.5"
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["max_completion_tokens"] == 512
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "plan",
            "schema": schema.schema_definition,
            "strict": True,
        },
    }


async def test_adapter_omits_optional_fields_when_unset() -> None:
    create = AsyncMock(return_value=_fake_completion())
    client = _client_with(create)

    await client.complete(
        ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="gpt-5.5-chat-latest",
        )
    )

    assert create.await_args is not None
    kwargs = create.await_args.kwargs
    assert "response_format" not in kwargs
    assert "reasoning_effort" not in kwargs
    assert "max_completion_tokens" not in kwargs


async def test_adapter_raises_typed_error_on_provider_failure() -> None:
    create = AsyncMock(side_effect=APIError("boom", request=MagicMock(), body=None))
    client = _client_with(create)

    with pytest.raises(LLMProviderError) as excinfo:
        await client.complete(
            ChatRequest(
                messages=[ChatMessage(role="user", content="hi")],
                model="gpt-5.5-chat-latest",
            )
        )
    assert "OpenAI Chat Completions failed" in str(excinfo.value)


async def test_adapter_requires_model() -> None:
    create = AsyncMock(return_value=_fake_completion())
    client = _client_with(create)

    with pytest.raises(ValueError, match="requires ChatRequest.model"):
        await client.complete(ChatRequest(messages=[ChatMessage(role="user", content="hi")]))


# --- embedding adapter --------------------------------------------------------


def _fake_embedding_response(
    *,
    vectors: list[list[float]] | None = None,
    model: str = "text-embedding-3-small",
    prompt_tokens: int = 5,
    total_tokens: int = 5,
) -> SimpleNamespace:
    """Minimal stand-in for openai's CreateEmbeddingResponse object."""
    payload = vectors if vectors is not None else [[0.1, 0.2, 0.3]]
    data = [SimpleNamespace(index=index, embedding=vec) for index, vec in enumerate(payload)]
    return SimpleNamespace(
        data=data,
        model=model,
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, total_tokens=total_tokens),
    )


def _embedding_client_with(create_mock: AsyncMock) -> OpenAIEmbeddingClient:
    sdk = MagicMock()
    sdk.embeddings.create = create_mock
    return OpenAIEmbeddingClient(api_key="test-key", client=sdk)


async def test_embedding_adapter_maps_response_and_token_usage() -> None:
    create = AsyncMock(
        return_value=_fake_embedding_response(
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            total_tokens=11,
        )
    )
    client = _embedding_client_with(create)

    response = await client.embed(
        EmbeddingRequest(texts=["a", "b"], model="text-embedding-3-small")
    )

    assert response.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert response.model == "text-embedding-3-small"
    assert response.usage.total_tokens == 11

    assert create.await_args is not None
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "text-embedding-3-small"
    assert kwargs["input"] == ["a", "b"]


async def test_embedding_adapter_orders_results_by_index() -> None:
    out_of_order = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[0.3, 0.4]),
            SimpleNamespace(index=0, embedding=[0.1, 0.2]),
        ],
        model="text-embedding-3-small",
        usage=SimpleNamespace(prompt_tokens=0, total_tokens=0),
    )
    create = AsyncMock(return_value=out_of_order)
    client = _embedding_client_with(create)

    response = await client.embed(
        EmbeddingRequest(texts=["a", "b"], model="text-embedding-3-small")
    )

    assert response.vectors == [[0.1, 0.2], [0.3, 0.4]]


async def test_embedding_adapter_short_circuits_on_empty_input() -> None:
    create = AsyncMock()
    client = _embedding_client_with(create)

    response = await client.embed(EmbeddingRequest(texts=[], model="text-embedding-3-small"))

    assert response.vectors == []
    assert response.model == "text-embedding-3-small"
    create.assert_not_awaited()


async def test_embedding_adapter_raises_typed_error_on_provider_failure() -> None:
    from openai import APIError

    create = AsyncMock(side_effect=APIError("boom", request=MagicMock(), body=None))
    client = _embedding_client_with(create)

    with pytest.raises(LLMProviderError) as excinfo:
        await client.embed(EmbeddingRequest(texts=["a"], model="text-embedding-3-small"))
    assert "OpenAI Embeddings failed" in str(excinfo.value)


async def test_embedding_adapter_requires_model() -> None:
    create = AsyncMock(return_value=_fake_embedding_response())
    client = _embedding_client_with(create)

    with pytest.raises(ValueError, match="requires EmbeddingRequest.model"):
        await client.embed(EmbeddingRequest(texts=["a"]))
