from app.llm.client import ChatMessage, ChatRequest, EmbeddingRequest
from app.llm.fake import FakeEmbeddingClient, FakeLLMClient


async def test_fake_llm_returns_configured_response_and_usage() -> None:
    client = FakeLLMClient(response_text="hello from fake")

    response = await client.complete(
        ChatRequest(messages=[ChatMessage(role="user", content="hi there")])
    )

    assert response.content == "hello from fake"
    assert response.model == "fake-chat-model"
    assert response.usage.prompt_tokens == 2
    assert response.usage.completion_tokens == 3
    assert response.usage.total_tokens == 5
    assert len(client.requests) == 1


async def test_fake_embeddings_are_stable_and_normalized() -> None:
    client = FakeEmbeddingClient(dimensions=4)

    first = await client.embed(EmbeddingRequest(texts=["reset password"]))
    second = await client.embed(EmbeddingRequest(texts=["reset password"]))

    assert first.vectors == second.vectors
    assert len(first.vectors[0]) == 4
    magnitude = sum(value * value for value in first.vectors[0]) ** 0.5
    assert magnitude == 1.0

