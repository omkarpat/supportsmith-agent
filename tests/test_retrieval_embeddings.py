import pytest

from app.llm.fake import FakeEmbeddingClient
from app.retrieval.embeddings import EmbeddingGenerator


async def test_embedding_generator_returns_one_vector_per_text() -> None:
    generator = EmbeddingGenerator(FakeEmbeddingClient(dimensions=16))

    vectors = await generator.embed_many(["reset password", "export data", "phishing"])

    assert len(vectors) == 3
    assert all(len(vector) == 16 for vector in vectors)


async def test_embedding_generator_respects_batch_size() -> None:
    client = FakeEmbeddingClient(dimensions=4)
    generator = EmbeddingGenerator(client, batch_size=2)

    await generator.embed_many(["a", "b", "c", "d", "e"])

    batches = [request.texts for request in client.requests]
    assert batches == [["a", "b"], ["c", "d"], ["e"]]


async def test_embedding_generator_returns_empty_for_no_inputs() -> None:
    generator = EmbeddingGenerator(FakeEmbeddingClient(dimensions=4))

    assert await generator.embed_many([]) == []


def test_embedding_generator_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError):
        EmbeddingGenerator(FakeEmbeddingClient(), batch_size=0)
