"""Embedding generation wrapper used by ingestion and retrieval.

The generator only knows the :class:`EmbeddingClient` Protocol, so callers
inject either the deterministic fake (Phase 2) or a future OpenAI adapter
without touching the seed or repository code paths.
"""

from app.llm.client import EmbeddingClient, EmbeddingRequest

DEFAULT_BATCH_SIZE = 32


class EmbeddingGenerator:
    """Batched wrapper around an :class:`EmbeddingClient`."""

    def __init__(
        self,
        client: EmbeddingClient,
        *,
        model: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.client = client
        self.model = model
        self.batch_size = batch_size

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input text, in input order."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = await self.client.embed(EmbeddingRequest(texts=batch, model=self.model))
            if len(response.vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding provider returned {len(response.vectors)} vectors "
                    f"for {len(batch)} inputs"
                )
            vectors.extend(response.vectors)
        return vectors
