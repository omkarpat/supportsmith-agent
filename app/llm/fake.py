"""Deterministic fake LLM clients for tests and local harnesses."""

import hashlib

from app.llm.client import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    TokenUsage,
)


class FakeLLMClient:
    """A deterministic chat client that returns configured text."""

    def __init__(self, response_text: str = "SupportSmith fake response") -> None:
        self.response_text = response_text
        self.requests: list[ChatRequest] = []

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Return a canned response while recording the request."""
        self.requests.append(request)
        prompt_tokens = sum(len(message.content.split()) for message in request.messages)
        completion_tokens = len(self.response_text.split())
        return ChatResponse(
            content=self.response_text,
            model=request.model or "fake-chat-model",
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


class FakeEmbeddingClient:
    """A deterministic embedding client with stable vectors per input string."""

    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions
        self.requests: list[EmbeddingRequest] = []

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Return stable pseudo-embeddings based on SHA-256 bytes."""
        self.requests.append(request)
        vectors = [self._vector_for_text(text) for text in request.texts]
        token_count = sum(len(text.split()) for text in request.texts)
        return EmbeddingResponse(
            vectors=vectors,
            model=request.model or "fake-embedding-model",
            usage=TokenUsage(total_tokens=token_count),
        )

    def _vector_for_text(self, text: str) -> list[float]:
        material = bytearray()
        counter = 0
        while len(material) < self.dimensions:
            block = hashlib.sha256(f"{text}|{counter}".encode()).digest()
            material.extend(block)
            counter += 1
        values = [material[index] / 255.0 for index in range(self.dimensions)]
        magnitude = sum(value * value for value in values) ** 0.5
        if magnitude == 0:
            return values
        return [value / magnitude for value in values]

