"""Source-agnostic chunking interface for retrieval ingestion.

FAQ rows arrive pre-chunked: each row is one citable atomic unit, so the
``FAQChunker`` is essentially identity. The Protocol exists so later sources
(scraped Knotch pages, long-form docs) can plug in paragraph-level chunkers
without changing the seed/upsert code path.
"""

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """One unit of content paired with its embedding text and citation title."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    embedding_text: str = Field(min_length=1)
    ordinal: int = 0


class Chunker(Protocol):
    """Splits source content into citable chunks."""

    def split(self, *, title: str, body: str) -> list[Chunk]:
        """Return one or more chunks from the supplied source unit."""


class FAQChunker:
    """One question + answer becomes one chunk."""

    def split(self, *, title: str, body: str) -> list[Chunk]:
        """Return a single chunk containing the FAQ question and its answer."""
        from app.retrieval.normalization import normalize_text

        content = f"Q: {title.strip()}\n\nA: {body.strip()}"
        embedding_text = f"{normalize_text(title)} {normalize_text(body)}".strip()
        return [Chunk(title=title.strip(), content=content, embedding_text=embedding_text)]
