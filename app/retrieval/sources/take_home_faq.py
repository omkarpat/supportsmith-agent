"""Load and validate the take-home FAQ knowledge base file."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.chunker import FAQChunker
from app.retrieval.models import RejectedItem, SeedDocument

KnowledgeBaseQuality = Literal["trusted", "low_quality", "ambiguous"]
SOURCE: Literal["take_home_faq"] = "take_home_faq"

_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


class RawKnowledgeBaseItem(BaseModel):
    """One raw row as it appears in ``data/knowledge-base.json``."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    category: str | None = None
    quality: KnowledgeBaseQuality


class KnowledgeBaseFile(BaseModel):
    """Top-level structure of the take-home knowledge base file."""

    model_config = ConfigDict(extra="forbid")

    knowledge_base_items: list[RawKnowledgeBaseItem]


class LoadedKnowledgeBase(BaseModel):
    """Result of loading and partitioning the knowledge base file."""

    model_config = ConfigDict(extra="forbid")

    documents: list[SeedDocument]
    rejected: list[RejectedItem]


def load_take_home_faq(path: Path) -> LoadedKnowledgeBase:
    """Read the knowledge base, validate each row, and partition by quality.

    Trusted rows become SeedDocuments. Non-trusted rows are returned as
    RejectedItems so the seed CLI can surface them in its run summary
    instead of silently dropping content.
    """
    raw = json.loads(path.read_text())
    file = KnowledgeBaseFile.model_validate(raw)

    chunker = FAQChunker()
    documents: list[SeedDocument] = []
    rejected: list[RejectedItem] = []

    for index, item in enumerate(file.knowledge_base_items):
        external_id = _build_external_id(item.question, ordinal=index)
        title = item.question.strip()

        if item.quality != "trusted":
            rejected.append(
                RejectedItem(
                    external_id=external_id,
                    title=title,
                    reason=f"quality={item.quality}",
                )
            )
            continue

        chunks = chunker.split(title=item.question, body=item.answer)
        if len(chunks) != 1:
            raise RuntimeError(
                f"FAQChunker returned {len(chunks)} chunks for one FAQ row; expected 1"
            )
        chunk = chunks[0]
        documents.append(
            SeedDocument(
                external_id=external_id,
                source=SOURCE,
                title=title,
                content=chunk.content,
                embedding_text=chunk.embedding_text,
                category=item.category,
                metadata={"ordinal": index},
            )
        )

    return LoadedKnowledgeBase(documents=documents, rejected=rejected)


def _build_external_id(question: str, *, ordinal: int) -> str:
    """Build a stable, human-recognizable external id for a FAQ row.

    Slugged form keeps log output legible; the question hash + ordinal suffix
    guarantees uniqueness when slugs collide (e.g., short or near-empty
    questions like the ambiguous ``x`` row).
    """
    slug = _NON_SLUG_RE.sub("-", question.lower()).strip("-")
    if not slug:
        slug = "row"
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]
    return f"{SOURCE}:{slug[:48]}-{digest}-{ordinal:03d}"
