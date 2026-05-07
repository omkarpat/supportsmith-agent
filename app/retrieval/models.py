"""Pydantic contracts for the retrieval and ingestion layer."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SeedSource = Literal["take_home_faq"]


class SeedDocument(BaseModel):
    """One validated, citable document destined for ``support_documents``.

    ``content`` preserves the original text so citations stay readable.
    ``embedding_text`` is the normalized form passed to the embedding model.
    """

    model_config = ConfigDict(extra="forbid")

    external_id: str = Field(min_length=1)
    source: SeedSource
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    embedding_text: str = Field(min_length=1)
    source_url: str | None = None
    category: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """One ranked search hit returned to retrieval callers."""

    model_config = ConfigDict(extra="forbid")

    external_id: str
    source: SeedSource
    title: str
    content: str
    source_url: str | None = None
    category: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float
    distance: float


class UpsertOutcome(BaseModel):
    """Per-document outcome from the repository upsert path."""

    model_config = ConfigDict(extra="forbid")

    external_id: str
    action: Literal["inserted", "updated", "unchanged"]
    embedded: bool


class RejectedItem(BaseModel):
    """One source row that was filtered out before reaching the repository."""

    model_config = ConfigDict(extra="forbid")

    external_id: str
    title: str
    reason: str


class UpsertSummary(BaseModel):
    """Aggregate result for a seed run."""

    model_config = ConfigDict(extra="forbid")

    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    embedded: int = 0
    rejected: list[RejectedItem] = Field(default_factory=list)
    outcomes: list[UpsertOutcome] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
