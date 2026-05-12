"""pgvector cosine search over ``support_documents``."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SupportDocument
from app.retrieval.models import RetrievalResult, SeedSource

DEFAULT_LIMIT = 5
DEFAULT_QUALITY: tuple[str, ...] = ("trusted",)
ALLOWED_SOURCES: frozenset[SeedSource] = frozenset(("faq", "website"))


class SupportDocumentSearch:
    """Cosine-similarity search returning typed retrieval results."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def search(
        self,
        embedding: list[float],
        *,
        limit: int = DEFAULT_LIMIT,
        category: str | None = None,
        min_score: float | None = None,
        sources: Sequence[SeedSource] | None = None,
        qualities: Sequence[str] | None = DEFAULT_QUALITY,
    ) -> list[RetrievalResult]:
        """Return the top-``limit`` documents ranked by cosine similarity.

        Cosine distance from pgvector lives in ``[0, 2]``; score is
        ``1 - distance`` so callers can apply a single ``min_score`` threshold.

        ``sources`` filters by document source (``faq``, ``website``). ``None``
        searches across all known sources. ``qualities`` filters by row quality
        and defaults to ``("trusted",)`` so stale or low-quality rows never
        reach the agent. Pass an empty tuple to disable the quality filter.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")

        distance = SupportDocument.embedding.cosine_distance(embedding).label("distance")
        stmt = (
            select(SupportDocument, distance)
            .where(SupportDocument.embedding.is_not(None))
            .order_by(distance)
            .limit(limit)
        )
        if category is not None:
            stmt = stmt.where(SupportDocument.category == category)
        if sources is not None:
            stmt = stmt.where(SupportDocument.source.in_(list(sources)))
        if qualities:
            stmt = stmt.where(SupportDocument.quality.in_(list(qualities)))

        result = await self.session.execute(stmt)
        results: list[RetrievalResult] = []
        for row, dist in result.all():
            score = 1.0 - float(dist)
            if min_score is not None and score < min_score:
                continue
            results.append(_to_retrieval_result(row, score=score, distance=float(dist)))
        return results

    async def list_categories(self) -> list[str]:
        """Return the distinct, non-null categories currently stored."""
        stmt = (
            select(SupportDocument.category)
            .where(SupportDocument.category.is_not(None))
            .distinct()
            .order_by(SupportDocument.category)
        )
        result = await self.session.execute(stmt)
        return [category for category in result.scalars().all() if category is not None]


def _to_retrieval_result(
    row: SupportDocument,
    *,
    score: float,
    distance: float,
) -> RetrievalResult:
    return RetrievalResult(
        external_id=row.external_id,
        source=_validated_source(row.source),
        title=row.title,
        content=row.content,
        source_url=row.source_url,
        category=row.category,
        metadata=row.metadata_,
        score=score,
        distance=distance,
    )


def _validated_source(value: str) -> SeedSource:
    if value == "faq":
        return "faq"
    if value == "website":
        return "website"
    raise ValueError(f"Unexpected source value in support_documents: {value!r}")
