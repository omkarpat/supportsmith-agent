"""pgvector cosine search over ``support_documents``."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SupportDocument
from app.retrieval.models import RetrievalResult, SeedSource

DEFAULT_LIMIT = 5


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
    ) -> list[RetrievalResult]:
        """Return the top-``limit`` documents ranked by cosine similarity.

        Cosine distance from pgvector lives in ``[0, 2]``; score is
        ``1 - distance`` so callers can apply a single ``min_score`` threshold.
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
    if value != "take_home_faq":
        raise ValueError(f"Unexpected source value in support_documents: {value!r}")
    return "take_home_faq"
