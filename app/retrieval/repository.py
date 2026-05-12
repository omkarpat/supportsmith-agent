"""Reads and writes for the ``support_documents`` table."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SupportDocument
from app.retrieval.models import SeedDocument, UpsertOutcome
from app.retrieval.normalization import compute_content_hash

EmbeddingFn = Callable[[SeedDocument], Awaitable[list[float]]]
UpsertAction = Literal["inserted", "updated", "unchanged"]


class SupportDocumentRepository:
    """Idempotent writer for ``support_documents`` rows."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(
        self,
        document: SeedDocument,
        *,
        embed: EmbeddingFn,
    ) -> UpsertOutcome:
        """Insert, update, or skip ``document`` based on its content hash.

        ``embed`` is invoked only when the row is new or its content has
        actually changed, so re-running ingestion does not re-embed unchanged
        rows.
        """
        new_hash = compute_content_hash(document)
        existing = await self._fetch_by_external_id(document.external_id)

        if existing is not None and existing.content_hash == new_hash:
            return UpsertOutcome(
                external_id=document.external_id,
                action="unchanged",
                embedded=False,
            )

        embedding = await embed(document)
        action: UpsertAction = "inserted" if existing is None else "updated"

        stmt = (
            pg_insert(SupportDocument)
            .values(
                external_id=document.external_id,
                source=document.source,
                source_url=document.source_url,
                title=document.title,
                content=document.content,
                content_hash=new_hash,
                category=document.category,
                quality="trusted",
                metadata_=document.metadata,
                embedding=embedding,
            )
            .on_conflict_do_update(
                index_elements=[SupportDocument.external_id],
                set_={
                    "source": document.source,
                    "source_url": document.source_url,
                    "title": document.title,
                    "content": document.content,
                    "content_hash": new_hash,
                    "category": document.category,
                    "quality": "trusted",
                    "metadata": document.metadata,
                    "embedding": embedding,
                },
            )
        )
        await self.session.execute(stmt)
        return UpsertOutcome(external_id=document.external_id, action=action, embedded=True)

    async def fetch_all(self) -> Sequence[SupportDocument]:
        """Return every persisted document. Used by tests and inspection tools."""
        result = await self.session.execute(select(SupportDocument))
        return result.scalars().all()

    async def fetch_website_external_ids(self, site_name: str) -> list[str]:
        """Return the external_ids previously stored for one website source."""
        stmt = select(SupportDocument.external_id).where(
            SupportDocument.source == "website",
            SupportDocument.metadata_["site_name"].astext == site_name,
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def mark_website_chunks_stale(
        self,
        *,
        site_name: str,
        keep_external_ids: Sequence[str],
    ) -> int:
        """Mark previously ingested website chunks not in ``keep_external_ids`` stale.

        Returns the number of rows updated. Stale rows are excluded from the
        default search query because :class:`SupportDocumentSearch` filters on
        ``quality="trusted"``.
        """
        stmt = (
            update(SupportDocument)
            .where(
                SupportDocument.source == "website",
                SupportDocument.metadata_["site_name"].astext == site_name,
                SupportDocument.external_id.notin_(list(keep_external_ids)),
                SupportDocument.quality == "trusted",
            )
            .values(quality="stale")
        )
        result = await self.session.execute(stmt)
        # ``CursorResult.rowcount`` is the typed accessor for affected rows;
        # the SQLAlchemy stub for the base ``Result`` doesn't expose it.
        rowcount = getattr(result, "rowcount", 0) or 0
        return int(rowcount)

    async def _fetch_by_external_id(self, external_id: str) -> SupportDocument | None:
        result = await self.session.execute(
            select(SupportDocument).where(SupportDocument.external_id == external_id)
        )
        return result.scalar_one_or_none()
