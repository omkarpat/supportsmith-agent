"""Repository and search tests against a real Postgres + pgvector instance.

Skipped unless ``SUPPORTSMITH_TEST_DATABASE_URL`` is set. When present, each
test runs in a transaction that's rolled back so the database stays clean.
Run locally with ``docker compose up -d postgres`` and then export
``SUPPORTSMITH_TEST_DATABASE_URL`` to the same Postgres URL used in
``.env`` before invoking ``uv run pytest tests/test_retrieval_repository.py``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.migrate import upgrade
from app.db.session import create_engine, create_session_factory
from app.llm.fake import FakeEmbeddingClient
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.models import SeedDocument
from app.retrieval.repository import SupportDocumentRepository
from app.retrieval.search import SupportDocumentSearch

DATABASE_URL = os.environ.get("SUPPORTSMITH_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="Set SUPPORTSMITH_TEST_DATABASE_URL to run repository/search tests against Postgres.",
)


@pytest.fixture(scope="module", autouse=True)
def _migrate_database() -> None:
    """Apply migrations once per module so the schema is current."""
    assert DATABASE_URL is not None
    upgrade(DATABASE_URL)


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """One transactional session per test; rolled back to keep the DB clean."""
    assert DATABASE_URL is not None
    engine = create_engine(DATABASE_URL)
    factory = create_session_factory(engine)
    async with factory() as inner_session, inner_session.begin():
        # Scope each test to a fresh slate without dropping the schema
        # that other devs may have populated.
        await inner_session.execute(_truncate_support_documents())
        yield inner_session
        await inner_session.rollback()
    await engine.dispose()


def _truncate_support_documents() -> Any:
    from sqlalchemy import text

    return text("truncate table support_documents restart identity")


def _seed(external_id: str, title: str, *, category: str = "security") -> SeedDocument:
    return SeedDocument(
        external_id=external_id,
        source="take_home_faq",
        title=title,
        content=f"Q: {title}\n\nA: ...",
        embedding_text=title.lower(),
        category=category,
        metadata={"ordinal": 0},
    )


def _embedder() -> EmbeddingGenerator:
    return EmbeddingGenerator(FakeEmbeddingClient(dimensions=1536))


async def _embed(generator: EmbeddingGenerator, document: SeedDocument) -> list[float]:
    return (await generator.embed_many([document.embedding_text]))[0]


async def test_upsert_inserts_then_marks_unchanged(session: AsyncSession) -> None:
    repo = SupportDocumentRepository(session)
    generator = _embedder()
    doc = _seed("take_home_faq:reset-password-aaa-000", "Reset my password")

    first = await repo.upsert(doc, embed=lambda d: _embed(generator, d))
    second = await repo.upsert(doc, embed=lambda d: _embed(generator, d))

    assert first.action == "inserted"
    assert first.embedded is True
    assert second.action == "unchanged"
    assert second.embedded is False


async def test_upsert_updates_when_content_changes_and_re_embeds(session: AsyncSession) -> None:
    repo = SupportDocumentRepository(session)
    generator = _embedder()
    original = _seed("take_home_faq:reset-password-aaa-000", "Reset my password")
    revised = original.model_copy(update={"content": "Q: Reset my password\n\nA: Updated."})

    await repo.upsert(original, embed=lambda d: _embed(generator, d))
    second = await repo.upsert(revised, embed=lambda d: _embed(generator, d))

    assert second.action == "updated"
    assert second.embedded is True


async def test_search_ranks_by_cosine_similarity(session: AsyncSession) -> None:
    repo = SupportDocumentRepository(session)
    generator = _embedder()
    documents = [
        _seed("take_home_faq:reset-password-001", "Reset my password"),
        _seed("take_home_faq:export-data-002", "How do I export my data?", category="privacy"),
        _seed("take_home_faq:phishing-003", "Recognize a phishing email"),
    ]
    for document in documents:
        await repo.upsert(document, embed=lambda d: _embed(generator, d))

    search = SupportDocumentSearch(session)
    query_embedding = await _embed(generator, documents[1])

    results = await search.search(query_embedding, limit=2)

    assert results[0].external_id == "take_home_faq:export-data-002"
    assert results[0].score > results[1].score


async def test_search_filters_by_category(session: AsyncSession) -> None:
    repo = SupportDocumentRepository(session)
    generator = _embedder()
    security = _seed("take_home_faq:reset-password-001", "Reset my password")
    privacy = _seed(
        "take_home_faq:export-data-002",
        "How do I export my data?",
        category="privacy",
    )
    await repo.upsert(security, embed=lambda d: _embed(generator, d))
    await repo.upsert(privacy, embed=lambda d: _embed(generator, d))

    search = SupportDocumentSearch(session)
    query_embedding = await _embed(generator, security)

    results = await search.search(query_embedding, limit=5, category="privacy")

    assert [result.external_id for result in results] == ["take_home_faq:export-data-002"]


async def test_search_lists_categories(session: AsyncSession) -> None:
    repo = SupportDocumentRepository(session)
    generator = _embedder()
    await repo.upsert(
        _seed("take_home_faq:reset-password-001", "Reset password"),
        embed=lambda d: _embed(generator, d),
    )
    await repo.upsert(
        _seed("take_home_faq:export-data-002", "Export data", category="privacy"),
        embed=lambda d: _embed(generator, d),
    )

    search = SupportDocumentSearch(session)
    categories = await search.list_categories()

    assert categories == ["privacy", "security"]
