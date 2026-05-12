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
        source="faq",
        title=title,
        content=f"Q: {title}\n\nA: ...",
        embedding_text=title.lower(),
        category=category,
        metadata={"ordinal": 0, "dataset": "take_home_faq"},
    )


def _website_seed(
    external_id: str,
    title: str,
    *,
    site_name: str = "knotch",
    page_type: str = "case_study",
) -> SeedDocument:
    return SeedDocument(
        external_id=external_id,
        source="website",
        title=title,
        content=f"Body for {title}",
        embedding_text=title.lower(),
        source_url=f"https://{site_name}.com/case-studies/{title.lower().replace(' ', '-')}",
        category=page_type,
        metadata={
            "site_name": site_name,
            "page_type": page_type,
            "customer_names": [title],
        },
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


async def test_search_filters_by_source(session: AsyncSession) -> None:
    repo = SupportDocumentRepository(session)
    generator = _embedder()
    faq = _seed("take_home_faq:reset-password-001", "Reset password")
    site = _website_seed("website:knotch:abc:000", "Acme case study")
    await repo.upsert(faq, embed=lambda d: _embed(generator, d))
    await repo.upsert(site, embed=lambda d: _embed(generator, d))

    search = SupportDocumentSearch(session)
    query_embedding = await _embed(generator, site)

    only_website = await search.search(query_embedding, limit=5, sources=["website"])
    only_faq = await search.search(query_embedding, limit=5, sources=["faq"])
    everything = await search.search(query_embedding, limit=5)

    assert [r.source for r in only_website] == ["website"]
    assert [r.source for r in only_faq] == ["faq"]
    assert {r.source for r in everything} == {"faq", "website"}


async def test_search_excludes_stale_website_rows(session: AsyncSession) -> None:
    repo = SupportDocumentRepository(session)
    generator = _embedder()
    keep = _website_seed("website:knotch:keep:000", "Acme case study")
    stale = _website_seed("website:knotch:stale:000", "Initech case study")
    for document in (keep, stale):
        await repo.upsert(document, embed=lambda d: _embed(generator, d))

    marked = await repo.mark_website_chunks_stale(
        site_name="knotch",
        keep_external_ids=[keep.external_id],
    )
    assert marked == 1

    search = SupportDocumentSearch(session)
    results = await search.search(
        await _embed(generator, stale),
        limit=5,
        sources=["website"],
    )
    assert [r.external_id for r in results] == [keep.external_id]


async def test_mark_stale_is_scoped_to_one_site(session: AsyncSession) -> None:
    repo = SupportDocumentRepository(session)
    generator = _embedder()
    knotch_doc = _website_seed("website:knotch:1:000", "Acme")
    other_doc = _website_seed("website:other:1:000", "Beta", site_name="other")
    for d in (knotch_doc, other_doc):
        await repo.upsert(d, embed=lambda doc: _embed(generator, doc))

    marked = await repo.mark_website_chunks_stale(
        site_name="knotch",
        keep_external_ids=[],
    )
    assert marked == 1

    other_after = await repo.fetch_website_external_ids("other")
    assert other_doc.external_id in other_after
