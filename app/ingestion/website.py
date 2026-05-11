"""Website ingestion orchestrator.

Glues the Firecrawl wrapper, markdown chunker, LLM classifier + extractor,
embedding generator, and repository upsert path into one async pipeline. The
CLI and the admin ingestion API both drive this module; neither knows about
Firecrawl, prompts, or the database schema directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.llm.client import EmbeddingClient
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.firecrawl import (
    CrawlOptions,
    FirecrawlClient,
    FirecrawlPage,
)
from app.retrieval.models import SeedDocument, UpsertSummary
from app.retrieval.repository import SupportDocumentRepository
from app.retrieval.sources.websites import WebsiteSourceConfig
from app.retrieval.url_utils import build_website_external_id, normalize_url
from app.retrieval.website_chunker import WebsiteChunk, split_page
from app.retrieval.website_classifier import (
    PRIORITY_PAGE_TYPES,
    CustomerNameExtractor,
    PageType,
    WebsitePageClassifier,
)

WebsiteIngestionPhase = Literal["map", "dry_run", "ingest"]


class CrawlPageSummary(BaseModel):
    """Per-page summary surfaced in the run output."""

    model_config = ConfigDict(extra="forbid")

    url: str
    page_title: str
    page_type: PageType
    priority: bool
    chunk_count: int
    skipped: bool = False
    skip_reason: str | None = None
    customer_names: list[str] = Field(default_factory=list)


class WebsiteIngestionSummary(BaseModel):
    """Full result of a website ingestion run; matches the CLI/API surface."""

    model_config = ConfigDict(extra="forbid")

    site_name: str
    base_url: str
    phase: WebsiteIngestionPhase
    crawl_job_id: str | None = None
    discovered_urls: int = 0
    crawled_pages: int = 0
    skipped_pages: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    embedded: int = 0
    stale_marked: int = 0
    total_chunks: int = 0
    pages: list[CrawlPageSummary] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class WebsiteIngestor:
    """Run one ingestion job end-to-end."""

    def __init__(
        self,
        *,
        firecrawl: FirecrawlClient,
        classifier: WebsitePageClassifier,
        extractor: CustomerNameExtractor,
        embedding_client: EmbeddingClient,
        embedding_model: str,
        session_factory: async_sessionmaker[Any],
        max_chunks_per_page: int = 40,
        max_total_chunks: int = 5000,
    ) -> None:
        self.firecrawl = firecrawl
        self.classifier = classifier
        self.extractor = extractor
        self.embeddings = EmbeddingGenerator(embedding_client, model=embedding_model)
        self.session_factory = session_factory
        self.max_chunks_per_page = max_chunks_per_page
        self.max_total_chunks = max_total_chunks

    async def map_only(self, config: WebsiteSourceConfig) -> WebsiteIngestionSummary:
        """Return the URL inventory Firecrawl discovers without crawling."""
        started = _utcnow()
        mapped = await self.firecrawl.map_site(config.base_url, limit=config.crawl.limit)
        return WebsiteIngestionSummary(
            site_name=config.name,
            base_url=config.base_url,
            phase="map",
            discovered_urls=len(mapped.urls),
            started_at=started,
            finished_at=_utcnow(),
        )

    async def run(
        self,
        config: WebsiteSourceConfig,
        *,
        dry_run: bool = False,
    ) -> WebsiteIngestionSummary:
        """Map, crawl, chunk, embed (when not dry-run), and upsert."""
        started = _utcnow()
        crawl = await self.firecrawl.crawl_site(
            config.base_url,
            options=_to_crawl_options(config),
        )
        summary = WebsiteIngestionSummary(
            site_name=config.name,
            base_url=config.base_url,
            phase="dry_run" if dry_run else "ingest",
            crawl_job_id=crawl.job_id,
            crawled_pages=len(crawl.pages),
            started_at=started,
        )

        seed_documents: list[SeedDocument] = []
        kept_external_ids: list[str] = []
        for page in crawl.pages:
            page_record = await self._process_page(page, config=config)
            summary.pages.append(page_record.summary)
            if page_record.summary.skipped:
                summary.skipped_pages += 1
                continue
            for document in page_record.documents:
                if len(seed_documents) >= self.max_total_chunks:
                    summary.skipped_pages += 1
                    break
                seed_documents.append(document)
                kept_external_ids.append(document.external_id)
        summary.total_chunks = len(seed_documents)

        if dry_run:
            summary.finished_at = _utcnow()
            return summary

        async with self.session_factory() as session, session.begin():
            repository = SupportDocumentRepository(session)
            for document in seed_documents:
                outcome = await repository.upsert(
                    document,
                    embed=lambda doc: self._embed(doc),
                )
                if outcome.action == "inserted":
                    summary.inserted += 1
                elif outcome.action == "updated":
                    summary.updated += 1
                else:
                    summary.unchanged += 1
                if outcome.embedded:
                    summary.embedded += 1
            summary.stale_marked = await repository.mark_website_chunks_stale(
                site_name=config.name,
                keep_external_ids=kept_external_ids,
            )

        summary.finished_at = _utcnow()
        return summary

    async def _embed(self, document: SeedDocument) -> list[float]:
        vectors = await self.embeddings.embed_many([document.embedding_text])
        return vectors[0]

    async def _process_page(
        self,
        page: FirecrawlPage,
        *,
        config: WebsiteSourceConfig,
    ) -> _PageRecord:
        split = split_page(page)
        path = urlparse(page.url).path or "/"
        priority = config.path_priority(path)
        if split.skip_reason or not split.chunks:
            return _PageRecord(
                summary=CrawlPageSummary(
                    url=page.url,
                    page_title=split.page_title,
                    page_type="unknown",
                    priority=priority,
                    chunk_count=0,
                    skipped=True,
                    skip_reason=split.skip_reason or "no_chunks",
                ),
                documents=[],
            )

        decision = await self.classifier.classify(
            url=page.url,
            title=split.page_title,
            description=split.page_description,
            markdown=page.markdown,
            priority_hint=priority,
        )
        # Aggregated raw signals across chunks (already on each chunk too)
        all_alt_text: tuple[str, ...] = tuple(
            alt for chunk in split.chunks for alt in chunk.asset_alt_text
        )
        all_captions: tuple[str, ...] = tuple(
            cap for chunk in split.chunks for cap in chunk.nearby_captions
        )
        all_headings: tuple[str, ...] = tuple(
            heading
            for chunk in split.chunks
            for heading in chunk.headings_path
        )

        customer = await self.extractor.extract(
            url=page.url,
            page_title=split.page_title,
            page_type=decision.page_type,
            section_headings=all_headings,
            asset_alt_text=all_alt_text,
            nearby_captions=all_captions,
            body_snippet=page.markdown,
        )

        normalized_url = normalize_url(page.url)
        documents: list[SeedDocument] = []
        for chunk in split.chunks[: self.max_chunks_per_page]:
            documents.append(
                _build_seed_document(
                    config=config,
                    page=page,
                    page_split_title=split.page_title,
                    page_description=split.page_description,
                    normalized_url=normalized_url,
                    chunk=chunk,
                    page_type=decision.page_type,
                    page_type_confidence=decision.confidence,
                    priority=priority or decision.page_type in PRIORITY_PAGE_TYPES,
                    customer_names=customer.customer_names,
                    customer_evidence=[str(e) for e in customer.evidence_types],
                )
            )

        return _PageRecord(
            summary=CrawlPageSummary(
                url=page.url,
                page_title=split.page_title,
                page_type=decision.page_type,
                priority=priority or decision.page_type in PRIORITY_PAGE_TYPES,
                chunk_count=len(documents),
                customer_names=customer.customer_names,
            ),
            documents=documents,
        )


class _PageRecord(BaseModel):
    """Internal carrier from page processing back to the run loop."""

    model_config = ConfigDict(extra="allow")

    summary: CrawlPageSummary
    documents: list[SeedDocument]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _to_crawl_options(config: WebsiteSourceConfig) -> CrawlOptions:
    return CrawlOptions(
        limit=config.crawl.limit,
        max_depth=config.crawl.max_depth,
        include_paths=config.include_paths,
        exclude_paths=config.exclude_paths,
        allow_subdomains=config.crawl.allow_subdomains,
        allow_external_links=config.crawl.allow_external_links,
        ignore_query_parameters=config.crawl.ignore_query_parameters,
        only_main_content=config.crawl.only_main_content,
    )


def _build_seed_document(
    *,
    config: WebsiteSourceConfig,
    page: FirecrawlPage,
    page_split_title: str,
    page_description: str | None,
    normalized_url: str,
    chunk: WebsiteChunk,
    page_type: PageType,
    page_type_confidence: float,
    priority: bool,
    customer_names: Sequence[str],
    customer_evidence: Sequence[str],
) -> SeedDocument:
    external_id = build_website_external_id(
        site_name=config.name,
        url=page.url,
        chunk_index=chunk.chunk_index,
    )
    citation_title = (
        chunk.section_heading
        if chunk.section_heading and chunk.section_heading != page_split_title
        else page_split_title
    )
    metadata: dict[str, Any] = {
        "site_name": config.name,
        "base_url": config.base_url,
        "canonical_url": page.canonical_url or normalized_url,
        "page_url": page.url,
        "page_title": page_split_title,
        "page_description": page_description,
        "page_type": page_type,
        "page_type_confidence": page_type_confidence,
        "priority": priority,
        "section_heading": chunk.section_heading,
        "headings_path": list(chunk.headings_path),
        "asset_alt_text": list(chunk.asset_alt_text),
        "nearby_captions": list(chunk.nearby_captions),
        "customer_names": list(customer_names),
        "customer_evidence": list(customer_evidence),
        "chunk_index": chunk.chunk_index,
        "chunk_count": chunk.chunk_count,
        "firecrawl_metadata": _strip_metadata(page.metadata),
        "crawled_at": _utcnow().isoformat(),
    }
    return SeedDocument(
        external_id=external_id,
        source="website",
        title=citation_title,
        content=chunk.content,
        embedding_text=chunk.embedding_text,
        source_url=page.url,
        category=page_type,
        metadata=metadata,
    )


def _strip_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Drop redundant or oversized fields from Firecrawl's metadata blob.

    We keep the useful fields (statusCode, language, etc.) but skip raw HTML,
    OG image arrays, and anything that wouldn't fit cleanly in JSONB.
    """
    keep_keys = {
        "statusCode",
        "language",
        "ogLocale",
        "siteName",
        "publishedTime",
        "modifiedTime",
        "robots",
    }
    return {k: v for k, v in metadata.items() if k in keep_keys}


def upsert_summary_from(summary: WebsiteIngestionSummary) -> UpsertSummary:
    """Adapt the website ingestion summary into the shared :class:`UpsertSummary`.

    Useful for CLI output that wants to display website ingestion results
    alongside FAQ seed results in the same shape.
    """
    return UpsertSummary(
        inserted=summary.inserted,
        updated=summary.updated,
        unchanged=summary.unchanged,
        embedded=summary.embedded,
        started_at=summary.started_at,
        finished_at=summary.finished_at,
    )
