"""Ingest a public website into ``support_documents``.

Run from the repo root::

    uv run --env-file .env supportsmith-ingest-website knotch --map-only
    uv run --env-file .env supportsmith-ingest-website knotch --dry-run
    uv run --env-file .env supportsmith-ingest-website knotch --limit 500
    uv run --env-file .env supportsmith-ingest-website --url https://example.com/ --name example

Requires ``SUPPORTSMITH_FIRECRAWL_API_KEY``. Live embeddings are the default;
``--fake-embeddings`` switches to a deterministic local embedder for offline
use (the live agent must match for the vectors to retrieve consistently).
"""

from __future__ import annotations

import argparse
import asyncio
import json

from app.core.config import get_settings
from app.db.session import create_engine, create_session_factory
from app.ingestion.website import WebsiteIngestionSummary, WebsiteIngestor
from app.llm.client import EmbeddingClient
from app.llm.fake import FakeEmbeddingClient
from app.llm.openai import OpenAIChatCompletionsClient, OpenAIEmbeddingClient
from app.retrieval.firecrawl import FirecrawlSDKClient
from app.retrieval.sources.websites import (
    CrawlConfig,
    WebsiteSourceConfig,
    load_website_source,
)
from app.retrieval.website_classifier import (
    CustomerNameExtractor,
    WebsiteLLMConfig,
    WebsitePageClassifier,
)

EMBEDDING_DIMENSIONS = 1536


def build_parser() -> argparse.ArgumentParser:
    """Build the website ingestion CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Crawl a public website and ingest it into pgvector.",
    )
    parser.add_argument(
        "name",
        nargs="?",
        help="Site config name under data/websites/. Required unless --url is given.",
    )
    parser.add_argument(
        "--url",
        help="Base URL to ingest. When supplied, --name is used as the site name.",
    )
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="List discovered URLs without crawling or writing to the database.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Crawl, classify, and chunk without embedding or writing rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override crawl.limit from the site config.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Override crawl.max_depth from the site config.",
    )
    parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help=(
            "Use a deterministic local embedder instead of OpenAI. The live "
            "agent must be configured with the matching fake embedder for "
            "retrieval to behave consistently."
        ),
    )
    return parser


def _resolve_config(args: argparse.Namespace) -> WebsiteSourceConfig:
    if args.url:
        if not args.name:
            raise SystemExit("--url requires a positional site name to label the source")
        config = WebsiteSourceConfig(
            name=args.name,
            base_url=args.url,
            description=None,
            crawl=CrawlConfig(),
        )
    else:
        if not args.name:
            raise SystemExit("Site name is required (or supply --url <url> --name <name>)")
        config = load_website_source(args.name)

    crawl_updates: dict[str, object] = {}
    if args.limit is not None:
        crawl_updates["limit"] = args.limit
    if args.max_depth is not None:
        crawl_updates["max_depth"] = args.max_depth
    if crawl_updates:
        config = config.model_copy(update={"crawl": config.crawl.model_copy(update=crawl_updates)})
    return config


async def _run(args: argparse.Namespace) -> dict[str, object]:
    settings = get_settings()
    config = _resolve_config(args)

    if not settings.firecrawl_api_key:
        raise SystemExit(
            "Firecrawl ingestion requires SUPPORTSMITH_FIRECRAWL_API_KEY (or FIRECRAWL_API_KEY)."
        )

    firecrawl = FirecrawlSDKClient(api_key=settings.firecrawl_api_key)

    # --map-only never crawls pages — Firecrawl alone is enough, no LLM, no DB.
    if args.map_only:
        from app.retrieval.firecrawl import FirecrawlMap

        mapped: FirecrawlMap = await firecrawl.map_site(config.base_url, limit=config.crawl.limit)
        return {
            "site_name": config.name,
            "base_url": config.base_url,
            "phase": "map",
            "discovered_urls": len(mapped.urls),
            "urls": mapped.urls,
        }

    if not settings.openai_api_key:
        raise SystemExit(
            "Website ingestion requires OPENAI_API_KEY for page classification "
            "and customer-name extraction. Use --map-only for an LLM-free URL "
            "inventory."
        )

    embedding_client: EmbeddingClient
    if args.fake_embeddings:
        embedding_client = FakeEmbeddingClient(dimensions=EMBEDDING_DIMENSIONS)
    else:
        embedding_client = OpenAIEmbeddingClient(api_key=settings.openai_api_key)

    llm = OpenAIChatCompletionsClient(api_key=settings.openai_api_key)
    classifier_config = WebsiteLLMConfig(
        classifier_model=settings.website_classifier_model,
        classifier_reasoning_effort=settings.website_classifier_reasoning_effort,
        classifier_max_completion_tokens=settings.website_classifier_max_completion_tokens,
        extractor_model=settings.website_extractor_model,
        extractor_reasoning_effort=settings.website_extractor_reasoning_effort,
        extractor_max_completion_tokens=settings.website_extractor_max_completion_tokens,
    )

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    try:
        ingestor = WebsiteIngestor(
            firecrawl=firecrawl,
            classifier=WebsitePageClassifier(llm, classifier_config),
            extractor=CustomerNameExtractor(llm, classifier_config),
            embedding_client=embedding_client,
            embedding_model=settings.embedding_model,
            session_factory=session_factory,
            max_chunks_per_page=settings.website_max_chunks_per_page,
            max_total_chunks=settings.website_max_total_chunks_per_job,
        )
        summary: WebsiteIngestionSummary = await ingestor.run(config, dry_run=args.dry_run)
    finally:
        await engine.dispose()

    return summary.model_dump(mode="json")


def main() -> None:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    result = asyncio.run(_run(args))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
