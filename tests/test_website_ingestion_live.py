"""One end-to-end smoke test that hits the real Firecrawl + OpenAI APIs.

Opt-in: marked ``live`` and excluded from the default pytest run. Spends a
small amount of Firecrawl and OpenAI credit so we keep it tightly scoped to
the map+dry-run path (no embedding writes, no recursive crawl).

Requires:
    - SUPPORTSMITH_FIRECRAWL_API_KEY (or FIRECRAWL_API_KEY) in the env
    - OPENAI_API_KEY (or SUPPORTSMITH_OPENAI_API_KEY) for the LLM-driven
      classifier and customer-name extractor
    - SUPPORTSMITH_TEST_DATABASE_URL for the engine factory wiring (no
      writes happen in dry-run mode)
"""

from __future__ import annotations

import os

import pytest

from app.core.config import get_settings
from app.db.session import create_engine, create_session_factory
from app.ingestion.website import WebsiteIngestor
from app.llm.fake import FakeEmbeddingClient
from app.llm.openai import OpenAIChatCompletionsClient
from app.retrieval.firecrawl import FirecrawlSDKClient
from app.retrieval.sources.websites import CrawlConfig, WebsiteSourceConfig
from app.retrieval.website_classifier import (
    CustomerNameExtractor,
    WebsiteLLMConfig,
    WebsitePageClassifier,
)

DATABASE_URL = os.environ.get("SUPPORTSMITH_TEST_DATABASE_URL")
FIRECRAWL_KEY = os.environ.get("SUPPORTSMITH_FIRECRAWL_API_KEY") or os.environ.get(
    "FIRECRAWL_API_KEY"
)
OPENAI_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("SUPPORTSMITH_OPENAI_API_KEY")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not FIRECRAWL_KEY, reason="Set SUPPORTSMITH_FIRECRAWL_API_KEY to run."),
    pytest.mark.skipif(not OPENAI_KEY, reason="Set OPENAI_API_KEY to run."),
    pytest.mark.skipif(not DATABASE_URL, reason="Set SUPPORTSMITH_TEST_DATABASE_URL to run."),
]


async def test_knotch_map_only_returns_urls() -> None:
    """Firecrawl ``map`` returns a non-empty URL inventory for the public site."""
    assert FIRECRAWL_KEY is not None
    firecrawl = FirecrawlSDKClient(api_key=FIRECRAWL_KEY)
    result = await firecrawl.map_site("https://knotch.com/", limit=20)
    assert result.site_url.startswith("https://knotch.com")
    assert result.urls, "map_site should return at least one discovered URL"


async def test_dry_run_classifies_and_chunks_a_small_crawl() -> None:
    """A capped Firecrawl crawl + LLM classifier round-trips against the live APIs.

    Dry-run skips the database write so no rows are inserted; the test only
    asserts the pipeline returns typed chunks + page-type metadata.
    """
    assert FIRECRAWL_KEY is not None
    assert OPENAI_KEY is not None
    assert DATABASE_URL is not None

    settings = get_settings()
    firecrawl = FirecrawlSDKClient(api_key=FIRECRAWL_KEY)
    llm = OpenAIChatCompletionsClient(api_key=OPENAI_KEY)
    config = WebsiteLLMConfig(
        classifier_model=settings.website_classifier_model,
        classifier_reasoning_effort=settings.website_classifier_reasoning_effort,
        classifier_max_completion_tokens=settings.website_classifier_max_completion_tokens,
        extractor_model=settings.website_extractor_model,
        extractor_reasoning_effort=settings.website_extractor_reasoning_effort,
        extractor_max_completion_tokens=settings.website_extractor_max_completion_tokens,
    )

    engine = create_engine(DATABASE_URL)
    session_factory = create_session_factory(engine)
    try:
        ingestor = WebsiteIngestor(
            firecrawl=firecrawl,
            classifier=WebsitePageClassifier(llm, config),
            extractor=CustomerNameExtractor(llm, config),
            embedding_client=FakeEmbeddingClient(dimensions=1536),
            embedding_model="text-embedding-3-small",
            session_factory=session_factory,
        )
        site_config = WebsiteSourceConfig(
            name="knotch",
            base_url="https://knotch.com/",
            crawl=CrawlConfig(limit=3),
        )
        summary = await ingestor.run(site_config, dry_run=True)
    finally:
        await engine.dispose()

    assert summary.phase == "dry_run"
    assert summary.crawled_pages >= 1
    expected_types = {"case_study", "blog", "product", "company"}
    assert any(page.page_type in expected_types for page in summary.pages)
