"""FastAPI application factory for SupportSmith."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.wiring import build_live_support_agent
from app.api.routes import conversations, health, website_ingestions
from app.api.security import configure_security
from app.core.config import Settings, get_settings
from app.db.postgres import PostgresDatabase
from app.ingestion.jobs import IngestionJobRegistry
from app.ingestion.website import WebsiteIngestor
from app.llm.client import EmbeddingClient
from app.llm.fake import FakeEmbeddingClient
from app.llm.openai import OpenAIChatCompletionsClient, OpenAIEmbeddingClient
from app.retrieval.firecrawl import FirecrawlClient, FirecrawlSDKClient
from app.retrieval.search import SupportDocumentSearch
from app.retrieval.website_classifier import (
    CustomerNameExtractor,
    WebsiteLLMConfig,
    WebsitePageClassifier,
)

EMBEDDING_DIMENSIONS = 1536


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize app-scoped dependencies used by route handlers."""
    settings: Settings = app.state.settings
    app.state.database = PostgresDatabase.from_settings(settings)
    await app.state.database.connect()
    app.state.ingestion_jobs = IngestionJobRegistry()
    app.state.website_ingestor_factory = _build_ingestor_factory(app, settings)
    try:
        if settings.environment != "test":
            session_factory = app.state.database.session_factory
            assert session_factory is not None, "database session factory must be initialized"
            search = await _build_search(session_factory)
            app.state.agent = await build_live_support_agent(settings, search=search)
        # In tests, the harness in tests/conftest.py injects ``app.state.agent``
        # before requests fly. The lifespan deliberately skips agent setup so a
        # misconfigured test fails loudly instead of silently using a stub.
        yield
    finally:
        await app.state.database.close()


async def _build_search(
    session_factory: async_sessionmaker[AsyncSession],
) -> SupportDocumentSearch:
    """Construct a SupportDocumentSearch dependency.

    The current implementation opens one session for the lifetime of the
    application; Phase 5 will swap this for a per-request session pattern when
    durable conversation persistence lands.
    """
    session = session_factory()
    return SupportDocumentSearch(session)


def _build_ingestor_factory(
    app: FastAPI,
    settings: Settings,
) -> Callable[[], Awaitable[WebsiteIngestor]]:
    """Defer constructing the ingestor until the first ingestion job lands.

    This keeps cold start cheap and means a missing ``FIRECRAWL_API_KEY`` is a
    per-request 503 from the admin route rather than a hard failure on boot.
    Tests override ``app.state.website_ingestor_factory`` directly.
    """

    async def factory() -> WebsiteIngestor:
        firecrawl: FirecrawlClient
        if not settings.firecrawl_api_key:
            raise RuntimeError(
                "Firecrawl ingestion requires SUPPORTSMITH_FIRECRAWL_API_KEY."
            )
        firecrawl = FirecrawlSDKClient(api_key=settings.firecrawl_api_key)
        if not settings.openai_api_key:
            raise RuntimeError(
                "Website ingestion requires OPENAI_API_KEY for page classification."
            )
        llm = OpenAIChatCompletionsClient(api_key=settings.openai_api_key)
        classifier_config = WebsiteLLMConfig(
            classifier_model=settings.website_classifier_model,
            classifier_reasoning_effort=settings.website_classifier_reasoning_effort,
            classifier_max_completion_tokens=settings.website_classifier_max_completion_tokens,
            extractor_model=settings.website_extractor_model,
            extractor_reasoning_effort=settings.website_extractor_reasoning_effort,
            extractor_max_completion_tokens=settings.website_extractor_max_completion_tokens,
        )
        database: PostgresDatabase = app.state.database
        session_factory = database.session_factory
        assert session_factory is not None
        # Live admin ingestion always uses real embeddings so the vectors match
        # the deployed agent's embedder. Fake embeddings are CLI-only.
        embeddings: EmbeddingClient
        if settings.environment == "test":
            embeddings = FakeEmbeddingClient(dimensions=EMBEDDING_DIMENSIONS)
        else:
            embeddings = OpenAIEmbeddingClient(api_key=settings.openai_api_key)
        return WebsiteIngestor(
            firecrawl=firecrawl,
            classifier=WebsitePageClassifier(llm, classifier_config),
            extractor=CustomerNameExtractor(llm, classifier_config),
            embedding_client=embeddings,
            embedding_model=settings.embedding_model,
            session_factory=session_factory,
            max_chunks_per_page=settings.website_max_chunks_per_page,
            max_total_chunks=settings.website_max_total_chunks_per_job,
        )

    return factory


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title=resolved_settings.service_name,
        version=resolved_settings.app_version,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings

    configure_security(app, settings=resolved_settings)

    app.include_router(health.router)
    app.include_router(conversations.router)
    app.include_router(website_ingestions.router)
    return app
