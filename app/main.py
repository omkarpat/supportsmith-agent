"""FastAPI application factory for SupportSmith."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.wiring import build_live_support_agent
from app.api.routes import conversations, health
from app.core.config import Settings, get_settings
from app.db.postgres import PostgresDatabase
from app.retrieval.search import SupportDocumentSearch


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize app-scoped dependencies used by route handlers."""
    settings: Settings = app.state.settings
    app.state.database = PostgresDatabase.from_settings(settings)
    await app.state.database.connect()
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


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title=resolved_settings.service_name,
        version=resolved_settings.app_version,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings

    app.include_router(health.router)
    app.include_router(conversations.router)
    return app
