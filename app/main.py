"""FastAPI application factory for SupportSmith."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent.harness import PhaseOneAgent
from app.api.routes import conversations, health
from app.core.config import Settings, get_settings
from app.db.postgres import PostgresDatabase


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize app-scoped dependencies used by route handlers."""
    settings: Settings = app.state.settings
    app.state.agent = PhaseOneAgent(service_name=settings.service_name)
    app.state.database = PostgresDatabase.from_settings(settings)
    await app.state.database.connect()
    try:
        yield
    finally:
        await app.state.database.close()


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
