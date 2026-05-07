"""SQLAlchemy-backed Postgres database client."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.session import create_engine, create_session_factory


class DatabaseHealth(BaseModel):
    """Database health payload exposed through the health endpoint."""

    model_config = ConfigDict(extra="forbid")

    status: str
    detail: str | None = None


class PostgresDatabase:
    """Small SQLAlchemy wrapper used until richer repositories land."""

    def __init__(
        self,
        database_url: str,
    ) -> None:
        self.database_url = database_url
        self.engine: AsyncEngine | None = None
        self.session_factory: async_sessionmaker[AsyncSession] | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> PostgresDatabase:
        """Create a database client from application settings."""
        return cls(database_url=settings.database_url)

    async def connect(self) -> None:
        """Create an async SQLAlchemy engine."""
        self.engine = create_engine(database_url=self.database_url)
        self.session_factory = create_session_factory(self.engine)

    async def close(self) -> None:
        """Dispose the engine if it was opened."""
        if self.engine is not None:
            await self.engine.dispose()
            self.engine = None
            self.session_factory = None

    async def health(self) -> DatabaseHealth:
        """Return database readiness without raising into route handlers."""
        if self.engine is None:
            return DatabaseHealth(status="not_connected")
        try:
            async with self.engine.connect() as connection:
                result = await connection.execute(text("select 1"))
                value = result.scalar_one_or_none()
        except Exception as exc:
            return DatabaseHealth(status="unavailable", detail=str(exc))
        if value == 1:
            return DatabaseHealth(status="ok")
        return DatabaseHealth(status="unavailable", detail="unexpected result")
