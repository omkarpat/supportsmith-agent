"""SQLAlchemy async engine and session helpers."""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def to_async_database_url(database_url: str) -> str:
    """Convert a standard Postgres URL to SQLAlchemy's async psycopg URL."""
    if database_url.startswith("postgresql+psycopg://"):
        return database_url
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


def to_sync_database_url(database_url: str) -> str:
    """Convert async psycopg URLs to a sync URL for Alembic offline rendering."""
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    return database_url


def create_engine(database_url: str) -> AsyncEngine:
    """Create the application async SQLAlchemy engine."""
    return create_async_engine(
        to_async_database_url(database_url),
        pool_pre_ping=True,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create the application async session factory."""
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autoflush=False,
    )
