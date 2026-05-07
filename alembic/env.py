"""Alembic environment for SupportSmith."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from app.core.config import get_settings
from app.db.models import Base
from app.db.session import to_async_database_url, to_sync_database_url

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_database_url() -> str:
    """Return the database URL from Alembic config or application settings."""
    configured_url = config.get_main_option("sqlalchemy.url")
    if configured_url:
        return configured_url
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL or SUPPORTSMITH_DATABASE_URL is required.")
    return settings.database_url


def run_migrations_offline() -> None:
    """Run migrations without opening a database connection."""
    context.configure(
        url=to_sync_database_url(get_database_url()),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations against an open connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations through SQLAlchemy's async engine."""
    connectable = async_engine_from_config(
        {"sqlalchemy.url": to_async_database_url(get_database_url())},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations online."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

