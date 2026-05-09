"""FastAPI dependencies for Phase 5+ persistence and agent access."""

from collections.abc import AsyncIterator

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.harness import Agent


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield one async SQLAlchemy session per request.

    The session is closed after the response is sent, even if a background
    task is still using a different session it opened on its own. Routes
    that mutate state should wrap their own ``async with session.begin()``
    block — this dependency only owns lifecycle.
    """
    session_factory = request.app.state.database.session_factory
    if session_factory is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    async with session_factory() as session:
        yield session


def get_agent(request: Request) -> Agent:
    """Return the lifespan-built ``SupportAgent`` (or a test-injected one)."""
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent  # type: ignore[no-any-return]


__all__ = ["get_db_session", "get_agent", "Depends"]
