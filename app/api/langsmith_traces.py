"""LangSmith read-through helpers for the conversation trace endpoints.

The trace endpoints are deliberately thin: they take a conversation id (and
optional turn number) and ask LangSmith for the matching root runs by
metadata. We do not persist the fetched payloads — LangSmith remains the
source of truth for trace data.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from langsmith import Client as LangSmithClient
from langsmith.utils import LangSmithError
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings

log = logging.getLogger(__name__)


class TraceSummary(BaseModel):
    """One LangSmith root-run summary, projected for our API responses."""

    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    name: str
    status: str | None = None
    turn_number: int | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    total_tokens: int | None = None
    error: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def require_langsmith_enabled(settings: Settings) -> LangSmithClient:
    """Return a configured LangSmith Client or raise 503.

    The trace endpoints are read-through to LangSmith, so we surface a
    503 (rather than 500) when LangSmith credentials aren't present —
    that's the right signal to a caller that traces are unavailable
    in this environment, not that the request itself was malformed.
    """
    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        raise HTTPException(status_code=503, detail="LangSmith tracing unavailable")
    return LangSmithClient(api_key=settings.langsmith_api_key)


def build_thread_filter(*, conversation_id: str, turn_number: int | None = None) -> str:
    """Return a LangSmith ``filter`` expression matching one conversation.

    LangSmith stores user metadata under the ``metadata`` field; the
    ``has(metadata, '<json>')`` predicate matches runs whose metadata is a
    superset of the supplied JSON object. We always match by ``thread_id``
    (== ``conversation_id`` in our convention) and optionally narrow by
    ``turn_number`` for the per-turn endpoint.
    """
    import json

    payload: dict[str, Any] = {"thread_id": conversation_id}
    if turn_number is not None:
        payload["turn_number"] = turn_number
    return f'and(eq(is_root, true), has(metadata, \'{json.dumps(payload)}\'))'


def list_thread_runs(
    *,
    client: LangSmithClient,
    settings: Settings,
    conversation_id: str,
    turn_number: int | None = None,
) -> list[TraceSummary]:
    """Query LangSmith for root runs matching this thread (and optional turn).

    Used by the conversation-level trace endpoint. The per-turn endpoint
    prefers :func:`read_run_by_id` because we persist the exact run UUID
    on the message row at turn time, which makes a direct lookup cheaper
    than filtering by metadata.
    """
    try:
        runs = list(
            client.list_runs(
                project_name=settings.langsmith_project,
                filter=build_thread_filter(
                    conversation_id=conversation_id,
                    turn_number=turn_number,
                ),
                is_root=True,
                limit=50,
            )
        )
    except LangSmithError as exc:
        log.exception("LangSmith trace query failed: %s", exc)
        raise HTTPException(status_code=503, detail="LangSmith trace query failed") from exc

    return [_to_summary(run) for run in runs]


def read_run_by_id(
    *,
    client: LangSmithClient,
    run_id: UUID,
) -> TraceSummary | None:
    """Direct LangSmith lookup by run UUID. Returns None when the run is missing.

    The chat flow persists the root run UUID on ``conversation_messages``
    before the @traceable decorators submit the run to LangSmith. The DB row
    therefore exists slightly before the LangSmith run is queryable; in
    practice ingestion is fast enough that by the time a client requests the
    turn trace the run is available, but if it isn't yet (or if tracing was
    disabled when the turn ran) we return ``None`` and the caller surfaces
    404 to the user.
    """
    try:
        run = client.read_run(run_id)
    except LangSmithError as exc:
        # Treat any not-found / forbidden / read failure as "no trace" rather
        # than 503; the conversation/turn exists locally so the missing
        # trace is the data shape we want to report.
        log.warning("LangSmith read_run(%s) failed: %s", run_id, exc)
        return None
    if run is None:
        return None
    return _to_summary(run)


def _to_summary(run: Any) -> TraceSummary:
    """Project a LangSmith Run into the smaller shape we expose externally."""
    metadata = (getattr(run, "extra", {}) or {}).get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    total_tokens = None
    usage = (getattr(run, "extra", {}) or {}).get("usage_metadata")
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if isinstance(total, int):
            total_tokens = total
    if total_tokens is None and getattr(run, "total_tokens", None) is not None:
        total_tokens = int(run.total_tokens)

    return TraceSummary(
        run_id=run.id,
        name=getattr(run, "name", "") or "",
        status=getattr(run, "status", None),
        turn_number=metadata.get("turn_number") if isinstance(metadata, dict) else None,
        start_time=getattr(run, "start_time", None),
        end_time=getattr(run, "end_time", None),
        total_tokens=total_tokens,
        error=getattr(run, "error", None),
        url=getattr(run, "url", None),
        metadata=metadata,
    )
