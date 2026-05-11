"""Conversation endpoints (Phase 5, extended in Phase 8).

Write surface (the *only* place that mutates conversation state):

  - ``POST /chat``                       — mint or resume by body id
  - ``POST /chat/{conversation_id}``     — resume by path id (404 if unknown)

Read surface:

  - ``GET /conversations``                          — sidebar list (Phase 8)
  - ``GET /conversations/{id}/messages``
  - ``GET /conversations/{id}/turns/{turn_number}/messages``
  - ``GET /conversations/{id}/trace``               — LangSmith read-through
  - ``GET /conversations/{id}/turns/{turn_number}/trace`` — LangSmith read-through
"""

from __future__ import annotations

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.harness import Agent, AgentResponse
from app.api.chat_flow import process_chat_turn
from app.api.dependencies import get_agent, get_db_session
from app.api.langsmith_traces import (
    TraceSummary,
    list_thread_runs,
    read_run_by_id,
    require_langsmith_enabled,
)
from app.core.config import Settings
from app.persistence import (
    ConversationNotFoundError,
    ConversationRepository,
    ConversationSummary,
    MessageRecord,
    MessageRepository,
)

CONVERSATIONS_LIST_DEFAULT_LIMIT = 50
CONVERSATIONS_LIST_MAX_LIMIT = 100

router = APIRouter(tags=["conversations"])


class ChatRequest(BaseModel):
    """Inbound chat payload. ``conversation_id`` resumes an existing thread."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    conversation_id: str | None = Field(default=None, min_length=1)


class MessagesResponse(BaseModel):
    """List of persisted messages for a conversation or one turn."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    messages: list[MessageRecord]


class ConversationsResponse(BaseModel):
    """Sidebar payload: recent conversations with last-message previews."""

    model_config = ConfigDict(extra="forbid")

    conversations: list[ConversationSummary]


class TraceResponse(BaseModel):
    """One turn's LangSmith root-run summary."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    turn_number: int
    trace: TraceSummary


class ConversationTraceResponse(BaseModel):
    """LangSmith root runs for a conversation, grouped by thread_id."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    traces: list[TraceSummary]


def _settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


async def _run_chat_turn(
    request: Request,
    agent: Agent,
    *,
    conversation_id: str | None,
    message: str,
) -> AgentResponse:
    """Shared core for the two write endpoints."""
    session_factory = request.app.state.database.session_factory
    if session_factory is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    settings = _settings(request)

    if conversation_id is None:
        # Mint a new conversation in its own short transaction so the create
        # is durable even if the rest of the turn fails downstream.
        async with session_factory() as session, session.begin():
            conversation_id = uuid4().hex
            await ConversationRepository(session).create(conversation_id)
    else:
        async with session_factory() as session:
            try:
                await ConversationRepository(session).require(conversation_id)
            except ConversationNotFoundError as exc:
                raise HTTPException(
                    status_code=404, detail="Conversation not found"
                ) from exc

    outcome = await process_chat_turn(
        session_factory=session_factory,
        agent=agent,
        conversation_id=conversation_id,
        user_message=message,
        context_user_turns=settings.context_user_turns,
    )
    response = outcome.response
    return response.model_copy(
        update={"conversation_id": conversation_id, "turn_number": outcome.turn_number}
    )


@router.post("/chat", response_model=AgentResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    agent: Annotated[Agent, Depends(get_agent)],
) -> AgentResponse:
    """Send a chat message. Mints a new conversation when no id is supplied."""
    return await _run_chat_turn(
        request=request,
        agent=agent,
        conversation_id=payload.conversation_id,
        message=payload.message,
    )


@router.post("/chat/{conversation_id}", response_model=AgentResponse)
async def chat_resume(
    conversation_id: Annotated[str, Path(min_length=1)],
    payload: ChatRequest,
    request: Request,
    agent: Annotated[Agent, Depends(get_agent)],
) -> AgentResponse:
    """Resume an existing conversation by path id. 404 when the id is unknown."""
    if payload.conversation_id and payload.conversation_id != conversation_id:
        raise HTTPException(
            status_code=400,
            detail="Body conversation_id does not match path conversation_id",
        )
    return await _run_chat_turn(
        request=request,
        agent=agent,
        conversation_id=conversation_id,
        message=payload.message,
    )


# --- read endpoints -----------------------------------------------------------


async def _require_conversation(session: AsyncSession, conversation_id: str) -> None:
    try:
        await ConversationRepository(session).require(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc


@router.get("/conversations", response_model=ConversationsResponse)
async def list_conversations(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=CONVERSATIONS_LIST_MAX_LIMIT,
            description="Maximum conversations to return (1-100).",
        ),
    ] = CONVERSATIONS_LIST_DEFAULT_LIMIT,
) -> ConversationsResponse:
    """Phase 8 sidebar feed: most recently updated conversations first."""
    summaries = await ConversationRepository(session).list_recent(limit=limit)
    return ConversationsResponse(conversations=summaries)


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=MessagesResponse,
)
async def list_messages(
    conversation_id: Annotated[str, Path(min_length=1)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> MessagesResponse:
    """Return every persisted message for a conversation in chronological order."""
    await _require_conversation(session, conversation_id)
    messages = await MessageRepository(session).list_for_conversation(conversation_id)
    return MessagesResponse(conversation_id=conversation_id, messages=messages)


@router.get(
    "/conversations/{conversation_id}/turns/{turn_number}/messages",
    response_model=MessagesResponse,
)
async def list_turn_messages(
    conversation_id: Annotated[str, Path(min_length=1)],
    turn_number: Annotated[int, Path(ge=1)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> MessagesResponse:
    """Return only the persisted messages for one turn. 404 when the turn is missing."""
    await _require_conversation(session, conversation_id)
    messages = await MessageRepository(session).list_for_turn(
        conversation_id=conversation_id,
        turn_number=turn_number,
    )
    if not messages:
        raise HTTPException(status_code=404, detail="Turn not found")
    return MessagesResponse(conversation_id=conversation_id, messages=messages)


@router.get(
    "/conversations/{conversation_id}/trace",
    response_model=ConversationTraceResponse,
)
async def conversation_trace(
    conversation_id: Annotated[str, Path(min_length=1)],
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ConversationTraceResponse:
    """Read-through to LangSmith: list root runs for this conversation thread.

    Returns ``503`` when LangSmith is not configured. Returns ``404`` when
    the local conversation exists but no LangSmith runs are tagged with this
    thread id.
    """
    await _require_conversation(session, conversation_id)
    settings = _settings(request)
    client = require_langsmith_enabled(settings)
    summaries = list_thread_runs(
        client=client,
        settings=settings,
        conversation_id=conversation_id,
    )
    if not summaries:
        raise HTTPException(status_code=404, detail="Trace not found")
    return ConversationTraceResponse(conversation_id=conversation_id, traces=summaries)


@router.get(
    "/conversations/{conversation_id}/turns/{turn_number}/trace",
    response_model=TraceResponse,
)
async def turn_trace(
    conversation_id: Annotated[str, Path(min_length=1)],
    turn_number: Annotated[int, Path(ge=1)],
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> TraceResponse:
    """Direct LangSmith lookup by the persisted root-run UUID.

    The chat flow persists the LangSmith root run UUID on the agent /
    compliance message row at turn time, so the per-turn trace endpoint
    can do an O(1) ``read_run`` call instead of the metadata filter scan
    used by the conversation-level endpoint.
    """
    await _require_conversation(session, conversation_id)
    # Check the LangSmith availability *before* checking for a persisted run
    # id so a tracing-off environment surfaces a clear 503 rather than
    # collapsing into a per-turn 404 the caller has to interpret.
    settings = _settings(request)
    client = require_langsmith_enabled(settings)

    local_messages = await MessageRepository(session).list_for_turn(
        conversation_id=conversation_id,
        turn_number=turn_number,
    )
    if not local_messages:
        raise HTTPException(status_code=404, detail="Turn not found")

    bot_message = next(
        (
            msg
            for msg in local_messages
            if msg.role in {"agent", "compliance"} and msg.langsmith_run_id is not None
        ),
        None,
    )
    if bot_message is None or bot_message.langsmith_run_id is None:
        # The turn predates LangSmith tracing, the agent message was never
        # persisted, or this specific turn ran with tracing disabled.
        raise HTTPException(status_code=404, detail="Trace not found")

    summary = read_run_by_id(client=client, run_id=bot_message.langsmith_run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return TraceResponse(
        conversation_id=conversation_id,
        turn_number=turn_number,
        trace=summary,
    )


__all__ = ["router"]
