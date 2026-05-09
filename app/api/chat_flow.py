"""Phase 5 chat-turn orchestration.

The flow per turn:

  1. Open a short DB transaction. Create the conversation (when the caller
     supplies no id; the route layer handles the create) or require it (when
     one is supplied). Compute the next ``turn_number`` and persist the user
     message. Load the most recent ``context_user_turns`` prior turns
     (oldest-first) for context. Commit and close the transaction.
  2. Outside the transaction, generate a UUID for the LangSmith root run of
     this turn and call ``agent.respond(...)`` with the loaded context. A
     retry+fallback wrapper handles transient OpenAI / network failures; on
     a second failure the user gets a deterministic fallback message and the
     persisted message metadata records ``status="failed_recovered"`` (or
     ``failed_unhandled`` when even the fallback path failed).
  3. Open a second short transaction to persist the visible agent / compliance
     response, including ``langsmith_run_id`` so operators can jump from a
     DB row directly to the matching LangSmith trace.

Trace storage lives in LangSmith — there is no local trace table. The two
``/conversations/...trace`` endpoints fetch traces dynamically from
LangSmith using the conversation id as the thread id.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agent.harness import Agent, AgentRequest, AgentResponse, AgentSource, CostSummary
from app.llm.openai import LLMProviderError
from app.persistence import (
    ConversationRepository,
    MessageRepository,
    MessageRole,
    PriorTurn,
)

log = logging.getLogger(__name__)

TurnStatus = Literal["completed", "failed_recovered", "failed_unhandled"]

FALLBACK_RESPONSE = (
    "I'm sorry, something went wrong while handling that request. "
    "I've logged the issue for review. Please try again or contact support if it continues."
)

# Exceptions that trigger one retry. Hard application errors (validation,
# missing resources, programmer errors) deliberately do not match — those
# should bubble to the caller as proper HTTP errors.
_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    LLMProviderError,
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    asyncio.TimeoutError,
    ConnectionError,
)


@dataclass(frozen=True)
class ChatFlowOutcome:
    """Result of one chat turn after persistence."""

    response: AgentResponse
    conversation_id: str
    turn_number: int
    status: TurnStatus
    langsmith_run_id: UUID | None


async def process_chat_turn(
    *,
    session_factory: async_sessionmaker[Any],
    agent: Agent,
    conversation_id: str,
    user_message: str,
    context_user_turns: int,
) -> ChatFlowOutcome:
    """Run one user-turn through persistence + the agent.

    Raises :class:`ConversationNotFoundError` when the supplied
    ``conversation_id`` does not exist; the route handler turns that into
    ``404``. All other agent failures route through the retry+fallback path
    inside ``_respond_with_retry`` and return a typed outcome.
    """
    async with session_factory() as session, session.begin():
        await ConversationRepository(session).require(conversation_id)
        messages_repo = MessageRepository(session)
        turn_number = await messages_repo.next_turn_number(conversation_id)
        await messages_repo.append(
            conversation_id=conversation_id,
            turn_number=turn_number,
            role="user",
            content=user_message,
            metadata={},
        )
        # Load the most recent N user turns *including* the one we just
        # persisted, then drop it so the agent only sees prior history.
        full_history = await messages_repo.load_prior_turns(
            conversation_id=conversation_id,
            limit=context_user_turns + 1,
        )
        prior_turns = [t for t in full_history if t.turn_number < turn_number]

    response, status = await _respond_with_retry(
        agent,
        AgentRequest(conversation_id=conversation_id, message=user_message),
        prior_turns=prior_turns,
        turn_number=turn_number,
    )

    # The runner captures the LangSmith-assigned root run UUID on
    # ``_captured_langsmith_run_id`` while inside the @traceable scope.
    # When LangSmith tracing is disabled this stays ``None`` and we
    # persist NULL on the message row.
    langsmith_run_id: UUID | None = getattr(agent, "_captured_langsmith_run_id", None)

    visible_role = _visible_role_for(response.source)
    visible_metadata: dict[str, Any] = {
        "source": response.source,
        "tools_used": response.tools_used,
        "matched_questions": response.matched_questions,
        "verified": response.verified,
        "status": status,
        "trace_id": response.trace_id,
        "total_tokens": response.cost.total_tokens,
    }

    async with session_factory() as session, session.begin():
        await MessageRepository(session).append(
            conversation_id=conversation_id,
            turn_number=turn_number,
            role=visible_role,
            content=response.response,
            metadata=visible_metadata,
            langsmith_run_id=langsmith_run_id,
        )

    return ChatFlowOutcome(
        response=response,
        conversation_id=conversation_id,
        turn_number=turn_number,
        status=status,
        langsmith_run_id=langsmith_run_id,
    )


async def _respond_with_retry(
    agent: Agent,
    request: AgentRequest,
    *,
    prior_turns: list[PriorTurn],
    turn_number: int,
) -> tuple[AgentResponse, TurnStatus]:
    """Run ``agent.respond`` with one retry on transient provider failures.

    Phase 5 plumbs prior_turns + turn_number onto the agent instance just
    before the call so the public ``respond`` Protocol stays single-method.
    ``SupportAgent.respond`` reads these to build the initial ``GraphState``
    and exposes ``_captured_langsmith_run_id`` once the @traceable boundary
    has assigned the root run UUID.
    """
    agent._pending_prior_turns = prior_turns  # type: ignore[attr-defined]
    agent._pending_turn_number = turn_number  # type: ignore[attr-defined]
    try:
        try:
            return await agent.respond(request), "completed"
        except _TRANSIENT_EXCEPTIONS as exc:
            log.warning("Transient agent failure on first try; retrying once: %s", exc)
            try:
                return await agent.respond(request), "completed"
            except _TRANSIENT_EXCEPTIONS as retry_exc:
                log.warning("Retry also failed: %s", retry_exc)
                return _fallback_response(request), "failed_recovered"
            except Exception as retry_exc:  # noqa: BLE001
                log.exception("Non-transient failure during retry: %s", retry_exc)
                return _fallback_response(request), "failed_unhandled"
        except Exception as exc:  # noqa: BLE001
            log.exception("Non-transient agent failure: %s", exc)
            return _fallback_response(request), "failed_unhandled"
    finally:
        for attr in ("_pending_prior_turns", "_pending_turn_number"):
            with contextlib.suppress(AttributeError):
                delattr(agent, attr)


def _fallback_response(request: AgentRequest) -> AgentResponse:
    """Build a deterministic fallback response when the agent itself fails."""
    return AgentResponse(
        conversation_id=request.conversation_id,
        response=FALLBACK_RESPONSE,
        source="agent",
        matched_questions=[],
        tools_used=[],
        verified=False,
        trace_id=f"trace_fallback_{int(datetime.now(UTC).timestamp())}",
        cost=CostSummary(),
    )


def _visible_role_for(source: AgentSource) -> MessageRole:
    """Map the response's source onto the persisted message role.

    Compliance refusals/overrides are stamped ``compliance``; every other
    user-facing response is ``agent``. The ``user`` role is reserved for
    inbound messages.
    """
    if source == "compliance":
        return "compliance"
    return "agent"
