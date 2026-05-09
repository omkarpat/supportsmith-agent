"""Conversation message helpers (visible transcript only)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ConversationMessage

MessageRole = Literal["user", "agent", "compliance"]


class TurnNotFoundError(LookupError):
    """Raised when a caller asks for a turn that does not exist on a conversation."""

    def __init__(self, *, conversation_id: str, turn_number: int) -> None:
        super().__init__(
            f"Turn not found: conversation_id={conversation_id} turn_number={turn_number}"
        )
        self.conversation_id = conversation_id
        self.turn_number = turn_number


class MessageRecord(BaseModel):
    """One persisted visible message, projected for API responses."""

    model_config = ConfigDict(extra="forbid")

    message_id: int
    conversation_id: str
    turn_number: int
    role: MessageRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    langsmith_run_id: UUID | None = None
    created_at: datetime


class PriorTurn(BaseModel):
    """One prior user turn + its visible bot reply, used for context loading."""

    model_config = ConfigDict(extra="forbid")

    turn_number: int
    user_message: str
    bot_reply: str | None = None
    bot_role: Literal["agent", "compliance"] | None = None
    bot_metadata: dict[str, Any] = Field(default_factory=dict)


class MessageRepository:
    """Read/write helpers for ``conversation_messages``."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def append(
        self,
        *,
        conversation_id: str,
        turn_number: int,
        role: MessageRole,
        content: str,
        metadata: dict[str, Any] | None = None,
        langsmith_run_id: UUID | None = None,
    ) -> ConversationMessage:
        """Insert one visible message and return the row."""
        message = ConversationMessage(
            conversation_id=conversation_id,
            turn_number=turn_number,
            role=role,
            content=content,
            metadata_=metadata or {},
            langsmith_run_id=langsmith_run_id,
        )
        self.session.add(message)
        await self.session.flush()
        return message

    async def list_for_conversation(self, conversation_id: str) -> list[MessageRecord]:
        """Return every persisted message for a conversation in chronological order."""
        result = await self.session.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at, ConversationMessage.id)
        )
        return [_to_record(row) for row in result.scalars().all()]

    async def list_for_turn(
        self, *, conversation_id: str, turn_number: int
    ) -> list[MessageRecord]:
        """Return the messages for one turn; empty list when the turn does not exist."""
        result = await self.session.execute(
            select(ConversationMessage)
            .where(
                ConversationMessage.conversation_id == conversation_id,
                ConversationMessage.turn_number == turn_number,
            )
            .order_by(ConversationMessage.created_at, ConversationMessage.id)
        )
        return [_to_record(row) for row in result.scalars().all()]

    async def next_turn_number(self, conversation_id: str) -> int:
        """Return the next user-turn number for a conversation (1-indexed)."""
        result = await self.session.execute(
            select(func.coalesce(func.max(ConversationMessage.turn_number), 0)).where(
                ConversationMessage.conversation_id == conversation_id
            )
        )
        current = result.scalar_one()
        return int(current) + 1

    async def load_prior_turns(
        self, *, conversation_id: str, limit: int
    ) -> list[PriorTurn]:
        """Return the last ``limit`` user turns + their bot replies, oldest-first.

        Tool calls and clarifications belong on the trace blob, not the visible
        transcript, so the ``conversation_messages`` table only carries
        ``user`` / ``agent`` / ``compliance`` roles. The result is fed into the
        next turn's ``GraphState.prior_user_turns`` so the planner / synthesizer
        see real conversation context.
        """
        if limit < 1:
            return []

        result = await self.session.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.turn_number, ConversationMessage.created_at)
        )
        all_messages = list(result.scalars().all())

        by_turn: dict[int, dict[str, ConversationMessage]] = {}
        for message in all_messages:
            by_turn.setdefault(message.turn_number, {})[message.role] = message

        ordered_turns = sorted(by_turn.keys())
        windowed = ordered_turns[-limit:]

        prior: list[PriorTurn] = []
        for turn_number in windowed:
            roles = by_turn[turn_number]
            user_msg = roles.get("user")
            if user_msg is None:
                # Defensive: every turn should have a user message; skip if not.
                continue
            bot_msg = roles.get("agent") or roles.get("compliance")
            prior.append(
                PriorTurn(
                    turn_number=turn_number,
                    user_message=user_msg.content,
                    bot_reply=bot_msg.content if bot_msg else None,
                    bot_role=("agent" if bot_msg and bot_msg.role == "agent" else "compliance")
                    if bot_msg
                    else None,
                    bot_metadata=bot_msg.metadata_ if bot_msg else {},
                )
            )
        return prior


def _to_record(row: ConversationMessage) -> MessageRecord:
    return MessageRecord(
        message_id=row.id,
        conversation_id=row.conversation_id,
        turn_number=row.turn_number,
        role=_validate_role(row.role),
        content=row.content,
        metadata=row.metadata_,
        langsmith_run_id=row.langsmith_run_id,
        created_at=row.created_at,
    )


def _validate_role(value: str) -> MessageRole:
    if value not in {"user", "agent", "compliance"}:
        raise ValueError(f"Unexpected role value persisted: {value!r}")
    return value  # type: ignore[return-value]
