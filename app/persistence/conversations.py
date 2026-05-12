"""Conversation row helpers."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, ConversationMessage
from app.persistence.messages import MessageRole

PREVIEW_MAX_CHARS = 200


class ConversationNotFoundError(LookupError):
    """Raised when a caller-supplied conversation id does not exist."""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(f"Conversation not found: {conversation_id}")
        self.conversation_id = conversation_id


class ConversationSummary(BaseModel):
    """Sidebar summary for the chat-UI conversation list (Phase 8)."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    created_at: datetime
    updated_at: datetime
    last_turn_number: int | None = None
    last_message_preview: str | None = None
    last_role: MessageRole | None = None


class ConversationRepository:
    """Read/write helpers for the ``conversations`` table."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, conversation_id: str) -> Conversation | None:
        """Return the conversation by id, or ``None`` when missing."""
        result = await self.session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        return result.scalar_one_or_none()

    async def require(self, conversation_id: str) -> Conversation:
        """Return the conversation by id or raise :class:`ConversationNotFoundError`."""
        conversation = await self.get(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        return conversation

    async def create(self, conversation_id: str) -> Conversation:
        """Insert a new conversation row and return it."""
        conversation = Conversation(id=conversation_id)
        self.session.add(conversation)
        await self.session.flush()
        return conversation

    async def get_or_create(self, conversation_id: str) -> Conversation:
        """Return the conversation, creating it when missing."""
        existing = await self.get(conversation_id)
        if existing is not None:
            return existing
        return await self.create(conversation_id)

    async def list_recent(self, limit: int) -> list[ConversationSummary]:
        """Return the ``limit`` most recently updated conversations.

        Each row carries the latest visible message's role + preview so the
        Phase 8 sidebar can render rows without a second per-conversation
        fetch. The preview comes from ``conversation_messages``, which means
        rows with no messages yet (a freshly minted but never-sent
        conversation) get ``None`` for the preview fields.
        """
        if limit < 1:
            return []

        conversations = (
            (
                await self.session.execute(
                    select(Conversation)
                    .order_by(desc(Conversation.updated_at), desc(Conversation.id))
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        if not conversations:
            return []

        ids = [c.id for c in conversations]
        # PostgreSQL ``DISTINCT ON`` picks the row with the largest
        # (created_at, id) per conversation in one round trip.
        latest_messages = (
            (
                await self.session.execute(
                    select(ConversationMessage)
                    .where(ConversationMessage.conversation_id.in_(ids))
                    .distinct(ConversationMessage.conversation_id)
                    .order_by(
                        ConversationMessage.conversation_id,
                        desc(ConversationMessage.created_at),
                        desc(ConversationMessage.id),
                    )
                )
            )
            .scalars()
            .all()
        )
        by_id: dict[str, ConversationMessage] = {
            m.conversation_id: m for m in latest_messages
        }

        return [
            _to_summary(conversation, by_id.get(conversation.id))
            for conversation in conversations
        ]


def _to_summary(
    conversation: Conversation,
    last_message: ConversationMessage | None,
) -> ConversationSummary:
    if last_message is None:
        return ConversationSummary(
            conversation_id=conversation.id,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )
    return ConversationSummary(
        conversation_id=conversation.id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        last_turn_number=last_message.turn_number,
        last_message_preview=_preview(last_message.content),
        last_role=_validate_role(last_message.role),
    )


def _preview(text: str) -> str:
    stripped = text.strip()
    if len(stripped) <= PREVIEW_MAX_CHARS:
        return stripped
    return stripped[: PREVIEW_MAX_CHARS - 1].rstrip() + "…"


def _validate_role(value: str) -> MessageRole:
    if value not in {"user", "agent", "compliance"}:
        raise ValueError(f"Unexpected role value persisted: {value!r}")
    return value  # type: ignore[return-value]
