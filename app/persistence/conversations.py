"""Conversation row helpers."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation


class ConversationNotFoundError(LookupError):
    """Raised when a caller-supplied conversation id does not exist."""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(f"Conversation not found: {conversation_id}")
        self.conversation_id = conversation_id


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
