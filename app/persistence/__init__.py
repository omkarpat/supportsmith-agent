"""Phase 5 persistence layer: typed repositories for conversations and messages.

LangSmith owns trace storage; the local DB only carries product state
(conversations, visible messages, escalations).
"""

from app.persistence.conversations import (
    ConversationNotFoundError,
    ConversationRepository,
)
from app.persistence.messages import (
    MessageRecord,
    MessageRepository,
    MessageRole,
    PriorTurn,
    TurnNotFoundError,
)

__all__ = [
    "ConversationNotFoundError",
    "ConversationRepository",
    "MessageRecord",
    "MessageRepository",
    "MessageRole",
    "PriorTurn",
    "TurnNotFoundError",
]
