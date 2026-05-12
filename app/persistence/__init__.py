"""Phase 5 persistence layer: typed repositories for conversations and messages.

LangSmith owns trace storage; the local DB only carries product state
(conversations, visible messages, escalations).
"""

from app.persistence.conversations import (
    ConversationNotFoundError,
    ConversationRepository,
    ConversationSummary,
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
    "ConversationSummary",
    "MessageRecord",
    "MessageRepository",
    "MessageRole",
    "PriorTurn",
    "TurnNotFoundError",
]
