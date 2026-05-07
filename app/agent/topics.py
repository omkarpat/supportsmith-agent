"""Support-topic examples used by the clarification flow.

These are *examples surfaced in the user-facing clarification message* (not
prompt few-shots). They live in code rather than in the system prompt so the
agent can show specific, current topics regardless of which model variant or
prompt revision is in use.
"""

from typing import Final

SUPPORT_TOPIC_EXAMPLES: Final[tuple[str, ...]] = (
    "password reset and 2FA",
    "billing, refunds, and subscriptions",
    "account security and locked accounts",
    "profile and settings",
    "data export and account deletion",
    "app crashes or site slowness",
)


def render_topic_examples() -> str:
    """Return a short bulleted list of topic examples for clarification prose."""
    return "\n".join(f"- {topic}" for topic in SUPPORT_TOPIC_EXAMPLES)
