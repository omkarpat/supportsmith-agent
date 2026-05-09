"""Public agent contracts: request/response types and the Agent Protocol."""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

AgentSource = Literal[
    "agent",
    "faq",
    "general",
    "clarify",
    "escalate",
    "refuse",
    "compliance",
]


class CostSummary(BaseModel):
    """Estimated cost and token usage for a response."""

    model_config = ConfigDict(extra="forbid")

    total_tokens: int = 0
    estimated_usd: float = 0.0


class AgentRequest(BaseModel):
    """Provider-neutral request into a support agent."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    message: str = Field(min_length=1)


class AgentResponse(BaseModel):
    """HTTP-facing support agent response.

    ``turn_number`` is set by the persistence flow (Phase 5+); for unit-style
    callers that build an ``AgentResponse`` directly without going through
    persistence, ``0`` indicates "not assigned a persisted turn number".
    """

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    turn_number: int = 0
    response: str
    source: AgentSource
    matched_questions: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    verified: bool
    trace_id: str
    cost: CostSummary = Field(default_factory=CostSummary)


class Agent(Protocol):
    """Anything that can respond to a single agent turn.

    The graph-driven :class:`app.agent.runner.SupportAgent` is the production
    implementation; tests and eval drivers can satisfy this Protocol with a
    scripted or stubbed object.
    """

    async def respond(self, request: AgentRequest) -> AgentResponse: ...
