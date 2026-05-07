"""Phase 1 agent harness."""

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

AgentSource = Literal["agent", "faq", "compliance", "general", "human"]


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
    """HTTP-facing support agent response."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    response: str
    source: AgentSource
    matched_questions: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    verified: bool
    trace_id: str
    cost: CostSummary = Field(default_factory=CostSummary)


class PhaseOneAgent:
    """Deterministic placeholder agent used while graph orchestration is built."""

    def __init__(self, service_name: str = "SupportSmith") -> None:
        self.service_name = service_name

    async def respond(self, request: AgentRequest) -> AgentResponse:
        """Return a typed scaffold response for Phase 1."""
        response = (
            f"{self.service_name} Phase 1 harness is online. "
            "The LangGraph support workflow will replace this scaffold in Phase 3."
        )
        return AgentResponse(
            conversation_id=request.conversation_id,
            response=response,
            source="agent",
            tools_used=["phase_one_agent"],
            verified=True,
            trace_id=f"trace_{uuid4().hex}",
        )
