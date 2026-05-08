"""LangGraph state schema, trace events, and structured plan shapes.

The state object travels through every node. Nodes append to ``trace_events``
but never mutate prior entries, so the trace is always a faithful chronological
record of what the graph did during the turn.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent.compliance import ComplianceDecision
from app.agent.tools import ToolName
from app.agent.verifier import VerifierOutput

PlanIntent = Literal[
    "use_tool",
    "clarify",
    "synthesize_now",
    "escalate",
    "refuse",
]

NodeName = Literal[
    "load_context",
    "precheck",
    "plan",
    "execute_tool",
    "observe",
    "synthesize",
    "verify",
    "postcheck",
    "finalize",
]


class Plan(BaseModel):
    """Structured plan emitted by the planner node.

    The planner LLM returns a JSON object matching this schema; the dispatch
    edge inspects ``intent`` (and ``tool_name`` when the intent is
    ``use_tool``) to choose the next node. ``rationale`` is a *short*
    operational note, not chain-of-thought, and is included in the trace so
    reviewers can audit decisions.
    """

    model_config = ConfigDict(extra="forbid")

    intent: PlanIntent
    tool_name: ToolName | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class ToolObservation(BaseModel):
    """Result of one tool invocation, captured for both state and trace."""

    model_config = ConfigDict(extra="forbid")

    tool_name: ToolName
    arguments: dict[str, Any]
    output: dict[str, Any]
    succeeded: bool
    error: str | None = None


class TraceTokenUsage(BaseModel):
    """Per-event token usage; mirrors :class:`app.llm.client.TokenUsage`."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class TraceEvent(BaseModel):
    """One per-node event recorded during a graph turn.

    Trace events are append-only. Sensitive prompt content is summarized into
    ``rationale`` so the persisted trace stays auditable without echoing the
    full LLM context.
    """

    model_config = ConfigDict(extra="forbid")

    node: NodeName
    started_at: datetime
    finished_at: datetime
    latency_ms: int
    model: str | None = None
    rationale: str = ""
    tokens: TraceTokenUsage | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class CandidateAnswer(BaseModel):
    """Tentative final answer produced by the synthesize node.

    ``compliance`` is a distinct source from ``refuse``: planner-driven refuses
    use ``refuse`` (cheap in-loop gatekeeper); compliance precheck/postcheck
    refusals use ``compliance`` so reviewers can distinguish the two paths.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    citations: list[str] = Field(default_factory=list)
    source: Literal[
        "faq",
        "general",
        "clarify",
        "escalate",
        "refuse",
        "compliance",
        "agent",
    ] = "agent"


class GraphState(BaseModel):
    """Single Pydantic model carried by every node.

    LangGraph supports both TypedDict and Pydantic state schemas; we use
    Pydantic so node return values are validated and so the trace events list
    is type-checked.
    """

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    turn_id: str
    user_message: str
    plan: Plan | None = None
    observations: list[ToolObservation] = Field(default_factory=list)
    tool_iterations: int = 0
    candidate_answer: CandidateAnswer | None = None
    compliance_precheck: ComplianceDecision | None = None
    compliance_postcheck: ComplianceDecision | None = None
    verification: VerifierOutput | None = None
    repair_attempts: int = 0
    trace_events: list[TraceEvent] = Field(default_factory=list)
    halted_reason: str | None = None
