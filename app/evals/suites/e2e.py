"""End-to-end behavior suite: full agent turns over the live graph.

Each case is a conversation (one or more user turns). The suite drives the
:class:`Agent` Protocol, accumulating prior turns locally so multi-turn
context lands on the next call exactly the way ``app.api.chat_flow`` would
hand it to the agent. Deterministic gates and metrics come from the
``AgentResponse`` shape (source, verified, tools_used). When the runner is
invoked with ``--judge llm``, suite cases that list rubric names get
additional :class:`Metric` entries scored by :class:`LLMJudge`.

One :class:`ScoreRecord` is emitted per turn; multi-turn cases become
``<case_id>:turn<N>`` records. This keeps the rollup uniform across suites
and lets the runner show which specific turn failed in a multi-turn case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.harness import Agent, AgentRequest, AgentResponse, AgentSource
from app.evals.judge import LLMJudge
from app.evals.scoring import (
    Evidence,
    Metric,
    ScoreRecord,
    build_record,
    make_gate,
)
from app.persistence.messages import PriorTurn

# Suite-local metric weights. Deterministic checks are weighted lower than
# rubric scores when both are present so the LLM judge can pull a borderline
# response up or down; with --judge none the deterministic weights drive the
# pass/fail decision on their own.
_DET_WEIGHTS: dict[str, float] = {
    "source_match": 0.20,
    "verified": 0.10,
    "tool_budget": 0.10,
    "expected_tools_called": 0.20,
}

# Each rubric metric gets this weight when emitted; the scorer normalizes
# across the full metric list so the relative balance stays consistent.
_RUBRIC_WEIGHT: float = 0.20

# Fallback string that ``app.agent.runner._to_agent_response`` returns when
# the graph could not produce a candidate. Matching this triggers the
# ``no_response`` gate.
_FALLBACK_TEXT_PREFIX = "I couldn't put together a confident answer"


class ExpectedToolCall(BaseModel):
    """One tool invocation the case expects the agent to make."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)


class E2EExpectations(BaseModel):
    """Deterministic per-turn expectations.

    Each field is optional; suites only check the fields the case author
    set. ``trace_contains_nodes`` is recorded in evidence but not enforced
    as a metric (the agent target uses ``tools_used`` as its observable
    signal; richer trace assertions live in LangSmith-backed runs).
    """

    model_config = ConfigDict(extra="forbid")

    source: AgentSource | None = None
    verified: bool | None = None
    max_tool_calls: int | None = None
    trace_contains_nodes: list[str] = Field(default_factory=list)


class E2ETurn(BaseModel):
    """One user turn inside an e2e case."""

    model_config = ConfigDict(extra="forbid")

    user: str = Field(min_length=1)
    reference_answer: str | None = None
    reference_docs: list[str] = Field(default_factory=list)
    expected_tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    rubrics: list[str] = Field(default_factory=list)
    expectations: E2EExpectations = Field(default_factory=E2EExpectations)


class E2ECase(BaseModel):
    """One e2e case; one or more user turns sharing a conversation id."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    conversation_id: str = Field(min_length=1)
    turns: list[E2ETurn] = Field(min_length=1)


@dataclass(frozen=True)
class E2ESuiteDeps:
    """Per-run dependencies for the e2e suite."""

    agent: Agent
    judge: LLMJudge | None  # None when --judge none


async def score_case(case: E2ECase, deps: E2ESuiteDeps) -> list[ScoreRecord]:
    """Run every turn in ``case`` and return one :class:`ScoreRecord` per turn."""
    records: list[ScoreRecord] = []
    prior_turns: list[PriorTurn] = []

    for index, turn in enumerate(case.turns):
        turn_number = index + 1
        _stash_context(deps.agent, prior_turns=prior_turns, turn_number=turn_number)
        try:
            response = await deps.agent.respond(
                AgentRequest(conversation_id=case.conversation_id, message=turn.user)
            )
        except Exception as exc:  # pragma: no cover - exercised in live runs
            records.append(
                _crash_record(case=case, turn=turn, turn_number=turn_number, error=str(exc))
            )
            break

        record = await _score_turn(
            case=case,
            turn=turn,
            turn_number=turn_number,
            response=response,
            judge=deps.judge,
        )
        records.append(record)

        prior_turns.append(
            PriorTurn(
                turn_number=turn_number,
                user_message=turn.user,
                bot_reply=response.response,
                bot_role="agent",
            )
        )

    return records


async def _score_turn(
    *,
    case: E2ECase,
    turn: E2ETurn,
    turn_number: int,
    response: AgentResponse,
    judge: LLMJudge | None,
) -> ScoreRecord:
    metrics: list[Metric] = []
    gates = []

    if not response.response or response.response.startswith(_FALLBACK_TEXT_PREFIX):
        gates.append(
            make_gate(
                "no_response",
                applied=True,
                reason="agent returned empty or fallback text",
            )
        )

    expectations = turn.expectations

    if expectations.source is not None:
        matched = response.source == expectations.source
        metrics.append(
            Metric(
                name="source_match",
                score=1.0 if matched else 0.0,
                weight=_DET_WEIGHTS["source_match"],
                judge="deterministic",
                rationale=f"expected={expectations.source} actual={response.source}",
            )
        )

    if expectations.verified is not None:
        matched = response.verified == expectations.verified
        metrics.append(
            Metric(
                name="verified",
                score=1.0 if matched else 0.0,
                weight=_DET_WEIGHTS["verified"],
                judge="deterministic",
                rationale=f"expected={expectations.verified} actual={response.verified}",
            )
        )

    if expectations.max_tool_calls is not None:
        within = len(response.tools_used) <= expectations.max_tool_calls
        metrics.append(
            Metric(
                name="tool_budget",
                score=1.0 if within else 0.0,
                weight=_DET_WEIGHTS["tool_budget"],
                judge="deterministic",
                rationale=(
                    f"used={len(response.tools_used)} budget={expectations.max_tool_calls}"
                ),
            )
        )

    if turn.expected_tool_calls:
        expected_names = {call.name for call in turn.expected_tool_calls}
        present = expected_names.intersection(response.tools_used)
        coverage = len(present) / len(expected_names)
        metrics.append(
            Metric(
                name="expected_tools_called",
                score=coverage,
                weight=_DET_WEIGHTS["expected_tools_called"],
                judge="deterministic",
                rationale=(
                    f"expected={sorted(expected_names)} called={sorted(response.tools_used)}"
                ),
            )
        )

    if judge is not None and turn.rubrics:
        rubric_metrics = await _score_rubrics(
            judge=judge,
            turn=turn,
            response=response,
        )
        metrics.extend(rubric_metrics)

    evidence = Evidence(
        response_text=response.response,
        source=response.source,
        verified=response.verified,
        tool_calls=[{"name": name} for name in response.tools_used],
        turn_number=turn_number,
        trace_nodes=expectations.trace_contains_nodes,
        extra={
            "trace_id": response.trace_id,
            "case_tags": case.tags,
            "reference_answer": turn.reference_answer,
        },
    )

    return build_record(
        case_id=_record_id(case.id, turn_number, len(case.turns)),
        suite="e2e",
        target="agent",
        metrics=metrics,
        gates=gates,
        evidence=evidence,
    )


async def _score_rubrics(
    *,
    judge: LLMJudge,
    turn: E2ETurn,
    response: AgentResponse,
) -> list[Metric]:
    context = {
        "user_message": turn.user,
        "reference_answer": turn.reference_answer,
        "reference_docs": turn.reference_docs,
        "expected_tool_calls": [call.model_dump() for call in turn.expected_tool_calls],
        "agent_response": {
            "text": response.response,
            "source": response.source,
            "verified": response.verified,
            "tools_used": response.tools_used,
            "matched_questions": response.matched_questions,
        },
    }
    metrics: list[Metric] = []
    for rubric_name in turn.rubrics:
        verdict = await judge.score(rubric=rubric_name, context=context)
        metrics.append(
            Metric(
                name=rubric_name,
                score=verdict.score,
                weight=_RUBRIC_WEIGHT,
                judge="llm",
                rationale=verdict.rationale,
                confidence=verdict.confidence,
            )
        )
    return metrics


def _crash_record(
    *, case: E2ECase, turn: E2ETurn, turn_number: int, error: str
) -> ScoreRecord:
    return build_record(
        case_id=_record_id(case.id, turn_number, len(case.turns)),
        suite="e2e",
        target="agent",
        metrics=[],
        gates=[make_gate("no_response", applied=True, reason=f"agent raised: {error}")],
        evidence=Evidence(
            response_text=None,
            turn_number=turn_number,
            extra={
                "case_tags": case.tags,
                "user_message": turn.user,
                "error": error,
            },
        ),
    )


def _record_id(case_id: str, turn_number: int, total_turns: int) -> str:
    if total_turns <= 1:
        return case_id
    return f"{case_id}:turn{turn_number}"


def _stash_context(
    agent: Agent, *, prior_turns: list[PriorTurn], turn_number: int
) -> None:
    """Drop multi-turn context onto the agent the way chat_flow would.

    The :class:`Agent` Protocol does not surface this in its signature, but
    the production :class:`app.agent.runner.SupportAgent` reads
    ``_pending_prior_turns`` and ``_pending_turn_number`` off ``self`` before
    each call. We mirror that so the eval drives the same code path as
    ``/chat``.
    """
    agent._pending_prior_turns = list(prior_turns)  # type: ignore[attr-defined]
    agent._pending_turn_number = turn_number  # type: ignore[attr-defined]
