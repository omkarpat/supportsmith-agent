"""LangGraph node implementations for the SupportSmith agent.

Each node is a coroutine ``(GraphState) -> dict``. Node functions take a
prebuilt :class:`NodeContext` carrying the LLM client, tool registry, and
model selection so the graph wiring can stay focused on edges.

Trace policy: every node appends exactly one :class:`TraceEvent` describing
what it did. Sensitive prompt text is summarized into the ``rationale`` field;
we never echo full LLM messages into the trace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from app.agent.compliance import ComplianceAgent, is_hard_block
from app.agent.policy import CANONICAL_REFUSAL
from app.agent.state import (
    CandidateAnswer,
    GraphState,
    NodeName,
    Plan,
    ToolObservation,
    TraceEvent,
    TraceTokenUsage,
)
from app.agent.tools import (
    TOOL_NAMES,
    AskUserClarificationOutput,
    EscalateToHumanOutput,
    GeneralKnowledgeLookupOutput,
    GetFAQByCategoryOutput,
    RefuseOutput,
    SearchFAQOutput,
    ToolRegistry,
)
from app.agent.verifier import VerifierOutput
from app.llm.client import (
    ChatMessage,
    ChatRequest,
    ChatResponseSchema,
    LLMClient,
    ReasoningEffort,
)
from app.llm.openai import LLMProviderError
from app.prompts import load_prompt


@dataclass(frozen=True)
class NodeContext:
    """Per-graph-instance dependencies. Built once at graph compile time."""

    llm: LLMClient
    tools: ToolRegistry
    compliance: ComplianceAgent
    chat_model: str
    reasoning_model: str
    planner_reasoning_effort: ReasoningEffort
    planner_max_completion_tokens: int
    synthesis_max_completion_tokens: int
    verifier_model: str
    verifier_reasoning_effort: ReasoningEffort | None
    verifier_max_completion_tokens: int
    max_tool_iterations: int
    max_repair_attempts: int = 1


# --- planner schema (passed to OpenAI as response_format=json_schema) ---------


_PLANNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["use_tool", "clarify", "synthesize_now", "escalate", "refuse"],
        },
        "tool_name": {
            "anyOf": [
                {"type": "string", "enum": list(TOOL_NAMES)},
                {"type": "null"},
            ],
        },
        "arguments": {
            "type": "object",
            "additionalProperties": True,
        },
        "rationale": {"type": "string"},
    },
    "required": ["intent", "tool_name", "arguments", "rationale"],
}

# System prompts for the planner and synthesizer live in YAML under prompts/.
# Loaded lazily so a typo in a prompt file fails near where it's used, not at
# module import time.

_SYNTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "text": {"type": "string"},
        "cited_titles": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["text", "cited_titles"],
}


class _SynthesisOutput(BaseModel):
    """Parsed JSON from the synthesizer response."""

    model_config = ConfigDict(extra="forbid")

    text: str
    cited_titles: list[str] = []


_VERIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "addresses_request": {"type": "boolean"},
        "grounding": {
            "type": "string",
            "enum": [
                "faq_grounded",
                "general_marked",
                "clarification",
                "escalation",
                "refusal",
                "unsupported",
            ],
        },
        "leakage_detected": {"type": "boolean"},
        "safe_source_label": {"type": "boolean"},
        "retry_recommendation": {
            "type": "string",
            "enum": ["accept", "repair", "escalate", "refuse"],
        },
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": [
        "addresses_request",
        "grounding",
        "leakage_detected",
        "safe_source_label",
        "retry_recommendation",
        "reason",
        "confidence",
    ],
}


# --- helpers ------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC)


def _make_event(
    *,
    node: NodeName,
    started_at: datetime,
    rationale: str = "",
    model: str | None = None,
    tokens: TraceTokenUsage | None = None,
    payload: dict[str, Any] | None = None,
) -> TraceEvent:
    finished_at = _now()
    latency_ms = int((finished_at - started_at).total_seconds() * 1000)
    return TraceEvent(
        node=node,
        started_at=started_at,
        finished_at=finished_at,
        latency_ms=latency_ms,
        model=model,
        rationale=rationale,
        tokens=tokens,
        payload=payload or {},
    )


def _output_to_dict(output: object) -> dict[str, Any]:
    if isinstance(
        output,
        (
            SearchFAQOutput,
            GetFAQByCategoryOutput,
            AskUserClarificationOutput,
            GeneralKnowledgeLookupOutput,
            EscalateToHumanOutput,
            RefuseOutput,
        ),
    ):
        return output.model_dump(mode="json")
    raise TypeError(f"Unhandled tool output type: {type(output).__name__}")


# --- nodes --------------------------------------------------------------------


async def load_context(state: GraphState, _ctx: NodeContext) -> dict[str, Any]:
    """Phase 3 stub: durable history loading lands in Phase 5.

    For now we just record a trace event so reviewers can see the entry point.
    """
    started = _now()
    event = _make_event(
        node="load_context",
        started_at=started,
        rationale="entered turn; durable history loading is deferred to Phase 5",
    )
    return {"trace_events": [*state.trace_events, event]}


async def plan(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
    """Ask the reasoning model for a structured next-step plan."""
    started = _now()
    request = ChatRequest(
        model=ctx.reasoning_model,
        reasoning_effort=ctx.planner_reasoning_effort,
        max_completion_tokens=ctx.planner_max_completion_tokens,
        response_schema=ChatResponseSchema(
            name="support_plan",
            schema_definition=_PLANNER_SCHEMA,
            strict=False,
        ),
        messages=[
            ChatMessage(role="system", content=load_prompt("planner").system),
            ChatMessage(role="user", content=_render_plan_prompt(state)),
        ],
    )
    response = await ctx.llm.complete(request)
    plan_obj = _parse_plan(response.content)

    event = _make_event(
        node="plan",
        started_at=started,
        model=response.model,
        tokens=TraceTokenUsage(**response.usage.model_dump()),
        rationale=plan_obj.rationale,
        payload={"intent": plan_obj.intent, "tool_name": plan_obj.tool_name},
    )
    return {
        "plan": plan_obj,
        "trace_events": [*state.trace_events, event],
    }


async def execute_tool(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
    """Validate the planned arguments and run the chosen tool."""
    started = _now()
    if state.plan is None or state.plan.tool_name is None:
        raise RuntimeError("execute_tool called without a tool_name in state.plan")

    tool_name = state.plan.tool_name
    arguments = dict(state.plan.arguments)
    if tool_name == "escalate_to_human" and "transcript" not in arguments:
        arguments["transcript"] = [
            {"role": "user", "content": state.user_message},
        ]

    succeeded = True
    error: str | None = None
    output_payload: dict[str, Any] = {}
    try:
        output = await ctx.tools.run(tool_name, arguments)
        output_payload = _output_to_dict(output)
    except LLMProviderError as exc:
        succeeded = False
        error = f"provider_error: {exc}"
    except (ValueError, RuntimeError) as exc:
        succeeded = False
        error = f"tool_error: {exc}"

    observation = ToolObservation(
        tool_name=tool_name,
        arguments=arguments,
        output=output_payload,
        succeeded=succeeded,
        error=error,
    )

    event = _make_event(
        node="execute_tool",
        started_at=started,
        rationale=(
            f"ran {tool_name}"
            if succeeded
            else f"{tool_name} failed: {error}"
        ),
        payload={"tool_name": tool_name, "succeeded": succeeded},
    )
    return {
        "observations": [*state.observations, observation],
        "tool_iterations": state.tool_iterations + 1,
        "trace_events": [*state.trace_events, event],
    }


async def observe(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
    """Decide whether the most recent observation is enough to synthesize.

    The plan node already routes intent broadly; observe is a lightweight check
    that maps tool outcomes into the next-loop signal. We do not call the LLM
    here in Phase 3 — keeping this deterministic keeps the loop predictable and
    is consistent with the doc's "small, readable graph" guardrail. Real LLM
    re-planning happens by routing back to the plan node.
    """
    started = _now()
    if not state.observations:
        raise RuntimeError("observe called with no observations recorded")

    last = state.observations[-1]
    rationale: str
    payload: dict[str, Any] = {"tool_name": last.tool_name, "succeeded": last.succeeded}

    if last.tool_name in {"ask_user_clarification", "escalate_to_human", "refuse"}:
        rationale = f"terminal observation from {last.tool_name}; route to synthesize"
    elif not last.succeeded:
        rationale = "tool failed; route back to plan to recover"
    elif last.tool_name == "search_faq":
        results = last.output.get("results", [])
        if results and float(results[0].get("score", 0.0)) >= 0.4:
            rationale = "search_faq returned a confident match; ready to synthesize"
        else:
            rationale = "search_faq low-confidence; route back to plan for fallback"
    else:
        rationale = f"{last.tool_name} produced output; ready to synthesize"

    payload["next_signal"] = rationale
    event = _make_event(node="observe", started_at=started, rationale=rationale, payload=payload)
    return {"trace_events": [*state.trace_events, event]}


async def synthesize(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
    """Compose the final answer using the chat model and observation context.

    Short-circuit: when the planner picked the ``refuse`` tool, skip the LLM
    call and stamp :data:`CANONICAL_REFUSAL` directly. The doc requires one
    refusal string for all refusals, and the ``refuse`` tool's output is
    just ``{reason: ...}`` — there is nothing for the synthesizer to compose.
    """
    started = _now()

    last = state.observations[-1] if state.observations else None
    if last is not None and last.succeeded and last.tool_name == "refuse":
        candidate = CandidateAnswer(
            text=CANONICAL_REFUSAL,
            citations=[],
            source="refuse",
        )
        event = _make_event(
            node="synthesize",
            started_at=started,
            rationale="planner refused; stamped canonical refusal without an LLM call",
            payload={"source": "refuse", "skipped_llm": True},
        )
        return {
            "candidate_answer": candidate,
            "trace_events": [*state.trace_events, event],
        }

    rendered_observations = _render_observations(state)

    response = await ctx.llm.complete(
        ChatRequest(
            model=ctx.chat_model,
            max_completion_tokens=ctx.synthesis_max_completion_tokens,
            response_schema=ChatResponseSchema(
                name="support_synthesis",
                schema_definition=_SYNTHESIS_SCHEMA,
                strict=False,
            ),
            messages=[
                ChatMessage(role="system", content=load_prompt("synthesizer").system),
                ChatMessage(
                    role="user",
                    content=(
                        f"User message: {state.user_message}\n\n"
                        f"Tool observations:\n{rendered_observations}"
                    ),
                ),
            ],
        )
    )

    parsed = _parse_synthesis(response.content)
    fetched_titles = _collect_fetched_titles(state)
    # Cross-check: only surface titles the model declared *and* that retrieval
    # actually returned, to defend against the synthesizer hallucinating a
    # title that never came back from search.
    cited_titles = [title for title in parsed.cited_titles if title in fetched_titles]
    candidate = CandidateAnswer(
        text=parsed.text.strip(),
        citations=cited_titles,
        source=_infer_source(state),
    )
    event = _make_event(
        node="synthesize",
        started_at=started,
        model=response.model,
        tokens=TraceTokenUsage(**response.usage.model_dump()),
        rationale=(
            f"composed candidate answer; "
            f"{len(cited_titles)} of {len(fetched_titles)} retrieved titles cited"
        ),
        payload={
            "fetched_titles": fetched_titles,
            "cited_titles": cited_titles,
            "source": candidate.source,
        },
    )
    return {
        "candidate_answer": candidate,
        "trace_events": [*state.trace_events, event],
    }


async def precheck(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
    """Run the compliance precheck on the user message before planning.

    Hard-blocks (prompt injection, harmful/illegal, severe sensitive-account)
    short-circuit to ``finalize`` via the conditional edge. Other categories
    pass through with the decision attached to state so the planner can route
    sensibly.
    """
    started = _now()
    decision = await ctx.compliance.precheck(state.user_message)

    update: dict[str, Any] = {"compliance_precheck": decision}
    rationale = f"category={decision.category} allowed={decision.allowed}"

    if is_hard_block(decision):
        update["candidate_answer"] = CandidateAnswer(
            text=CANONICAL_REFUSAL,
            citations=[],
            source="compliance",
        )
        update["halted_reason"] = f"compliance precheck blocked: {decision.category}"
        rationale = f"hard-block on {decision.category}; routing to finalize"

    event = _make_event(
        node="precheck",
        started_at=started,
        model=ctx.compliance.config.model,
        rationale=rationale,
        payload={
            "mode": "precheck",
            "category": decision.category,
            "allowed": decision.allowed,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "override_applied": False,
        },
    )
    update["trace_events"] = [*state.trace_events, event]
    return update


async def verify(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
    """Run the verifier on the candidate answer and decide accept/repair/reject.

    Per the Phase 4 fail-fast policy:
    - ``accept`` → forward to postcheck.
    - ``repair`` → bounce back to synthesize **at most once** for fixable
      wording/source-label issues. If we've already repaired, treat as
      ``escalate`` to avoid loops.
    - ``escalate`` → stamp the answer as an escalation message and forward.
    - ``refuse`` → replace the answer with the canonical refusal and forward.
    """
    started = _now()
    if state.candidate_answer is None:
        raise RuntimeError("verify called without a candidate_answer in state")

    rendered_observations = _render_observations(state)
    verifier_user_content = (
        f'User message: "{state.user_message}"\n\n'
        f"Candidate answer source label: {state.candidate_answer.source}\n\n"
        f"Candidate answer text:\n{state.candidate_answer.text}\n\n"
        f"Tool observations:\n{rendered_observations}"
    )
    response = await ctx.llm.complete(
        ChatRequest(
            model=ctx.verifier_model,
            reasoning_effort=ctx.verifier_reasoning_effort,
            max_completion_tokens=ctx.verifier_max_completion_tokens,
            response_schema=ChatResponseSchema(
                name="verifier_verdict",
                schema_definition=_VERIFIER_SCHEMA,
                strict=False,
            ),
            messages=[
                ChatMessage(role="system", content=load_prompt("verifier").system),
                ChatMessage(role="user", content=verifier_user_content),
            ],
        )
    )
    verdict = _parse_verifier(response.content)
    rationale = (
        f"grounding={verdict.grounding} retry={verdict.retry_recommendation} "
        f"leakage={verdict.leakage_detected}"
    )
    payload: dict[str, Any] = {
        "addresses_request": verdict.addresses_request,
        "grounding": verdict.grounding,
        "leakage_detected": verdict.leakage_detected,
        "safe_source_label": verdict.safe_source_label,
        "retry_recommendation": verdict.retry_recommendation,
        "confidence": verdict.confidence,
        "reason": verdict.reason,
    }

    effective = verdict.retry_recommendation
    if effective == "repair" and state.repair_attempts >= ctx.max_repair_attempts:
        effective = "escalate"
        rationale += "; repair budget exhausted, escalating"
        payload["effective_recommendation"] = "escalate"

    # Store the *effective* verdict so the router doesn't loop us back to
    # synthesize after a budget-exhausted repair was already converted to
    # escalate above. The raw verifier verdict is preserved in the trace
    # payload (retry_recommendation field) for auditability.
    update: dict[str, Any] = {
        "verification": verdict.model_copy(update={"retry_recommendation": effective})
    }

    if effective == "repair":
        update["repair_attempts"] = state.repair_attempts + 1
    elif effective == "refuse":
        update["candidate_answer"] = state.candidate_answer.model_copy(
            update={"text": CANONICAL_REFUSAL, "source": "refuse", "citations": []}
        )
    elif effective == "escalate":
        update["candidate_answer"] = state.candidate_answer.model_copy(
            update={
                "text": (
                    "I'm not confident enough to answer this on my own. "
                    "I'll route you to a human agent who can help."
                ),
                "source": "escalate",
            }
        )

    event = _make_event(
        node="verify",
        started_at=started,
        model=response.model,
        tokens=TraceTokenUsage(**response.usage.model_dump()),
        rationale=rationale,
        payload=payload,
    )
    update["trace_events"] = [*state.trace_events, event]
    return update


async def postcheck(state: GraphState, ctx: NodeContext) -> dict[str, Any]:
    """Run the compliance postcheck on the verified candidate answer.

    When the postcheck blocks, replace the candidate text with the agent's
    ``override_response`` (if provided) or :data:`CANONICAL_REFUSAL` and stamp
    the source as ``compliance``.

    Terminal candidates (refuse / escalate / clarify) skip the LLM call: the
    answer is already a known-safe template (canonical refusal, escalation
    handoff, or a clarification question with topic examples), so paying for
    another compliance pass would only risk the LLM relabelling a refusal as
    a "compliance"-source override and burning tokens for no behavior change.
    """
    started = _now()
    if state.candidate_answer is None:
        raise RuntimeError("postcheck called without a candidate_answer in state")

    if state.candidate_answer.source in {"refuse", "escalate", "clarify"}:
        skip_event = _make_event(
            node="postcheck",
            started_at=started,
            rationale=(
                f"skipped postcheck LLM call: candidate is terminal "
                f"({state.candidate_answer.source})"
            ),
            payload={
                "mode": "postcheck",
                "skipped": True,
                "source": state.candidate_answer.source,
            },
        )
        return {"trace_events": [*state.trace_events, skip_event]}

    decision = await ctx.compliance.postcheck(
        user_message=state.user_message,
        candidate_answer=state.candidate_answer.text,
        candidate_source=state.candidate_answer.source,
    )

    update: dict[str, Any] = {"compliance_postcheck": decision}
    override_applied = False
    rationale = f"category={decision.category} allowed={decision.allowed}"

    if not decision.allowed:
        replacement_text = (
            decision.override_response
            if (decision.override_response and decision.override_response.strip())
            else CANONICAL_REFUSAL
        )
        update["candidate_answer"] = state.candidate_answer.model_copy(
            update={"text": replacement_text, "source": "compliance", "citations": []}
        )
        override_applied = True
        rationale = f"postcheck blocked ({decision.category}); applied override"

    event = _make_event(
        node="postcheck",
        started_at=started,
        model=ctx.compliance.config.model,
        rationale=rationale,
        payload={
            "mode": "postcheck",
            "category": decision.category,
            "allowed": decision.allowed,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "override_applied": override_applied,
        },
    )
    update["trace_events"] = [*state.trace_events, event]
    return update


async def finalize(state: GraphState, _ctx: NodeContext) -> dict[str, Any]:
    """Terminal node: emits a trace event noting the outcome."""
    started = _now()
    rationale = (
        f"halted: {state.halted_reason}"
        if state.halted_reason
        else "delivered candidate answer"
    )
    event = _make_event(
        node="finalize",
        started_at=started,
        rationale=rationale,
        payload={"has_answer": state.candidate_answer is not None},
    )
    return {"trace_events": [*state.trace_events, event]}


# --- prompt rendering helpers -------------------------------------------------


def _render_plan_prompt(state: GraphState) -> str:
    if not state.observations:
        return f'User message: "{state.user_message}"\n\nNo prior observations.'
    return (
        f'User message: "{state.user_message}"\n\n'
        f"Prior observations:\n{_render_observations(state)}"
    )


def _render_observations(state: GraphState) -> str:
    if not state.observations:
        return "(none)"
    lines: list[str] = []
    for index, obs in enumerate(state.observations, start=1):
        if obs.succeeded:
            lines.append(
                f"{index}. {obs.tool_name}({_compact(obs.arguments)}) -> "
                f"{_compact(obs.output)}"
            )
        else:
            lines.append(f"{index}. {obs.tool_name} FAILED: {obs.error}")
    return "\n".join(lines)


def _compact(value: Any) -> str:
    text = json.dumps(value, default=str)
    if len(text) > 500:
        return text[:497] + "..."
    return text


def _collect_fetched_titles(state: GraphState) -> list[str]:
    """Return every distinct FAQ title returned by successful search observations."""
    titles: list[str] = []
    for obs in state.observations:
        if not obs.succeeded:
            continue
        if obs.tool_name not in {"search_faq", "get_faq_by_category"}:
            continue
        for hit in obs.output.get("results", []):
            title = hit.get("title")
            if title and title not in titles:
                titles.append(title)
    return titles


def _parse_synthesis(content: str) -> _SynthesisOutput:
    """Parse the synthesizer's structured JSON output.

    Raises :class:`RuntimeError` when the response is non-JSON or fails the
    Pydantic shape; that propagates as a 500 so reviewers see the broken model
    output instead of a silently empty reply.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"synthesizer returned non-JSON content: {content!r}") from exc
    return _SynthesisOutput.model_validate(data)


def _parse_verifier(content: str) -> VerifierOutput:
    """Parse the verifier's structured JSON output."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"verifier returned non-JSON content: {content!r}") from exc
    return VerifierOutput.model_validate(data)


def _infer_source(
    state: GraphState,
) -> Literal["faq", "general", "clarify", "escalate", "refuse", "agent"]:
    last = state.observations[-1] if state.observations else None
    if last is None:
        return "agent"
    if last.tool_name in {"search_faq", "get_faq_by_category"}:
        return "faq"
    if last.tool_name == "general_knowledge_lookup":
        return "general"
    if last.tool_name == "ask_user_clarification":
        return "clarify"
    if last.tool_name == "escalate_to_human":
        return "escalate"
    if last.tool_name == "refuse":
        return "refuse"
    return "agent"


def _parse_plan(content: str) -> Plan:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"planner returned non-JSON content: {content!r}") from exc
    return Plan.model_validate(data)
