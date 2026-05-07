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

from app.agent.state import (
    CandidateAnswer,
    ComplianceDecision,
    GraphState,
    NodeName,
    Plan,
    ToolObservation,
    TraceEvent,
    TraceTokenUsage,
    VerificationResult,
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
from app.llm.client import (
    ChatMessage,
    ChatRequest,
    ChatResponseSchema,
    LLMClient,
    ReasoningEffort,
)
from app.llm.openai import LLMProviderError


@dataclass(frozen=True)
class NodeContext:
    """Per-graph-instance dependencies. Built once at graph compile time."""

    llm: LLMClient
    tools: ToolRegistry
    chat_model: str
    reasoning_model: str
    planner_reasoning_effort: ReasoningEffort
    planner_max_completion_tokens: int
    synthesis_max_completion_tokens: int
    max_tool_iterations: int


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

_PLANNER_SYSTEM_PROMPT = """\
You are SupportSmith's planner. Decide the next action for one turn.

Choose exactly one intent:
- "use_tool": invoke a tool from the allowed list. Set tool_name and arguments.
- "clarify": ask the user a short clarification question via the
  ask_user_clarification tool. Set tool_name="ask_user_clarification" and put
  the user-facing question into arguments.question.
- "synthesize_now": observations are sufficient for a final answer.
- "escalate": route to a human; set tool_name="escalate_to_human" and provide
  arguments.reason. The runtime will attach the transcript.
- "refuse": the request is out of scope or unsafe; set tool_name="refuse" and
  provide arguments.reason.

Allowed tools: search_faq, get_faq_by_category, ask_user_clarification,
general_knowledge_lookup, escalate_to_human, refuse.

Routing guidance:
- Prefer search_faq for any specific support question.
- Use general_knowledge_lookup only after FAQ retrieval has run and returned
  no high-confidence match.
- For ambiguous, malformed, or single-character user messages: clarify.
- For account compromise, lockout, or sensitive security incidents you cannot
  resolve via FAQ: escalate.

Keep rationale to one short sentence. Do not include chain-of-thought.
Return JSON matching the schema; no extra fields.
"""


_SYNTHESIS_SYSTEM_PROMPT = """\
You are SupportSmith's writer. Compose a concise, accurate support reply
grounded in the supplied tool observations. Do not invent product behavior.

Output rules:
- Return JSON matching the schema: {"text": "<user-facing reply>",
  "cited_titles": ["<exact FAQ title>", ...]}.
- Put the entire user-facing answer in "text". Do not include source labels,
  inline "Source: ..." lines, or any reference to FAQ titles inside "text".
- Put the FAQ titles you actually used into "cited_titles", word-for-word as
  they appear in the observations. If you did not use a search result, omit
  its title.
- If the observations are insufficient, say so plainly in "text" and return
  "cited_titles": [].
"""

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
            ChatMessage(role="system", content=_PLANNER_SYSTEM_PROMPT),
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
    """Compose the final answer using the chat model and observation context."""
    started = _now()
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
                ChatMessage(role="system", content=_SYNTHESIS_SYSTEM_PROMPT),
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


async def verify(state: GraphState, _ctx: NodeContext) -> dict[str, Any]:
    """Phase 4 placeholder: pass-through verification."""
    started = _now()
    event = _make_event(
        node="verify",
        started_at=started,
        rationale="phase 4 placeholder; pass-through",
    )
    return {
        "verification": VerificationResult(passed=True),
        "compliance": ComplianceDecision(allowed=True),
        "trace_events": [*state.trace_events, event],
    }


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
