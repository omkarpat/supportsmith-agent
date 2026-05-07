"""SupportAgent: thin runner that wraps the compiled LangGraph workflow."""

from __future__ import annotations

from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph

from app.agent.harness import AgentRequest, AgentResponse, AgentSource, CostSummary
from app.agent.state import CandidateAnswer, GraphState

_SOURCE_FALLBACK: AgentSource = "agent"

_FALLBACK_TEXT = (
    "I couldn't put together a confident answer this turn. Please tell me a bit "
    "more about what you're trying to do, or try rephrasing the question."
)


class SupportAgent:
    """Drive a single user turn through the compiled support graph."""

    def __init__(self, graph: CompiledStateGraph) -> None:  # type: ignore[type-arg]
        self.graph = graph

    async def respond(self, request: AgentRequest) -> AgentResponse:
        """Run the graph for one turn and project the final state into AgentResponse."""
        turn_id = f"turn_{uuid4().hex}"
        initial = GraphState(
            conversation_id=request.conversation_id,
            turn_id=turn_id,
            user_message=request.message,
        )
        result = await self.graph.ainvoke(initial)
        final = GraphState.model_validate(result)
        return _to_agent_response(final, conversation_id=request.conversation_id)


def _to_agent_response(state: GraphState, *, conversation_id: str) -> AgentResponse:
    candidate = state.candidate_answer
    response_text = candidate.text if candidate and candidate.text else _FALLBACK_TEXT
    source = _project_source(candidate)
    tools_used: list[str] = []
    for observation in state.observations:
        if observation.tool_name not in tools_used:
            tools_used.append(observation.tool_name)
    cost = CostSummary(
        total_tokens=sum(
            (event.tokens.total_tokens if event.tokens else 0)
            for event in state.trace_events
        ),
    )
    return AgentResponse(
        conversation_id=conversation_id,
        response=response_text,
        source=source,
        matched_questions=list(candidate.citations) if candidate else [],
        tools_used=tools_used,
        verified=state.verification.passed,
        trace_id=state.turn_id,
        cost=cost,
    )


def _project_source(candidate: CandidateAnswer | None) -> AgentSource:
    if candidate is None:
        return _SOURCE_FALLBACK
    # CandidateAnswer.source uses an internal vocabulary that overlaps with
    # AgentSource's public vocabulary. Map and let invalid values fall back so
    # the API contract stays predictable even if the synthesizer drifts.
    mapping: dict[str, AgentSource] = {
        "agent": "agent",
        "faq": "faq",
        "general": "general",
        "clarify": "clarify",
        "escalate": "escalate",
        "refuse": "refuse",
    }
    return mapping.get(candidate.source, _SOURCE_FALLBACK)
