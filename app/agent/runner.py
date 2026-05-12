"""SupportAgent: thin runner that wraps the compiled LangGraph workflow.

LangSmith integration: ``respond`` reads the chat-flow-supplied root run id
+ thread metadata from instance attributes (set just before each call) and
threads them through a ``@traceable``-decorated inner method so every turn
becomes one root LangSmith run with our caller-chosen UUID. Child node runs
inherit the parent automatically via the ``traceable`` context.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

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
        """Run the graph for one turn and project the final state into AgentResponse.

        The chat-flow orchestrator (``app.api.chat_flow``) stashes context
        (prior user turns and the persisted ``turn_number``) on the agent
        instance before calling this method. After the traced inner call we
        also stash the LangSmith-assigned root run UUID on
        ``_captured_langsmith_run_id`` so the orchestrator can persist it
        on the agent message row. The public ``respond(request)`` signature
        stays provider-neutral.
        """
        prior_turns = list(getattr(self, "_pending_prior_turns", []))
        turn_number = int(getattr(self, "_pending_turn_number", 0))

        langsmith_extra: dict[str, Any] = {
            "metadata": {
                "thread_id": request.conversation_id,
                "conversation_id": request.conversation_id,
                "turn_number": turn_number,
            }
        }
        # Reset the captured slot so a previous turn's id can't leak forward
        # if LangSmith is disabled this turn.
        self._captured_langsmith_run_id = None

        return await self._respond_traced(
            request,
            prior_turns=prior_turns,
            turn_number=turn_number,
            langsmith_extra=langsmith_extra,  # type: ignore[arg-type]
        )

    @traceable(name="chat_turn", run_type="chain")
    async def _respond_traced(
        self,
        request: AgentRequest,
        *,
        prior_turns: list[Any],
        turn_number: int,
    ) -> AgentResponse:
        """Inner traced entry point.

        LangSmith mints the run UUID itself; we capture it from the active
        run tree so the orchestrator can persist it. When tracing is
        disabled, ``get_current_run_tree()`` returns ``None`` and we leave
        the capture slot ``None`` to signal "no LangSmith run for this turn".
        """
        run_tree = get_current_run_tree()
        if run_tree is not None:
            self._captured_langsmith_run_id = run_tree.id

        turn_id = f"turn_{uuid4().hex}"
        initial = GraphState(
            conversation_id=request.conversation_id,
            turn_id=turn_id,
            turn_number=turn_number,
            user_message=request.message,
            prior_user_turns=prior_turns,
        )
        result = await self.graph.ainvoke(initial)
        final = GraphState.model_validate(result)
        return _to_agent_response(
            final,
            conversation_id=request.conversation_id,
            turn_number=turn_number,
        )


def _to_agent_response(
    state: GraphState,
    *,
    conversation_id: str,
    turn_number: int = 0,
) -> AgentResponse:
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
        turn_number=turn_number,
        response=response_text,
        source=source,
        matched_questions=list(candidate.citations) if candidate else [],
        tools_used=tools_used,
        verified=_project_verified(state),
        trace_id=state.turn_id,
        cost=cost,
    )


def _project_verified(state: GraphState) -> bool:
    """Did the candidate clear every gate that ran this turn?

    True when each gate that executed approved the answer:
    - precheck (always runs unless missing) said allowed
    - verify (skipped on precheck hard-block) said retry_recommendation == accept
    - postcheck (skipped on precheck hard-block) said allowed
    """
    precheck_ok = state.compliance_precheck is None or state.compliance_precheck.allowed
    verify_ok = state.verification is None or state.verification.retry_recommendation == "accept"
    postcheck_ok = state.compliance_postcheck is None or state.compliance_postcheck.allowed
    return precheck_ok and verify_ok and postcheck_ok


def _project_source(candidate: CandidateAnswer | None) -> AgentSource:
    if candidate is None:
        return _SOURCE_FALLBACK
    # CandidateAnswer.source uses an internal vocabulary that overlaps with
    # AgentSource's public vocabulary. Map and let invalid values fall back so
    # the API contract stays predictable even if the synthesizer drifts.
    mapping: dict[str, AgentSource] = {
        "agent": "agent",
        "faq": "faq",
        "website": "website",
        "general": "general",
        "clarify": "clarify",
        "escalate": "escalate",
        "refuse": "refuse",
        "compliance": "compliance",
    }
    return mapping.get(candidate.source, _SOURCE_FALLBACK)
