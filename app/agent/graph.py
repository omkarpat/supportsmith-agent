"""Compile the SupportSmith LangGraph state machine.

Edges (read top to bottom):

::

  START
    -> load_context
    -> precheck
       compliance_precheck.allowed=false AND hard-block category
                                          -> finalize  (CANONICAL_REFUSAL stamped)
       otherwise                          -> plan
    -> plan
       intent==use_tool                   -> execute_tool
       intent in (clarify, escalate,
                  refuse)                  -> execute_tool
       intent==synthesize_now              -> synthesize
    -> execute_tool
       (always)                            -> observe
    -> observe
       most recent tool was terminal
         (clarify | escalate | refuse)    -> synthesize
       max_tool_iterations reached        -> halt -> synthesize
       tool_iterations < cap              -> plan  (loop)
    -> synthesize
    -> verify
       retry_recommendation==repair AND
       repair_attempts < cap              -> synthesize  (single repair)
       otherwise                          -> postcheck
    -> postcheck
       (always)                           -> finalize  (override may have been
                                                       applied to candidate)
    -> finalize
    -> END

The cap on ``tool_iterations`` is enforced inside the observe→next router so
the graph cannot exceed ``ctx.max_tool_iterations``. The cap on repair
attempts is enforced inside the verify→next router so the verifier cannot
loop the synthesizer indefinitely.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.compliance import is_hard_block
from app.agent.nodes import (
    NodeContext,
    execute_tool,
    finalize,
    load_context,
    observe,
    plan,
    postcheck,
    precheck,
    synthesize,
    verify,
)
from app.agent.state import GraphState

NodeFn = Callable[[GraphState, NodeContext], Awaitable[dict[str, Any]]]
BoundNodeFn = Callable[[GraphState], Awaitable[dict[str, Any]]]


def build_graph(ctx: NodeContext) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Compile the SupportSmith agent graph with ``ctx`` injected per node."""
    builder = StateGraph(GraphState)

    builder.add_node("load_context", _bind(load_context, ctx))
    builder.add_node("precheck", _bind(precheck, ctx))
    builder.add_node("plan", _bind(plan, ctx))
    builder.add_node("execute_tool", _bind(execute_tool, ctx))
    builder.add_node("observe", _bind(observe, ctx))
    builder.add_node("halt", _halt_node)
    builder.add_node("synthesize", _bind(synthesize, ctx))
    builder.add_node("verify", _bind(verify, ctx))
    builder.add_node("postcheck", _bind(postcheck, ctx))
    builder.add_node("finalize", _bind(finalize, ctx))

    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "precheck")

    builder.add_conditional_edges(
        "precheck",
        _route_from_precheck,
        {
            "plan": "plan",
            "finalize": "finalize",
        },
    )
    builder.add_conditional_edges(
        "plan",
        _route_from_plan,
        {
            "execute_tool": "execute_tool",
            "synthesize": "synthesize",
        },
    )
    builder.add_edge("execute_tool", "observe")

    builder.add_conditional_edges(
        "observe",
        lambda state: _route_from_observe(state, ctx),
        {
            "plan": "plan",
            "synthesize": "synthesize",
            "halt": "halt",
        },
    )
    builder.add_edge("halt", "synthesize")
    builder.add_edge("synthesize", "verify")

    builder.add_conditional_edges(
        "verify",
        lambda state: _route_from_verify(state, ctx),
        {
            "synthesize": "synthesize",
            "postcheck": "postcheck",
        },
    )
    builder.add_edge("postcheck", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


# --- routers ------------------------------------------------------------------


def _route_from_precheck(state: GraphState) -> str:
    if state.compliance_precheck and is_hard_block(state.compliance_precheck):
        return "finalize"
    return "plan"


def _route_from_plan(state: GraphState) -> str:
    if state.plan is None:
        # Defensive: planner failed; finalize will surface the trace.
        return "synthesize"
    if state.plan.intent == "use_tool":
        if state.plan.tool_name is None:
            return "synthesize"
        return "execute_tool"
    if state.plan.intent in {"clarify", "escalate", "refuse"}:
        # These intents still produce a structured tool observation, then
        # synthesize emits a user-facing message describing what happened.
        return "execute_tool"
    return "synthesize"


def _route_from_observe(state: GraphState, ctx: NodeContext) -> str:
    if not state.observations:
        return "synthesize"
    last = state.observations[-1]

    if last.tool_name in {"ask_user_clarification", "escalate_to_human", "refuse"}:
        return "synthesize"

    if state.tool_iterations >= ctx.max_tool_iterations:
        return "halt"

    if not last.succeeded:
        return "plan"

    if last.tool_name == "search_faq":
        results = last.output.get("results", [])
        if results and float(results[0].get("score", 0.0)) >= 0.4:
            return "synthesize"
        return "plan"

    return "synthesize"


def _route_from_verify(state: GraphState, ctx: NodeContext) -> str:
    """Verifier recommends repair → synthesize once; otherwise → postcheck.

    The verify node enforces the budget by transforming repair → escalate
    when ``repair_attempts >= max`` and stores the *effective* recommendation
    on state. So if we see ``rec == "repair"`` here, the budget is still
    spendable and we route back to synthesize for the actual repair pass.
    """
    if state.verification is None:
        return "postcheck"
    if state.verification.retry_recommendation == "repair":
        return "synthesize"
    return "postcheck"


# --- helpers ------------------------------------------------------------------


async def _halt_node(state: GraphState) -> dict[str, Any]:
    """Stamp a halt reason when the iteration cap is reached."""
    return {
        "halted_reason": (
            f"hit max_tool_iterations={state.tool_iterations}; returning best-effort answer"
        )
    }


def _bind(node_fn: NodeFn, ctx: NodeContext) -> Any:
    """Bind a node function's NodeContext so LangGraph sees a (state) -> dict.

    Returns ``Any`` so LangGraph's heavily overloaded ``add_node`` does not
    bind ``NodeInputT`` to ``Never`` under mypy. The runtime behavior is a
    plain ``Callable[[GraphState], Awaitable[dict[str, Any]]]``.
    """

    async def _runner(state: GraphState) -> dict[str, Any]:
        return await node_fn(state, ctx)

    _runner.__name__ = getattr(node_fn, "__name__", "node")
    return _runner
