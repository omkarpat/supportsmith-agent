"""Compile the SupportSmith LangGraph state machine.

Edges (read top to bottom):

::

  START
    -> load_context
    -> plan
       intent==use_tool                -> execute_tool
       intent in (clarify, escalate,
                  refuse, synthesize_now) -> synthesize
    -> execute_tool
       (always) -> observe
    -> observe
       most recent tool was terminal
         (clarify | escalate | refuse)  -> synthesize
       max_tool_iterations reached      -> halt -> synthesize
       tool_iterations < cap            -> plan  (loop)
    -> synthesize
    -> verify
    -> finalize
    -> END

The cap on ``tool_iterations`` is enforced inside the observe→next router so
the graph cannot exceed ``ctx.max_tool_iterations`` regardless of what the
planner emits. When we cap out we stamp ``halted_reason`` and route to
synthesize so the user still gets a graceful response.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes import (
    NodeContext,
    execute_tool,
    finalize,
    load_context,
    observe,
    plan,
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
    builder.add_node("plan", _bind(plan, ctx))
    builder.add_node("execute_tool", _bind(execute_tool, ctx))
    builder.add_node("observe", _bind(observe, ctx))
    builder.add_node("halt", _halt_node)
    builder.add_node("synthesize", _bind(synthesize, ctx))
    builder.add_node("verify", _bind(verify, ctx))
    builder.add_node("finalize", _bind(finalize, ctx))

    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "plan")

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
    builder.add_edge("verify", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


# --- routers ------------------------------------------------------------------


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
