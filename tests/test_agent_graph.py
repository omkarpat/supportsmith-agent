"""Graph-level scenarios with mocked LLM responses (no OpenAI calls)."""

from __future__ import annotations

import json

import pytest

from app.agent.harness import AgentRequest
from tests.conftest import build_support_agent_harness, faq_result


def _plan(intent: str, *, tool_name: str | None = None, **arguments: object) -> str:
    return json.dumps(
        {
            "intent": intent,
            "tool_name": tool_name,
            "arguments": arguments,
            "rationale": "scripted",
        }
    )


def _synth(text: str, *, cited_titles: list[str] | None = None) -> str:
    return json.dumps({"text": text, "cited_titles": cited_titles or []})


def _client_post(harness, conversation_id: str, message: str):  # type: ignore[no-untyped-def]
    return harness.client.post(
        "/chat",
        json={"conversation_id": conversation_id, "message": message},
    )


def test_general_knowledge_runs_only_after_low_confidence_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[
            # 1. Planner: try the FAQ first
            _plan("use_tool", tool_name="search_faq", query="cosmic ray hardening"),
            # 2. Planner: low-confidence search; fall back to general knowledge
            _plan("use_tool", tool_name="general_knowledge_lookup", query="cosmic ray hardening"),
            # 3. general_knowledge_lookup tool LLM call (the tool itself; not JSON)
            "Cosmic rays cause bit flips; ECC RAM helps.",
            # 4. Synthesizer (structured)
            _synth("Here is general guidance on cosmic ray hardening."),
        ],
        canned_search_results=[
            faq_result(score=0.05, distance=0.95, title="unrelated", category="profile")
        ],
    )

    response = _client_post(harness, "demo", "What protects against cosmic rays?")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tools_used"] == ["search_faq", "general_knowledge_lookup"]
    assert payload["source"] == "general"


def test_escalation_returns_structured_mock_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[
            _plan(
                "escalate",
                tool_name="escalate_to_human",
                reason="user reports account compromise",
            ),
            _synth("I've routed this to our security team for human review."),
        ],
    )

    response = _client_post(harness, "demo", "my account was hacked, help!")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tools_used"] == ["escalate_to_human"]
    assert payload["source"] == "escalate"


def test_refusal_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Synthesize short-circuits for the refuse tool (stamps CANONICAL_REFUSAL
    # without an LLM call), so no synth response needs scripting here.
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[_plan("refuse", tool_name="refuse", reason="off-topic")],
    )

    response = _client_post(harness, "demo", "write me a poem about cats")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tools_used"] == ["refuse"]
    assert payload["source"] == "refuse"


def test_loop_limit_produces_graceful_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the planner keeps low-confidence-searching forever, the graph halts."""
    # Build a script that loops search_faq with low-confidence results until the
    # graph hits max_tool_iterations. We need (max_iters + 1) plan responses
    # (one extra plan call after the cap is hit, then the synthesizer).
    # max_tool_iterations defaults to 6 → exactly 6 plan calls before the
    # observe router sends us to halt → synthesize (one LLM call).
    plans = [_plan("use_tool", tool_name="search_faq", query="loop")] * 6
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[*plans, _synth("I couldn't put together a confident answer.")],
        canned_search_results=[
            faq_result(score=0.05, distance=0.95, title="unrelated", category="profile")
        ],
    )

    response = _client_post(harness, "demo", "trigger loop")

    assert response.status_code == 200
    payload = response.json()
    # The graph should terminate cleanly even when the planner refuses to converge.
    assert payload["verified"] is True
    # tools_used should reflect the bounded loop, not exceed the cap of 6.
    assert len(payload["tools_used"]) <= 1  # only one distinct tool name
    assert payload["tools_used"] == ["search_faq"]


def test_planner_uses_reasoning_model_and_synthesizer_uses_chat_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify per-node model selection (chat vs reasoning) is correct."""
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[
            _plan("use_tool", tool_name="search_faq", query="reset password"),
            _synth("Go to Settings > Security and follow the prompts."),
        ],
        canned_search_results=[
            faq_result(
                score=0.92,
                distance=0.08,
                title="What steps do I take to reset my password?",
            ),
        ],
    )

    _client_post(harness, "demo", "How do I reset my password?")

    # Five LLM calls: precheck, plan, synthesize, verify, postcheck.
    assert len(harness.llm.requests) == 5
    precheck_request, planner_request, synthesis_request, verifier_request, postcheck_request = (
        harness.llm.requests
    )
    assert precheck_request.model == "scripted-routing-model"
    assert precheck_request.reasoning_effort == "low"
    assert planner_request.model == "scripted-reasoning-model"
    assert planner_request.reasoning_effort == "high"
    assert planner_request.response_schema is not None
    assert planner_request.response_schema.name == "support_plan"
    assert synthesis_request.model == "scripted-chat-model"
    assert synthesis_request.response_schema is not None
    assert synthesis_request.response_schema.name == "support_synthesis"
    assert verifier_request.model == "scripted-reasoning-model"
    assert verifier_request.reasoning_effort == "medium"
    assert postcheck_request.model == "scripted-routing-model"
    assert postcheck_request.reasoning_effort == "low"


def test_matched_questions_only_includes_titles_synthesizer_actually_cited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retrieval may surface 5 candidates, but matched_questions only lists
    the ones the synthesizer declared in its structured cited_titles list."""
    cited = "Can I get a refund?"
    not_cited = "Where can I download invoices?"
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[
            _plan("use_tool", tool_name="search_faq", query="refund"),
            _synth(
                "Refunds are available within 14 days for eligible plans.",
                cited_titles=[cited],
            ),
        ],
        canned_search_results=[
            faq_result(external_id="a", title=cited, category="billing", score=0.9),
            faq_result(external_id="b", title=not_cited, category="billing", score=0.8),
        ],
    )

    response = _client_post(harness, "demo", "can I get a refund?")
    payload = response.json()

    assert payload["matched_questions"] == [cited]
    assert not_cited not in payload["matched_questions"]
    # The user-facing text must NOT contain the citation; that's metadata only.
    assert cited not in payload["response"]


def test_matched_questions_is_empty_when_synthesizer_does_not_cite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[
            _plan("use_tool", tool_name="search_faq", query="refund"),
            _synth(
                "I don't have enough info to confidently answer that.",
                cited_titles=[],
            ),
        ],
        canned_search_results=[
            faq_result(external_id="a", title="Can I get a refund?", score=0.9),
        ],
    )

    response = _client_post(harness, "demo", "refund?")
    payload = response.json()

    assert payload["matched_questions"] == []


def test_matched_questions_drops_titles_the_synthesizer_hallucinated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the synthesizer claims a title that retrieval never returned, drop it."""
    real = "Can I get a refund?"
    hallucinated = "How do I summon a sandwich?"
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[
            _plan("use_tool", tool_name="search_faq", query="refund"),
            _synth("Refunds within 14 days.", cited_titles=[real, hallucinated]),
        ],
        canned_search_results=[
            faq_result(external_id="a", title=real, category="billing", score=0.9),
        ],
    )

    response = _client_post(harness, "demo", "refund?")
    payload = response.json()

    assert payload["matched_questions"] == [real]


async def test_agent_emits_trace_id_per_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[
            _plan("clarify", tool_name="ask_user_clarification", question="more?"),
            _synth("Could you tell me more?"),
        ],
    )

    response = _client_post(harness, "demo", "x")

    payload = response.json()
    assert payload["trace_id"].startswith("turn_")
    assert payload["conversation_id"] == "demo"


_ = AgentRequest  # imported so type-check sees the public contract
