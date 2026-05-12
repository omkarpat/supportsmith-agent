"""Graph-level scenarios with mocked LLM responses (no OpenAI calls)."""

from __future__ import annotations

import json

import pytest

from app.agent.harness import AgentRequest
from tests.conftest import build_support_agent_harness, faq_result, website_result


def _plan(intent: str, *, tool_name: str | None = None, **arguments: object) -> str:
    return json.dumps(
        {
            "intent": intent,
            "tool_name": tool_name,
            "arguments": arguments,
            "rationale": "scripted",
        }
    )


def _synth(text: str, *, cited_chunk_ids: list[int] | None = None) -> str:
    return json.dumps({"text": text, "cited_chunk_ids": cited_chunk_ids or []})


async def test_general_knowledge_runs_only_after_low_confidence_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[
            # 1. Planner: try the FAQ first
            _plan("use_tool", tool_name="search_kb", query="cosmic ray hardening"),
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

    response = await harness.agent.respond(
        AgentRequest(conversation_id="demo", message="What protects against cosmic rays?")
    )

    assert response.tools_used == ["search_kb", "general_knowledge_lookup"]
    assert response.source == "general"


async def test_escalation_returns_structured_mock_handoff(
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

    response = await harness.agent.respond(
        AgentRequest(conversation_id="demo", message="my account was hacked, help!")
    )

    assert response.tools_used == ["escalate_to_human"]
    assert response.source == "escalate"


async def test_refusal_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Synthesize short-circuits for the refuse tool (stamps CANONICAL_REFUSAL
    # without an LLM call), so no synth response needs scripting here.
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[_plan("refuse", tool_name="refuse", reason="off-topic")],
    )

    response = await harness.agent.respond(
        AgentRequest(conversation_id="demo", message="write me a poem about cats")
    )

    assert response.tools_used == ["refuse"]
    assert response.source == "refuse"


async def test_loop_limit_produces_graceful_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the planner keeps low-confidence-searching forever, the graph halts."""
    # max_tool_iterations defaults to 6 → exactly 6 plan calls before the
    # observe router sends us to halt → synthesize (one LLM call).
    plans = [_plan("use_tool", tool_name="search_kb", query="loop")] * 6
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[*plans, _synth("I couldn't put together a confident answer.")],
        canned_search_results=[
            faq_result(score=0.05, distance=0.95, title="unrelated", category="profile")
        ],
    )

    response = await harness.agent.respond(
        AgentRequest(conversation_id="demo", message="trigger loop")
    )

    # The graph should terminate cleanly even when the planner refuses to converge.
    assert response.verified is True
    # tools_used should reflect the bounded loop, not exceed the cap of 6.
    assert response.tools_used == ["search_kb"]


async def test_planner_uses_reasoning_model_and_synthesizer_uses_chat_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify per-node model selection (chat vs reasoning) is correct."""
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[
            _plan("use_tool", tool_name="search_kb", query="reset password"),
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

    await harness.agent.respond(
        AgentRequest(conversation_id="demo", message="How do I reset my password?")
    )

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


async def test_matched_questions_only_includes_titles_synthesizer_actually_cited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrieval may surface multiple candidates, but matched_questions only
    lists the titles for chunk ids the synthesizer declared cited."""
    cited_title = "Can I get a refund?"
    not_cited = "Where can I download invoices?"
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[
            _plan("use_tool", tool_name="search_kb", query="refund"),
            _synth(
                "Refunds are available within 14 days for eligible plans.",
                cited_chunk_ids=[0],
            ),
        ],
        canned_search_results=[
            faq_result(external_id="a", title=cited_title, category="billing", score=0.9),
            faq_result(external_id="b", title=not_cited, category="billing", score=0.8),
        ],
    )

    response = await harness.agent.respond(
        AgentRequest(conversation_id="demo", message="can I get a refund?")
    )

    assert response.matched_questions == [cited_title]
    assert not_cited not in response.matched_questions
    # The user-facing text must NOT contain the citation title; that's metadata only.
    assert cited_title not in response.response


async def test_matched_questions_is_empty_when_synthesizer_does_not_cite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[
            _plan("use_tool", tool_name="search_kb", query="refund"),
            _synth(
                "I don't have enough info to confidently answer that.",
                cited_chunk_ids=[],
            ),
        ],
        canned_search_results=[
            faq_result(external_id="a", title="Can I get a refund?", score=0.9),
        ],
    )

    response = await harness.agent.respond(
        AgentRequest(conversation_id="demo", message="refund?")
    )

    assert response.matched_questions == []


async def test_synthesizer_invalid_chunk_ids_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out-of-range chunk ids are silently dropped — chunk-id citations are
    deterministically resolved from observations, not the LLM's vocabulary."""
    real = "Can I get a refund?"
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[
            _plan("use_tool", tool_name="search_kb", query="refund"),
            # cite id 0 (real) and id 99 (out of range; must be dropped)
            _synth("Refunds within 14 days.", cited_chunk_ids=[0, 99]),
        ],
        canned_search_results=[
            faq_result(external_id="a", title=real, category="billing", score=0.9),
        ],
    )

    response = await harness.agent.respond(
        AgentRequest(conversation_id="demo", message="refund?")
    )

    assert response.matched_questions == [real]


async def test_website_chunk_citation_renders_url_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Citing a website chunk should attach a [title](url) Sources line to text,
    set source='website', and keep matched_questions empty."""
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[
            _plan("use_tool", tool_name="search_kb", query="acme customer"),
            _synth("Knotch worked with Acme to boost engagement.", cited_chunk_ids=[0]),
        ],
        canned_search_results=[
            website_result(
                title="Acme case study",
                source_url="https://knotch.com/case-studies/acme",
                score=0.9,
            ),
        ],
    )

    response = await harness.agent.respond(
        AgentRequest(conversation_id="demo", message="who is Acme?")
    )

    assert response.source == "website"
    assert response.matched_questions == []
    assert "https://knotch.com/case-studies/acme" in response.response
    assert "Acme case study" in response.response


async def test_agent_emits_trace_id_per_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[
            _plan("clarify", tool_name="ask_user_clarification", question="more?"),
            _synth("Could you tell me more?"),
        ],
    )

    response = await harness.agent.respond(
        AgentRequest(conversation_id="demo", message="x")
    )

    assert response.trace_id.startswith("turn_")
    assert response.conversation_id == "demo"
