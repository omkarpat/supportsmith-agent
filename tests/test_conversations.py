"""Graph-driven conversation behavior with scripted LLM responses.

These tests bypass FastAPI and call ``agent.respond(...)`` directly. The
HTTP / persistence surface is exercised by ``tests/test_chat_persistence.py``,
which is gated on a real Postgres URL.
"""

from __future__ import annotations

import json

import pytest

from app.agent.harness import AgentRequest
from tests.conftest import build_support_agent_harness


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


async def test_chat_routes_ambiguous_message_through_clarify_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[
            _plan(
                "clarify",
                tool_name="ask_user_clarification",
                question="Could you tell me which area you need help with?",
            ),
            _synth("Could you share more about what you need help with today?"),
        ],
    )

    response = await harness.agent.respond(
        AgentRequest(conversation_id="demo", message="x")
    )

    assert response.source == "clarify"
    assert response.tools_used == ["ask_user_clarification"]
    assert response.verified is True
    assert response.trace_id.startswith("turn_")
    assert "more about what you need" in response.response.lower()


async def test_chat_routes_password_reset_through_search_faq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.conftest import faq_result

    title = "What steps do I take to reset my password?"
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[
            _plan("use_tool", tool_name="search_faq", query="reset password"),
            _synth(
                "Go to Settings > Security and select Change Password.",
                cited_titles=[title],
            ),
        ],
        canned_search_results=[
            faq_result(
                external_id="take_home_faq:reset-password-001",
                title=title,
                category="security",
                score=0.92,
                distance=0.08,
            )
        ],
    )

    response = await harness.agent.respond(
        AgentRequest(conversation_id="demo", message="How do I reset my password?")
    )

    assert response.conversation_id == "demo"
    assert response.source == "faq"
    assert response.tools_used == ["search_faq"]
    assert response.matched_questions == [title]
    assert response.response == "Go to Settings > Security and select Change Password."
    assert title not in response.response


async def test_conversation_id_round_trips_through_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The agent echoes back the conversation_id it was given on the request."""
    harness = build_support_agent_harness(
        monkeypatch,
        llm_responses=[
            _plan(
                "clarify",
                tool_name="ask_user_clarification",
                question="Tell me more.",
            ),
            _synth("Could you tell me more about your account question?"),
        ],
    )

    response = await harness.agent.respond(
        AgentRequest(conversation_id="my-conversation", message="Hello")
    )

    assert response.conversation_id == "my-conversation"
    assert response.source == "clarify"
