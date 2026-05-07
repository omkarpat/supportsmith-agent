from uuid import UUID

import pytest

from tests.conftest import build_support_agent_harness, faq_result


def _plan(intent: str, *, tool_name: str | None = None, **arguments: object) -> str:
    """Render the JSON plan payload that the planner LLM is mocked to return."""
    import json

    return json.dumps(
        {
            "intent": intent,
            "tool_name": tool_name,
            "arguments": arguments,
            "rationale": "scripted",
        }
    )


def _synth(text: str, *, cited_titles: list[str] | None = None) -> str:
    """Render the JSON the synthesizer LLM is mocked to return."""
    import json

    return json.dumps({"text": text, "cited_titles": cited_titles or []})


def test_chat_routes_ambiguous_message_through_clarify_tool(
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

    response = harness.client.post("/chat", json={"message": "x"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "clarify"
    assert payload["tools_used"] == ["ask_user_clarification"]
    assert payload["verified"] is True
    assert payload["trace_id"].startswith("turn_")
    assert "more about what you need" in payload["response"].lower()


def test_chat_routes_password_reset_through_search_faq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    response = harness.client.post(
        "/chat",
        json={"conversation_id": "demo", "message": "How do I reset my password?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"] == "demo"
    assert payload["source"] == "faq"
    assert payload["tools_used"] == ["search_faq"]
    assert payload["matched_questions"] == [title]
    assert payload["response"] == "Go to Settings > Security and select Change Password."
    # Make sure the inline citation noise is gone — the title shouldn't leak
    # into the user-facing prose.
    assert title not in payload["response"]


def test_chat_mints_conversation_id_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
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

    response = harness.client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 200
    payload = response.json()
    assert UUID(payload["conversation_id"])
    assert payload["source"] == "clarify"


def test_conversation_message_rejects_empty_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = build_support_agent_harness(monkeypatch, llm_responses=[])

    response = harness.client.post("/conversations/abc-123/messages", json={"message": ""})

    assert response.status_code == 422
