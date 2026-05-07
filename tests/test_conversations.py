from uuid import UUID

from fastapi.testclient import TestClient


def test_conversation_message_uses_phase_one_agent(client: TestClient) -> None:
    response = client.post(
        "/conversations/abc-123/messages",
        json={"message": "How do I reset my account?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"] == "abc-123"
    assert "Phase 1 harness is online" in payload["response"]
    assert payload["source"] == "agent"
    assert payload["tools_used"] == ["phase_one_agent"]
    assert payload["verified"] is True
    assert payload["trace_id"].startswith("trace_")
    assert payload["cost"] == {"total_tokens": 0, "estimated_usd": 0.0}


def test_conversation_message_rejects_empty_payload(client: TestClient) -> None:
    response = client.post("/conversations/abc-123/messages", json={"message": ""})

    assert response.status_code == 422


def test_chat_mints_conversation_id_when_missing(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 200
    payload = response.json()
    assert UUID(payload["conversation_id"])
    assert payload["source"] == "agent"
    assert payload["tools_used"] == ["phase_one_agent"]


def test_chat_uses_supplied_conversation_id(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"conversation_id": "existing-conversation", "message": "Continue"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"] == "existing-conversation"
    assert "Phase 1 harness is online" in payload["response"]
