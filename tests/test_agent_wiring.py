"""Startup wiring tests: probe failure modes, fallback selection, key required."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.agent.wiring import (
    StartupConfigurationError,
    build_live_support_agent,
    probe_chat_model,
)
from app.core.config import Settings
from app.llm.client import ChatRequest, ChatResponse, TokenUsage
from app.llm.openai import LLMProviderError


class _RecordingClient:
    """Tiny LLMClient that records calls and returns scripted responses or errors."""

    def __init__(self, *, model_responses: dict[str, Any]) -> None:
        self.model_responses = model_responses
        self.calls: list[ChatRequest] = []

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        if request.model is None:
            raise LLMProviderError("model is required for the recording client")
        outcome = self.model_responses.get(request.model)
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is None:
            raise LLMProviderError(f"no scripted outcome for {request.model}")
        return ChatResponse(
            content=outcome,
            model=request.model,
            usage=TokenUsage(),
        )


async def test_probe_returns_first_working_model() -> None:
    client = _RecordingClient(
        model_responses={
            "gpt-5.5-chat-latest": "ok",
            "gpt-5.4": "ok",
        }
    )

    chosen = await probe_chat_model(client, candidates=["gpt-5.5-chat-latest", "gpt-5.4"])

    assert chosen == "gpt-5.5-chat-latest"
    assert [call.model for call in client.calls] == ["gpt-5.5-chat-latest"]


async def test_probe_falls_through_to_next_candidate() -> None:
    client = _RecordingClient(
        model_responses={
            "gpt-5.5-chat-latest": LLMProviderError("model not found"),
            "gpt-5.4": "ok",
        }
    )

    chosen = await probe_chat_model(client, candidates=["gpt-5.5-chat-latest", "gpt-5.4"])

    assert chosen == "gpt-5.4"
    assert [call.model for call in client.calls] == ["gpt-5.5-chat-latest", "gpt-5.4"]


async def test_probe_raises_when_no_candidate_works() -> None:
    client = _RecordingClient(
        model_responses={
            "gpt-5.5-chat-latest": LLMProviderError("model not found"),
            "gpt-5.4": LLMProviderError("rate limited"),
        }
    )

    with pytest.raises(StartupConfigurationError) as excinfo:
        await probe_chat_model(client, candidates=["gpt-5.5-chat-latest", "gpt-5.4"])

    assert "No configured Chat Completions model is usable" in str(excinfo.value)


async def test_probe_raises_on_empty_candidate_list() -> None:
    client = _RecordingClient(model_responses={})

    with pytest.raises(StartupConfigurationError, match="No chat model candidates"):
        await probe_chat_model(client, candidates=[])


async def test_build_live_support_agent_requires_openai_key() -> None:
    settings = Settings(
        environment="local",
        database_url="postgresql://supportsmith:supportsmith@localhost:55432/supportsmith",
    )
    # No OPENAI_API_KEY in test env (conftest doesn't set it on Settings directly).
    settings = settings.model_copy(update={"openai_api_key": None})

    with pytest.raises(StartupConfigurationError, match="OPENAI_API_KEY"):
        await build_live_support_agent(settings, search=AsyncMock())
