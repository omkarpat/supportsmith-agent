"""Tool registry behavior with mocked LLM and search."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.agent.tools import (
    AskUserClarificationOutput,
    EscalateToHumanOutput,
    GeneralKnowledgeLookupOutput,
    GetFAQByCategoryOutput,
    RefuseOutput,
    SearchKBOutput,
    ToolDependencies,
    ToolRegistry,
)
from app.agent.topics import SUPPORT_TOPIC_EXAMPLES
from app.llm.client import ChatMessage
from app.llm.fake import FakeEmbeddingClient, FakeLLMClient
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.models import RetrievalResult


@dataclass
class _FakeSearch:
    canned: list[RetrievalResult]
    last_kwargs: dict[str, Any] | None = None

    async def search(
        self,
        embedding: list[float],
        *,
        limit: int = 5,
        category: str | None = None,
        min_score: float | None = None,
        sources: list[str] | None = None,
        qualities: list[str] | None = None,
    ) -> list[RetrievalResult]:
        self.last_kwargs = {
            "limit": limit,
            "category": category,
            "min_score": min_score,
            "sources": list(sources) if sources is not None else None,
            "embedding_dims": len(embedding),
        }
        results = self.canned
        if category is not None:
            results = [r for r in results if r.category == category]
        if sources is not None:
            results = [r for r in results if r.source in sources]
        return results[:limit]


def _result(**overrides: Any) -> RetrievalResult:
    base: dict[str, Any] = {
        "external_id": "take_home_faq:reset-password-001",
        "source": "faq",
        "title": "Reset password",
        "content": "Q: ...\n\nA: ...",
        "source_url": None,
        "category": "security",
        "metadata": {},
        "score": 0.9,
        "distance": 0.1,
    }
    base.update(overrides)
    return RetrievalResult.model_validate(base)


def _registry(
    *,
    canned_results: Iterable[RetrievalResult] | None = None,
    llm: FakeLLMClient | None = None,
) -> tuple[ToolRegistry, _FakeSearch, FakeLLMClient]:
    search = _FakeSearch(canned=list(canned_results or [_result()]))
    fake_llm = llm or FakeLLMClient(response_text="general knowledge answer")
    deps = ToolDependencies(
        llm=fake_llm,
        embeddings=EmbeddingGenerator(FakeEmbeddingClient(dimensions=1536)),
        search=search,  # type: ignore[arg-type]
        chat_model="gpt-5.5-chat-latest",
    )
    return ToolRegistry(deps), search, fake_llm


async def test_search_kb_validates_args_and_returns_typed_results() -> None:
    registry, search, _ = _registry()

    output = await registry.run("search_kb", {"query": "reset password", "limit": 3})

    assert isinstance(output, SearchKBOutput)
    assert output.results
    assert search.last_kwargs == {
        "limit": 3,
        "category": None,
        "min_score": None,
        "sources": None,
        "embedding_dims": 1536,
    }


async def test_search_kb_filters_by_source() -> None:
    registry, search, _ = _registry(
        canned_results=[
            _result(external_id="faq-1", source="faq"),
            _result(external_id="web-1", source="website", source_url="https://x/y"),
        ]
    )

    output = await registry.run(
        "search_kb",
        {"query": "knotch customers", "sources": ["website"], "limit": 3},
    )

    assert isinstance(output, SearchKBOutput)
    assert search.last_kwargs is not None
    assert search.last_kwargs["sources"] == ["website"]
    assert all(result.source == "website" for result in output.results)


async def test_search_kb_rejects_invalid_args() -> None:
    registry, _, _ = _registry()

    with pytest.raises(ValueError):
        await registry.run("search_kb", {"query": "", "limit": 3})


async def test_get_faq_by_category_filters_results() -> None:
    registry, _, _ = _registry(
        canned_results=[
            _result(external_id="a", category="security"),
            _result(external_id="b", category="billing"),
            _result(external_id="c", category="security"),
        ]
    )

    output = await registry.run("get_faq_by_category", {"category": "security"})

    assert isinstance(output, GetFAQByCategoryOutput)
    assert output.category == "security"
    assert {r.external_id for r in output.results} == {"a", "c"}


async def test_ask_user_clarification_returns_topic_examples_from_config() -> None:
    registry, _, _ = _registry()

    output = await registry.run(
        "ask_user_clarification",
        {"question": "Could you tell me more about what you need?"},
    )

    assert isinstance(output, AskUserClarificationOutput)
    assert output.topic_examples == list(SUPPORT_TOPIC_EXAMPLES)
    assert output.question.startswith("Could you")


async def test_general_knowledge_lookup_calls_chat_model_and_marks_not_grounded() -> None:
    fake_llm = FakeLLMClient(response_text="Use a strong unique password.")
    registry, _, _ = _registry(llm=fake_llm)

    output = await registry.run(
        "general_knowledge_lookup",
        {"query": "password best practices"},
    )

    assert isinstance(output, GeneralKnowledgeLookupOutput)
    assert output.answer == "Use a strong unique password."
    assert output.grounded_in_kb is False
    assert fake_llm.requests, "general_knowledge_lookup should call the LLM"
    assert fake_llm.requests[-1].model == "gpt-5.5-chat-latest"


async def test_escalate_to_human_returns_typed_handoff_record() -> None:
    registry, _, _ = _registry()

    output = await registry.run(
        "escalate_to_human",
        {
            "reason": "user reports account compromise",
            "transcript": [ChatMessage(role="user", content="my account is hacked").model_dump()],
        },
    )

    assert isinstance(output, EscalateToHumanOutput)
    assert output.status == "queued"
    assert output.handoff_id.startswith("escalation_")
    assert output.reason == "user reports account compromise"


async def test_refuse_returns_typed_reason() -> None:
    registry, _, _ = _registry()

    output = await registry.run("refuse", {"reason": "out of scope"})

    assert isinstance(output, RefuseOutput)
    assert output.reason == "out of scope"


async def test_unknown_tool_raises() -> None:
    registry, _, _ = _registry()

    with pytest.raises(ValueError, match="Unknown tool"):
        await registry.run("delete_database", {})  # type: ignore[arg-type]


def test_tool_input_schemas_are_strict() -> None:
    registry, _, _ = _registry()

    schema = registry.input_schema("search_kb")

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "query" in schema["properties"]


# Avoid an unused-import lint flag for fixtures that test mocking surface only.
_ = AsyncMock
