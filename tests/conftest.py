from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.agent.graph import build_graph
from app.agent.nodes import NodeContext
from app.agent.runner import SupportAgent
from app.agent.tools import ToolDependencies, ToolRegistry
from app.core.config import Settings
from app.db.postgres import DatabaseHealth, PostgresDatabase
from app.llm.client import ChatResponse, TokenUsage
from app.llm.fake import FakeEmbeddingClient, ScriptedLLMClient
from app.main import create_app
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.models import RetrievalResult


class FakeDatabase:
    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def health(self) -> DatabaseHealth:
        return DatabaseHealth(status="ok")


@dataclass
class FakeSupportSearch:
    """In-memory stand-in for SupportDocumentSearch used by graph tests."""

    canned: list[RetrievalResult]
    last_kwargs: dict[str, Any] | None = None

    async def search(
        self,
        embedding: list[float],
        *,
        limit: int = 5,
        category: str | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        self.last_kwargs = {"limit": limit, "category": category, "min_score": min_score}
        if category is None:
            return self.canned[:limit]
        return [hit for hit in self.canned if hit.category == category][:limit]

    async def list_categories(self) -> list[str]:
        return sorted({hit.category for hit in self.canned if hit.category})


def _test_settings() -> Settings:
    return Settings(
        environment="test",
        database_url="postgresql://supportsmith:supportsmith@localhost:55432/supportsmith_test",
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(
        PostgresDatabase,
        "from_settings",
        classmethod(lambda cls, settings: FakeDatabase()),
    )
    app = create_app(_test_settings())
    with TestClient(app) as test_client:
        yield test_client


@dataclass
class SupportAgentHarness:
    """A TestClient + ScriptedLLMClient pair for graph-driven endpoint tests."""

    client: TestClient
    llm: ScriptedLLMClient
    search: FakeSupportSearch


def _scripted_responses(payloads: Iterable[str]) -> list[ChatResponse]:
    return [
        ChatResponse(
            content=payload,
            model="scripted-test-model",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        for payload in payloads
    ]


def build_support_agent_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    llm_responses: Iterable[str],
    canned_search_results: Iterable[RetrievalResult] | None = None,
) -> SupportAgentHarness:
    """Build a TestClient whose ``/chat`` flows through the SupportAgent graph.

    ``llm_responses`` is the ordered sequence of LLM completions a turn will
    consume (typically: planner JSON, then synthesizer prose). The test-mode
    lifespan does not install an agent, so we set ``app.state.agent`` after
    the TestClient enters lifespan but before any request flies.
    """
    monkeypatch.setattr(
        PostgresDatabase,
        "from_settings",
        classmethod(lambda cls, settings: FakeDatabase()),
    )

    llm = ScriptedLLMClient(_scripted_responses(llm_responses))
    search = FakeSupportSearch(canned=list(canned_search_results or []))
    deps = ToolDependencies(
        llm=llm,
        embeddings=EmbeddingGenerator(FakeEmbeddingClient(dimensions=1536)),
        search=search,  # type: ignore[arg-type]
        chat_model="scripted-chat-model",
    )
    ctx = NodeContext(
        llm=llm,
        tools=ToolRegistry(deps),
        chat_model="scripted-chat-model",
        reasoning_model="scripted-reasoning-model",
        planner_reasoning_effort="high",
        planner_max_completion_tokens=512,
        synthesis_max_completion_tokens=256,
        max_tool_iterations=6,
    )
    agent = SupportAgent(build_graph(ctx))

    app = create_app(_test_settings())
    test_client = TestClient(app)
    test_client.__enter__()
    app.state.agent = agent
    return SupportAgentHarness(client=test_client, llm=llm, search=search)


def faq_result(**overrides: Any) -> RetrievalResult:
    """Convenience builder for canned RetrievalResult rows in tests."""
    base: dict[str, Any] = {
        "external_id": "take_home_faq:reset-password-001",
        "source": "take_home_faq",
        "title": "What steps do I take to reset my password?",
        "content": "Q: ...\n\nA: ...",
        "source_url": None,
        "category": "security",
        "metadata": {},
        "score": 0.9,
        "distance": 0.1,
    }
    base.update(overrides)
    return RetrievalResult.model_validate(base)


# Re-export so tests can import from conftest directly without circular imports.
__all__ = [
    "FakeDatabase",
    "FakeSupportSearch",
    "SupportAgentHarness",
    "build_support_agent_harness",
    "faq_result",
]


# Avoid an unused-import lint flag on AsyncMock; tests import it from this module.
_ = AsyncMock
