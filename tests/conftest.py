import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.agent.compliance import ComplianceAgent, ComplianceConfig
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
    # ``None`` so dependencies that ``raise HTTPException(503, ...)`` on a
    # missing factory surface a clean 503 in tests instead of an
    # AttributeError. Tests that need real persistence point at a Postgres
    # URL and bypass this stub.
    session_factory = None

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
    """SupportAgent + scripted LLM pair for graph-level tests.

    Tests call ``await harness.agent.respond(AgentRequest(...))`` directly,
    bypassing FastAPI / persistence. Persistence is exercised by the
    Postgres-gated integration tests in ``tests/test_chat_persistence.py``.
    """

    agent: SupportAgent
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


def compliance_decision_json(
    *,
    allowed: bool = True,
    category: str = "support_allowed",
    reason: str = "scripted",
    override_response: str | None = None,
    confidence: float = 0.95,
) -> str:
    """Render a ComplianceDecision JSON payload for scripted tests."""
    return json.dumps(
        {
            "allowed": allowed,
            "category": category,
            "reason": reason,
            "override_response": override_response,
            "confidence": confidence,
        }
    )


def verifier_verdict_json(
    *,
    addresses_request: bool = True,
    grounding: str = "faq_grounded",
    leakage_detected: bool = False,
    safe_source_label: bool = True,
    retry_recommendation: str = "accept",
    reason: str = "scripted",
    confidence: float = 0.95,
) -> str:
    """Render a VerifierOutput JSON payload for scripted tests."""
    return json.dumps(
        {
            "addresses_request": addresses_request,
            "grounding": grounding,
            "leakage_detected": leakage_detected,
            "safe_source_label": safe_source_label,
            "retry_recommendation": retry_recommendation,
            "reason": reason,
            "confidence": confidence,
        }
    )


def build_support_agent_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    llm_responses: Iterable[str],
    canned_search_results: Iterable[RetrievalResult] | None = None,
    wrap_gates: bool = True,
) -> SupportAgentHarness:
    """Build a SupportAgent backed by a scripted LLM client.

    By default, ``wrap_gates=True`` automatically prepends a "support_allowed"
    precheck and appends an "accept" verifier verdict + "support_allowed"
    postcheck. Tests that focus on the planner / synthesizer flow can keep
    ``llm_responses`` short and not worry about the gates. Compliance- and
    verifier-specific tests pass ``wrap_gates=False`` and script every
    response explicitly.

    Returns the agent directly (no FastAPI / DB). Tests call
    ``await harness.agent.respond(...)`` to exercise the graph.
    """
    # ``monkeypatch`` is accepted for symmetry with the older harness signature
    # and in case a test wants to swap something at the module level; we don't
    # use it directly here because no FastAPI app is built.
    del monkeypatch

    inner = list(llm_responses)
    if wrap_gates:
        scripted = [
            compliance_decision_json(),
            *inner,
            verifier_verdict_json(),
            compliance_decision_json(),
        ]
    else:
        scripted = inner

    llm = ScriptedLLMClient(_scripted_responses(scripted))
    search = FakeSupportSearch(canned=list(canned_search_results or []))
    deps = ToolDependencies(
        llm=llm,
        embeddings=EmbeddingGenerator(FakeEmbeddingClient(dimensions=1536)),
        search=search,  # type: ignore[arg-type]
        chat_model="scripted-chat-model",
    )
    compliance = ComplianceAgent(
        llm=llm,
        config=ComplianceConfig(model="scripted-routing-model", reasoning_effort="low"),
    )
    ctx = NodeContext(
        llm=llm,
        tools=ToolRegistry(deps),
        compliance=compliance,
        chat_model="scripted-chat-model",
        reasoning_model="scripted-reasoning-model",
        planner_reasoning_effort="high",
        planner_max_completion_tokens=512,
        synthesis_max_completion_tokens=256,
        verifier_model="scripted-reasoning-model",
        verifier_reasoning_effort="medium",
        verifier_max_completion_tokens=512,
        max_tool_iterations=6,
        max_repair_attempts=1,
    )
    agent = SupportAgent(build_graph(ctx))
    return SupportAgentHarness(agent=agent, llm=llm, search=search)


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
    "compliance_decision_json",
    "faq_result",
    "verifier_verdict_json",
]


# Avoid an unused-import lint flag on AsyncMock; tests import it from this module.
_ = AsyncMock
