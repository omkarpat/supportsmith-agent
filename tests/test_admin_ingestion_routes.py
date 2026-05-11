"""Admin ingestion route tests.

The Firecrawl client is replaced by a fixture; the page classifier and
customer extractor LLMs are replaced by scripted clients. No network.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.postgres import PostgresDatabase
from app.ingestion.website import WebsiteIngestor
from app.llm.client import ChatResponse, TokenUsage
from app.llm.fake import FakeEmbeddingClient, ScriptedLLMClient
from app.main import create_app
from app.retrieval.firecrawl import (
    FirecrawlCrawl,
    FirecrawlMap,
    FixtureFirecrawlClient,
)
from app.retrieval.website_classifier import (
    CustomerNameExtractor,
    WebsiteLLMConfig,
    WebsitePageClassifier,
)
from tests.conftest import FakeDatabase

FIXTURE = Path("tests/fixtures/firecrawl_knotch_sample.json")


def _settings() -> Settings:
    return Settings.model_validate(
        {
            "environment": "test",
            "database_url": "postgresql://supportsmith:supportsmith@localhost/test",
            "api_bearer_token": "demo-token",
            "admin_api_key": "admin-key",
            "allowed_ingestion_hosts": "knotch.com",
        }
    )


def _scripted(payload: dict[str, Any]) -> ChatResponse:
    return ChatResponse(content=json.dumps(payload), model="scripted", usage=TokenUsage())


def _ingestor_factory(monkeypatch_target: Any) -> Any:
    """Build a fixture-backed WebsiteIngestor factory."""
    raw = json.loads(FIXTURE.read_text())
    firecrawl_map = FirecrawlMap.model_validate(raw["map"])
    firecrawl_crawl = FirecrawlCrawl.model_validate(raw["crawl"])

    # Scripted LLM: one classifier + one extractor call per page (2 pages → 4 calls)
    scripted = ScriptedLLMClient(
        [
            _scripted({"page_type": "case_study", "confidence": 0.95, "reason": "fixture"}),
            _scripted(
                {
                    "customer_names": ["Acme"],
                    "evidence_types": ["case_study_body"],
                    "confidence": 0.9,
                }
            ),
            _scripted({"page_type": "blog", "confidence": 0.9, "reason": "fixture"}),
            _scripted({"customer_names": [], "evidence_types": [], "confidence": 0.7}),
        ]
        * 10  # spare responses if the test re-runs ingestion
    )
    config = WebsiteLLMConfig(
        classifier_model="scripted",
        classifier_reasoning_effort="low",
        classifier_max_completion_tokens=128,
        extractor_model="scripted",
        extractor_reasoning_effort="low",
        extractor_max_completion_tokens=256,
    )

    class _NoSessionFactory:
        def __call__(self) -> Any:  # pragma: no cover - not used in dry_run
            raise AssertionError("session factory should not be invoked in dry_run tests")

    async def factory() -> WebsiteIngestor:
        return WebsiteIngestor(
            firecrawl=FixtureFirecrawlClient(
                map_result=firecrawl_map,
                crawl_result=firecrawl_crawl,
            ),
            classifier=WebsitePageClassifier(scripted, config),
            extractor=CustomerNameExtractor(scripted, config),
            embedding_client=FakeEmbeddingClient(dimensions=1536),
            embedding_model="fake",
            session_factory=_NoSessionFactory(),  # type: ignore[arg-type]
        )

    return factory


@pytest.fixture
def admin_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(
        PostgresDatabase,
        "from_settings",
        classmethod(lambda cls, settings: FakeDatabase()),
    )
    app = create_app(_settings())
    with TestClient(app) as client:
        app.state.website_ingestor_factory = _ingestor_factory(monkeypatch)
        yield client


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": "Bearer demo-token",
        "X-Admin-Api-Key": "admin-key",
    }


def test_queue_returns_202_and_job_id(admin_client: TestClient) -> None:
    response = admin_client.post(
        "/admin/website-ingestions",
        headers=_auth_headers(),
        json={
            "url": "https://knotch.com/",
            "name": "knotch",
            "dry_run": True,
            "limit": 50,
        },
    )
    assert response.status_code == 202
    body = response.json()
    assert body["job_id"].startswith("website_ingest_")
    assert body["status"] in {"queued", "running", "succeeded"}
    assert body["base_url"].startswith("https://knotch.com")


def test_queue_rejects_disallowed_host(admin_client: TestClient) -> None:
    response = admin_client.post(
        "/admin/website-ingestions",
        headers=_auth_headers(),
        json={
            "url": "https://evil.example/",
            "name": "evil",
            "dry_run": True,
        },
    )
    assert response.status_code == 400
    assert "allowlist" in response.json()["detail"].lower()


def test_queue_rejects_loopback(admin_client: TestClient) -> None:
    response = admin_client.post(
        "/admin/website-ingestions",
        headers=_auth_headers(),
        json={
            "url": "http://localhost/",
            "name": "local",
            "dry_run": True,
        },
    )
    assert response.status_code == 400


def test_get_job_returns_404_for_unknown_id(admin_client: TestClient) -> None:
    response = admin_client.get(
        "/admin/website-ingestions/does-not-exist",
        headers=_auth_headers(),
    )
    assert response.status_code == 404


def test_dry_run_completes_and_summary_lists_pages(admin_client: TestClient) -> None:
    response = admin_client.post(
        "/admin/website-ingestions",
        headers=_auth_headers(),
        json={
            "url": "https://knotch.com/",
            "name": "knotch",
            "dry_run": True,
        },
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    # The background task runs synchronously after the response in TestClient.
    # Give it a moment for any pending awaits to resolve.
    asyncio.run(asyncio.sleep(0))

    follow_up = admin_client.get(
        f"/admin/website-ingestions/{job_id}",
        headers=_auth_headers(),
    )
    assert follow_up.status_code == 200
    body = follow_up.json()
    assert body["status"] == "succeeded"
    assert body["summary"]["site_name"] == "knotch"
    assert body["summary"]["crawled_pages"] == 2
    pages = body["summary"]["pages"]
    assert any(page["page_type"] == "case_study" for page in pages)
    assert any(page["page_type"] == "blog" for page in pages)
