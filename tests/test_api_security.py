"""Bearer-token middleware and admin-API-key dependency tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.postgres import PostgresDatabase
from app.main import create_app
from tests.conftest import FakeDatabase


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "test",
        "database_url": "postgresql://supportsmith:supportsmith@localhost/test",
    }
    base.update(overrides)
    return Settings.model_validate(base)


@pytest.fixture
def secured_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(
        PostgresDatabase,
        "from_settings",
        classmethod(lambda cls, settings: FakeDatabase()),
    )
    app = create_app(
        _settings(
            api_bearer_token="demo-token",
            admin_api_key="admin-key",
            allowed_ingestion_hosts="knotch.com",
        )
    )
    with TestClient(app) as client:
        yield client


def test_health_is_open_even_when_bearer_token_is_required(
    secured_client: TestClient,
) -> None:
    response = secured_client.get("/health")
    assert response.status_code == 200


def test_protected_route_rejects_missing_bearer(secured_client: TestClient) -> None:
    response = secured_client.post("/chat", json={"message": "hi"})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_protected_route_accepts_valid_bearer(secured_client: TestClient) -> None:
    # Hit /chat with a malformed body. The bearer middleware lets the
    # request through; FastAPI returns 422 from validation. Anything other
    # than 401 proves the middleware is the only gate this request crossed.
    response = secured_client.post(
        "/chat",
        json={},
        headers={"Authorization": "Bearer demo-token"},
    )
    assert response.status_code != 401


def test_admin_route_returns_401_when_bearer_is_valid_but_admin_key_missing(
    secured_client: TestClient,
) -> None:
    response = secured_client.get(
        "/admin/website-ingestions",
        headers={"Authorization": "Bearer demo-token"},
    )
    assert response.status_code == 401
    assert "admin api key" in response.json()["detail"].lower()


def test_admin_route_requires_admin_api_key(secured_client: TestClient) -> None:
    response = secured_client.get(
        "/admin/website-ingestions",
        headers={"Authorization": "Bearer demo-token"},
    )
    assert response.status_code == 401
    assert "admin api key" in response.json()["detail"].lower()


def test_admin_route_accepts_correct_admin_key(secured_client: TestClient) -> None:
    response = secured_client.get(
        "/admin/website-ingestions",
        headers={"Authorization": "Bearer demo-token", "X-Admin-Api-Key": "admin-key"},
    )
    assert response.status_code == 200
    assert response.json() == []


def test_admin_route_503_when_admin_key_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PostgresDatabase,
        "from_settings",
        classmethod(lambda cls, settings: FakeDatabase()),
    )
    app = create_app(_settings())  # no admin key, no bearer
    with TestClient(app) as client:
        response = client.get("/admin/website-ingestions")
        assert response.status_code == 503
