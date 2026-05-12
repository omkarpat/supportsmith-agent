"""Bearer-token middleware tests for the Phase 8 demo gate.

When ``SUPPORTSMITH_API_BEARER_TOKEN`` is configured, every API route
must require a matching ``Authorization: Bearer <token>`` header — except
``/health`` (for platform health checks) and the static chat UI at ``/``
and ``/ui*`` (so a visitor can load the page and paste a token).

Phase 7 will additively layer the admin-API-key dependency on top of
this; that case is not exercised here because the admin routes don't
exist on this branch.
"""

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
    app = create_app(_settings(api_bearer_token="demo-token"))
    with TestClient(app) as client:
        yield client


def test_health_is_open_even_when_bearer_required(
    secured_client: TestClient,
) -> None:
    response = secured_client.get("/health")
    assert response.status_code == 200


def test_root_is_open_so_landing_page_loads(secured_client: TestClient) -> None:
    """The bare-domain redirect must work without a token — otherwise a
    visitor can't even reach the page that lets them paste one."""
    response = secured_client.get("/", follow_redirects=False)
    assert response.status_code in (307, 308)
    assert response.headers["location"] == "/ui"


def test_ui_shell_and_assets_are_open(secured_client: TestClient) -> None:
    for path in ("/ui", "/ui/assets/app.js", "/ui/assets/styles.css"):
        response = secured_client.get(path)
        assert response.status_code == 200, path


def test_protected_route_rejects_missing_bearer(secured_client: TestClient) -> None:
    response = secured_client.post("/chat", json={"message": "hi"})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"
    # The middleware never echoes the configured token in error responses.
    assert "demo-token" not in response.text


def test_protected_route_rejects_wrong_bearer(secured_client: TestClient) -> None:
    response = secured_client.post(
        "/chat",
        json={"message": "hi"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401


def test_protected_route_accepts_valid_bearer(secured_client: TestClient) -> None:
    # /chat with a body that fails Pydantic validation. The bearer middleware
    # lets the request through; FastAPI returns 422. Anything other than 401
    # proves the middleware is the only gate this request crossed.
    response = secured_client.post(
        "/chat",
        json={},
        headers={"Authorization": "Bearer demo-token"},
    )
    assert response.status_code != 401


def test_conversations_list_requires_bearer(secured_client: TestClient) -> None:
    """Phase 8's sidebar feed must be behind the gate too."""
    unauth = secured_client.get("/conversations")
    assert unauth.status_code == 401

    auth = secured_client.get(
        "/conversations",
        headers={"Authorization": "Bearer demo-token"},
    )
    # 503 ("Database not initialized") is fine here — the point is that we
    # passed the bearer gate. With the FakeDatabase fixture the session
    # factory is absent, which surfaces as 503 from get_db_session.
    assert auth.status_code != 401


def test_no_token_configured_means_no_enforcement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local dev / tests with no token configured stay unauthenticated."""
    monkeypatch.setattr(
        PostgresDatabase,
        "from_settings",
        classmethod(lambda cls, settings: FakeDatabase()),
    )
    app = create_app(_settings())  # no api_bearer_token
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        # A missing header must NOT 401 when the gate is disabled.
        ui = client.get("/ui")
        assert ui.status_code == 200


def test_malformed_authorization_header_is_rejected(
    secured_client: TestClient,
) -> None:
    for header in (
        "demo-token",  # missing scheme
        "Basic demo-token",  # wrong scheme
        "Bearer",  # missing token
        "Bearer  ",  # whitespace only
    ):
        response = secured_client.post(
            "/chat",
            json={"message": "hi"},
            headers={"Authorization": header},
        )
        assert response.status_code == 401, header
