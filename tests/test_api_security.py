"""Bearer-token middleware + admin-API-key dependency tests.

When ``SUPPORTSMITH_API_BEARER_TOKEN`` is configured, every API route
must require a matching ``Authorization: Bearer <token>`` header — except
``/health`` (for platform health checks) and the static chat UI at ``/``
and ``/ui*`` (so a visitor can load the page and paste a token).

Admin ingestion routes additionally require ``X-Admin-Api-Key`` when
``SUPPORTSMITH_ADMIN_API_KEY`` is configured. When that key is not
configured, those routes return 503 rather than running unauthenticated.
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
    # ``_env_file=None`` keeps the test hermetic. ``model_validate`` would
    # still trigger pydantic-settings env loading (via __init__) and pick
    # up SUPPORTSMITH_API_BEARER_TOKEN / SUPPORTSMITH_ADMIN_API_KEY from
    # the developer's ``.env``, which would silently switch the bearer or
    # admin gates on for tests that expect them off.
    base: dict[str, object] = {
        "environment": "test",
        "database_url": "postgresql://supportsmith:supportsmith@localhost/test",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg,arg-type]


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


# --- bearer middleware --------------------------------------------------------


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
    app = create_app(_settings())  # no api_bearer_token, no admin_api_key
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


# --- admin api key gate -------------------------------------------------------


def test_admin_route_returns_401_when_bearer_is_valid_but_admin_key_missing(
    secured_client: TestClient,
) -> None:
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
