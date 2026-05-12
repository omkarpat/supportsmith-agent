"""Static-route tests for the Phase 8 chat-UI demo console.

Exercises the bare HTTP contract — root redirect, HTML shell, JS/CSS
asset whitelist — using the FakeDatabase test client. Conversation-API
behavior is covered separately in ``test_chat_persistence.py``.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_redirects_to_ui(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (307, 308)
    assert response.headers["location"] == "/ui"


def test_ui_shell_returns_html(client: TestClient) -> None:
    response = client.get("/ui")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "<title>SupportSmith demo console</title>" in body
    assert "/ui/assets/app.js" in body
    assert "/ui/assets/styles.css" in body


def test_ui_assets_serve_javascript_and_css(client: TestClient) -> None:
    js = client.get("/ui/assets/app.js")
    assert js.status_code == 200
    assert js.headers["content-type"].startswith("application/javascript")
    # Anchor on a stable identifier so a future rename surfaces a test diff.
    assert "SIDEBAR_LIMIT" in js.text

    css = client.get("/ui/assets/styles.css")
    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")
    assert ".composer" in css.text


def test_ui_asset_whitelist_rejects_other_paths(client: TestClient) -> None:
    """Unknown asset names 404 — the route never reads an arbitrary file."""
    for name in (
        "index.html",
        "../app.py",
        "secret.txt",
        "..%2Fapp.py",
    ):
        response = client.get(f"/ui/assets/{name}")
        assert response.status_code == 404, name


def test_ui_shell_does_not_embed_bearer_token(client: TestClient) -> None:
    """The shell must never hard-code or echo a server-side token.

    The phase-8 doc is explicit: the UI should not render the token back
    into the DOM except inside the input field, and should not log it.
    The "Bearer token" *label* on the input is fine — what we're
    forbidding is a serialized ``Authorization: Bearer <value>`` header.
    """
    body = client.get("/ui").text
    assert "Authorization: Bearer" not in body
    assert "SUPPORTSMITH_API_BEARER_TOKEN" not in body
