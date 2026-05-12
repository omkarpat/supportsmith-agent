"""Bearer-token gate for the public demo deployment.

The bearer token is a *demo* protection layer, not production auth. When
``api_bearer_token`` is configured, every API route except a small public
set requires ``Authorization: Bearer <token>``. The token is compared with
:func:`hmac.compare_digest` to avoid timing side channels and is never
logged.

The public set covers ``/health`` (so platform health checks keep working
without a token) and the chat-UI demo console at ``/`` + ``/ui*`` (so a
fresh visitor can load the page and paste a token; chat actions stay
disabled until they do).

Phase 7 will additively layer an ``X-Admin-Api-Key`` dependency on top of
this for ingestion routes — see ``project_phase_scope`` memory.
"""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.core.config import Settings

PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        "/",
        "/ui",
        "/ui/assets/app.js",
        "/ui/assets/styles.css",
    }
)


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """Reject requests without the configured bearer token.

    No-op when ``api_bearer_token`` is not set, so local development and
    the test suite (which never sets the token) keep working unchanged.
    Mounted early in the middleware stack.
    """

    def __init__(self, app: ASGIApp, *, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)
        header = request.headers.get("authorization", "")
        if not _valid_bearer(header, self._token):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing or invalid bearer token."},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)


def configure_security(app: FastAPI, *, settings: Settings) -> None:
    """Attach the bearer-token middleware when a token is configured."""
    if settings.api_bearer_token:
        app.add_middleware(BearerTokenMiddleware, token=settings.api_bearer_token)


def _valid_bearer(header: str, token: str) -> bool:
    if not header.lower().startswith("bearer "):
        return False
    provided = header.split(" ", 1)[1].strip()
    return _constant_time_eq(provided, token)


def _constant_time_eq(provided: str, expected: str) -> bool:
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


__all__ = [
    "BearerTokenMiddleware",
    "PUBLIC_PATHS",
    "configure_security",
]
