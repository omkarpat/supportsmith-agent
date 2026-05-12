"""Static routes for the Phase 8 chat-UI demo.

Three public surfaces:

  - ``GET /``                 — friendly redirect to ``/ui``
  - ``GET /ui``               — single-page HTML shell
  - ``GET /ui/assets/{name}`` — ``app.js`` / ``styles.css``

The bearer-token middleware (Phase 7) treats ``/ui*`` as protected by
default, but the shell is harmless without a token: chat actions stay
disabled until the user supplies one in the in-page input.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, Response
from starlette import status

router = APIRouter(tags=["web"])

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Whitelist of files we will serve from ``/ui/assets/{name}``. Anything not
# in this map returns 404 so the route can never be coerced into reading an
# arbitrary path on disk.
_ASSETS: dict[str, tuple[str, str]] = {
    "app.js": ("app.js", "application/javascript; charset=utf-8"),
    "styles.css": ("styles.css", "text/css; charset=utf-8"),
}


@router.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Send bare-domain hits to the demo UI so Railway has a friendly landing."""
    return RedirectResponse(url="/ui", status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@router.get("/ui", include_in_schema=False)
async def ui_index() -> FileResponse:
    """Serve the single-page HTML shell."""
    return FileResponse(
        STATIC_DIR / "index.html",
        media_type="text/html; charset=utf-8",
    )


@router.get("/ui/assets/{asset_name}", include_in_schema=False)
async def ui_asset(asset_name: str) -> Response:
    """Serve the JS/CSS that ``/ui`` references."""
    entry = _ASSETS.get(asset_name)
    if entry is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    filename, media_type = entry
    return FileResponse(STATIC_DIR / filename, media_type=media_type)


__all__ = ["router", "STATIC_DIR"]
