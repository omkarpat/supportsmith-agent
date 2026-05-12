"""Phase 8 chat-UI demo module.

Serves a minimal HTML/CSS/JS chat console that drives the existing ``/chat``
and ``/conversations`` endpoints. The UI is intentionally static — no Node
build step — so the same Uvicorn process that serves the API also serves
the demo console on Railway.
"""

from app.web.routes import router

__all__ = ["router"]
