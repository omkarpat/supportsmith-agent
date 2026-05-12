"""One end-to-end smoke test that hits the real OpenAI API.

This test is opt-in: it's marked ``live`` and is excluded from the default
pytest run (see ``addopts`` in ``pyproject.toml``). It spends a small amount
of OpenAI credit on every run, so we keep it tightly scoped and run it
manually as a pre-merge / pre-release sanity check.

To run, start the compose stack with the seeded Postgres and then export
the test DSN before invoking pytest with the live marker — see the
README's "Test policy" section for the exact commands.

Requires:
    - OPENAI_API_KEY (or SUPPORTSMITH_OPENAI_API_KEY) set in the environment
    - SUPPORTSMITH_TEST_DATABASE_URL pointing at a Postgres that has already
      been seeded with live OpenAI embeddings (compose's seed service does
      this on `docker compose up`)

The test never runs in the default pytest invocation. Adapter and graph
behavior are exhaustively covered by mocked tests; this one only verifies
that the live wiring works end-to-end against real services.
"""

from __future__ import annotations

import os

import pytest

from app.agent.harness import AgentRequest
from app.agent.wiring import build_live_support_agent
from app.core.config import Settings
from app.db.session import create_engine, create_session_factory
from app.retrieval.search import SupportDocumentSearch

DATABASE_URL = os.environ.get("SUPPORTSMITH_TEST_DATABASE_URL")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("SUPPORTSMITH_OPENAI_API_KEY")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not DATABASE_URL,
        reason="Set SUPPORTSMITH_TEST_DATABASE_URL to a seeded Postgres to run live tests.",
    ),
    pytest.mark.skipif(
        not OPENAI_KEY,
        reason="Set OPENAI_API_KEY to run live tests.",
    ),
]


async def test_password_reset_question_returns_grounded_faq_answer() -> None:
    """A real planner + retrieval + synthesizer turn lands the password-reset FAQ.

    Asserts the contract that matters end-to-end:
    - The agent picks up the question and routes through search_kb
    - Live embeddings rank the password-reset FAQ as the top match
    - The synthesizer cites it via matched_questions (structured output)
    - The user-facing prose is non-empty and free of inline citation noise
    """
    assert DATABASE_URL is not None
    assert OPENAI_KEY is not None

    settings = Settings(
        environment="local",
        database_url=DATABASE_URL,
        openai_api_key=OPENAI_KEY,
    )

    engine = create_engine(DATABASE_URL)
    factory = create_session_factory(engine)
    session = factory()
    try:
        search = SupportDocumentSearch(session)
        agent = await build_live_support_agent(settings, search=search)

        response = await agent.respond(
            AgentRequest(conversation_id="live-e2e", message="How do I reset my password?")
        )
    finally:
        await session.close()
        await engine.dispose()

    assert response.source == "faq", f"expected source=faq, got {response.source}"
    assert response.tools_used == ["search_kb"], (
        f"expected tools_used=[search_kb], got {response.tools_used}"
    )
    assert "What steps do I take to reset my password?" in response.matched_questions, (
        f"expected the password-reset FAQ in matched_questions, got {response.matched_questions}"
    )
    assert len(response.response) > 20, f"response prose too short: {response.response!r}"
    # The new structured-synthesis path puts citations in matched_questions, not in
    # the body. Make sure no inline source marker leaked into the user-facing text.
    assert 'Source: "' not in response.response
    assert response.verified is True
    assert response.trace_id.startswith("turn_")
    assert response.cost.total_tokens > 0, "live turn should report token usage"
