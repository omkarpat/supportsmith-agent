"""SupportSmith eval CLI.

Runs the e2e and retrieval suites against the live agent and prints a single
JSON :class:`RunSummary`. The runner always exercises the real agent — the
``--judge`` flag only controls whether LLM-as-judge rubric scores augment
the deterministic gates.

Required environment for any run:

- ``SUPPORTSMITH_OPENAI_API_KEY`` (or ``OPENAI_API_KEY``)
- ``SUPPORTSMITH_DATABASE_URL`` (or ``DATABASE_URL``) pointing at a seeded
  Postgres + pgvector instance.

The runner exits non-zero if any case's ``passed`` field is false. Reviewer
quickstart:

::

    uv run supportsmith-eval                              # all suites, deterministic
    uv run supportsmith-eval --suite retrieval            # IR metrics only
    uv run supportsmith-eval --suite e2e --judge llm      # add rubric scoring
    uv run supportsmith-eval --target api \\
        --base-url http://127.0.0.1:8000                  # smoke a running service
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Literal

import yaml

from app.agent.harness import Agent, AgentRequest, AgentResponse, CostSummary
from app.core.config import Settings, get_settings
from app.db.session import create_engine, create_session_factory
from app.evals.judge import LLMJudge, build_default_judge
from app.evals.scoring import (
    RunSummary,
    ScoreRecord,
    SuiteSummary,
    summarize_suite,
)
from app.evals.suites.e2e import E2ECase, E2ESuiteDeps
from app.evals.suites.e2e import score_case as score_e2e_case
from app.evals.suites.retrieval import RetrievalCase, RetrievalSuiteDeps
from app.evals.suites.retrieval import score_case as score_retrieval_case
from app.llm.openai import OpenAIChatCompletionsClient, OpenAIEmbeddingClient
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.search import SupportDocumentSearch

SuiteChoice = Literal["e2e", "retrieval", "all"]
JudgeChoice = Literal["none", "llm"]
TargetChoice = Literal["agent", "api"]

DEFAULT_CASES_PATH: Path = Path("evals/cases.yaml")
DEFAULT_BASE_URL: str = "http://127.0.0.1:8000"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="supportsmith-eval",
        description="Run SupportSmith eval suites and print a JSON summary.",
    )
    parser.add_argument(
        "--suite",
        choices=("e2e", "retrieval", "all"),
        default="all",
        help="Which suite to run (default: all).",
    )
    parser.add_argument(
        "--judge",
        choices=("none", "llm"),
        default="none",
        help="Add LLM-as-judge rubric scoring on top of deterministic gates.",
    )
    parser.add_argument(
        "--target",
        choices=("agent", "api"),
        default="agent",
        help="Drive the in-process agent or a running FastAPI service.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the LLM judge model (does not affect the agent's own models).",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=f"Path to the YAML case file (default: {DEFAULT_CASES_PATH}).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL for --target api (default: {DEFAULT_BASE_URL}).",
    )
    return parser


# --- case loading ------------------------------------------------------------


def load_cases(path: Path) -> tuple[list[E2ECase], list[RetrievalCase]]:
    """Load and validate ``e2e_cases`` and ``retrieval_cases`` from a YAML file."""
    if not path.is_file():
        raise FileNotFoundError(f"Eval cases file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Eval cases YAML must be a mapping at the top level: {path}")

    e2e_block = raw.get("e2e_cases", []) or []
    ret_block = raw.get("retrieval_cases", []) or []
    if not isinstance(e2e_block, list) or not isinstance(ret_block, list):
        raise ValueError("e2e_cases and retrieval_cases must be lists")

    e2e_cases = [E2ECase.model_validate(item) for item in e2e_block]
    retrieval_cases = [RetrievalCase.model_validate(item) for item in ret_block]
    return e2e_cases, retrieval_cases


# --- live agent / api target -------------------------------------------------


async def _build_agent_target(settings: Settings) -> Agent:
    """Construct the in-process live agent (used for ``--target agent``)."""
    # Imported lazily so a CLI invocation that only targets the API doesn't
    # pay the live-agent import cost (and so retrieval-only runs can skip
    # the chat-model probe).
    from app.agent.wiring import build_live_support_agent

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    # Hold one long-lived session for the duration of the eval run so the
    # retrieval search and the live agent share the same pgvector view.
    session = session_factory()
    search = SupportDocumentSearch(session)
    agent = await build_live_support_agent(settings, search=search)
    return _SessionHoldingAgent(agent=agent, session=session, engine=engine)


class _SessionHoldingAgent:
    """Wraps a SupportAgent and keeps the DB engine/session alive for the run."""

    def __init__(self, *, agent: Agent, session: Any, engine: Any) -> None:
        self._agent = agent
        self._session = session
        self._engine = engine

    async def respond(self, request: AgentRequest) -> AgentResponse:
        return await self._agent.respond(request)

    async def aclose(self) -> None:
        await self._session.close()
        await self._engine.dispose()


class HttpAgent:
    """Agent Protocol adapter that posts to a running ``/chat`` endpoint."""

    def __init__(self, base_url: str) -> None:
        try:
            import httpx
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised in live runs
            raise RuntimeError(
                "--target api requires the `httpx` package. Install it with "
                "`uv add httpx` (or `pip install httpx`)."
            ) from exc
        self._httpx = httpx
        self._base_url = base_url.rstrip("/")
        self._client: Any | None = None

    async def _client_or_create(self) -> Any:
        if self._client is None:
            self._client = self._httpx.AsyncClient(base_url=self._base_url, timeout=60.0)
        return self._client

    async def respond(self, request: AgentRequest) -> AgentResponse:
        client = await self._client_or_create()
        resp = await client.post(
            "/chat",
            json={
                "conversation_id": request.conversation_id,
                "message": request.message,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        return AgentResponse(
            conversation_id=body.get("conversation_id", request.conversation_id),
            turn_number=body.get("turn_number", 0),
            response=body.get("response", ""),
            source=body.get("source", "agent"),
            matched_questions=body.get("matched_questions", []) or [],
            tools_used=body.get("tools_used", []) or [],
            verified=bool(body.get("verified", False)),
            trace_id=body.get("trace_id", ""),
            cost=CostSummary.model_validate(body.get("cost", {}) or {}),
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


# --- runner ------------------------------------------------------------------


async def _run_e2e(
    cases: list[E2ECase],
    agent: Agent,
    judge: LLMJudge | None,
) -> list[ScoreRecord]:
    deps = E2ESuiteDeps(agent=agent, judge=judge)
    records: list[ScoreRecord] = []
    for case in cases:
        case_records = await score_e2e_case(case, deps)
        records.extend(case_records)
    return records


async def _run_retrieval(
    cases: list[RetrievalCase],
    settings: Settings,
) -> list[ScoreRecord]:
    if not settings.openai_api_key:
        raise RuntimeError("Retrieval suite requires OPENAI_API_KEY for query embeddings.")
    embedding_client = OpenAIEmbeddingClient(api_key=settings.openai_api_key)
    embeddings = EmbeddingGenerator(embedding_client, model=settings.embedding_model)

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    records: list[ScoreRecord] = []
    try:
        async with session_factory() as session:
            deps = RetrievalSuiteDeps(session=session, embeddings=embeddings)
            for case in cases:
                records.append(await score_retrieval_case(case, deps))
    finally:
        await engine.dispose()
    return records


async def run(
    *,
    suite: SuiteChoice,
    judge_mode: JudgeChoice,
    target: TargetChoice,
    cases_path: Path,
    base_url: str,
    model_override: str | None,
    settings: Settings | None = None,
) -> RunSummary:
    """Drive the requested suites and return the assembled :class:`RunSummary`."""
    settings = settings or get_settings()
    _check_required_env(settings)

    e2e_cases, retrieval_cases = load_cases(cases_path)
    suite_summaries: list[SuiteSummary] = []

    # E2E suite — needs the agent.
    if suite in ("e2e", "all") and e2e_cases:
        judge: LLMJudge | None = None
        if judge_mode == "llm":
            judge_client = OpenAIChatCompletionsClient(api_key=settings.openai_api_key or "")
            judge = build_default_judge(
                judge_client, model_override=model_override, settings=settings
            )
        agent = await _build_target(target=target, base_url=base_url, settings=settings)
        try:
            e2e_records = await _run_e2e(e2e_cases, agent, judge)
        finally:
            await _maybe_close(agent)
        suite_summaries.append(summarize_suite(suite="e2e", records=e2e_records))

    # Retrieval suite — needs a DB session + embeddings, not the agent.
    if suite in ("retrieval", "all") and retrieval_cases:
        ret_records = await _run_retrieval(retrieval_cases, settings)
        suite_summaries.append(summarize_suite(suite="retrieval", records=ret_records))

    return RunSummary(
        target=target,
        judge=judge_mode,
        suites=suite_summaries,
    )


async def _build_target(
    *, target: TargetChoice, base_url: str, settings: Settings
) -> Agent:
    if target == "api":
        return HttpAgent(base_url)
    return await _build_agent_target(settings)


async def _maybe_close(agent: Agent) -> None:
    closer = getattr(agent, "aclose", None)
    if closer is not None:
        await closer()


def _check_required_env(settings: Settings) -> None:
    missing: list[str] = []
    if not settings.openai_api_key:
        missing.append("SUPPORTSMITH_OPENAI_API_KEY (or OPENAI_API_KEY)")
    if not settings.database_url:
        missing.append("SUPPORTSMITH_DATABASE_URL (or DATABASE_URL)")
    if missing:
        raise SystemExit(
            "Missing required environment variables for eval run: " + ", ".join(missing)
        )


def _to_payload(summary: RunSummary) -> dict[str, Any]:
    return summary.model_dump(mode="json")


def main() -> None:
    """CLI entrypoint. Prints a JSON :class:`RunSummary` and exits non-zero on failure."""
    args = build_parser().parse_args()
    try:
        summary = asyncio.run(
            run(
                suite=args.suite,
                judge_mode=args.judge,
                target=args.target,
                cases_path=args.cases,
                base_url=args.base_url,
                model_override=args.model,
            )
        )
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - reviewer-facing error path
        print(f"supportsmith-eval failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(json.dumps(_to_payload(summary), indent=2, default=str))
    if not summary.all_passed:
        raise SystemExit(1)


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_CASES_PATH",
    "HttpAgent",
    "build_parser",
    "load_cases",
    "main",
    "run",
]


if __name__ == "__main__":
    main()
