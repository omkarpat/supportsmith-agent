"""Eval runner unit tests using an in-line stub agent.

The runner only requires ``Agent``-shaped objects, so the tests use a tiny
``StubAgent`` that emits canned responses. Phase 6 will land a real live-agent
runner; this test file just covers loading + scoring + summary plumbing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.harness import AgentRequest, AgentResponse
from app.evals.runner import build_parser, load_cases, main, run_cases, run_file


class StubAgent:
    def __init__(self, *, response_text: str = "stub answer") -> None:
        self.response_text = response_text
        self.calls: list[AgentRequest] = []

    async def respond(self, request: AgentRequest) -> AgentResponse:
        self.calls.append(request)
        return AgentResponse(
            conversation_id=request.conversation_id,
            response=self.response_text,
            source="agent",
            tools_used=["stub"],
            verified=True,
            trace_id=f"trace_{len(self.calls):04d}",
        )


def test_load_cases_parses_yaml(tmp_path: Path) -> None:
    eval_file = tmp_path / "cases.yaml"
    eval_file.write_text(
        """
cases:
  - id: smoke
    conversation_id: eval-1
    message: hello
    expectations:
      response_contains: stub answer
      source: agent
      tools_used: [stub]
      verified: true
""".strip()
    )

    cases = load_cases(eval_file)

    assert len(cases) == 1
    assert cases[0].id == "smoke"


async def test_run_cases_scores_passes_against_stub_agent(tmp_path: Path) -> None:
    eval_file = tmp_path / "cases.yaml"
    eval_file.write_text(
        """
cases:
  - id: stub-pass
    conversation_id: eval-1
    message: hello
    expectations:
      response_contains: stub answer
      verified: true
""".strip()
    )
    cases = load_cases(eval_file)

    summary = await run_cases(StubAgent(), cases)

    assert summary.total == 1
    assert summary.passed == 1
    assert summary.failed == 0


async def test_run_cases_reports_failure_when_response_does_not_contain_expected() -> None:
    cases = load_cases_from_inline(
        """
cases:
  - id: stub-fail
    conversation_id: eval-1
    message: hello
    expectations:
      response_contains: not in the stub answer
""".strip()
    )

    summary = await run_cases(StubAgent(), cases)

    assert summary.passed == 0
    assert summary.failed == 1
    assert "did not contain" in summary.results[0].failures[0]


async def test_run_file_dispatches_through_supplied_agent(tmp_path: Path) -> None:
    eval_file = tmp_path / "cases.yaml"
    eval_file.write_text(
        """
cases:
  - id: dispatch
    conversation_id: eval-1
    message: hi
    expectations:
      response_contains: stub answer
""".strip()
    )

    summary = await run_file(eval_file, StubAgent())

    assert summary.passed == 1


def test_main_refuses_to_run_in_phase_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["supportsmith-eval"])

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert "Phase 6" in str(excinfo.value)


def test_build_parser_defaults_to_evals_cases_yaml() -> None:
    parser = build_parser()
    args = parser.parse_args([])

    assert args.path == "evals/cases.yaml"


def load_cases_from_inline(text: str) -> list:  # type: ignore[type-arg]
    """Helper to load cases from an inline string without writing to disk."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(text)
        path = Path(handle.name)
    return load_cases(path)
