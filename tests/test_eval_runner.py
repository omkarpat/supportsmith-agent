from pathlib import Path

from app.agent.harness import PhaseOneAgent
from app.evals.runner import load_cases, run_cases, run_file


def test_eval_runner_loads_and_scores_cases(tmp_path: Path) -> None:
    eval_file = tmp_path / "cases.yaml"
    eval_file.write_text(
        """
cases:
  - id: phase-one-smoke
    conversation_id: eval-1
    message: hello
    expectations:
      response_contains: Phase 1 harness is online
      source: agent
      tools_used:
        - phase_one_agent
      verified: true
""".strip()
    )

    cases = load_cases(eval_file)

    assert len(cases) == 1
    assert cases[0].id == "phase-one-smoke"


async def test_eval_runner_reports_passes() -> None:
    cases = load_cases(Path("evals/cases.yaml"))

    summary = await run_cases(PhaseOneAgent(), cases)

    assert summary.total == 1
    assert summary.passed == 1
    assert summary.failed == 0


async def test_run_file_uses_phase_one_agent() -> None:
    summary = await run_file(Path("evals/cases.yaml"))

    assert summary.failed == 0

