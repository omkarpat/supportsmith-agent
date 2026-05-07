"""Deterministic Phase 1 eval runner."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.agent.harness import AgentRequest, AgentResponse, PhaseOneAgent


class EvalExpectations(BaseModel):
    """Expected properties for one eval case."""

    model_config = ConfigDict(extra="forbid")

    response_contains: str | None = None
    source: str | None = None
    tools_used: list[str] | None = None
    verified: bool | None = None


class EvalCase(BaseModel):
    """One deterministic eval case."""

    model_config = ConfigDict(extra="forbid")

    id: str
    conversation_id: str
    message: str = Field(min_length=1)
    expectations: EvalExpectations


class EvalCaseResult(BaseModel):
    """Result for one eval case."""

    model_config = ConfigDict(extra="forbid")

    id: str
    passed: bool
    failures: list[str] = Field(default_factory=list)
    response: AgentResponse


class EvalSummary(BaseModel):
    """Aggregate eval run summary."""

    model_config = ConfigDict(extra="forbid")

    total: int
    passed: int
    failed: int
    results: list[EvalCaseResult]


def load_cases(path: Path) -> list[EvalCase]:
    """Load eval cases from a YAML file."""
    raw = yaml.safe_load(path.read_text()) or {}
    cases = raw.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("Eval file must contain a list at `cases`.")
    return [EvalCase.model_validate(case) for case in cases]


async def run_cases(agent: PhaseOneAgent, cases: list[EvalCase]) -> EvalSummary:
    """Run eval cases against a support agent."""
    results = []
    for case in cases:
        response = await agent.respond(
            AgentRequest(conversation_id=case.conversation_id, message=case.message)
        )
        failures = score_response(response, case.expectations)
        results.append(
            EvalCaseResult(
                id=case.id,
                passed=not failures,
                failures=failures,
                response=response,
            )
        )

    passed = sum(1 for result in results if result.passed)
    return EvalSummary(
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        results=results,
    )


def score_response(response: AgentResponse, expectations: EvalExpectations) -> list[str]:
    """Return expectation failures for one agent response."""
    failures: list[str] = []
    if expectations.response_contains and expectations.response_contains not in response.response:
        failures.append(f"response did not contain {expectations.response_contains!r}")
    if expectations.source and response.source != expectations.source:
        failures.append(f"source expected {expectations.source!r}, got {response.source!r}")
    if expectations.tools_used is not None and response.tools_used != expectations.tools_used:
        failures.append(
            f"tools_used expected {expectations.tools_used!r}, got {response.tools_used!r}"
        )
    if expectations.verified is not None and response.verified is not expectations.verified:
        failures.append(f"verified expected {expectations.verified!r}, got {response.verified!r}")
    return failures


async def run_file(path: Path) -> EvalSummary:
    """Run eval cases from a YAML file using the Phase 1 fake agent."""
    cases = load_cases(path)
    return await run_cases(PhaseOneAgent(), cases)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run SupportSmith eval cases.")
    parser.add_argument(
        "path",
        nargs="?",
        default="evals/cases.yaml",
        help="Path to eval YAML cases.",
    )
    return parser


def main() -> None:
    """CLI entrypoint for deterministic evals."""
    args = build_parser().parse_args()
    summary = asyncio.run(run_file(Path(args.path)))
    payload: dict[str, Any] = summary.model_dump(mode="json")
    print(json.dumps(payload, indent=2))
    if summary.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
