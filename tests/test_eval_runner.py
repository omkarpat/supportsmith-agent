"""Tests for the Phase 6 eval CLI and suite plumbing.

The runner always exercises the live agent in production, but the suite
scoring logic and the case-loader plumbing are deterministic functions that
we can cover without a live OpenAI key or a Postgres instance. Live runs
are exercised by ``-m live`` tests and the manual ``supportsmith-eval``
command, not by this file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.harness import AgentRequest, AgentResponse
from app.evals.runner import build_parser, load_cases
from app.evals.scoring import (
    PASS_THRESHOLD,
    Evidence,
    Gate,
    Metric,
    build_record,
    compute_overall_score,
    did_pass,
    make_gate,
    summarize_suite,
)
from app.evals.suites.e2e import (
    E2ECase,
    E2ESuiteDeps,
    ExpectedToolCall,
)
from app.evals.suites.e2e import score_case as score_e2e_case
from app.evals.suites.retrieval import (
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

CASES_PATH = Path(__file__).resolve().parent.parent / "evals" / "cases.yaml"


# --- scoring core ------------------------------------------------------------


def test_compute_overall_score_weighted_average() -> None:
    metrics = [
        Metric(name="a", score=1.0, weight=2.0, judge="deterministic"),
        Metric(name="b", score=0.0, weight=2.0, judge="deterministic"),
    ]
    assert compute_overall_score(metrics, []) == 0.5


def test_compute_overall_score_no_metrics_returns_one() -> None:
    """Deterministic-only paths pass on gate-clearance alone."""
    assert compute_overall_score([], []) == 1.0


def test_compute_overall_score_applies_strictest_gate_cap() -> None:
    metrics = [Metric(name="a", score=1.0, weight=1.0, judge="deterministic")]
    gates = [
        make_gate("hallucinated_citation", applied=True),
        make_gate("critical_guardrail_miss", applied=True),
    ]
    # critical_guardrail_miss (0.30) is stricter than hallucinated_citation (0.60).
    assert compute_overall_score(metrics, gates) == pytest.approx(0.30)


def test_unapplied_gate_does_not_cap() -> None:
    metrics = [Metric(name="a", score=1.0, weight=1.0, judge="deterministic")]
    gates = [make_gate("invalid_schema", applied=False)]
    assert compute_overall_score(metrics, gates) == 1.0


def test_did_pass_threshold() -> None:
    assert did_pass(PASS_THRESHOLD) is True
    assert did_pass(PASS_THRESHOLD - 0.0001) is False


def test_build_record_sets_passed_and_rounds_score() -> None:
    record = build_record(
        case_id="c1",
        suite="retrieval",
        target="support_documents",
        metrics=[Metric(name="m", score=0.875, weight=1.0, judge="deterministic")],
    )
    assert record.overall_score == 0.875
    assert record.passed is True
    assert record.suite == "retrieval"


def test_summarize_suite_counts_pass_and_fail() -> None:
    high = build_record(
        case_id="pass",
        suite="e2e",
        target="agent",
        metrics=[Metric(name="m", score=1.0, weight=1.0, judge="deterministic")],
    )
    low = build_record(
        case_id="fail",
        suite="e2e",
        target="agent",
        metrics=[Metric(name="m", score=0.0, weight=1.0, judge="deterministic")],
    )
    summary = summarize_suite(suite="e2e", records=[high, low])
    assert summary.passed == 1
    assert summary.failed == 1
    assert summary.average_score == 0.5


def test_make_gate_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        make_gate("not-a-real-gate", applied=True)


def test_make_gate_returns_typed_gate() -> None:
    gate = make_gate("no_response", applied=True, reason="empty")
    assert isinstance(gate, Gate)
    assert gate.cap == 0.0
    assert gate.applied is True


# --- retrieval IR metric math ------------------------------------------------


def test_mrr_at_k_picks_first_relevant_position() -> None:
    assert mrr_at_k(["a", "b", "c"], {"c"}, k=5) == pytest.approx(1 / 3)
    assert mrr_at_k(["a", "b"], {"c"}, k=5) == 0.0


def test_recall_at_k_handles_empty_reference() -> None:
    assert recall_at_k(["a"], set(), k=5) == 1.0


def test_recall_at_k_counts_top_k_only() -> None:
    assert recall_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0
    assert recall_at_k(["a", "b", "c"], {"c"}, k=3) == 1.0


def test_precision_at_k_zero_when_no_hits() -> None:
    assert precision_at_k([], {"a"}, k=5) == 0.0
    assert precision_at_k(["a", "b"], {"a"}, k=2) == 0.5


def test_ndcg_at_k_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)


def test_ndcg_at_k_penalizes_lower_rank() -> None:
    perfect = ndcg_at_k(["a", "b"], {"a"}, k=2)
    deferred = ndcg_at_k(["b", "a"], {"a"}, k=2)
    assert deferred < perfect


# --- e2e suite scoring -------------------------------------------------------


class _StubAgent:
    """Minimal Agent Protocol stub used to exercise e2e scoring deterministically."""

    def __init__(self, response: AgentResponse) -> None:
        self._response = response
        self.calls: list[AgentRequest] = []

    async def respond(self, request: AgentRequest) -> AgentResponse:
        self.calls.append(request)
        return self._response


def _agent_response(
    *,
    text: str = "Reset your password from Account Settings.",
    source: str = "faq",
    tools_used: list[str] | None = None,
    verified: bool = True,
) -> AgentResponse:
    return AgentResponse(
        conversation_id="eval",
        turn_number=1,
        response=text,
        source=source,  # type: ignore[arg-type]
        matched_questions=[],
        tools_used=tools_used or ["search_kb"],
        verified=verified,
        trace_id="trace-1",
    )


async def test_e2e_score_case_passes_when_all_expectations_met() -> None:
    case = E2ECase.model_validate(
        {
            "id": "stub-faq",
            "tags": ["happy_path"],
            "conversation_id": "eval-stub",
            "turns": [
                {
                    "user": "How do I reset my password?",
                    "expected_tool_calls": [{"name": "search_kb"}],
                    "expectations": {
                        "source": "faq",
                        "verified": True,
                        "max_tool_calls": 2,
                    },
                }
            ],
        }
    )
    agent = _StubAgent(_agent_response())

    records = await score_e2e_case(case, E2ESuiteDeps(agent=agent, judge=None))

    assert len(records) == 1
    assert records[0].passed is True
    assert records[0].overall_score == pytest.approx(1.0)
    assert records[0].evidence.source == "faq"


async def test_e2e_score_case_fails_when_source_mismatches() -> None:
    case = E2ECase.model_validate(
        {
            "id": "stub-source-mismatch",
            "conversation_id": "eval-stub",
            "turns": [
                {
                    "user": "How do I reset my password?",
                    "expectations": {"source": "faq", "verified": True},
                }
            ],
        }
    )
    agent = _StubAgent(_agent_response(source="general"))

    records = await score_e2e_case(case, E2ESuiteDeps(agent=agent, judge=None))

    assert records[0].passed is False
    source_metric = next(m for m in records[0].metrics if m.name == "source_match")
    assert source_metric.score == 0.0


async def test_e2e_score_case_applies_no_response_gate_on_empty_text() -> None:
    case = E2ECase.model_validate(
        {
            "id": "stub-empty",
            "conversation_id": "eval-stub",
            "turns": [{"user": "hi"}],
        }
    )
    agent = _StubAgent(_agent_response(text=""))

    records = await score_e2e_case(case, E2ESuiteDeps(agent=agent, judge=None))

    assert records[0].passed is False
    assert any(g.applied and g.name == "no_response" for g in records[0].gates)
    assert records[0].overall_score == 0.0


async def test_e2e_score_case_propagates_prior_turns_across_a_case() -> None:
    case = E2ECase.model_validate(
        {
            "id": "stub-multi",
            "conversation_id": "eval-stub-multi",
            "turns": [
                {"user": "first ask", "expectations": {"verified": True}},
                {"user": "follow-up", "expectations": {"verified": True}},
            ],
        }
    )
    agent = _StubAgent(_agent_response())

    records = await score_e2e_case(case, E2ESuiteDeps(agent=agent, judge=None))

    # Both turns produce records; second turn must have run with prior context
    # populated on the agent.
    assert len(records) == 2
    assert agent.calls[0].message == "first ask"
    assert agent.calls[1].message == "follow-up"
    assert agent._pending_turn_number == 2  # type: ignore[attr-defined]
    assert len(agent._pending_prior_turns) == 1  # type: ignore[attr-defined]


async def test_e2e_expected_tools_metric_uses_set_intersection() -> None:
    case = E2ECase.model_validate(
        {
            "id": "stub-tools",
            "conversation_id": "eval-stub",
            "turns": [
                {
                    "user": "go",
                    "expected_tool_calls": [
                        ExpectedToolCall(name="search_kb").model_dump(),
                        ExpectedToolCall(name="get_faq_by_category").model_dump(),
                    ],
                }
            ],
        }
    )
    agent = _StubAgent(_agent_response(tools_used=["search_kb"]))

    records = await score_e2e_case(case, E2ESuiteDeps(agent=agent, judge=None))

    coverage = next(m for m in records[0].metrics if m.name == "expected_tools_called")
    assert coverage.score == 0.5


# --- runner plumbing ---------------------------------------------------------


def test_load_cases_parses_real_cases_file() -> None:
    e2e_cases, retrieval_cases = load_cases(CASES_PATH)
    assert len(e2e_cases) >= 13, "expected at least 13 e2e cases per the Phase 6 plan"
    assert any("happy_path" in case.tags for case in e2e_cases)
    assert any("multi_turn" in case.tags for case in e2e_cases)
    assert any("malicious" in case.tags for case in e2e_cases)
    assert any("ambiguous" in case.tags for case in e2e_cases)
    assert any("off_topic" in case.tags for case in e2e_cases)
    assert retrieval_cases, "retrieval suite must have at least one case"


def test_load_cases_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "missing.yaml")


def test_build_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.suite == "all"
    assert args.judge == "none"
    assert args.target == "agent"
    assert args.model is None
    assert str(args.cases) == "evals/cases.yaml"


def test_evidence_default_shape() -> None:
    evidence = Evidence()
    assert evidence.trace_nodes == []
    assert evidence.tool_calls == []
    assert evidence.retrieved_doc_ids == []
    assert evidence.response_text is None
