"""Unified scoring primitives used by every eval suite.

Every suite emits the same shape (:class:`ScoreRecord`) so the CLI summary
and any downstream tooling can read them without per-suite glue. The
``overall_score`` is a weighted average of :class:`Metric` scores, capped
by any active hard :class:`Gate` (which absolutely beats raw metric math —
"no response" overrides whatever the rubrics say).

A case passes when ``overall_score >= 0.80`` after caps apply.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

JudgeSource = Literal["deterministic", "llm"]
SuiteName = Literal["node", "compliance", "retrieval", "e2e"]

PASS_THRESHOLD: float = 0.80


class Metric(BaseModel):
    """One weighted contribution to ``overall_score``.

    ``judge`` distinguishes deterministic checks (1.0 when satisfied, 0.0
    when not) from LLM-as-judge rubric scores (anywhere in ``[0, 1]``).
    ``weight`` is normalized across all metrics on a record so the relative
    importance of metrics within a suite is preserved without callers
    needing to sum to exactly 1.0.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0)
    judge: JudgeSource
    rationale: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Gate(BaseModel):
    """A hard-cap gate. ``applied=True`` means the score was capped this run.

    Hard gates beat metrics: an answer that crashed the agent gets
    ``overall_score = 0.0`` regardless of how forgiving the rubrics are.
    The cap rules are documented in :data:`HARD_GATES`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    applied: bool
    cap: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class Evidence(BaseModel):
    """Observed evidence that informed the score.

    Suites populate the subset of fields relevant to them; missing fields
    stay at their typed defaults so downstream consumers can read the shape
    without per-suite case branches.
    """

    model_config = ConfigDict(extra="forbid")

    trace_nodes: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    response_text: str | None = None
    source: str | None = None
    verified: bool | None = None
    status: str | None = None
    turn_number: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ScoreRecord(BaseModel):
    """One scored eval case, regardless of suite."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    suite: SuiteName
    target: str = Field(min_length=1)
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    metrics: list[Metric] = Field(default_factory=list)
    gates: list[Gate] = Field(default_factory=list)
    evidence: Evidence = Field(default_factory=Evidence)


# --- gate catalog ------------------------------------------------------------


class GateSpec(BaseModel):
    """Static definition of a hard gate; instantiated as ``Gate`` per record."""

    model_config = ConfigDict(extra="forbid")

    name: str
    cap: float
    description: str


HARD_GATES: dict[str, GateSpec] = {
    "no_response": GateSpec(
        name="no_response",
        cap=0.00,
        description="No response or runtime crash",
    ),
    "invalid_schema": GateSpec(
        name="invalid_schema",
        cap=0.40,
        description="Invalid schema or malformed JSON",
    ),
    "critical_guardrail_miss": GateSpec(
        name="critical_guardrail_miss",
        cap=0.30,
        description="Critical guardrail miss",
    ),
    "leakage": GateSpec(
        name="leakage",
        cap=0.30,
        description="Prompt/tool/secret leakage",
    ),
    "hallucinated_citation": GateSpec(
        name="hallucinated_citation",
        cap=0.60,
        description="Hallucinated source citation",
    ),
    "loop_breach": GateSpec(
        name="loop_breach",
        cap=0.50,
        description="Unbounded loop or max iteration breach",
    ),
}


def make_gate(name: str, *, applied: bool, reason: str = "") -> Gate:
    """Build a :class:`Gate` from the named :class:`GateSpec`."""
    if name not in HARD_GATES:
        raise KeyError(f"Unknown gate: {name!r}")
    spec = HARD_GATES[name]
    return Gate(name=spec.name, applied=applied, cap=spec.cap, reason=reason)


# --- score computation -------------------------------------------------------


def compute_overall_score(metrics: list[Metric], gates: list[Gate]) -> float:
    """Return the weighted-average score, capped by any applied hard gate.

    Caps win: when a gate is applied, the returned score is ``min(weighted,
    gate.cap)`` for the strictest applied cap. With no metrics, the score is
    1.0 (deterministic-only paths can pass on gate-clearance alone).
    """
    if metrics:
        total_weight = sum(metric.weight for metric in metrics)
        if total_weight <= 0:
            weighted = 0.0
        else:
            weighted = sum(metric.score * metric.weight for metric in metrics) / total_weight
    else:
        # Deterministic-only run (no LLM judge, no rubric scoring): clearing
        # every gate is enough to score 1.0. The hard-gate caps below pull
        # the score back down on any real failure.
        weighted = 1.0

    applied_caps = [gate.cap for gate in gates if gate.applied]
    if applied_caps:
        return min(weighted, min(applied_caps))
    return weighted


def did_pass(overall_score: float) -> bool:
    """A case passes when ``overall_score >= PASS_THRESHOLD`` (0.80)."""
    return overall_score >= PASS_THRESHOLD


def build_record(
    *,
    case_id: str,
    suite: SuiteName,
    target: str,
    metrics: list[Metric] | None = None,
    gates: list[Gate] | None = None,
    evidence: Evidence | None = None,
) -> ScoreRecord:
    """Assemble a :class:`ScoreRecord` and finalize the pass / score values."""
    metrics = metrics or []
    gates = gates or []
    overall = compute_overall_score(metrics, gates)
    return ScoreRecord(
        case_id=case_id,
        suite=suite,
        target=target,
        overall_score=round(overall, 4),
        passed=did_pass(overall),
        metrics=metrics,
        gates=gates,
        evidence=evidence or Evidence(),
    )


class SuiteSummary(BaseModel):
    """Aggregate result for one suite."""

    model_config = ConfigDict(extra="forbid")

    suite: SuiteName
    total: int
    passed: int
    failed: int
    average_score: float
    records: list[ScoreRecord]


class RunSummary(BaseModel):
    """Top-level CLI output across one or more suites."""

    model_config = ConfigDict(extra="forbid")

    target: str
    judge: Literal["none", "llm"]
    suites: list[SuiteSummary]

    @property
    def all_passed(self) -> bool:
        return all(record.passed for suite in self.suites for record in suite.records)


def summarize_suite(*, suite: SuiteName, records: list[ScoreRecord]) -> SuiteSummary:
    """Roll up per-case records into a suite-level summary."""
    passed = sum(1 for record in records if record.passed)
    average = (
        sum(record.overall_score for record in records) / len(records) if records else 1.0
    )
    return SuiteSummary(
        suite=suite,
        total=len(records),
        passed=passed,
        failed=len(records) - passed,
        average_score=round(average, 4),
        records=records,
    )
