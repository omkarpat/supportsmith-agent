"""LLM-as-judge client for eval suites.

The judge is a thin wrapper around :class:`app.llm.client.LLMClient`. Each
call loads a rubric YAML from ``evals/rubrics/``, sends it as a system
prompt, hands the rendered case evidence to the model as the user message,
and returns a typed ``JudgeVerdict``. The verdict shape (``score``,
``rationale``, ``confidence``) is constant across rubrics — what varies is
the system prompt that tells the model *what* to grade.

Configuration comes from ``Settings.judge_model`` / ``_reasoning_effort`` /
``_max_completion_tokens``. The CLI ``--model`` flag overrides only the
model id. Suites should construct one :class:`LLMJudge` per run and reuse
it across cases; callers are expected to skip judge calls entirely when
``--judge none`` is in effect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings, get_settings
from app.llm.client import (
    ChatMessage,
    ChatRequest,
    ChatResponseSchema,
    LLMClient,
    ReasoningEffort,
)

RUBRICS_DIR: Path = Path(__file__).resolve().parent.parent.parent / "evals" / "rubrics"


class Rubric(BaseModel):
    """One reviewable rubric for LLM-as-judge scoring.

    Same shape as :class:`app.prompts.Prompt` so reviewers can scan rubrics
    the same way they scan agent prompts. ``criteria`` is a short
    reviewer-facing list of the dimensions the system prompt should grade;
    it is documentation only — the LLM reads ``system``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: int = Field(ge=1)
    system: str = Field(min_length=1)
    criteria: list[str] = Field(default_factory=list)
    notes: str = ""


@lru_cache
def load_rubric(name: str) -> Rubric:
    """Load a YAML rubric by name (e.g. ``correctness``)."""
    full_path = RUBRICS_DIR / f"{name}.yaml"
    if not full_path.is_file():
        raise FileNotFoundError(f"Rubric not found: {full_path}")
    raw = yaml.safe_load(full_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Rubric YAML must be a mapping at the top level: {full_path}")
    raw.setdefault("name", name)
    return Rubric.model_validate(raw)


class JudgeVerdict(BaseModel):
    """Structured judge output. The shared shape across every rubric."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["score", "rationale", "confidence"],
}


@dataclass(frozen=True)
class JudgeConfig:
    """Per-run judge config; populated from settings, optionally overridden by CLI."""

    model: str
    reasoning_effort: ReasoningEffort | None = "low"
    max_completion_tokens: int = 1024

    @classmethod
    def from_settings(
        cls, settings: Settings, *, model_override: str | None = None
    ) -> JudgeConfig:
        return cls(
            model=model_override or settings.judge_model,
            reasoning_effort=settings.judge_reasoning_effort,
            max_completion_tokens=settings.judge_max_completion_tokens,
        )


class LLMJudge:
    """Score one rubric over one case-evidence payload."""

    def __init__(self, llm: LLMClient, config: JudgeConfig) -> None:
        self.llm = llm
        self.config = config

    async def score(
        self,
        *,
        rubric: str | Rubric,
        context: dict[str, Any],
    ) -> JudgeVerdict:
        """Grade ``context`` against ``rubric`` and return a typed verdict.

        ``context`` is rendered as compact JSON in the user turn. Suites are
        responsible for choosing which evidence fields are relevant for the
        rubric (e.g. correctness needs ``reference_answer``; tool_efficiency
        needs ``expected_tool_calls`` + ``tool_calls``).
        """
        rubric_obj = rubric if isinstance(rubric, Rubric) else load_rubric(rubric)
        user_content = json.dumps(context, ensure_ascii=False, indent=2, default=str)
        response = await self.llm.complete(
            ChatRequest(
                model=self.config.model,
                reasoning_effort=self.config.reasoning_effort,
                max_completion_tokens=self.config.max_completion_tokens,
                response_schema=ChatResponseSchema(
                    name=f"rubric_{rubric_obj.name}",
                    schema_definition=_JUDGE_SCHEMA,
                    strict=False,
                ),
                messages=[
                    ChatMessage(role="system", content=rubric_obj.system),
                    ChatMessage(role="user", content=user_content),
                ],
            )
        )
        return _parse_verdict(response.content)


def _parse_verdict(content: str) -> JudgeVerdict:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"judge returned non-JSON content: {content!r}") from exc
    return JudgeVerdict.model_validate(data)


def build_default_judge(
    llm: LLMClient,
    *,
    model_override: str | None = None,
    settings: Settings | None = None,
) -> LLMJudge:
    """Construct an :class:`LLMJudge` from app settings + optional CLI override."""
    settings = settings or get_settings()
    return LLMJudge(llm, JudgeConfig.from_settings(settings, model_override=model_override))
