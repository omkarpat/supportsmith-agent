"""ComplianceAgent: pre-graph and post-graph safety / policy gate.

The compliance agent is intentionally separate from the planner's ``refuse``
tool. ``refuse`` is a cheap in-loop gatekeeper the planner can pick to skip
unnecessary work; the compliance agent is a dedicated safety check that runs
twice per turn — once before planning (precheck) and once after synthesis
(postcheck) — with its own prompts, schema, and trace events.

Per the Phase 4 doc, compliance precheck and postcheck have different jobs:
  - precheck evaluates the *user message* and hard-blocks injection / harm
    / severe sensitive-account requests so the agent never plans on them.
  - postcheck evaluates the *candidate answer* for safety, leakage, and
    policy override; quality and grounding belong to the verifier.

Both run on the configured routing model with low reasoning effort so the
gate is fast and cheap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.llm.client import (
    ChatMessage,
    ChatRequest,
    ChatResponseSchema,
    LLMClient,
    ReasoningEffort,
)
from app.prompts import load_prompt

ComplianceCategory = Literal[
    "support_allowed",
    "support_adjacent_general",
    "ambiguous_support",
    "unsupported_off_topic",
    "prompt_injection",
    "sensitive_account",
    "harmful_or_illegal",
    "unknown",
]

ComplianceMode = Literal["precheck", "postcheck"]

# Categories that hard-block the user turn at precheck. The doc lists prompt
# injection, harmful/illegal, and severe sensitive-account exfiltration. We
# include ``sensitive_account`` here only when the precheck itself sets
# ``allowed=false``; the precheck prompt is responsible for distinguishing
# benign account questions from severe ones.
HARD_BLOCK_CATEGORIES: frozenset[ComplianceCategory] = frozenset(
    {
        "prompt_injection",
        "harmful_or_illegal",
    }
)


class ComplianceDecision(BaseModel):
    """Structured output from a compliance precheck or postcheck."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    category: ComplianceCategory
    reason: str = ""
    override_response: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


_COMPLIANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "allowed": {"type": "boolean"},
        "category": {
            "type": "string",
            "enum": [
                "support_allowed",
                "support_adjacent_general",
                "ambiguous_support",
                "unsupported_off_topic",
                "prompt_injection",
                "sensitive_account",
                "harmful_or_illegal",
                "unknown",
            ],
        },
        "reason": {"type": "string"},
        "override_response": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["allowed", "category", "reason", "confidence"],
}


@dataclass(frozen=True)
class ComplianceConfig:
    """Per-agent runtime config; injected once at graph compile time."""

    model: str
    reasoning_effort: ReasoningEffort | None = "low"
    max_completion_tokens: int = 512


class ComplianceAgent:
    """Run a precheck on the user message or a postcheck on the candidate."""

    def __init__(self, llm: LLMClient, config: ComplianceConfig) -> None:
        self.llm = llm
        self.config = config

    async def precheck(self, user_message: str) -> ComplianceDecision:
        """Classify the user message before the main graph runs."""
        return await self._run(
            mode="precheck",
            user_content=f'User message: "{user_message}"',
            schema_name="compliance_precheck",
        )

    async def postcheck(
        self,
        *,
        user_message: str,
        candidate_answer: str,
        candidate_source: str,
    ) -> ComplianceDecision:
        """Classify the candidate answer before it goes back to the user."""
        rendered = (
            f'User message: "{user_message}"\n\n'
            f"Candidate answer source label: {candidate_source}\n\n"
            f"Candidate answer text:\n{candidate_answer}"
        )
        return await self._run(
            mode="postcheck",
            user_content=rendered,
            schema_name="compliance_postcheck",
        )

    async def _run(
        self,
        *,
        mode: ComplianceMode,
        user_content: str,
        schema_name: str,
    ) -> ComplianceDecision:
        prompt = load_prompt(f"compliance.{mode}")
        response = await self.llm.complete(
            ChatRequest(
                model=self.config.model,
                reasoning_effort=self.config.reasoning_effort,
                max_completion_tokens=self.config.max_completion_tokens,
                response_schema=ChatResponseSchema(
                    name=schema_name,
                    schema_definition=_COMPLIANCE_SCHEMA,
                    strict=False,
                ),
                messages=[
                    ChatMessage(role="system", content=prompt.system),
                    ChatMessage(role="user", content=user_content),
                ],
            )
        )
        return _parse_decision(response.content)


def _parse_decision(content: str) -> ComplianceDecision:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"compliance returned non-JSON content: {content!r}") from exc
    return ComplianceDecision.model_validate(data)


def is_hard_block(decision: ComplianceDecision) -> bool:
    """Return True when a precheck decision should short-circuit to refusal."""
    if decision.allowed:
        return False
    return decision.category in HARD_BLOCK_CATEGORIES or decision.category == "sensitive_account"
