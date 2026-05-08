"""Structured-output types for the verifier.

The verifier itself is inlined into the ``verify`` graph node in
:mod:`app.agent.nodes` (matching the planner / synthesizer pattern: single
caller, single LLM call). Only the typed output shape lives here so other
modules — like :class:`app.agent.state.GraphState` — can reference the
verdict without pulling in node code.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

GroundingLabel = Literal[
    "faq_grounded",
    "general_marked",
    "clarification",
    "escalation",
    "refusal",
    "unsupported",
]

RetryRecommendation = Literal["accept", "repair", "escalate", "refuse"]


class VerifierOutput(BaseModel):
    """Structured verifier verdict on one synthesized candidate."""

    model_config = ConfigDict(extra="forbid")

    addresses_request: bool
    grounding: GroundingLabel
    leakage_detected: bool
    safe_source_label: bool
    retry_recommendation: RetryRecommendation
    reason: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
