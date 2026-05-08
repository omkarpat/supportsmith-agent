"""Prompt loader for SupportSmith.

Prompts live as YAML under ``prompts/`` so the wording is reviewable in
isolation from the orchestration code. JSON schemas stay alongside the
Pydantic models that consume them — only the natural-language system
prompt and a small block of reviewer-facing metadata move into YAML.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROMPTS_DIR: Path = Path(__file__).resolve().parent.parent / "prompts"


class Prompt(BaseModel):
    """One reviewable prompt.

    ``name`` is the dotted path used by callers (e.g. ``planner`` or
    ``compliance.precheck``). ``version`` is bumped when the wording changes
    in a way reviewers should re-read; it is not used for behavior gating.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: int = Field(ge=1)
    system: str = Field(min_length=1)
    notes: str = ""


@lru_cache
def load_prompt(name: str) -> Prompt:
    """Load a YAML prompt by dotted name (e.g. ``compliance.precheck``)."""
    relative_path = name.replace(".", "/") + ".yaml"
    full_path = PROMPTS_DIR / relative_path
    if not full_path.is_file():
        raise FileNotFoundError(f"Prompt not found: {full_path}")
    raw = yaml.safe_load(full_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Prompt YAML must be a mapping at the top level: {full_path}")
    raw.setdefault("name", name)
    return Prompt.model_validate(raw)
