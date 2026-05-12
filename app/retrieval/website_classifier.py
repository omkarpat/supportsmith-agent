"""LLM-driven page classifier + customer-name extractor for website ingestion.

Both classes accept an :class:`LLMClient` Protocol so unit tests can drop in
a scripted client and a captured fixture without an API key. The raw signals
the extractor consumes (alt text, captions, headings) are deterministically
attached to every chunk by :mod:`app.retrieval.website_chunker` so the
synthesizer can reach for the same data without re-prompting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from app.llm.client import (
    ChatMessage,
    ChatRequest,
    ChatResponseSchema,
    LLMClient,
    ReasoningEffort,
)
from app.prompts import load_prompt

PageType = Literal[
    "case_study",
    "blog",
    "resource",
    "product",
    "service",
    "customer_logo",
    "company",
    "legal",
    "unknown",
]

EvidenceType = Literal[
    "logo_alt_text",
    "case_study_title",
    "case_study_body",
    "customer_page_text",
    "nearby_caption",
]

PAGE_TYPE_VALUES: tuple[PageType, ...] = (
    "case_study",
    "blog",
    "resource",
    "product",
    "service",
    "customer_logo",
    "company",
    "legal",
    "unknown",
)

PRIORITY_PAGE_TYPES: frozenset[PageType] = frozenset(
    {"case_study", "blog", "resource", "product", "service", "customer_logo"}
)

MAX_CLASSIFIER_SNIPPET_CHARS = 2000
MAX_EXTRACTOR_SNIPPET_CHARS = 1500


class PageTypeDecision(BaseModel):
    """Structured output from :class:`WebsitePageClassifier`."""

    model_config = ConfigDict(extra="forbid")

    page_type: PageType
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reason: str = ""


class CustomerExtractionDecision(BaseModel):
    """Structured output from :class:`CustomerNameExtractor`."""

    model_config = ConfigDict(extra="forbid")

    customer_names: list[str] = Field(default_factory=list)
    evidence_types: list[EvidenceType] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


@dataclass(frozen=True)
class WebsiteLLMConfig:
    """Model + reasoning settings for both classifier and extractor."""

    classifier_model: str
    classifier_reasoning_effort: ReasoningEffort | None
    classifier_max_completion_tokens: int
    extractor_model: str
    extractor_reasoning_effort: ReasoningEffort | None
    extractor_max_completion_tokens: int


_CLASSIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "page_type": {"type": "string", "enum": list(PAGE_TYPE_VALUES)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
    },
    "required": ["page_type", "confidence", "reason"],
}

_EXTRACTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "customer_names": {
            "type": "array",
            "items": {"type": "string"},
        },
        "evidence_types": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "logo_alt_text",
                    "case_study_title",
                    "case_study_body",
                    "customer_page_text",
                    "nearby_caption",
                ],
            },
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["customer_names", "evidence_types", "confidence"],
}


class WebsitePageClassifier:
    """Label one Firecrawl page with a :data:`PageType` value via an LLM."""

    def __init__(self, llm: LLMClient, config: WebsiteLLMConfig) -> None:
        self.llm = llm
        self.config = config

    async def classify(
        self,
        *,
        url: str,
        title: str,
        description: str | None,
        markdown: str,
        priority_hint: bool = False,
    ) -> PageTypeDecision:
        prompt = load_prompt("website.classifier")
        snippet = markdown.strip()[:MAX_CLASSIFIER_SNIPPET_CHARS]
        user_content = (
            f"URL: {url}\n"
            f"Title: {title}\n"
            f"Description: {description or '(none)'}\n"
            f"Operator-flagged priority path: {priority_hint}\n\n"
            f"Page content snippet:\n{snippet}"
        )
        response = await self.llm.complete(
            ChatRequest(
                model=self.config.classifier_model,
                reasoning_effort=self.config.classifier_reasoning_effort,
                max_completion_tokens=self.config.classifier_max_completion_tokens,
                response_schema=ChatResponseSchema(
                    name="website_page_classifier",
                    schema_definition=_CLASSIFIER_SCHEMA,
                    strict=False,
                ),
                messages=[
                    ChatMessage(role="system", content=prompt.system),
                    ChatMessage(role="user", content=user_content),
                ],
            )
        )
        return _parse_decision(response.content, PageTypeDecision)


class CustomerNameExtractor:
    """Pull likely customer names off a page using only on-page evidence."""

    def __init__(self, llm: LLMClient, config: WebsiteLLMConfig) -> None:
        self.llm = llm
        self.config = config

    async def extract(
        self,
        *,
        url: str,
        page_title: str,
        page_type: PageType,
        section_headings: tuple[str, ...],
        asset_alt_text: tuple[str, ...],
        nearby_captions: tuple[str, ...],
        body_snippet: str,
    ) -> CustomerExtractionDecision:
        prompt = load_prompt("website.customer_extractor")
        user_content = (
            f"URL: {url}\n"
            f"Page title: {page_title}\n"
            f"Page type: {page_type}\n"
            f"Section headings: {_render_list(section_headings)}\n"
            f"Image alt text: {_render_list(asset_alt_text)}\n"
            f"Nearby captions: {_render_list(nearby_captions)}\n\n"
            f"Body snippet:\n{body_snippet.strip()[:MAX_EXTRACTOR_SNIPPET_CHARS]}"
        )
        response = await self.llm.complete(
            ChatRequest(
                model=self.config.extractor_model,
                reasoning_effort=self.config.extractor_reasoning_effort,
                max_completion_tokens=self.config.extractor_max_completion_tokens,
                response_schema=ChatResponseSchema(
                    name="website_customer_extractor",
                    schema_definition=_EXTRACTOR_SCHEMA,
                    strict=False,
                ),
                messages=[
                    ChatMessage(role="system", content=prompt.system),
                    ChatMessage(role="user", content=user_content),
                ],
            )
        )
        decision = _parse_decision(response.content, CustomerExtractionDecision)
        decision.customer_names = _dedupe_names(decision.customer_names)
        return decision


def _render_list(items: tuple[str, ...]) -> str:
    cleaned = [item.strip() for item in items if item.strip()]
    if not cleaned:
        return "(none)"
    return "; ".join(cleaned[:25])


def _dedupe_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for name in names:
        cleaned = name.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


_TDecision = TypeVar("_TDecision", PageTypeDecision, CustomerExtractionDecision)


def _parse_decision(content: str, model: type[_TDecision]) -> _TDecision:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{model.__name__} returned non-JSON content: {content!r}"
        ) from exc
    return model.model_validate(data)
