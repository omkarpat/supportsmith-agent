"""LLM classifier + customer extractor tests using a scripted LLM client."""

from __future__ import annotations

import json

import pytest

from app.llm.client import ChatResponse, TokenUsage
from app.llm.fake import ScriptedLLMClient
from app.retrieval.website_classifier import (
    CustomerNameExtractor,
    WebsiteLLMConfig,
    WebsitePageClassifier,
)


def _config() -> WebsiteLLMConfig:
    return WebsiteLLMConfig(
        classifier_model="scripted",
        classifier_reasoning_effort="low",
        classifier_max_completion_tokens=256,
        extractor_model="scripted",
        extractor_reasoning_effort="low",
        extractor_max_completion_tokens=512,
    )


def _response(payload: dict[str, object]) -> ChatResponse:
    return ChatResponse(
        content=json.dumps(payload),
        model="scripted",
        usage=TokenUsage(),
    )


async def test_classifier_returns_typed_decision_for_each_label() -> None:
    llm = ScriptedLLMClient(
        [
            _response(
                {"page_type": "case_study", "confidence": 0.9, "reason": "named customer"}
            )
        ]
    )
    classifier = WebsitePageClassifier(llm, _config())

    decision = await classifier.classify(
        url="https://knotch.com/case-studies/acme",
        title="Acme case study",
        description="How Acme used Knotch",
        markdown="# Acme case study\n\nAcme drove engagement.",
    )

    assert decision.page_type == "case_study"
    assert decision.confidence == pytest.approx(0.9)


async def test_classifier_rejects_invalid_label_value() -> None:
    llm = ScriptedLLMClient(
        [_response({"page_type": "homepage", "confidence": 0.5, "reason": "x"})]
    )
    classifier = WebsitePageClassifier(llm, _config())

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        await classifier.classify(
            url="https://example.com/", title="x", description=None, markdown="x"
        )


async def test_extractor_deduplicates_and_preserves_evidence() -> None:
    llm = ScriptedLLMClient(
        [
            _response(
                {
                    "customer_names": ["Acme", "acme", "Initech"],
                    "evidence_types": ["case_study_body", "logo_alt_text"],
                    "confidence": 0.8,
                }
            )
        ]
    )
    extractor = CustomerNameExtractor(llm, _config())

    decision = await extractor.extract(
        url="https://knotch.com/case-studies/acme",
        page_title="Acme case study",
        page_type="case_study",
        section_headings=("Results", "Background"),
        asset_alt_text=("Acme logo",),
        nearby_captions=("Acme is a customer",),
        body_snippet="Acme worked with Knotch.",
    )

    assert decision.customer_names == ["Acme", "Initech"]
    assert "case_study_body" in decision.evidence_types
