from pathlib import Path

import pytest

from app.retrieval.sources.take_home_faq import KnowledgeBaseFile, load_take_home_faq

FIXTURE = Path("tests/fixtures/take_home_faq_subset.json")


def test_loader_keeps_trusted_rows_and_records_rejections() -> None:
    loaded = load_take_home_faq(FIXTURE)

    assert {doc.title for doc in loaded.documents} == {
        "What steps do I take to reset my password?",
        "How do I export my data?",
    }
    assert {item.title for item in loaded.rejected} == {
        "help!!! my account is locked",
        "x",
    }
    assert {item.reason for item in loaded.rejected} == {
        "quality=low_quality",
        "quality=ambiguous",
    }


def test_loader_assigns_stable_external_ids_and_metadata() -> None:
    first = load_take_home_faq(FIXTURE)
    second = load_take_home_faq(FIXTURE)

    assert [doc.external_id for doc in first.documents] == [
        doc.external_id for doc in second.documents
    ]
    assert all(doc.source == "faq" for doc in first.documents)
    assert all(doc.external_id.startswith("take_home_faq:") for doc in first.documents)
    assert all(doc.metadata["dataset"] == "take_home_faq" for doc in first.documents)
    assert {doc.metadata["ordinal"] for doc in first.documents} == {0, 1}


def test_loader_preserves_category_and_normalizes_embedding_text() -> None:
    loaded = load_take_home_faq(FIXTURE)

    by_title = {doc.title: doc for doc in loaded.documents}
    reset = by_title["What steps do I take to reset my password?"]

    assert reset.category == "security"
    assert reset.content.startswith("Q: ")
    assert "A: " in reset.content
    assert reset.embedding_text == reset.embedding_text.lower()
    assert "  " not in reset.embedding_text


def test_loader_rejects_unknown_quality_values() -> None:
    with pytest.raises(ValueError):
        KnowledgeBaseFile.model_validate(
            {
                "knowledge_base_items": [
                    {
                        "question": "q",
                        "answer": "a",
                        "category": "x",
                        "quality": "made_up",
                    }
                ]
            }
        )
