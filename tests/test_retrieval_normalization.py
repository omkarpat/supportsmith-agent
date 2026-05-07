from app.retrieval.models import SeedDocument
from app.retrieval.normalization import compute_content_hash, normalize_text


def test_normalize_collapses_whitespace_and_lowercases() -> None:
    assert normalize_text("  Reset   Password  \n now ") == "reset password now"


def test_normalize_is_unicode_stable() -> None:
    composed = "café"
    decomposed = "café"

    assert normalize_text(composed) == normalize_text(decomposed)


def _seed(**overrides: object) -> SeedDocument:
    base: dict[str, object] = {
        "external_id": "take_home_faq:reset-password-001",
        "source": "take_home_faq",
        "title": "Reset password",
        "content": "Q: Reset password\n\nA: Use settings.",
        "embedding_text": "reset password use settings",
        "category": "security",
    }
    base.update(overrides)
    return SeedDocument.model_validate(base)


def test_content_hash_is_stable_across_calls() -> None:
    seed = _seed()

    assert compute_content_hash(seed) == compute_content_hash(seed)


def test_content_hash_changes_when_meaningful_field_changes() -> None:
    base = _seed()
    different_title = _seed(title="Reset your password")
    different_content = _seed(content="Q: Reset password\n\nA: Different answer.")
    different_embedding_text = _seed(embedding_text="reset password different answer")
    different_category = _seed(category="profile")

    base_hash = compute_content_hash(base)

    assert compute_content_hash(different_title) != base_hash
    assert compute_content_hash(different_content) != base_hash
    assert compute_content_hash(different_embedding_text) != base_hash
    assert compute_content_hash(different_category) != base_hash


def test_content_hash_is_unaffected_by_metadata() -> None:
    base = _seed()
    with_metadata = _seed(metadata={"ordinal": 0})

    assert compute_content_hash(base) == compute_content_hash(with_metadata)
