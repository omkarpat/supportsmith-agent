"""Website markdown chunker tests."""

from __future__ import annotations

from app.retrieval.firecrawl import FirecrawlPage
from app.retrieval.website_chunker import split_page


def _page(markdown: str, **kwargs: object) -> FirecrawlPage:
    base = {
        "url": "https://knotch.com/case-studies/acme",
        "title": "Acme case study",
        "description": "How Acme drove engagement with Knotch",
        "markdown": markdown,
    }
    base.update(kwargs)  # type: ignore[arg-type]
    return FirecrawlPage.model_validate(base)


def test_skips_low_value_pages() -> None:
    page = _page("# Tiny\n\nshort")
    split = split_page(page)
    assert split.skip_reason == "empty_or_low_value"
    assert split.chunks == []


def test_preserves_headings_in_chunk_content_and_embedding_text() -> None:
    markdown = (
        "# Acme case study\n\n"
        "Acme is a customer.\n\n"
        "## Results\n\n"
        + "Acme increased engagement by 40%. " * 30
    )
    split = split_page(_page(markdown))
    assert split.chunks, "expected at least one chunk"
    chunk = split.chunks[-1]
    assert chunk.title == "Acme case study"
    assert chunk.section_heading == "Results"
    assert "Acme case study" in chunk.content
    assert "results" in chunk.embedding_text
    assert "acme" in chunk.embedding_text


def test_chunker_indexes_and_counts_chunks() -> None:
    long_body = "Paragraph " + "x " * 600 + "\n\n"  # exceeds MAX_CHARS
    markdown = "# Big page\n\n" + long_body * 6
    split = split_page(_page(markdown))
    assert len(split.chunks) >= 2
    assert split.chunks[0].chunk_index == 0
    assert split.chunks[0].chunk_count == len(split.chunks)
    assert {chunk.chunk_count for chunk in split.chunks} == {len(split.chunks)}


def test_extracts_alt_text_and_captions_onto_chunks() -> None:
    markdown = (
        "# Knotch customers\n\n"
        "![Acme logo](https://cdn.example/acme.png)\n\n"
        "*Acme uses Knotch to drive engagement.*\n\n"
        + "More body text " * 80
    )
    split = split_page(_page(markdown))
    assert split.chunks
    first = split.chunks[0]
    assert "Acme logo" in first.asset_alt_text
    assert any("Acme" in caption for caption in first.nearby_captions)


def test_image_only_pages_are_skipped() -> None:
    markdown = "# logos\n\n" + "\n\n".join(
        f"![logo{i}](https://x/{i}.png)" for i in range(30)
    )
    split = split_page(_page(markdown))
    assert split.skip_reason == "empty_or_low_value"
