"""Convert Firecrawl pages to citable :class:`SeedDocument` chunks.

A page's markdown is split on heading boundaries, then long sections are
sliced into ~800-1,200-token chunks with light overlap. Every chunk carries
the page title and section heading at the top of ``embedding_text`` so
retrieval surfaces the right page even for fragmentary semantic matches.

The chunker also surfaces the raw signals the spec requires us to persist
deterministically — image alt text, captions, section headings — onto every
chunk's content + metadata. The LLM customer-name extractor and the
synthesizer can then operate on data that's already on disk, not on data the
LLM has to re-derive at answer time.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.firecrawl import FirecrawlPage
from app.retrieval.normalization import normalize_text

CHUNK_TARGET_TOKENS = 1000
CHUNK_MAX_TOKENS = 1200
CHUNK_MIN_TOKENS = 80
CHUNK_OVERLAP_TOKENS = 120
TOKEN_TO_CHAR_RATIO = 4  # rough text-to-token heuristic; avoids a tokenizer dep
TARGET_CHARS = CHUNK_TARGET_TOKENS * TOKEN_TO_CHAR_RATIO
MAX_CHARS = CHUNK_MAX_TOKENS * TOKEN_TO_CHAR_RATIO
MIN_CHARS = CHUNK_MIN_TOKENS * TOKEN_TO_CHAR_RATIO
OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * TOKEN_TO_CHAR_RATIO

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_CAPTION_RE = re.compile(r"^\s*[*_]([^\n*_]{4,200})[*_]\s*$", re.MULTILINE)
_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)^\s*cookie", re.MULTILINE),
    re.compile(r"(?i)^\s*we use cookies"),
    re.compile(r"(?i)privacy policy", re.MULTILINE),
)


class WebsiteChunk(BaseModel):
    """One chunk of a website page, ready for embed + upsert."""

    model_config = ConfigDict(extra="forbid")

    chunk_index: int
    chunk_count: int
    title: str
    content: str
    embedding_text: str
    section_heading: str | None
    headings_path: tuple[str, ...] = ()
    asset_alt_text: tuple[str, ...] = ()
    nearby_captions: tuple[str, ...] = ()


class WebsitePageSplit(BaseModel):
    """All chunks produced from one page, with shared page-level metadata."""

    model_config = ConfigDict(extra="forbid")

    url: str
    page_title: str
    page_description: str | None
    chunks: list[WebsiteChunk] = Field(default_factory=list)
    skip_reason: str | None = None


def split_page(page: FirecrawlPage) -> WebsitePageSplit:
    """Return the chunked representation of a Firecrawl page."""
    title = (page.title or "").strip() or _derive_title_from_markdown(page.markdown)
    cleaned = _strip_boilerplate(page.markdown)
    if _is_low_value(cleaned):
        return WebsitePageSplit(
            url=page.url,
            page_title=title,
            page_description=page.description,
            skip_reason="empty_or_low_value",
        )

    sections = _split_into_sections(cleaned)
    raw_chunks: list[dict[str, Any]] = []
    for section in sections:
        for body in _slice_section(section["body"]):
            raw_chunks.append(
                {
                    "section_heading": section["heading"],
                    "headings_path": section["headings_path"],
                    "body": body,
                }
            )

    if not raw_chunks:
        return WebsitePageSplit(
            url=page.url,
            page_title=title,
            page_description=page.description,
            skip_reason="no_chunks",
        )

    chunk_count = len(raw_chunks)
    chunks: list[WebsiteChunk] = []
    for index, raw in enumerate(raw_chunks):
        body = raw["body"]
        alt_texts = tuple(alt for alt, _ in _IMAGE_RE.findall(body) if alt.strip())
        captions = tuple(match.strip() for match in _CAPTION_RE.findall(body))
        section_heading = raw["section_heading"]
        headings_path = tuple(raw["headings_path"])
        chunk_content = _render_chunk_content(
            title=title,
            section_heading=section_heading,
            body=body,
        )
        embedding_text = _render_embedding_text(
            title=title,
            section_heading=section_heading,
            body=body,
        )
        chunks.append(
            WebsiteChunk(
                chunk_index=index,
                chunk_count=chunk_count,
                title=title,
                content=chunk_content,
                embedding_text=embedding_text,
                section_heading=section_heading,
                headings_path=headings_path,
                asset_alt_text=alt_texts,
                nearby_captions=captions,
            )
        )
    return WebsitePageSplit(
        url=page.url,
        page_title=title,
        page_description=page.description,
        chunks=chunks,
    )


def _split_into_sections(markdown: str) -> list[dict[str, Any]]:
    """Split markdown into heading-led sections preserving the heading path."""
    matches = list(_HEADING_RE.finditer(markdown))
    if not matches:
        return [{"heading": None, "headings_path": (), "body": markdown.strip()}]

    sections: list[dict[str, Any]] = []
    # Preamble before the first heading still counts as a section.
    preamble = markdown[: matches[0].start()].strip()
    if preamble:
        sections.append({"heading": None, "headings_path": (), "body": preamble})

    heading_stack: list[tuple[int, str]] = []
    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading_text))
        headings_path = tuple(text for _, text in heading_stack)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[match.end():end].strip()
        if not body:
            continue
        sections.append(
            {
                "heading": heading_text,
                "headings_path": headings_path,
                "body": body,
            }
        )
    return sections


def _slice_section(text: str) -> list[str]:
    """Split a long section into chunks with overlap on paragraph boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= MAX_CHARS:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0
    for paragraph in paragraphs:
        if buffer_len + len(paragraph) > TARGET_CHARS and buffer:
            chunks.append("\n\n".join(buffer))
            # Carry the tail of the previous chunk forward as overlap so
            # cross-paragraph claims stay retrievable.
            overlap = _tail(buffer, OVERLAP_CHARS)
            buffer = [overlap] if overlap else []
            buffer_len = len(overlap)
        buffer.append(paragraph)
        buffer_len += len(paragraph) + 2
    if buffer:
        chunks.append("\n\n".join(buffer))
    return [chunk for chunk in chunks if len(chunk) >= MIN_CHARS]


def _tail(buffer: list[str], limit: int) -> str:
    """Return the last ``limit`` chars of joined ``buffer``."""
    if not buffer:
        return ""
    joined = "\n\n".join(buffer)
    if len(joined) <= limit:
        return joined
    return joined[-limit:]


def _strip_boilerplate(markdown: str) -> str:
    cleaned = markdown
    for pattern in _BOILERPLATE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned.strip()


def _is_low_value(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 200:
        return True
    # If headings + images make up 90%+ of the text, treat as nav/menu garbage.
    informative = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", stripped)
    informative = re.sub(r"^[#*\-\s]+$", "", informative, flags=re.MULTILINE)
    return len(informative.strip()) < 150


def _derive_title_from_markdown(markdown: str) -> str:
    for match in _HEADING_RE.finditer(markdown):
        if int(len(match.group(1))) == 1:
            return match.group(2).strip()
    # Fall back to first heading of any level, else a placeholder.
    first_heading = next(_HEADING_RE.finditer(markdown), None)
    if first_heading is not None:
        return first_heading.group(2).strip()
    return "Untitled page"


def _render_chunk_content(
    *,
    title: str,
    section_heading: str | None,
    body: str,
) -> str:
    parts = [f"# {title}"]
    if section_heading and section_heading != title:
        parts.append(f"## {section_heading}")
    parts.append(body.strip())
    return "\n\n".join(parts)


def _render_embedding_text(
    *,
    title: str,
    section_heading: str | None,
    body: str,
) -> str:
    header = title
    if section_heading and section_heading != title:
        header = f"{title} - {section_heading}"
    return normalize_text(f"{header}\n{body}")
