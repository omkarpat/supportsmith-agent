"""Stable text normalization and content hashing."""

import hashlib
import re
import unicodedata

from app.retrieval.models import SeedDocument

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Return a stable, embedding-friendly form of ``text``.

    Lowercases, NFKC-normalizes unicode, and collapses internal whitespace.
    Punctuation is preserved so that questions remain readable to the
    embedding model; surrounding whitespace and case are not meaningful for
    semantic search and would otherwise destabilize content hashes.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE_RE.sub(" ", folded).strip()


def compute_content_hash(document: SeedDocument) -> str:
    """Return a sha256 hash that changes only when meaningful fields change."""
    parts = [
        document.source,
        document.external_id,
        document.title,
        document.content,
        document.embedding_text,
        document.source_url or "",
        document.category or "",
    ]
    payload = "␞".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
