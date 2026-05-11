"""Retrieval suite: traditional IR metrics over pgvector cosine search.

Each case is a natural-language query plus the ``reference_docs`` (external
ids of documents that should rank at or near the top). The suite embeds the
query with the same embedding model the live agent uses, runs cosine search
over ``support_documents``, and reports MRR@k, nDCG@k, recall@k, and
precision@k. These four scores become the metrics on a
:class:`ScoreRecord`; the case passes when the weighted score clears
``PASS_THRESHOLD``.

The suite emits a deterministic gate for empty retrieval (``no_response``
cap) when the query returned zero hits — that path is almost always a
configuration or seeding problem, not a ranking miss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.evals.scoring import (
    Evidence,
    Metric,
    ScoreRecord,
    build_record,
    make_gate,
)
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.models import RetrievalResult
from app.retrieval.search import SupportDocumentSearch

DEFAULT_K: int = 5

# Suite-local metric weights. Recall is weighted highest because the agent
# needs the right doc to *be* in the candidate pool; ranking-aware metrics
# (MRR, nDCG) come next; precision is informational on a small KB.
_METRIC_WEIGHTS: dict[str, float] = {
    "recall_at_k": 0.40,
    "mrr_at_k": 0.25,
    "ndcg_at_k": 0.25,
    "precision_at_k": 0.10,
}


class RetrievalCase(BaseModel):
    """One retrieval eval case."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    reference_docs: list[str] = Field(default_factory=list)
    category: str | None = None
    k: int = Field(default=DEFAULT_K, ge=1, le=50)


@dataclass(frozen=True)
class RetrievalSuiteDeps:
    """Per-run dependencies for the retrieval suite."""

    session: AsyncSession
    embeddings: EmbeddingGenerator


def mrr_at_k(retrieved_ids: list[str], reference_ids: set[str], k: int) -> float:
    """Mean reciprocal rank of the first relevant doc in the top-k."""
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in reference_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: list[str], reference_ids: set[str], k: int) -> float:
    """Fraction of reference docs that appear in the top-k."""
    if not reference_ids:
        return 1.0
    top = set(retrieved_ids[:k])
    return len(top & reference_ids) / len(reference_ids)


def precision_at_k(retrieved_ids: list[str], reference_ids: set[str], k: int) -> float:
    """Fraction of the top-k that are relevant."""
    if k == 0:
        return 0.0
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    return sum(1 for doc_id in top if doc_id in reference_ids) / len(top)


def ndcg_at_k(retrieved_ids: list[str], reference_ids: set[str], k: int) -> float:
    """Normalized DCG with binary relevance over the top-k."""
    if not reference_ids:
        return 1.0
    dcg = 0.0
    for index, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in reference_ids:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(reference_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


async def score_case(case: RetrievalCase, deps: RetrievalSuiteDeps) -> ScoreRecord:
    """Run one retrieval case and return its scored record."""
    [embedding] = await deps.embeddings.embed_many([case.query])
    search = SupportDocumentSearch(deps.session)
    hits = await search.search(embedding, limit=case.k, category=case.category)
    retrieved_ids = [hit.external_id for hit in hits]
    reference_ids = set(case.reference_docs)

    gates = []
    if not hits:
        gates.append(
            make_gate(
                "no_response",
                applied=True,
                reason="retrieval returned zero hits",
            )
        )

    metrics = [
        Metric(
            name="recall_at_k",
            score=recall_at_k(retrieved_ids, reference_ids, case.k),
            weight=_METRIC_WEIGHTS["recall_at_k"],
            judge="deterministic",
            rationale=f"recall@{case.k}",
        ),
        Metric(
            name="mrr_at_k",
            score=mrr_at_k(retrieved_ids, reference_ids, case.k),
            weight=_METRIC_WEIGHTS["mrr_at_k"],
            judge="deterministic",
            rationale=f"mrr@{case.k}",
        ),
        Metric(
            name="ndcg_at_k",
            score=ndcg_at_k(retrieved_ids, reference_ids, case.k),
            weight=_METRIC_WEIGHTS["ndcg_at_k"],
            judge="deterministic",
            rationale=f"ndcg@{case.k}",
        ),
        Metric(
            name="precision_at_k",
            score=precision_at_k(retrieved_ids, reference_ids, case.k),
            weight=_METRIC_WEIGHTS["precision_at_k"],
            judge="deterministic",
            rationale=f"precision@{case.k}",
        ),
    ]

    evidence = Evidence(
        retrieved_doc_ids=retrieved_ids,
        extra={
            "query": case.query,
            "reference_docs": sorted(reference_ids),
            "k": case.k,
            "category": case.category,
            "top_hit_scores": [_hit_summary(hit) for hit in hits[:3]],
        },
    )
    return build_record(
        case_id=case.id,
        suite="retrieval",
        target="support_documents",
        metrics=metrics,
        gates=gates,
        evidence=evidence,
    )


def _hit_summary(hit: RetrievalResult) -> dict[str, float | str]:
    return {
        "external_id": hit.external_id,
        "score": round(hit.score, 4),
        "title": hit.title,
    }
