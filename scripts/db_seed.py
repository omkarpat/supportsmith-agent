"""Seed ``support_documents`` with the take-home FAQ corpus.

Run from the repo root::

    uv run --env-file .env supportsmith-seed
    uv run --env-file .env supportsmith-seed --input data/knowledge-base.json
    uv run supportsmith-seed --fake-embeddings   # CI / tests; no API key

Defaults to live OpenAI embeddings so the seeded vectors match the model the
live agent uses for queries. Pass ``--fake-embeddings`` for deterministic
local seeds where no API key is configured (the agent wiring also has to use
the matching fake embedder for those vectors to compare meaningfully).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import get_settings
from app.db.session import create_engine, create_session_factory
from app.llm.client import EmbeddingClient
from app.llm.fake import FakeEmbeddingClient
from app.llm.openai import OpenAIEmbeddingClient
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.models import SeedDocument, UpsertSummary
from app.retrieval.repository import SupportDocumentRepository
from app.retrieval.sources.take_home_faq import load_take_home_faq

DEFAULT_INPUT = Path("data/knowledge-base.json")
EMBEDDING_DIMENSIONS = 1536


async def seed(
    *,
    input_path: Path,
    database_url: str,
    embedding_client: EmbeddingClient,
    embedding_model: str,
) -> UpsertSummary:
    """Load, embed, and upsert the take-home FAQ into ``support_documents``."""
    loaded = load_take_home_faq(input_path)
    summary = UpsertSummary(rejected=list(loaded.rejected), started_at=_utcnow())

    generator = EmbeddingGenerator(embedding_client, model=embedding_model)

    engine = create_engine(database_url)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session, session.begin():
            repository = SupportDocumentRepository(session)
            for document in loaded.documents:
                outcome = await repository.upsert(
                    document,
                    embed=lambda doc, gen=generator: _embed_document(gen, doc),
                )
                summary.outcomes.append(outcome)
                if outcome.action == "inserted":
                    summary.inserted += 1
                elif outcome.action == "updated":
                    summary.updated += 1
                else:
                    summary.unchanged += 1
                if outcome.embedded:
                    summary.embedded += 1
    finally:
        await engine.dispose()

    summary.finished_at = _utcnow()
    return summary


async def _embed_document(generator: EmbeddingGenerator, document: SeedDocument) -> list[float]:
    return (await generator.embed_many([document.embedding_text]))[0]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def build_parser() -> argparse.ArgumentParser:
    """Build the seed CLI argument parser."""
    parser = argparse.ArgumentParser(description="Seed pgvector with the take-home FAQ corpus.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the knowledge-base JSON file. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help=(
            "Use deterministic fake embeddings instead of OpenAI. Useful for "
            "CI and tests. The live agent must be configured with the matching "
            "fake embedder for retrieval to behave consistently."
        ),
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    settings = get_settings()

    embedding_client: EmbeddingClient
    if args.fake_embeddings:
        embedding_client = FakeEmbeddingClient(dimensions=EMBEDDING_DIMENSIONS)
    else:
        if not settings.openai_api_key:
            raise SystemExit(
                "Live embeddings require OPENAI_API_KEY. Re-run with "
                "--fake-embeddings to seed deterministic vectors instead."
            )
        embedding_client = OpenAIEmbeddingClient(api_key=settings.openai_api_key)

    summary = asyncio.run(
        seed(
            input_path=args.input,
            database_url=settings.database_url,
            embedding_client=embedding_client,
            embedding_model=settings.embedding_model,
        )
    )
    print(json.dumps(summary.model_dump(mode="json"), indent=2, default=str))


if __name__ == "__main__":
    main()
