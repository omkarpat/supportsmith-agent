# SupportSmith Agent

A traceable multi-tool customer support agent with compliance guardrails, self-verification, and multi-turn memory.

SupportSmith is being built for the Knotch AI Engineering take-home. Phase 1 shipped the environment, FastAPI skeleton, LLM/eval/agent harness interfaces, Docker support, and required Postgres/pgvector infrastructure so the project can be run locally or hosted on a Railway-style platform early. Phase 2 adds the retrieval layer: typed knowledge-base ingestion, idempotent pgvector seeding with hash-based change detection, and HNSW cosine-similarity search over `support_documents`.

## What To Review First

- FastAPI app factory: `app.main:create_app`
- Eval runner: `app.evals.runner`
- Postgres migration: `alembic/versions/20260507_0001_init_pgvector.py`
- Retrieval layer: `app.retrieval.{models,normalization,chunker,embeddings,repository,search}`
- FAQ source loader: `app.retrieval.sources.take_home_faq`
- Seed CLI: `scripts/db_seed.py` (entrypoint `supportsmith-seed`)

## Local Development

Use this path when you want FastAPI running directly on your machine, with only Postgres running in Docker.

```bash
uv sync
cp .env.example .env
docker compose up -d postgres
uv run --env-file .env supportsmith-migrate
uv run uvicorn app.main:create_app --factory --reload
```

The local `.env` should use the host-mapped Docker database URL:

```bash
DATABASE_URL=postgresql://supportsmith:supportsmith@localhost:55432/supportsmith
```

If the Dockerized app is already running on port `8000`, stop just that service before starting local uvicorn:

```bash
docker compose stop app
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Phase 1 conversation scaffold:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"How do I reset my account?"}'
```

Continue an existing conversation through the convenience chat endpoint:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"demo","message":"I forgot my password"}'
```

Explicit conversation endpoint:

```bash
curl -X POST http://127.0.0.1:8000/conversations/demo/messages \
  -H "Content-Type: application/json" \
  -d '{"message":"How do I reset my account?"}'
```

## Checks

```bash
uv run pytest
uv run ruff check .
uv run mypy app tests alembic
uv run supportsmith-eval
```

The repository and search tests in `tests/test_retrieval_repository.py` are skipped unless a Postgres URL is provided. To exercise them locally:

```bash
docker compose up -d postgres
SUPPORTSMITH_TEST_DATABASE_URL=postgresql://supportsmith:supportsmith@localhost:55432/supportsmith \
    uv run pytest tests/test_retrieval_repository.py
```

## Docker Development

Use this path when you want Postgres, migrations, and the API all managed by Compose.

```bash
docker compose up --build
```

Compose includes one-shot `migrate` and `seed` services, so the local database is upgraded and populated with the take-home FAQ corpus before the API starts. Both services are idempotent: re-running `docker compose up` does not duplicate rows or re-embed unchanged content.

Inside Docker Compose, the app uses the service hostname instead of `localhost`:

```bash
DATABASE_URL=postgresql://supportsmith:supportsmith@postgres:5432/supportsmith
```

Check the Dockerized API the same way:

```bash
curl http://127.0.0.1:8000/health
```

## Standalone Docker Image

The image listens on `${PORT:-8000}` so Railway can inject `PORT`. For a manually run container against the Compose Postgres database:

```bash
docker build -t supportsmith-agent .
docker run --rm -p 8000:8000 \
  -e DATABASE_URL=postgresql://supportsmith:supportsmith@host.docker.internal:55432/supportsmith \
  supportsmith-agent
```

## Postgres + pgvector

SupportSmith always expects Postgres with pgvector in runtime environments. Tests and deterministic evals patch the database boundary with fakes, but the API requires `DATABASE_URL` or `SUPPORTSMITH_DATABASE_URL` when it starts.

Start only the database:

```bash
docker compose up -d postgres
uv run --env-file .env supportsmith-migrate
```

Run the API against local Postgres:

```bash
uv run uvicorn app.main:create_app --factory --reload
```

The initial migration creates:

- `support_documents` with `embedding vector(1536)` for FAQ and website/document chunks, indexed with HNSW cosine ops for fast similarity search at the scale this project actually runs at.
- `conversations` and `conversation_messages`.
- `trace_events` for agent observability.
- `escalations` for handoff records.

## Seeding The Knowledge Base

Phase 2 seeds `support_documents` from a knowledge-base JSON file. The take-home FAQ corpus lives in `data/knowledge-base.json` (committed) with each row carrying a `quality` label of `trusted`, `low_quality`, or `ambiguous`. Only `trusted` rows are ingested; the other rows are surfaced in the run summary so reviewers can see what was filtered and why.

`docker compose up` runs the seed automatically as a one-shot service after `migrate` and before `app`, so a fresh stack comes up with the FAQ corpus already in pgvector. To run the seed manually against a host-side Postgres:

```bash
uv run --env-file .env supportsmith-seed
uv run --env-file .env supportsmith-seed --input path/to/your/knowledge-base.json
```

The CLI prints a JSON run summary with `inserted` / `updated` / `unchanged` / `embedded` counts, the per-row outcomes, and any rejected rows (rows whose `quality` is not `trusted`). Re-running the seed is idempotent: rows whose content hash is unchanged are not re-embedded or rewritten.

Phase 2 ships only the deterministic fake-embedding path so seeding does not spend the take-home OpenAI key. Cosine retrieval is wired end-to-end against pgvector, but ranking quality across user-typed questions requires real embeddings — the OpenAI embedding adapter and a `--live-embeddings` flag are tracked for a later phase.

## Example Data Policy

SupportSmith treats pgvector as a source of truth for citable support knowledge, not as a dumping ground for every take-home example.

- Trusted, answerable FAQ rows are seeded into `support_documents`.
- Pure junk, malformed examples, prompt injections, and routing-only examples are not embedded or stored as support documents.
- Messy inputs such as `x` belong in evals and prompt few-shots as ambiguous-input behavior cases.
- Sensitive examples such as `help!!! my account is locked` belong in compliance/routing evals, not in the retrieval KB as answer sources.
- Future Knotch website chunks should only enter pgvector after cleaning, chunking, and source metadata validation.

This keeps retrieval grounded in citable facts while still using the messy examples to test how the agent clarifies, refuses, or escalates.

For Railway, provision Postgres, set `DATABASE_URL`, deploy the Dockerfile, then run `uv run supportsmith-migrate` as a one-off command.

## Environment

Copy `.env.example` to `.env` for local configuration. Phase 1 uses fake clients by default, so an OpenAI key is not required for tests or the deterministic eval runner.
