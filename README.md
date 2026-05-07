# SupportSmith Agent

A traceable multi-tool customer support agent with compliance guardrails, self-verification, and multi-turn memory.

SupportSmith is being built for the Knotch AI Engineering take-home. Phase 1 ships the environment, FastAPI skeleton, LLM/eval/agent harness interfaces, Docker support, and required Postgres/pgvector infrastructure so the project can be run locally or hosted on a Railway-style platform early.

## What To Review First

- FastAPI app factory: `app.main:create_app`
- Eval runner: `app.evals.runner`
- Postgres migration: `alembic/versions/20260507_0001_init_pgvector.py`

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

## Docker Development

Use this path when you want Postgres, migrations, and the API all managed by Compose.

```bash
docker compose up --build
```

Compose includes a one-shot `migrate` service, so the local database is upgraded before the API starts.

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

- `support_documents` with `embedding vector(1536)` for FAQ and website/document chunks.
- `conversations` and `conversation_messages`.
- `trace_events` for agent observability.
- `escalations` for handoff records.

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
