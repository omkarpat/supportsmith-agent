# SupportSmith Agent

A traceable multi-tool customer support agent with compliance guardrails, self-verification, and multi-turn memory.

SupportSmith is being built for the Knotch AI Engineering take-home. Phase 1 shipped the environment, FastAPI skeleton, LLM/eval/agent harness interfaces, Docker support, and required Postgres/pgvector infrastructure so the project can be run locally or hosted on a Railway-style platform early. Phase 2 added the retrieval layer: typed knowledge-base ingestion, idempotent pgvector seeding with hash-based change detection, and HNSW cosine-similarity search over `support_documents`. Phase 3 wired the LangGraph agent: a structured plan → execute → observe → synthesize → verify → finalize loop driven by the OpenAI Chat Completions API, with six typed tools, a 6-iteration cap, and per-turn structured traces. Phase 4 adds compliance and verification: a separate Compliance Agent runs precheck (before the graph) and postcheck (after synthesis) to gate on safety / leakage / policy, an inlined verifier checks grounding and recommends accept / repair / escalate / refuse with a single-repair budget, and every refusal path emits one canonical string while the response payload distinguishes which gate caught the request.

## What To Review First

- FastAPI app factory: `app.main:create_app`
- Agent graph: `app.agent.{state,nodes,graph,runner}` — read `graph.py` first for the edge map
- Tool registry: `app.agent.tools` — six typed tools dispatched by JSON-schema-constrained plans
- Compliance Agent: `app.agent.compliance` — precheck + postcheck gates, stateless LLM wrapper
- Verifier: inlined in the `verify` node in `app.agent.nodes`; structured-output types live in `app.agent.verifier`
- Refusal policy: `app.agent.policy` — `CANONICAL_REFUSAL` shared by every refusal path
- Prompts: `prompts/` — planner, synthesizer, general_knowledge, verifier, compliance/{precheck,postcheck}
- OpenAI adapter: `app.llm.openai` (Chat Completions + Embeddings)
- Startup probe + live wiring: `app.agent.wiring`
- Retrieval layer: `app.retrieval.{models,normalization,chunker,embeddings,repository,search}`
- FAQ source loader: `app.retrieval.sources.take_home_faq`
- Seed CLI: `scripts/db_seed.py` (entrypoint `supportsmith-seed`)
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

## Agent Graph

`/chat` runs through a bounded LangGraph state machine with three independent gates wrapped around the planning loop:

```text
START -> load_context -> precheck
            precheck ─ hard-block (prompt_injection / harmful /
                                   severe sensitive_account)  ─> finalize  (CANONICAL_REFUSAL)
            precheck ─ otherwise                              ─> plan
            plan ─ use_tool ─────────> execute_tool -> observe
                                                       observe ─ continue ─> plan
                                                       observe ─ ready ────> synthesize
                                                       observe ─ cap-hit ──> halt -> synthesize
            plan ─ clarify | escalate
                 | refuse | synthesize_now ──────────> execute_tool -> observe -> synthesize
synthesize -> verify
            verify ─ retry == repair AND repair_attempts < cap ─> synthesize  (single repair)
            verify ─ otherwise                                  ─> postcheck
postcheck -> finalize -> END
```

### Nodes

- **`precheck`** runs the **Compliance Agent** in precheck mode (routing model + low reasoning effort). Hard-blocks `prompt_injection`, `harmful_or_illegal`, and `sensitive_account` requests that need account-specific access we can't verify. Other categories pass through with the classification attached to state. Hard-blocks short-circuit to `finalize` and stamp `CANONICAL_REFUSAL` with `source=compliance` — zero tool / planner / synthesizer calls.
- **`plan`** runs the *reasoning* model (`SUPPORTSMITH_REASONING_MODEL`, default `gpt-5.5`) with `reasoning_effort=high` and a JSON-schema response format. The planner emits a typed `Plan{intent, tool_name, arguments, rationale}`; LangChain `bind_tools` is *not* used so the dispatch surface stays explicit.
- **`execute_tool`** validates `arguments` against the matching Pydantic input model and dispatches to one of six tools: `search_faq`, `get_faq_by_category`, `ask_user_clarification`, `general_knowledge_lookup`, `escalate_to_human`, `refuse`. The `refuse` tool is a cheap planner-level gatekeeper, distinct from compliance.
- **`observe`** is a deterministic post-tool router (no LLM call) that flips between `synthesize` and another `plan` round, capping at `SUPPORTSMITH_MAX_TOOL_ITERATIONS` (default 6).
- **`synthesize`** runs the *chat* model (`SUPPORTSMITH_CHAT_MODEL`, default `gpt-5.5`) and returns structured JSON `{text, cited_titles}`. Short-circuits the LLM call when the planner picked `refuse` and stamps `CANONICAL_REFUSAL` directly. The user sees only `text`; cited titles flow through `matched_questions` as metadata, with a cross-check that drops any title the synthesizer hallucinated.
- **`verify`** runs the **Verifier** (reasoning model + medium reasoning effort, inlined into the node). Structured checks: addresses-user, grounding label (`faq_grounded` / `general_marked` / `clarification` / `escalation` / `refusal` / `unsupported`), leakage detection, safe source label, and a retry recommendation (`accept` / `repair` / `escalate` / `refuse`). Fail-fast: at most **one synthesize-only repair** for fixable wording or source-label issues. Budget exhaustion converts a second `repair` recommendation to `escalate` so the loop can never extend.
- **`postcheck`** runs the **Compliance Agent** in postcheck mode on the verified candidate. Last gate before the response goes back. When `allowed=false` it replaces the candidate text with `override_response` (when supplied) or `CANONICAL_REFUSAL`, stamping `source=compliance`. Skips the LLM call entirely for terminal candidates (`refuse` / `escalate` / `clarify`) — those are already known-safe templates and the LLM has nothing to add.
- **`finalize`** stamps the `trace_id` and exits.

### Three refusal mechanisms, one canonical string

Per the Phase 4 refusal policy, every refusal path produces the same user-facing string (`CANONICAL_REFUSAL`, declared in `app/agent/policy.py`). The response payload distinguishes which gate caught the request via `source`:

| Mechanism | When it fires | Cost (LLM calls) | `source` |
|---|---|---|---|
| Compliance **precheck** hard-block | Pre-graph block on injection / harm / severe sensitive-account | 1 (precheck only) | `compliance` |
| Planner `refuse` tool | Planner picks `refuse` mid-loop for off-topic | 4 (precheck, plan, verify, postcheck-skipped) | `refuse` |
| **Verifier** refusal | Verifier detects leakage / unsafe content | 5 (full graph) | `refuse` |
| Compliance **postcheck** override | Last-chance block on the synthesized answer | 5 + override | `compliance` |

The planner-`refuse` path skips the synthesizer LLM call (the canonical string is stamped directly), and the postcheck skips its LLM call when the candidate is already terminal — both keep cost off the cheap-gatekeeper paths.

### Compliance Agent and Verifier: stateless wrappers

`ComplianceAgent` (`app/agent/compliance.py`) is a plain Python class with `precheck(user_message)` and `postcheck(user_message, candidate_answer, candidate_source)` methods. It has no internal state, no graph awareness — just an `LLMClient` plus prompt loading. The graph nodes own the handoff: each calls into the agent, writes the resulting `ComplianceDecision` to `GraphState.compliance_precheck` / `compliance_postcheck`, emits a structured trace event, and routes via the conditional edge.

The verifier follows the same shape but is inlined directly into the `verify` node since it has only one caller. Promoting it back to a class is mechanical if Phase 6 evals or post-hoc audit need a standalone verifier surface.

### Prompts in YAML

All system prompts live under `prompts/` so the wording is reviewable in isolation from orchestration code. JSON schemas stay alongside the Pydantic models that consume them.

```text
prompts/
├── planner.yaml
├── synthesizer.yaml
├── general_knowledge.yaml
├── verifier.yaml
└── compliance/
    ├── precheck.yaml
    └── postcheck.yaml
```

`app.prompts.load_prompt("compliance.precheck")` returns a typed `Prompt{name, version, system, notes}` with `lru_cache` so the YAML is read once.

Every node appends a `TraceEvent` (node name, latency, model, token usage, short rationale) to graph state. Trace events are returned with the response under a stable `trace_id`; durable trace persistence lands in Phase 5.

### OpenAI integration

Phase 3 uses **Chat Completions and Embeddings** (`openai.AsyncOpenAI`, wrapped by `app.llm.openai.OpenAIChatCompletionsClient` and `OpenAIEmbeddingClient`). The Responses API is deferred — Chat Completions stays predictable under LangGraph orchestration. At startup the app probes the configured chat and reasoning models with a tiny ping, falling back through `SUPPORTSMITH_FALLBACK_CHAT_MODELS`. If no candidate works, startup fails with a typed `StartupConfigurationError` rather than coming up half-broken.

### Test policy

The default `uv run pytest` mocks OpenAI:

- Adapter tests (`test_openai_adapter.py`) mock `openai.AsyncOpenAI.chat.completions.create` and `embeddings.create`.
- Graph and node tests (`test_agent_graph.py`, `test_conversations.py`, `test_agent_wiring.py`) inject `ScriptedLLMClient` at the `LLMClient` Protocol seam.

One opt-in end-to-end smoke test (`tests/test_agent_live.py`) hits the real OpenAI API against the seeded compose Postgres. It's marked `live` and excluded from the default run via `addopts = ["-m", "not live"]`. To run it manually:

```bash
docker compose up -d   # postgres healthy + auto-seed with live embeddings
SUPPORTSMITH_TEST_DATABASE_URL=postgresql://supportsmith:supportsmith@localhost:55432/supportsmith \
    uv run --env-file .env pytest -m live tests/test_agent_live.py
```

Broader live behavior tests (eval suites) land in `evals/` in a later phase.

## Seeding The Knowledge Base

Phase 2 seeds `support_documents` from a knowledge-base JSON file. The take-home FAQ corpus lives in `data/knowledge-base.json` (committed) with each row carrying a `quality` label of `trusted`, `low_quality`, or `ambiguous`. Only `trusted` rows are ingested; the other rows are surfaced in the run summary so reviewers can see what was filtered and why.

`docker compose up` runs the seed automatically as a one-shot service after `migrate` and before `app`, so a fresh stack comes up with the FAQ corpus already in pgvector. To run the seed manually against a host-side Postgres:

```bash
uv run --env-file .env supportsmith-seed                           # live OpenAI embeddings
uv run --env-file .env supportsmith-seed --fake-embeddings         # deterministic, no API key
uv run --env-file .env supportsmith-seed --input path/to/file.json
```

The CLI prints a JSON run summary with `inserted` / `updated` / `unchanged` / `embedded` counts, the per-row outcomes, and any rejected rows (rows whose `quality` is not `trusted`). Re-running the seed is idempotent: rows whose content hash is unchanged are not re-embedded or rewritten.

Default seeding uses live OpenAI embeddings (`text-embedding-3-small`, 1536 dims) so semantic ranking works end-to-end against the running agent. `--fake-embeddings` keeps deterministic vectors available for CI and offline use; if you choose fake on the seed side, configure the agent to use the matching fake embedder so retrieval stays consistent.

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

Copy `.env.example` to `.env` for local configuration. Tests and the deterministic eval runner mock OpenAI, so they don't need a key. The live agent (`docker compose up` or local uvicorn outside the test environment) requires `OPENAI_API_KEY` for chat completions and embeddings; the startup probe fails fast if it's missing.

Notable settings:

| Var | Default | Purpose |
|---|---|---|
| `SUPPORTSMITH_CHAT_MODEL` | `gpt-5.5` | Synthesis + general-knowledge tool |
| `SUPPORTSMITH_REASONING_MODEL` | `gpt-5.5` | Planner |
| `SUPPORTSMITH_FALLBACK_CHAT_MODELS` | `gpt-5.4,gpt-5.1,gpt-5` | Walked when the primary model is unavailable at startup |
| `SUPPORTSMITH_EMBEDDING_MODEL` | `text-embedding-3-small` | Retrieval embeddings (seed + agent must match) |
| `SUPPORTSMITH_MAX_TOOL_ITERATIONS` | `6` | Hard cap on plan→execute→observe loops per turn |
| `SUPPORTSMITH_PLANNER_REASONING_EFFORT` | `high` | Planner reasoning depth |
