"""Build the live SupportAgent from settings.

This is the seam used by ``app.main`` at startup. Tests build a SupportAgent
directly with a scripted LLM client and skip the live probe; production
startup goes through :func:`build_live_support_agent`, which fails fast when
no configured Chat Completions model is reachable.
"""

from __future__ import annotations

import logging

from app.agent.graph import build_graph
from app.agent.nodes import NodeContext
from app.agent.runner import SupportAgent
from app.agent.tools import ToolDependencies, ToolRegistry
from app.core.config import Settings
from app.llm.client import ChatMessage, ChatRequest, LLMClient
from app.llm.openai import (
    LLMProviderError,
    OpenAIChatCompletionsClient,
    OpenAIEmbeddingClient,
)
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.search import SupportDocumentSearch

log = logging.getLogger(__name__)


class StartupConfigurationError(RuntimeError):
    """Raised when no configured Chat Completions model can be reached."""


async def probe_chat_model(client: LLMClient, *, candidates: list[str]) -> str:
    """Return the first ``candidates`` model that responds to a 1-token ping.

    Raises :class:`StartupConfigurationError` when every candidate fails so the
    application refuses to start with a half-broken LLM configuration.
    """
    if not candidates:
        raise StartupConfigurationError("No chat model candidates supplied to probe")

    failures: list[str] = []
    for model in candidates:
        try:
            await client.complete(
                ChatRequest(
                    model=model,
                    max_completion_tokens=8,
                    messages=[ChatMessage(role="user", content="ok")],
                )
            )
            return model
        except LLMProviderError as exc:
            failures.append(f"{model}: {exc}")
            log.warning("Chat model probe failed for %s: %s", model, exc)
    raise StartupConfigurationError(
        "No configured Chat Completions model is usable. Tried: " + "; ".join(failures)
    )


async def build_live_support_agent(
    settings: Settings,
    *,
    search: SupportDocumentSearch,
) -> SupportAgent:
    """Construct a SupportAgent backed by live OpenAI Chat Completions.

    Only called outside the test environment. The startup probe walks the
    primary chat model then any fallbacks. The reasoning model is probed
    separately because the planner uses a different model and reasoning effort.
    """
    if not settings.openai_api_key:
        raise StartupConfigurationError(
            "OPENAI_API_KEY (or SUPPORTSMITH_OPENAI_API_KEY) is required outside tests."
        )

    client = OpenAIChatCompletionsClient(api_key=settings.openai_api_key)
    chat_candidates = [settings.chat_model, *settings.fallback_chat_models]
    chosen_chat = await probe_chat_model(client, candidates=chat_candidates)
    if chosen_chat != settings.chat_model:
        log.warning(
            "Configured chat_model %s unavailable; using fallback %s",
            settings.chat_model,
            chosen_chat,
        )

    # The reasoning model is critical for the planner; we check it but reuse
    # the chosen chat model as a degraded fallback if it fails so the service
    # can still come up. Phase 4 may want to harden this further.
    chosen_reasoning = settings.reasoning_model
    try:
        await client.complete(
            ChatRequest(
                model=settings.reasoning_model,
                max_completion_tokens=64,
                reasoning_effort=settings.routing_reasoning_effort,
                messages=[ChatMessage(role="user", content="ok")],
            )
        )
    except LLMProviderError as exc:
        log.warning(
            "Reasoning model %s unavailable (%s); falling back to chat model %s",
            settings.reasoning_model,
            exc,
            chosen_chat,
        )
        chosen_reasoning = chosen_chat

    embedding_client = OpenAIEmbeddingClient(api_key=settings.openai_api_key)
    deps = ToolDependencies(
        llm=client,
        embeddings=EmbeddingGenerator(embedding_client, model=settings.embedding_model),
        search=search,
        chat_model=chosen_chat,
    )
    ctx = NodeContext(
        llm=client,
        tools=ToolRegistry(deps),
        chat_model=chosen_chat,
        reasoning_model=chosen_reasoning,
        planner_reasoning_effort=settings.planner_reasoning_effort,
        planner_max_completion_tokens=settings.planner_max_completion_tokens,
        synthesis_max_completion_tokens=settings.synthesis_max_completion_tokens,
        max_tool_iterations=settings.max_tool_iterations,
    )
    return SupportAgent(build_graph(ctx))
