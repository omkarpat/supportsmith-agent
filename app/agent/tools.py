"""Agent tool registry: typed inputs, typed outputs, async executors.

Tools are plain Python functions with Pydantic Input / Output schemas. The
plan node asks the LLM to emit a structured plan whose ``tool_name`` is one
of :data:`TOOL_NAMES` and whose ``arguments`` validate against the matching
Input model. The execute_tool node dispatches via :class:`ToolRegistry`.

We deliberately do **not** use LangChain's ``bind_tools`` machinery here.
Plans come from JSON-schema-constrained Chat Completions output, dispatch is
explicit, and tests can assert on every tool call without traversing
LangChain's adapter layer.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent.topics import SUPPORT_TOPIC_EXAMPLES
from app.llm.client import ChatMessage, ChatRequest, LLMClient
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.models import RetrievalResult
from app.retrieval.normalization import normalize_text
from app.retrieval.search import SupportDocumentSearch

ToolName = Literal[
    "search_faq",
    "get_faq_by_category",
    "ask_user_clarification",
    "general_knowledge_lookup",
    "escalate_to_human",
    "refuse",
]

TOOL_NAMES: tuple[ToolName, ...] = (
    "search_faq",
    "get_faq_by_category",
    "ask_user_clarification",
    "general_knowledge_lookup",
    "escalate_to_human",
    "refuse",
)

DEFAULT_SEARCH_LIMIT = 5


# --- input / output schemas ---------------------------------------------------


class SearchFAQInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1)
    category_filter: str | None = None
    limit: int = Field(default=DEFAULT_SEARCH_LIMIT, ge=1, le=20)


class SearchFAQOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[RetrievalResult]


class GetFAQByCategoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str = Field(min_length=1)
    limit: int = Field(default=DEFAULT_SEARCH_LIMIT, ge=1, le=20)


class GetFAQByCategoryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str
    results: list[RetrievalResult]


class AskUserClarificationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)


class AskUserClarificationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    topic_examples: list[str]


class GeneralKnowledgeLookupInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1)


class GeneralKnowledgeLookupOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    model: str
    grounded_in_kb: bool = False


class EscalateToHumanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=1)
    transcript: list[ChatMessage] = Field(default_factory=list)


class EscalateToHumanOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    handoff_id: str
    reason: str
    status: Literal["queued"] = "queued"


class RefuseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=1)


class RefuseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str


ToolOutput = (
    SearchFAQOutput
    | GetFAQByCategoryOutput
    | AskUserClarificationOutput
    | GeneralKnowledgeLookupOutput
    | EscalateToHumanOutput
    | RefuseOutput
)


# --- registry -----------------------------------------------------------------


@dataclass(frozen=True)
class ToolDependencies:
    """Shared per-turn dependencies passed to tool executors.

    Plain dataclass (not Pydantic) because the values are runtime infrastructure
    references — :class:`LLMClient` is a Protocol and Pydantic's isinstance
    validation rejects Protocol types.
    """

    llm: LLMClient
    embeddings: EmbeddingGenerator
    search: SupportDocumentSearch
    chat_model: str
    general_knowledge_max_tokens: int = 512


ToolExecutor = Callable[[Any, ToolDependencies], Awaitable[ToolOutput]]


class ToolRegistry:
    """Validate-and-dispatch facade used by the execute_tool node."""

    def __init__(self, dependencies: ToolDependencies) -> None:
        self.dependencies = dependencies
        self._executors: dict[ToolName, ToolExecutor] = {
            "search_faq": _run_search_faq,
            "get_faq_by_category": _run_get_faq_by_category,
            "ask_user_clarification": _run_ask_user_clarification,
            "general_knowledge_lookup": _run_general_knowledge_lookup,
            "escalate_to_human": _run_escalate_to_human,
            "refuse": _run_refuse,
        }
        self._input_models: dict[ToolName, type[BaseModel]] = {
            "search_faq": SearchFAQInput,
            "get_faq_by_category": GetFAQByCategoryInput,
            "ask_user_clarification": AskUserClarificationInput,
            "general_knowledge_lookup": GeneralKnowledgeLookupInput,
            "escalate_to_human": EscalateToHumanInput,
            "refuse": RefuseInput,
        }

    async def run(self, tool_name: ToolName, arguments: dict[str, Any]) -> ToolOutput:
        """Validate ``arguments`` against ``tool_name``'s input model and dispatch."""
        if tool_name not in self._executors:
            raise ValueError(f"Unknown tool: {tool_name!r}")
        validated = self._input_models[tool_name].model_validate(arguments)
        executor = self._executors[tool_name]
        return await executor(validated, self.dependencies)

    def input_schema(self, tool_name: ToolName) -> dict[str, Any]:
        """Return the JSON schema for ``tool_name``'s arguments."""
        return self._input_models[tool_name].model_json_schema()


# --- executors ----------------------------------------------------------------


async def _run_search_faq(
    inputs: SearchFAQInput,
    deps: ToolDependencies,
) -> SearchFAQOutput:
    embedding = await deps.embeddings.embed_many([normalize_text(inputs.query)])
    results = await deps.search.search(
        embedding[0],
        limit=inputs.limit,
        category=inputs.category_filter,
    )
    return SearchFAQOutput(results=results)


async def _run_get_faq_by_category(
    inputs: GetFAQByCategoryInput,
    deps: ToolDependencies,
) -> GetFAQByCategoryOutput:
    # Category-only browsing: use a zero embedding query so cosine becomes a
    # tie-breaker rather than a ranker, then return whatever the index sorts up.
    seed_embedding = await deps.embeddings.embed_many([inputs.category])
    results = await deps.search.search(
        seed_embedding[0],
        limit=inputs.limit,
        category=inputs.category,
    )
    return GetFAQByCategoryOutput(category=inputs.category, results=results)


async def _run_ask_user_clarification(
    inputs: AskUserClarificationInput,
    _deps: ToolDependencies,
) -> AskUserClarificationOutput:
    return AskUserClarificationOutput(
        question=inputs.question,
        topic_examples=list(SUPPORT_TOPIC_EXAMPLES),
    )


async def _run_general_knowledge_lookup(
    inputs: GeneralKnowledgeLookupInput,
    deps: ToolDependencies,
) -> GeneralKnowledgeLookupOutput:
    response = await deps.llm.complete(
        ChatRequest(
            model=deps.chat_model,
            max_completion_tokens=deps.general_knowledge_max_tokens,
            messages=[
                ChatMessage(
                    role="system",
                    content=(
                        "You are SupportSmith's fallback general-knowledge helper. "
                        "Answer support-adjacent questions concisely. Do not invent "
                        "company-specific procedures or product details. If the "
                        "question is outside general support knowledge, say so."
                    ),
                ),
                ChatMessage(role="user", content=inputs.query),
            ],
        )
    )
    return GeneralKnowledgeLookupOutput(
        answer=response.content,
        model=response.model,
        grounded_in_kb=False,
    )


async def _run_escalate_to_human(
    inputs: EscalateToHumanInput,
    _deps: ToolDependencies,
) -> EscalateToHumanOutput:
    # Mock handoff id; durable persistence lands in Phase 5.
    handoff_id = f"escalation_{abs(hash((inputs.reason, len(inputs.transcript)))):x}"
    return EscalateToHumanOutput(handoff_id=handoff_id, reason=inputs.reason)


async def _run_refuse(
    inputs: RefuseInput,
    _deps: ToolDependencies,
) -> RefuseOutput:
    return RefuseOutput(reason=inputs.reason)
