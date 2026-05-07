"""Conversation routes."""

from uuid import uuid4

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from app.agent.harness import Agent, AgentRequest, AgentResponse

router = APIRouter(tags=["conversations"])


class MessageRequest(BaseModel):
    """Incoming user message payload."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """Convenience chat payload that can optionally continue a conversation."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    conversation_id: str | None = Field(default=None, min_length=1)


async def run_agent(
    request: Request,
    conversation_id: str,
    message: str,
) -> AgentResponse:
    """Route a message through the configured support agent."""
    agent: Agent = request.app.state.agent
    return await agent.respond(AgentRequest(conversation_id=conversation_id, message=message))


@router.post("/conversations/{conversation_id}/messages", response_model=AgentResponse)
async def create_message(
    conversation_id: str,
    payload: MessageRequest,
    request: Request,
) -> AgentResponse:
    """Append a user message and return the Phase 1 agent scaffold response."""
    return await run_agent(
        request=request,
        conversation_id=conversation_id,
        message=payload.message,
    )


@router.post("/chat", response_model=AgentResponse)
async def chat(payload: ChatRequest, request: Request) -> AgentResponse:
    """Send a message, minting a conversation id when one is not supplied."""
    conversation_id = payload.conversation_id or str(uuid4())
    return await run_agent(
        request=request,
        conversation_id=conversation_id,
        message=payload.message,
    )
