"""Health check routes."""

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict

from app.core.config import Settings
from app.db.postgres import DatabaseHealth, PostgresDatabase

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Service health response."""

    model_config = ConfigDict(extra="forbid")

    service: str
    version: str
    environment: str
    status: str
    database: DatabaseHealth


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Return service health and deploy metadata."""
    settings: Settings = request.app.state.settings
    database: PostgresDatabase = request.app.state.database
    database_health = await database.health()
    return HealthResponse(
        service=settings.service_name,
        version=settings.app_version,
        environment=settings.environment,
        status="ok" if database_health.status == "ok" else "degraded",
        database=database_health,
    )
