"""Admin-only website ingestion routes.

Posting to ``/admin/website-ingestions`` queues a Firecrawl-backed crawl as a
FastAPI background task and returns immediately with a job id. Status and the
final ingestion summary are exposed via the read endpoints. State is in-memory
and lost on process restart — durability of the *ingested documents* lives in
Postgres.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.security import AdminApiKey
from app.core.config import Settings
from app.ingestion.jobs import (
    ACTIVE_STATUSES,
    DuplicateJobError,
    IngestionJob,
    IngestionJobRegistry,
    serialize_job,
)
from app.ingestion.website import WebsiteIngestionSummary, WebsiteIngestor
from app.retrieval.sources.websites import (
    CrawlConfig,
    WebsiteSourceConfig,
    load_website_source,
)
from app.retrieval.url_utils import UnsafeUrlError, validate_public_url

IngestorFactory = Callable[[], Awaitable[WebsiteIngestor]]

router = APIRouter(prefix="/admin/website-ingestions", tags=["admin-ingestion"])


class WebsiteIngestionRequest(BaseModel):
    """JSON body for a POST to ``/admin/website-ingestions``."""

    model_config = ConfigDict(extra="forbid")

    url: str
    name: str = Field(min_length=1)
    limit: int | None = Field(default=None, ge=1)
    max_depth: int | None = Field(default=None, ge=1)
    include_paths: list[str] | None = None
    exclude_paths: list[str] | None = None
    priority_paths: list[str] | None = None
    allow_subdomains: bool = False
    dry_run: bool = False

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        stripped = value.strip().lower()
        if not stripped.replace("-", "").replace("_", "").isalnum():
            raise ValueError("name must be alphanumeric with dashes/underscores only")
        return stripped


class WebsiteIngestionResponse(BaseModel):
    """Response surfaced by the queue + read endpoints."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    site_name: str
    base_url: str
    source: str = "website"
    status: str
    summary: dict[str, Any] | None = None
    error: str | None = None
    started_at: Any = None
    finished_at: Any = None


@router.post(
    "",
    response_model=WebsiteIngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[AdminApiKey],
)
async def queue_website_ingestion(
    payload: WebsiteIngestionRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> WebsiteIngestionResponse:
    """Queue a website crawl and return immediately with a job id."""
    settings: Settings = request.app.state.settings
    registry: IngestionJobRegistry = _get_registry(request)
    ingestor_factory = _get_ingestor_factory(request)

    try:
        normalized_url = validate_public_url(
            payload.url,
            allowed_hosts=settings.allowed_ingestion_hosts,
            allow_any=settings.allow_any_website_ingestion,
        )
    except UnsafeUrlError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    config = _resolve_config(payload, normalized_url, settings)

    try:
        job = await registry.create(site_name=config.name, base_url=config.base_url)
    except DuplicateJobError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    background_tasks.add_task(
        _run_ingestion_job,
        job_id=job.job_id,
        config=config,
        dry_run=payload.dry_run,
        registry=registry,
        ingestor_factory=ingestor_factory,
    )

    return _to_response(job)


@router.get(
    "",
    response_model=list[WebsiteIngestionResponse],
    dependencies=[AdminApiKey],
)
async def list_jobs(request: Request) -> list[WebsiteIngestionResponse]:
    registry: IngestionJobRegistry = _get_registry(request)
    return [_to_response(job) for job in registry.list_jobs()]


@router.get(
    "/{job_id}",
    response_model=WebsiteIngestionResponse,
    dependencies=[AdminApiKey],
)
async def get_job(job_id: str, request: Request) -> WebsiteIngestionResponse:
    registry: IngestionJobRegistry = _get_registry(request)
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return _to_response(job)


@router.post(
    "/{job_id}/cancel",
    response_model=WebsiteIngestionResponse,
    dependencies=[AdminApiKey],
)
async def cancel_job(job_id: str, request: Request) -> WebsiteIngestionResponse:
    registry: IngestionJobRegistry = _get_registry(request)
    job = await registry.request_cancel(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return _to_response(job)


def _get_registry(request: Request) -> IngestionJobRegistry:
    registry = getattr(request.app.state, "ingestion_jobs", None)
    if not isinstance(registry, IngestionJobRegistry):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion subsystem not initialized.",
        )
    return registry


def _get_ingestor_factory(request: Request) -> IngestorFactory:
    factory = getattr(request.app.state, "website_ingestor_factory", None)
    if factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Website ingestion is not configured on this deployment.",
        )
    return factory  # type: ignore[no-any-return]


def _resolve_config(
    payload: WebsiteIngestionRequest,
    normalized_url: str,
    settings: Settings,
) -> WebsiteSourceConfig:
    """Merge the request body with any matching site-config defaults."""
    try:
        base_config = load_website_source(payload.name)
    except FileNotFoundError:
        base_config = WebsiteSourceConfig(
            name=payload.name,
            base_url=normalized_url,
            description=None,
            crawl=CrawlConfig(),
        )

    # Always use the validated URL the caller submitted (not the YAML default
    # — operators sometimes ingest a subdirectory of a configured site).
    overrides: dict[str, Any] = {"base_url": normalized_url}
    if payload.include_paths is not None:
        overrides["include_paths"] = tuple(payload.include_paths)
    if payload.exclude_paths is not None:
        overrides["exclude_paths"] = tuple(payload.exclude_paths)
    if payload.priority_paths is not None:
        overrides["priority_paths"] = tuple(payload.priority_paths)

    crawl_updates: dict[str, Any] = {}
    requested_limit = payload.limit if payload.limit is not None else base_config.crawl.limit
    crawl_updates["limit"] = min(requested_limit, settings.website_max_pages_per_job)
    if payload.max_depth is not None:
        crawl_updates["max_depth"] = payload.max_depth
    if payload.allow_subdomains:
        crawl_updates["allow_subdomains"] = True

    overrides["crawl"] = base_config.crawl.model_copy(update=crawl_updates)
    return base_config.model_copy(update=overrides)


async def _run_ingestion_job(
    *,
    job_id: str,
    config: WebsiteSourceConfig,
    dry_run: bool,
    registry: IngestionJobRegistry,
    ingestor_factory: IngestorFactory,
) -> None:
    """Background task that drives one ingestion job to completion."""
    await registry.mark_running(job_id)
    if registry.cancel_requested(job_id):
        return
    try:
        ingestor: WebsiteIngestor = await ingestor_factory()
        summary: WebsiteIngestionSummary = await ingestor.run(config, dry_run=dry_run)
        await registry.mark_succeeded(job_id, summary.model_dump(mode="json"))
    except Exception as exc:  # noqa: BLE001 - background task must capture all
        await registry.mark_failed(job_id, f"{type(exc).__name__}: {exc}")


def _to_response(job: IngestionJob) -> WebsiteIngestionResponse:
    data = serialize_job(job)
    return WebsiteIngestionResponse(**data)


def hostname_of(url: str) -> str:
    """Helper used by tests + callers to read the host part."""
    return urlparse(url).netloc.lower()


__all__ = [
    "WebsiteIngestionRequest",
    "WebsiteIngestionResponse",
    "router",
]


# Re-export for typing convenience.
ActiveStatuses = ACTIVE_STATUSES
