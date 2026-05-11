"""Process-local registry for website ingestion jobs.

Phase 7 deliberately keeps job state in memory — the spec calls out that the
ingested documents themselves are durable in Postgres, so losing in-flight job
status across a process restart is acceptable. We never add a database table
for jobs.
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
ACTIVE_STATUSES: frozenset[JobStatus] = frozenset({"queued", "running"})


class IngestionJob(BaseModel):
    """One ingestion job's state."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    site_name: str
    base_url: str
    source: Literal["website"] = "website"
    status: JobStatus = "queued"
    summary: dict[str, Any] | None = None
    error: str | None = None
    cancel_requested: bool = False
    started_at: datetime
    finished_at: datetime | None = None


class DuplicateJobError(RuntimeError):
    """Raised when an active job for the same base_url is already in flight."""


class IngestionJobRegistry:
    """In-memory job registry; one instance is attached to ``app.state``."""

    def __init__(self) -> None:
        self._jobs: dict[str, IngestionJob] = {}
        self._lock = asyncio.Lock()

    async def create(self, *, site_name: str, base_url: str) -> IngestionJob:
        """Register a new queued job. Fails on duplicate active base_url."""
        async with self._lock:
            self._reject_duplicate(base_url)
            job = IngestionJob(
                job_id=_new_job_id(),
                site_name=site_name,
                base_url=base_url,
                started_at=_utcnow(),
            )
            self._jobs[job.job_id] = job
            return job

    async def mark_running(self, job_id: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.cancel_requested:
                job.status = "cancelled"
                job.finished_at = _utcnow()
                return
            job.status = "running"

    async def mark_succeeded(self, job_id: str, summary: dict[str, Any]) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "cancelled" if job.cancel_requested else "succeeded"
            job.summary = summary
            job.finished_at = _utcnow()

    async def mark_failed(self, job_id: str, error: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "cancelled" if job.cancel_requested else "failed"
            job.error = error
            job.finished_at = _utcnow()

    async def request_cancel(self, job_id: str) -> IngestionJob | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in ACTIVE_STATUSES:
                job.cancel_requested = True
            return job

    def get(self, job_id: str) -> IngestionJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[IngestionJob]:
        return sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)

    def _reject_duplicate(self, base_url: str) -> None:
        for job in self._jobs.values():
            if job.base_url == base_url and job.status in ACTIVE_STATUSES:
                raise DuplicateJobError(
                    f"Active job {job.job_id!r} already running for {base_url}"
                )

    def cancel_requested(self, job_id: str) -> bool:
        """Snapshot of the cancel flag — safe to read without the lock."""
        job = self._jobs.get(job_id)
        return bool(job and job.cancel_requested)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_job_id() -> str:
    return f"website_ingest_{secrets.token_hex(8)}"


def serialize_jobs(jobs: Iterable[IngestionJob]) -> list[dict[str, Any]]:
    return [job.model_dump(mode="json") for job in jobs]


_JOB_RESPONSE_FIELDS = {
    "job_id",
    "site_name",
    "base_url",
    "source",
    "status",
    "summary",
    "error",
    "started_at",
    "finished_at",
}


def serialize_job(job: IngestionJob) -> dict[str, Any]:
    return job.model_dump(mode="json", include=_JOB_RESPONSE_FIELDS)
