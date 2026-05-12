"""In-memory ingestion-job registry tests."""

from __future__ import annotations

import pytest

from app.ingestion.jobs import DuplicateJobError, IngestionJobRegistry


async def test_create_assigns_unique_job_id() -> None:
    registry = IngestionJobRegistry()
    a = await registry.create(site_name="knotch", base_url="https://knotch.com/")
    b = await registry.create(site_name="acme", base_url="https://acme.com/")
    assert a.job_id != b.job_id
    assert a.status == "queued"
    assert b.site_name == "acme"


async def test_duplicate_active_job_is_rejected() -> None:
    registry = IngestionJobRegistry()
    await registry.create(site_name="knotch", base_url="https://knotch.com/")
    with pytest.raises(DuplicateJobError):
        await registry.create(site_name="knotch", base_url="https://knotch.com/")


async def test_succeeded_job_does_not_block_new_run() -> None:
    registry = IngestionJobRegistry()
    first = await registry.create(site_name="knotch", base_url="https://knotch.com/")
    await registry.mark_succeeded(first.job_id, {"inserted": 1})
    again = await registry.create(site_name="knotch", base_url="https://knotch.com/")
    assert again.job_id != first.job_id


async def test_cancel_marks_running_job_for_cancellation() -> None:
    registry = IngestionJobRegistry()
    job = await registry.create(site_name="knotch", base_url="https://knotch.com/")
    await registry.mark_running(job.job_id)
    snapshot = await registry.request_cancel(job.job_id)
    assert snapshot is not None and snapshot.cancel_requested is True
    # mark_succeeded after cancel still produces a cancelled terminal status
    await registry.mark_succeeded(job.job_id, {"inserted": 0})
    final = registry.get(job.job_id)
    assert final is not None and final.status == "cancelled"


async def test_cancel_returns_none_for_unknown_id() -> None:
    registry = IngestionJobRegistry()
    assert await registry.request_cancel("does-not-exist") is None


async def test_list_jobs_orders_newest_first() -> None:
    registry = IngestionJobRegistry()
    await registry.create(site_name="a", base_url="https://a.com/")
    await registry.create(site_name="b", base_url="https://b.com/")
    listing = registry.list_jobs()
    assert [job.site_name for job in listing][0] == "b"
