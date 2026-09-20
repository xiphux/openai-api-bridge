"""``run_video_job``'s one promise: a job never stays queued or in_progress.

Clients poll ``GET /v1/videos/{id}`` until the status is terminal. A runner
that exits without writing ``completed`` or ``failed`` leaves them polling
forever, and nothing on the server would ever say so — ``mark_stale_failed``
only runs at startup. These tests drive the runner directly, with a stub
backend, through each way it can end.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from openai_api_bridge.api._videos_runner import run_video_job
from openai_api_bridge.backends.base import Backend, GeneratedAsset, ModelEntry
from openai_api_bridge.dispatcher import BackendDispatcher
from openai_api_bridge.errors import UpstreamError
from openai_api_bridge.infra.filestore import FileStore
from openai_api_bridge.infra.jobstore import JobStore

_ASSET = GeneratedAsset(data=b"\x00\x00\x00\x18ftypmp42", content_type="video/mp4", kind="video")


class _StubBackend(Backend):
    """A backend whose ``generate_video`` runs the given coroutine function."""

    def __init__(self, generate: Any) -> None:
        self._generate = generate

    async def list_models(self) -> list[ModelEntry]:
        return []

    async def generate_video(self, **kwargs: Any) -> GeneratedAsset:  # type: ignore[override]
        result: GeneratedAsset = await self._generate(**kwargs)
        return result


class _StubDispatcher:
    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    def for_provider(self, provider_id: str) -> Backend:
        return self._backend


async def _started(event: asyncio.Event, task: asyncio.Task[None]) -> None:
    """Wait for the stub render to begin, bounded — if the runner dies before
    reaching the backend, surface that instead of waiting forever."""
    waiter = asyncio.create_task(event.wait())
    done, _ = await asyncio.wait({waiter, task}, timeout=5, return_when=asyncio.FIRST_COMPLETED)
    if waiter not in done:
        waiter.cancel()
        if task in done:
            task.result()  # re-raise whatever killed the runner
        raise AssertionError("the stub render never started")


async def _run(generate: Any, jobstore: JobStore, filestore: FileStore) -> None:
    await jobstore.create(
        job_id="job1", model="p/m", prompt="a cat", size=None, aspect_ratio=None, seconds=None
    )
    await run_video_job(
        job_id="job1",
        provider_id="p",
        model_slug="m",
        prompt="a cat",
        size=None,
        aspect_ratio=None,
        seconds=None,
        input_reference=None,
        input_reference_content_type=None,
        dispatcher=cast(BackendDispatcher, _StubDispatcher(_StubBackend(generate))),
        jobstore=jobstore,
        filestore=filestore,
    )


async def test_an_unexpected_exception_still_fails_the_job(
    jobstore: JobStore, filestore: FileStore
) -> None:
    """Not every failure is a BridgeError: an adapter bug raises KeyError."""

    async def generate(**_: Any) -> GeneratedAsset:
        raise KeyError("video")

    await _run(generate, jobstore, filestore)  # contained, not propagated

    job = await jobstore.get("job1")
    assert job is not None
    assert job.status == "failed"
    assert job.error_message == "Internal error: KeyError: 'video'"


async def test_a_bridge_error_fails_the_job_with_its_own_message(
    jobstore: JobStore, filestore: FileStore
) -> None:
    async def generate(**_: Any) -> GeneratedAsset:
        raise UpstreamError("fal render failed: out of credits")

    await _run(generate, jobstore, filestore)

    job = await jobstore.get("job1")
    assert job is not None
    assert job.status == "failed"
    assert job.error_message == "fal render failed: out of credits"


async def test_a_failed_upstream_id_write_does_not_fail_the_render(
    jobstore: JobStore, filestore: FileStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The upstream id is bookkeeping for cancellation; losing it must not
    throw away a render that went on to finish (and was paid for)."""
    real_update = jobstore.update

    async def update(job_id: str, **fields: Any) -> None:
        if "upstream_id" in fields:
            raise RuntimeError("database is locked")
        await real_update(job_id, **fields)

    monkeypatch.setattr(jobstore, "update", update)

    async def generate(*, on_upstream_id: Any, **_: Any) -> GeneratedAsset:
        await on_upstream_id("upstream-123")
        return _ASSET

    await _run(generate, jobstore, filestore)

    job = await jobstore.get("job1")
    assert job is not None
    assert job.status == "completed"
    assert job.file_id is not None
    assert job.upstream_id is None


async def test_cancellation_marks_the_job_failed_and_still_propagates(
    jobstore: JobStore, filestore: FileStore
) -> None:
    started = asyncio.Event()

    async def generate(**_: Any) -> GeneratedAsset:
        started.set()
        await asyncio.Event().wait()  # a render that never finishes
        raise AssertionError("unreachable")

    task = asyncio.create_task(_run(generate, jobstore, filestore))
    await _started(started, task)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    job = await jobstore.get("job1")
    assert job is not None
    assert job.status == "failed"
    assert job.error_message == "Job cancelled"


async def test_cancellation_propagates_even_if_marking_the_job_fails(
    jobstore: JobStore, filestore: FileStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shutdown cancels runners while the database is closing; the scheduler
    has to see the cancellation, not a secondary error from the write."""
    started = asyncio.Event()

    async def fail_if_active(job_id: str, message: str) -> bool:
        raise RuntimeError("database is closed")

    monkeypatch.setattr(jobstore, "fail_if_active", fail_if_active)

    async def generate(**_: Any) -> GeneratedAsset:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    task = asyncio.create_task(_run(generate, jobstore, filestore))
    await _started(started, task)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
