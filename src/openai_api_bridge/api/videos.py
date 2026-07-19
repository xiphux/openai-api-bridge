"""``POST /v1/videos`` (async submit), ``GET /v1/videos/{id}``, ``GET /v1/videos/{id}/content``."""

from __future__ import annotations

import logging
import secrets
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse

from ..auth import require_api_key
from ..config import BridgeSettings, parse_model_id
from ..dispatcher import BackendDispatcher
from ..errors import (
    InvalidRequest,
    JobNotFound,
    JobNotReady,
    UpstreamError,
)
from ..infra.filestore import FileStore
from ..infra.jobstore import JobStore, VideoJob
from ..infra.tasks import TaskScheduler
from ._videos_runner import run_video_job

log = logging.getLogger(__name__)

router = APIRouter()


def _video_to_dict(job: VideoJob) -> dict:
    """Render a VideoJob row in the OpenAI-compatible Sora video object shape."""
    completed_at = job.updated_at if job.status in ("completed", "failed") else None
    error = None
    if job.status == "failed" and job.error_message:
        error = {"message": job.error_message, "type": "api_error", "code": None}
    return {
        "id": job.id,
        "object": "video",
        "model": job.model,
        "status": job.status,
        "progress": job.progress_pct,
        "seconds": job.seconds,
        "size": job.size,
        "created_at": job.created_at,
        "completed_at": completed_at,
        "error": error,
    }


@router.post("/v1/videos", dependencies=[Depends(require_api_key)])
async def videos_create(
    request: Request,
    model: Annotated[str, Form()],
    prompt: Annotated[str, Form()],
    size: Annotated[str | None, Form()] = None,
    seconds: Annotated[float | None, Form()] = None,
    input_reference: Annotated[
        UploadFile | None,
        File(description="Optional input image for image-to-video"),
    ] = None,
) -> dict:
    provider_id, model_slug = parse_model_id(model)
    dispatcher: BackendDispatcher = request.app.state.dispatcher
    # Eagerly fail with 404 if the provider is unknown — friendlier than
    # accepting the job and immediately failing inside the runner.
    dispatcher.for_provider(provider_id)

    if seconds is not None and seconds <= 0:
        raise InvalidRequest("seconds must be positive", param="seconds")

    input_ref_bytes: bytes | None = None
    input_ref_ct: str | None = None
    if input_reference is not None:
        input_ref_bytes = await input_reference.read()
        if not input_ref_bytes:
            raise InvalidRequest("input_reference upload was empty", param="input_reference")
        input_ref_ct = input_reference.content_type or "image/png"

    job_id = secrets.token_hex(16)
    jobstore: JobStore = request.app.state.jobstore
    job = await jobstore.create(
        job_id=job_id,
        model=model,
        prompt=prompt,
        size=size,
        seconds=seconds,
    )

    scheduler: TaskScheduler = request.app.state.scheduler
    settings: BridgeSettings = request.app.state.settings
    filestore: FileStore = request.app.state.filestore

    scheduler.submit(
        run_video_job(
            job_id=job_id,
            provider_id=provider_id,
            model_slug=model_slug,
            full_model_id=model,
            prompt=prompt,
            size=size,
            seconds=seconds,
            input_reference=input_ref_bytes,
            input_reference_content_type=input_ref_ct,
            dispatcher=dispatcher,
            jobstore=jobstore,
            filestore=filestore,
            settings=settings,
        ),
        name=f"video-job-{job_id}",
    )

    return _video_to_dict(job)


@router.get("/v1/videos/{video_id}", dependencies=[Depends(require_api_key)])
async def videos_get(video_id: str, request: Request) -> dict:
    jobstore: JobStore = request.app.state.jobstore
    job = await jobstore.get(video_id)
    if job is None:
        raise JobNotFound(f"Video job {video_id!r} not found")
    return _video_to_dict(job)


@router.delete("/v1/videos/{video_id}", dependencies=[Depends(require_api_key)])
async def videos_cancel(video_id: str, request: Request) -> dict:
    """Cancel a queued or in-progress video job.

    Returns the job's current state (200) regardless of whether cancellation
    was actually possible. Already-terminal jobs are returned as-is. For
    in-flight jobs we ask the scheduler to cancel the asyncio.Task; the
    runner's CancelledError handler then marks the row failed and releases
    the semaphore permit. If the task can't be found (e.g. the bridge was
    restarted since the job was submitted), we still mark the DB row failed
    so callers see a consistent terminal state.
    """
    jobstore: JobStore = request.app.state.jobstore
    scheduler: TaskScheduler = request.app.state.scheduler

    job = await jobstore.get(video_id)
    if job is None:
        raise JobNotFound(f"Video job {video_id!r} not found")
    if job.status in ("completed", "failed"):
        return _video_to_dict(job)

    # Mark the row failed synchronously so the caller's poll sees a terminal
    # state immediately, but only while it's still active: the runner can
    # complete between the read above and this write, and an unconditional
    # update would flip a finished render to failed and orphan its file.
    await jobstore.fail_if_active(video_id, "Cancelled by user")
    scheduler.cancel(f"video-job-{video_id}")

    refreshed = await jobstore.get(video_id)
    assert refreshed is not None
    return _video_to_dict(refreshed)


@router.get("/v1/videos/{video_id}/content", dependencies=[Depends(require_api_key)])
async def videos_get_content(video_id: str, request: Request) -> FileResponse:
    jobstore: JobStore = request.app.state.jobstore
    filestore: FileStore = request.app.state.filestore

    job = await jobstore.get(video_id)
    if job is None:
        raise JobNotFound(f"Video job {video_id!r} not found")
    if job.status != "completed":
        raise JobNotReady(f"Video job is {job.status}, not completed")
    if not job.file_id:
        raise UpstreamError("Job marked completed but has no file_id")

    result = await filestore.open_for_read(job.file_id)
    if result is None:
        # The file was evicted before the client fetched it. We could re-pin
        # at completion time to prevent this, but for v1 we accept it.
        raise JobNotFound("Video file no longer available (evicted from cache)")
    abs_path, meta = result
    return FileResponse(abs_path, media_type=meta.content_type)


# unused import suppressor (time is referenced indirectly via VideoJob.updated_at;
# kept here so future code edits that timestamp jobs don't have to re-import).
_ = time
