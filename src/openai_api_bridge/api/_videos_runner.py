"""Background runner that drives a video job from queued → completed/failed.

Runs as an asyncio task scheduled by the ``TaskScheduler``. The task itself
holds the input_reference bytes (transient, in-memory). If the bridge restarts
mid-flight, the row is cleaned up by ``JobStore.mark_stale_failed`` at startup.
"""

from __future__ import annotations

import asyncio
import logging

from ..config import BridgeSettings
from ..dispatcher import BackendDispatcher
from ..errors import BridgeError
from ..infra.filestore import FileStore
from ..infra.jobstore import JobStore

log = logging.getLogger(__name__)


async def run_video_job(
    *,
    job_id: str,
    provider_id: str,
    model_slug: str,
    full_model_id: str,
    prompt: str,
    size: str | None,
    seconds: float | None,
    input_reference: bytes | None,
    input_reference_content_type: str | None,
    dispatcher: BackendDispatcher,
    jobstore: JobStore,
    filestore: FileStore,
    settings: BridgeSettings,
) -> None:
    """Drive one video generation; persist state transitions to ``video_jobs``."""
    del settings, full_model_id  # reserved for future telemetry

    async def _persist_upstream_id(upstream_id: str) -> None:
        try:
            await jobstore.update(job_id, upstream_id=upstream_id)
        except Exception:
            log.exception("Failed to persist upstream_id for job %s", job_id)

    try:
        await jobstore.update(job_id, status="in_progress")
        backend = dispatcher.for_provider(provider_id)
        asset = await backend.generate_video(
            model_slug=model_slug,
            prompt=prompt,
            size=size,
            seconds=seconds,
            input_reference=input_reference,
            input_reference_content_type=input_reference_content_type,
            on_upstream_id=_persist_upstream_id,
        )
        file_id = await filestore.put(
            asset.data,
            content_type=asset.content_type,
            kind=asset.kind,
            source_backend=provider_id,
            source_model=model_slug,
            prompt_excerpt=prompt,
            pinned=False,
        )
        await jobstore.update(job_id, status="completed", file_id=file_id, progress_pct=100)
        log.info("Video job %s completed; file_id=%s", job_id, file_id)
    except asyncio.CancelledError:
        log.info("Video job %s cancelled; marking failed", job_id)
        # Best-effort: mark the row failed so it doesn't sit in_progress
        # forever. We swallow errors here so the cancellation propagates
        # cleanly even if the DB write itself races with shutdown.
        try:
            await jobstore.update(job_id, status="failed", error_message="Job cancelled")
        except Exception:
            log.exception("Failed to mark cancelled job %s as failed", job_id)
        raise
    except BridgeError as e:
        log.warning("Video job %s failed (bridge error): %s", job_id, e.message)
        await jobstore.update(job_id, status="failed", error_message=e.message)
    except Exception as e:
        log.exception("Video job %s failed (unexpected)", job_id)
        await jobstore.update(
            job_id,
            status="failed",
            error_message=f"Internal error: {type(e).__name__}: {e}",
        )
