"""SQLite-backed video_jobs CRUD + state transitions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import aiosqlite

from .db import Database

JobStatus = Literal["queued", "in_progress", "completed", "failed"]


class _Unset:
    """Sentinel for :meth:`JobStore.update`: "leave this column alone".

    The patch convention here is "only fields explicitly passed (not None) are
    written", which cannot express "write NULL". That is fine for every field
    that only ever gains a value, and wrong for ``aspect_ratio``, which has to be
    able to go BACK to NULL: it is seeded from the request at creation, and a
    backend that doesn't deal in ratios reports none, at which point the row must
    stop claiming the request was honoured.
    """


_UNSET = _Unset()


@dataclass(slots=True, frozen=True)
class VideoJob:
    id: str
    status: JobStatus
    model: str
    prompt: str
    size: str | None
    # The canonical aspect ratio this job renders at. Seeded from the request,
    # then settled on completion to what the backend actually used — snapping
    # means a model may render the nearest ratio it offers rather than the one
    # asked for, and a backend that deals in no ratios at all reports none, which
    # clears this back to NULL rather than leaving the unhonoured request
    # standing. So on a COMPLETED job this is what was rendered, or NULL when
    # nothing reported a ratio; while queued it is only what was asked for.
    aspect_ratio: str | None
    seconds: float | None
    # **Always None.** The column exists and is read back here, but nothing
    # writes it: an image-to-video reference lives in memory for the life of
    # the runner task (see `_videos_runner`) and is never put in the FileStore,
    # which is also why a job can't survive a restart. Kept as schema and as a
    # field because persisting the reference is the missing half of resumable
    # video jobs, and dropping the column needs a migration to get back.
    # Same honesty as `FileStore.set_pinned`.
    input_reference_file_id: str | None
    file_id: str | None
    upstream_id: str | None
    error_message: str | None
    created_at: int
    updated_at: int
    progress_pct: int | None


class JobStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def _row_to_job(row: aiosqlite.Row) -> VideoJob:
        return VideoJob(
            id=row["id"],
            status=row["status"],
            model=row["model"],
            prompt=row["prompt"],
            size=row["size"],
            aspect_ratio=row["aspect_ratio"],
            seconds=row["seconds"],
            input_reference_file_id=row["input_reference_file_id"],
            file_id=row["file_id"],
            upstream_id=row["upstream_id"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            progress_pct=row["progress_pct"],
        )

    async def create(
        self,
        *,
        job_id: str,
        model: str,
        prompt: str,
        size: str | None,
        aspect_ratio: str | None,
        seconds: float | None,
    ) -> VideoJob:
        """Insert a queued job.

        ``input_reference_file_id`` is deliberately not a parameter — see the
        note on :class:`VideoJob`. It took one, defaulted to ``None``, and no
        caller ever passed it, so the column read as live state that was in
        fact always NULL.
        """
        now = int(time.time())
        await self.db.execute(
            """INSERT INTO video_jobs (
                   id, status, model, prompt, size, aspect_ratio, seconds,
                   created_at, updated_at
               ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?)""",
            (job_id, model, prompt, size, aspect_ratio, seconds, now, now),
        )
        job = await self.get(job_id)
        assert job is not None  # we just inserted it
        return job

    async def get(self, job_id: str) -> VideoJob | None:
        row = await self.db.fetchone("SELECT * FROM video_jobs WHERE id = ?", (job_id,))
        return self._row_to_job(row) if row else None

    async def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        upstream_id: str | None = None,
        file_id: str | None = None,
        error_message: str | None = None,
        progress_pct: int | None = None,
        aspect_ratio: str | _Unset | None = _UNSET,
    ) -> None:
        """Patch-style update. Only fields explicitly passed (not None) are written.

        ``aspect_ratio`` is the exception: it takes a sentinel default so that
        passing ``None`` WRITES NULL rather than meaning "skip". See :class:`_Unset`.
        """
        sets: list[str] = ["updated_at = ?"]
        params: list[Any] = [int(time.time())]
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if upstream_id is not None:
            sets.append("upstream_id = ?")
            params.append(upstream_id)
        if file_id is not None:
            sets.append("file_id = ?")
            params.append(file_id)
        if error_message is not None:
            sets.append("error_message = ?")
            params.append(error_message)
        if progress_pct is not None:
            sets.append("progress_pct = ?")
            params.append(progress_pct)
        if not isinstance(aspect_ratio, _Unset):
            sets.append("aspect_ratio = ?")
            params.append(aspect_ratio)
        params.append(job_id)
        await self.db.execute(
            f"UPDATE video_jobs SET {', '.join(sets)} WHERE id = ?",
            tuple(params),
        )

    async def fail_if_active(self, job_id: str, message: str) -> bool:
        """Mark a job failed only while it is still queued or in_progress.

        Returns whether the row was actually transitioned. Cancellation races
        the runner: an unconditional write can flip a job the runner *just*
        completed to failed (leaving its file orphaned and the client told a
        finished render failed), and the runner's own CancelledError handler
        can overwrite the more specific message the canceller recorded.
        Guarding on the current status makes the first writer win, which is
        the intended semantics for a terminal state.
        """
        now = int(time.time())
        async with self.db.transaction() as conn:
            cur = await conn.execute(
                "UPDATE video_jobs SET status = 'failed', error_message = ?, updated_at = ?"
                " WHERE id = ? AND status IN ('queued', 'in_progress')",
                (message, now, job_id),
            )
            changed = bool(cur.rowcount > 0)
            await cur.close()
        return changed

    async def mark_stale_failed(self, message: str) -> int:
        """Mark all queued/in_progress jobs as failed. Called on startup to
        reap jobs whose runner died with the previous process."""
        now = int(time.time())
        async with self.db.transaction() as conn:
            cur = await conn.execute(
                "UPDATE video_jobs SET status = 'failed', error_message = ?, updated_at = ?"
                " WHERE status IN ('queued', 'in_progress')",
                (message, now),
            )
            count = cur.rowcount
            await cur.close()
        return count
