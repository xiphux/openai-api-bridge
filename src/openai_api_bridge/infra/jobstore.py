"""SQLite-backed video_jobs CRUD + state transitions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from .db import Database

JobStatus = Literal["queued", "in_progress", "completed", "failed"]


@dataclass(slots=True, frozen=True)
class VideoJob:
    id: str
    status: JobStatus
    model: str
    prompt: str
    size: str | None
    seconds: float | None
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
    def _row_to_job(row) -> VideoJob:
        return VideoJob(
            id=row["id"],
            status=row["status"],
            model=row["model"],
            prompt=row["prompt"],
            size=row["size"],
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
        seconds: float | None,
        input_reference_file_id: str | None = None,
    ) -> VideoJob:
        now = int(time.time())
        await self.db.execute(
            """INSERT INTO video_jobs (
                   id, status, model, prompt, size, seconds, input_reference_file_id,
                   created_at, updated_at
               ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?)""",
            (job_id, model, prompt, size, seconds, input_reference_file_id, now, now),
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
    ) -> None:
        """Patch-style update. Only fields explicitly passed (not None) are written."""
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
