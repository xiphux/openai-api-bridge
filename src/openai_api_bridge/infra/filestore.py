"""Disk-backed file store with SQLite metadata.

Layout: ``${FILES_DIR}/{id[0:2]}/{id[2:4]}/{id}{ext}``. Two-level shard prevents
one giant directory. ``id`` is 32 hex chars (``secrets.token_hex(16)``).

Atomicity guarantees:
  * Writes go to ``<path>.tmp`` and are renamed with ``Path.replace`` (atomic on
    the same filesystem). The DB row is inserted *after* the rename completes,
    so a row never points at a partial file.
  * Reads update ``last_accessed_at`` and return an absolute path. Because the
    caller (``FileResponse``) opens the path *later*, a row whose bytes are
    gone must be reported as absent rather than handed out: see
    :meth:`FileStore.open_for_read`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .db import Database

log = logging.getLogger(__name__)


def _write_atomic(tmp_path: Path, final_path: Path, data: bytes) -> None:
    """Write ``data`` to ``final_path`` via a temp file. Runs in a worker thread."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(data)
    tmp_path.replace(final_path)  # atomic on same FS


def _unlink_all(paths: list[Path]) -> None:
    """Unlink every path, ignoring ones already gone. Runs in a worker thread."""
    for path in paths:
        with contextlib.suppress(FileNotFoundError, OSError):
            path.unlink()


# SQLite's default parameter ceiling is 999; stay well under it so a large
# sweep can't blow the limit on the IN (...) clauses below.
_DELETE_CHUNK = 400


_EXT_BY_TYPE: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
}


@dataclass(slots=True, frozen=True)
class FileMetadata:
    id: str
    storage_path: str  # relative to files_dir
    content_type: str
    byte_size: int
    kind: str  # "image" | "video"
    source_backend: str
    source_model: str
    prompt_excerpt: str | None
    created_at: int
    last_accessed_at: int
    pinned: bool


class FileStore:
    def __init__(self, db: Database, files_dir: Path) -> None:
        self.db = db
        self.files_dir = files_dir

    # --- internals ---------------------------------------------------------

    def _ext_for(self, content_type: str) -> str:
        return _EXT_BY_TYPE.get(content_type.lower(), "")

    def _disk_path(self, file_id: str, ext: str) -> Path:
        return self.files_dir / file_id[0:2] / file_id[2:4] / f"{file_id}{ext}"

    def _absolute(self, storage_path: str) -> Path:
        return self.files_dir / storage_path

    @staticmethod
    def _row_to_metadata(row) -> FileMetadata:
        return FileMetadata(
            id=row["id"],
            storage_path=row["storage_path"],
            content_type=row["content_type"],
            byte_size=row["byte_size"],
            kind=row["kind"],
            source_backend=row["source_backend"],
            source_model=row["source_model"],
            prompt_excerpt=row["prompt_excerpt"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            pinned=bool(row["pinned"]),
        )

    # --- public API --------------------------------------------------------

    async def put(
        self,
        data: bytes,
        *,
        content_type: str,
        kind: str,
        source_backend: str,
        source_model: str,
        prompt_excerpt: str | None = None,
        pinned: bool = False,
    ) -> str:
        """Persist bytes + metadata; return the new file_id."""
        if kind not in ("image", "video"):
            raise ValueError(f"kind must be 'image' or 'video', got {kind!r}")

        file_id = secrets.token_hex(16)
        ext = self._ext_for(content_type)
        abs_path = self._disk_path(file_id, ext)
        tmp_path = abs_path.with_suffix(abs_path.suffix + ".tmp")

        # Off the event loop: a generated video is routinely hundreds of MB,
        # and the bridge runs a single uvicorn worker, so a synchronous write
        # stalls every other client for its duration (measured ~58ms for
        # 200MB on local SSD, and it degrades badly on network-backed storage).
        await asyncio.to_thread(_write_atomic, tmp_path, abs_path, data)

        relative = str(abs_path.relative_to(self.files_dir))
        now = int(time.time())
        excerpt = (prompt_excerpt or "")[:500] or None

        await self.db.execute(
            """INSERT INTO generated_files (
                   id, storage_path, content_type, byte_size, kind,
                   source_backend, source_model, prompt_excerpt,
                   created_at, last_accessed_at, pinned
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                file_id,
                relative,
                content_type,
                len(data),
                kind,
                source_backend,
                source_model,
                excerpt,
                now,
                now,
                1 if pinned else 0,
            ),
        )
        return file_id

    async def get_metadata(self, file_id: str) -> FileMetadata | None:
        row = await self.db.fetchone("SELECT * FROM generated_files WHERE id = ?", (file_id,))
        return self._row_to_metadata(row) if row else None

    async def open_for_read(self, file_id: str) -> tuple[Path, FileMetadata] | None:
        """Return (absolute_path, metadata) and bump ``last_accessed_at``.

        Returns ``None`` when the row exists but its bytes don't. The caller
        opens the path *after* we return it (``FileResponse`` stats it at send
        time), so handing back a path to a missing file surfaces as a
        ``RuntimeError`` and a 500 rather than the 404 the caller expects.
        Three ways a row outlives its file: ``FILES_DIR`` wiped while the DB
        persists (tmpfs, a recreated volume), a crash between the row DELETE
        and the unlink in :meth:`delete`, and an eviction pass landing between
        our metadata read and the caller's open.

        The orphan row is dropped on the way out — it describes bytes that no
        longer exist, and leaving it would keep its ``byte_size`` in the
        eviction sweeper's total.
        """
        meta = await self.get_metadata(file_id)
        if meta is None:
            return None
        abs_path = self._absolute(meta.storage_path)
        if not abs_path.is_file():
            log.warning(
                "File %s has a metadata row but no bytes at %s; reaping the row",
                file_id,
                abs_path,
            )
            await self.db.execute("DELETE FROM generated_files WHERE id = ?", (file_id,))
            return None
        await self.db.execute(
            "UPDATE generated_files SET last_accessed_at = ? WHERE id = ?",
            (int(time.time()), file_id),
        )
        return abs_path, meta

    async def set_pinned(self, file_id: str, pinned: bool) -> None:
        await self.db.execute(
            "UPDATE generated_files SET pinned = ? WHERE id = ?",
            (1 if pinned else 0, file_id),
        )

    async def delete(self, file_id: str) -> None:
        meta = await self.get_metadata(file_id)
        if meta is None:
            return
        await self.db.execute("DELETE FROM generated_files WHERE id = ?", (file_id,))
        with contextlib.suppress(FileNotFoundError):
            self._absolute(meta.storage_path).unlink()

    async def delete_many(self, file_ids: Sequence[str]) -> int:
        """Delete many files at once. Returns how many rows were removed.

        The eviction sweeper retires files in bulk, and doing that one at a
        time cost a SELECT, a DELETE, a commit and a blocking unlink each —
        a few thousand round trips and commits per pass, all on the single
        event loop. Batch the SQL and push the unlinks to a worker thread.
        """
        removed = 0
        for start in range(0, len(file_ids), _DELETE_CHUNK):
            chunk = list(file_ids[start : start + _DELETE_CHUNK])
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            rows = await self.db.fetchall(
                f"SELECT storage_path FROM generated_files WHERE id IN ({placeholders})",
                tuple(chunk),
            )
            if not rows:
                continue
            await self.db.execute(
                f"DELETE FROM generated_files WHERE id IN ({placeholders})",
                tuple(chunk),
            )
            paths = [self._absolute(row["storage_path"]) for row in rows]
            await asyncio.to_thread(_unlink_all, paths)
            removed += len(rows)
        return removed

    async def total_byte_size(self) -> int:
        row = await self.db.fetchone(
            "SELECT COALESCE(SUM(byte_size), 0) AS total FROM generated_files"
        )
        return int(row["total"]) if row else 0
