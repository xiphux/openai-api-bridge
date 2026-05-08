"""Disk-backed file store with SQLite metadata.

Layout: ``${FILES_DIR}/{id[0:2]}/{id[2:4]}/{id}{ext}``. Two-level shard prevents
one giant directory. ``id`` is 32 hex chars (``secrets.token_hex(16)``).

Atomicity guarantees:
  * Writes go to ``<path>.tmp`` and are renamed with ``Path.replace`` (atomic on
    the same filesystem). The DB row is inserted *after* the rename completes,
    so a row never points at a partial file.
  * Reads update ``last_accessed_at`` and return an absolute path. The eviction
    sweeper only deletes files whose row was already removed via DELETE...
    in-flight readers that opened the FD before the unlink keep streaming
    fine on Linux.
"""

from __future__ import annotations

import contextlib
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from .db import Database

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
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = abs_path.with_suffix(abs_path.suffix + ".tmp")
        tmp_path.write_bytes(data)
        tmp_path.replace(abs_path)  # atomic on same FS

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
        row = await self.db.fetchone(
            "SELECT * FROM generated_files WHERE id = ?", (file_id,)
        )
        return self._row_to_metadata(row) if row else None

    async def open_for_read(self, file_id: str) -> tuple[Path, FileMetadata] | None:
        """Return (absolute_path, metadata) and bump ``last_accessed_at``.

        The caller is expected to immediately open the file (e.g. via FileResponse)
        so an evictor unlinking concurrently doesn't break the in-flight read.
        """
        meta = await self.get_metadata(file_id)
        if meta is None:
            return None
        await self.db.execute(
            "UPDATE generated_files SET last_accessed_at = ? WHERE id = ?",
            (int(time.time()), file_id),
        )
        return self._absolute(meta.storage_path), meta

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

    async def total_byte_size(self) -> int:
        row = await self.db.fetchone(
            "SELECT COALESCE(SUM(byte_size), 0) AS total FROM generated_files"
        )
        return int(row["total"]) if row else 0
