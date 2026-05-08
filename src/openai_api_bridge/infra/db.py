"""SQLite + aiosqlite wrapper with WAL mode and a tiny inline migration runner.

Single shared connection for the whole process. WAL allows concurrent reads
across asyncio tasks; writes serialize on the connection's internal lock.
This is fine for our scale (low write rate, single-process).

Migrations are embedded as a list of ``(version, sql)`` tuples below — no
filesystem lookup, so the bridge ships fully self-contained whether deployed
as a wheel, a Docker image, or a checked-out source tree.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

_PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA busy_timeout=5000",
    "PRAGMA foreign_keys=ON",
)


class Database:
    """Async wrapper around a single aiosqlite Connection."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self._conn is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(self.path))
        for pragma in _PRAGMAS:
            await conn.execute(pragma)
        await conn.commit()
        conn.row_factory = aiosqlite.Row
        self._conn = conn

    async def close(self) -> None:
        if self._conn is not None:
            # Flush the WAL into the main .db and reset the -wal file to zero
            # bytes. The -wal/-shm files persist (SQLite recreates them on
            # next open), but a backup of state.db taken right after shutdown
            # is now self-contained — no out-of-band WAL replay needed.
            try:
                await self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                await self._conn.commit()
            except Exception:
                # Don't block shutdown on a checkpoint failure; SQLite will
                # recover the WAL on next open regardless.
                pass
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected — call connect() first")
        return self._conn

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        await self.conn.execute(sql, params)
        await self.conn.commit()

    async def fetchone(
        self, sql: str, params: Sequence[Any] = ()
    ) -> aiosqlite.Row | None:
        async with self.conn.execute(sql, params) as cur:
            return await cur.fetchone()

    async def fetchall(
        self, sql: str, params: Sequence[Any] = ()
    ) -> list[aiosqlite.Row]:
        async with self.conn.execute(sql, params) as cur:
            return list(await cur.fetchall())

    @asynccontextmanager
    async def transaction(self):
        """Group multiple statements into a single commit. Rolls back on error."""
        try:
            yield self.conn
            await self.conn.commit()
        except Exception:
            await self.conn.rollback()
            raise


_MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE generated_files (
          id                  TEXT    PRIMARY KEY,
          storage_path        TEXT    NOT NULL UNIQUE,
          content_type        TEXT    NOT NULL,
          byte_size           INTEGER NOT NULL,
          kind                TEXT    NOT NULL CHECK (kind IN ('image','video')),
          source_backend      TEXT    NOT NULL,
          source_model        TEXT    NOT NULL,
          prompt_excerpt      TEXT,
          created_at          INTEGER NOT NULL,
          last_accessed_at    INTEGER NOT NULL,
          pinned              INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX idx_files_last_accessed ON generated_files(last_accessed_at);
        CREATE INDEX idx_files_created       ON generated_files(created_at);

        CREATE TABLE video_jobs (
          id                       TEXT    PRIMARY KEY,
          status                   TEXT    NOT NULL CHECK (status IN ('queued','in_progress','completed','failed')),
          model                    TEXT    NOT NULL,
          prompt                   TEXT    NOT NULL,
          size                     TEXT,
          seconds                  REAL,
          input_reference_file_id  TEXT,
          file_id                  TEXT,
          upstream_id              TEXT,
          error_message            TEXT,
          created_at               INTEGER NOT NULL,
          updated_at               INTEGER NOT NULL,
          progress_pct             INTEGER
        );
        CREATE INDEX idx_jobs_status_updated ON video_jobs(status, updated_at);
        """,
    ),
]


async def run_migrations(db: Database) -> int:
    """Apply any embedded migrations newer than the recorded schema_version.

    Returns the version after applying. Add new migrations to ``_MIGRATIONS``
    in ascending version order; never edit a previously-shipped migration.
    """
    await db.execute(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    row = await db.fetchone("SELECT value FROM meta WHERE key = 'schema_version'")
    current = int(row["value"]) if row else 0

    for version, sql in _MIGRATIONS:
        if version <= current:
            continue
        await db.conn.executescript(sql)
        await db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(version),),
        )
        current = version
    return current


def rows_to_dicts(rows: Iterable[aiosqlite.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]
