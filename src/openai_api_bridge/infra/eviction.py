"""Cache eviction sweeper.

Runs a periodic asyncio task that combines two policies:

1. **TTL** — delete files older than ``retention_seconds``.
2. **LRU-by-size** — if the total still exceeds ``max_cache_bytes``, delete
   least-recently-accessed files until back under cap.

Both sweeps skip rows with ``pinned`` set — but **nothing sets it**. See
:meth:`~openai_api_bridge.infra.filestore.FileStore.set_pinned`: the flag is
schema and filter with no writer, so in practice every stored file is
evictable, including a video whose job just completed. The filters are kept
because they are the cheap half of the mechanism and removing them would mean
a migration to get back.

The sweeper deletes the SQLite row first; readers that already opened the FD
keep streaming fine on Linux even after the unlink.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time

from .filestore import FileStore

log = logging.getLogger(__name__)


async def run_eviction_pass(
    filestore: FileStore,
    *,
    retention_seconds: int,
    max_cache_bytes: int,
) -> tuple[int, int]:
    """Run one full pass. Returns ``(ttl_deleted, lru_deleted)``."""
    now = int(time.time())
    cutoff = now - retention_seconds

    ttl_rows = await filestore.db.fetchall(
        "SELECT id FROM generated_files WHERE pinned = 0 AND created_at < ?",
        (cutoff,),
    )
    # Batched: a sweep retiring a few thousand files used to issue a SELECT,
    # a DELETE, a commit and a blocking unlink apiece, all on the event loop.
    ttl_deleted = await filestore.delete_many([row["id"] for row in ttl_rows])

    lru_deleted = 0
    total = await filestore.total_byte_size()
    if total > max_cache_bytes:
        candidates = await filestore.db.fetchall(
            "SELECT id, byte_size FROM generated_files WHERE pinned = 0"
            " ORDER BY last_accessed_at ASC"
        )
        # Pick the victims first, then delete them in one batch, so the cap
        # arithmetic stays identical to the row-at-a-time version.
        victims: list[str] = []
        for row in candidates:
            if total <= max_cache_bytes:
                break
            victims.append(row["id"])
            total -= int(row["byte_size"])
        lru_deleted = await filestore.delete_many(victims)

    return ttl_deleted, lru_deleted


class EvictionLoop:
    def __init__(
        self,
        filestore: FileStore,
        *,
        retention_seconds: int,
        max_cache_bytes: int,
        interval_seconds: float,
    ) -> None:
        self.filestore = filestore
        self.retention_seconds = retention_seconds
        self.max_cache_bytes = max_cache_bytes
        self.interval_seconds = interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    def start(self) -> None:
        if self._task is not None:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._loop(), name="eviction-loop")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=10.0)
        except TimeoutError:
            log.warning("Eviction loop did not stop in time; cancelling")
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        self._task = None

    async def _loop(self) -> None:
        await self._safe_pass()
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.interval_seconds,
                )
                return  # stop signal received
            except TimeoutError:
                pass  # interval elapsed — fall through to next pass
            await self._safe_pass()

    async def _safe_pass(self) -> None:
        try:
            ttl, lru = await run_eviction_pass(
                self.filestore,
                retention_seconds=self.retention_seconds,
                max_cache_bytes=self.max_cache_bytes,
            )
            if ttl or lru:
                log.info("Eviction pass: TTL=%d LRU=%d", ttl, lru)
            else:
                log.debug("Eviction pass: nothing to do")
        except Exception:
            log.exception("Eviction pass failed; will retry next interval")
