"""Bounded asyncio task pool.

We avoid FastAPI's BackgroundTasks because those are tied to a single request's
response lifecycle and can be cancelled if the client disconnects mid-response.
Video jobs need to outlive the HTTP request that kicks them off.

The pool keeps a strong reference to every spawned task in a module-level set so
the task isn't garbage-collected before it completes (a known asyncio footgun).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from typing import Any

log = logging.getLogger(__name__)


class TaskScheduler:
    def __init__(self, max_concurrent: int) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._sem = asyncio.Semaphore(max_concurrent)
        self._tasks: set[asyncio.Task[Any]] = set()

    def submit(self, coro: Awaitable[Any], *, name: str | None = None) -> asyncio.Task[Any]:
        """Schedule a coroutine to run with concurrency capped by the semaphore."""

        async def _runner() -> None:
            async with self._sem:
                try:
                    await coro
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("Background task %r failed", name)

        task = asyncio.create_task(_runner(), name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    @property
    def in_flight(self) -> int:
        return len(self._tasks)

    async def shutdown(self, *, timeout: float = 30.0) -> None:
        """Wait for all in-flight tasks to finish, then cancel any stragglers."""
        if not self._tasks:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True),
                timeout=timeout,
            )
        except TimeoutError:
            log.warning(
                "Scheduler shutdown timed out with %d tasks still running; cancelling",
                len(self._tasks),
            )
            for t in list(self._tasks):
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
