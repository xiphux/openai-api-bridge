"""Bounded asyncio task pool.

We avoid FastAPI's BackgroundTasks because those are tied to a single request's
response lifecycle and can be cancelled if the client disconnects mid-response.
Video jobs need to outlive the HTTP request that kicks them off.

The pool keeps a strong reference to every spawned task in a module-level set so
the task isn't garbage-collected before it completes (a known asyncio footgun).
A secondary index by ``name`` enables external cancellation (used by the
``DELETE /v1/videos/{id}`` endpoint).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from typing import Any

log = logging.getLogger(__name__)

# Defensive upper bound on how long any one background task can hold its
# semaphore permit. Even with the comfyui poll loop bounded by its own
# generation timeout, this gives us a guaranteed permit-release floor in case
# a future backend introduces an unbounded await.
DEFAULT_TASK_TIMEOUT_SECONDS = 1800.0


class TaskScheduler:
    def __init__(
        self,
        max_concurrent: int,
        *,
        default_task_timeout_seconds: float = DEFAULT_TASK_TIMEOUT_SECONDS,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._sem = asyncio.Semaphore(max_concurrent)
        self._tasks: set[asyncio.Task[Any]] = set()
        self._tasks_by_name: dict[str, asyncio.Task[Any]] = {}
        self._default_timeout = default_task_timeout_seconds

    def submit(
        self,
        coro: Awaitable[Any],
        *,
        name: str | None = None,
        timeout_seconds: float | None = None,
    ) -> asyncio.Task[Any]:
        """Schedule a coroutine to run with concurrency capped by the semaphore.

        ``timeout_seconds`` overrides ``default_task_timeout_seconds`` for this
        one task. The timeout is enforced via ``asyncio.wait_for``, so the
        coroutine is cancelled (and the semaphore permit released) if it
        exceeds the budget.
        """
        timeout = self._default_timeout if timeout_seconds is None else timeout_seconds

        async def _runner() -> None:
            async with self._sem:
                try:
                    await asyncio.wait_for(coro, timeout=timeout)
                except TimeoutError:
                    log.error(
                        "Background task %r exceeded hard timeout of %.0fs; cancelled",
                        name,
                        timeout,
                    )
                except asyncio.CancelledError:
                    log.info("Background task %r cancelled", name)
                    raise
                except Exception:
                    log.exception("Background task %r failed", name)

        task = asyncio.create_task(_runner(), name=name)
        self._tasks.add(task)
        if name is not None:
            self._tasks_by_name[name] = task

        def _cleanup(t: asyncio.Task[Any]) -> None:
            self._tasks.discard(t)
            if name is not None and self._tasks_by_name.get(name) is t:
                del self._tasks_by_name[name]

        task.add_done_callback(_cleanup)
        return task

    def cancel(self, name: str) -> bool:
        """Cancel a running task by its submit-time ``name``.

        Returns ``True`` if a live task was found and cancellation was
        requested. ``False`` means the task either never existed or already
        finished — in both cases the caller's job state has already moved on.
        """
        task = self._tasks_by_name.get(name)
        if task is None or task.done():
            return False
        task.cancel()
        return True

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
