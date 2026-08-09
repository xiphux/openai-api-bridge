"""Background work that outlives the request which started it.

Two shapes, both here because they share the same hazard: a task nobody holds a
reference to can be garbage-collected mid-flight, and a task nobody drains is
still pending when the loop is torn down.

* :class:`TaskScheduler` — a bounded pool for video jobs. We avoid FastAPI's
  BackgroundTasks because those are tied to a single request's response
  lifecycle and can be cancelled if the client disconnects mid-response.
  Video jobs need to outlive the HTTP request that kicks them off. A secondary
  index by ``name`` enables external cancellation (used by
  ``DELETE /v1/videos/{id}``).
* :class:`SingleFlight` — at most one live task per key, for work a request
  starts, stops waiting on, and wants the *next* request to join rather than
  duplicate.

Both remove entries with an identity check rather than by key. That is not
incidental: a key can be reused while the previous task is still running, and
an unconditional delete lets a finishing task evict its live successor —
leaving that successor untracked, undrained, and holding the very resource the
map exists to protect.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
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


class SingleFlight[T]:
    """At most one live task per key; later callers join it rather than duplicate it.

    For work a request starts but may stop waiting on — the model-catalogue
    fetches behind ``GET /v1/models``, where a slow provider is dropped from
    one listing while its fetch keeps running to warm that backend's cache.

    The entry is created *with* the task, not after the starter gives up.
    Registering later leaves a window — the whole time the starter is waiting —
    in which every concurrent caller sees an empty map, starts its own task,
    and then overwrites the previous entry on the way out. What made that
    subtle rather than obvious is that the survivors are fine: it is the
    *untracked* task that ends up holding the upstream request, so a drain
    awaits the wrong one and the connection it needs is closed underneath it.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[T]] = {}

    def join_or_start(
        self,
        key: str,
        factory: Callable[[], Coroutine[Any, Any, T]],
        *,
        name: str | None = None,
    ) -> asyncio.Task[T]:
        """The live task for ``key``, starting one from ``factory`` if there isn't one.

        A factory rather than a coroutine so nothing is constructed when an
        existing task is joined — an un-awaited coroutine would warn.
        """
        existing = self._tasks.get(key)
        if existing is not None:
            return existing
        task = asyncio.create_task(factory(), name=name)
        self._tasks[key] = task

        def _forget(finished: asyncio.Task[T]) -> None:
            # Identity, not key: by the time this fires the slot may hold a
            # newer task, and removing that one would untrack a live fetch.
            if self._tasks.get(key) is finished:
                del self._tasks[key]

        task.add_done_callback(_forget)
        return task

    def __len__(self) -> int:
        return len(self._tasks)

    def keys(self) -> list[str]:
        return list(self._tasks)

    async def cancel_all(self) -> None:
        """Cancel every live task and wait for the cancellations to land.

        Cancel rather than wait-then-cancel. Everything in here is by
        construction work some request already gave up on, and what it would
        finish for — a cache inside a backend the process is about to close —
        does not survive the shutdown either. Waiting first only adds its
        grace period to every restart.
        """
        tasks = list(self._tasks.values())
        if not tasks:
            return
        log.info("Cancelling %d in-flight background fetch(es)", len(tasks))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
