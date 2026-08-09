"""Run independent per-request work concurrently, all-or-nothing."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any


async def run_all[T](factories: list[Callable[[], Coroutine[Any, Any, T]]]) -> list[T]:
    """Await every coroutine concurrently, preserving order.

    For a request whose ``n`` units of work are genuinely independent — OpenAI's
    ``n`` on an image endpoint, against a provider with no server-side batch
    parameter. Running those in a ``for`` loop makes the client wait for the
    *sum* of n generations inside one synchronous request, which at the
    permitted n=4 and a typical render is well past what OpenAI clients allow.

    On the first failure the remaining tasks are cancelled and awaited before
    the error propagates, rather than being left to run detached. Plain
    ``gather`` surfaces the first error immediately but does not cancel its
    siblings, so they would keep generating — costing money on a paid provider
    — and keep buffering assets into memory, long after the request that owns
    them has failed. ``return_exceptions=True`` is not the fix: it cancels
    nothing and makes the caller wait out every sibling before seeing the
    error.

    ComfyUI's batch path deliberately doesn't use this. It has to know *which*
    runs failed so it can ask the upstream to drop their queued prompts, which
    needs the task handles this hides.
    """
    tasks = [asyncio.create_task(factory()) for factory in factories]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
