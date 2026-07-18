"""A small single-value cache with a TTL, for upstream model catalogues.

``GET /v1/models`` fans out to every configured provider on every request, so
an uncached backend pays an upstream round trip each time a client refreshes
its model picker — and Venice pays two, since its catalogue is split across
listings.

A TTL rather than a permanent cache: a model added upstream should appear
without restarting the bridge. Fetches run under the lock, so a burst of
concurrent callers collapses into one fetch that they all wait for, rather than
each firing its own.

Two ways in:

* :meth:`get` is the whole pattern for a backend whose listing is one call and
  whose result is always cacheable.
* :attr:`lock` / :meth:`fresh` / :meth:`store` are for a backend that has to
  decide *whether* the result is worth caching — Venice deliberately declines
  to store a listing whose second half failed, since that would pin its edit
  routing unresolved for the whole TTL.

A failure is remembered for ``failure_cooldown_seconds`` and re-raised to
callers arriving inside that window. That matters because the fetch runs *under
the lock*: without it, a burst arriving during an upstream hang would each
re-acquire and start their own fetch, so the Nth caller waits N x timeout —
worse than the uncached behaviour, where they at least hung concurrently. With
it, the first caller pays the timeout and the rest fail fast.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable


class AsyncTTLCache[T]:
    def __init__(self, ttl_seconds: float, failure_cooldown_seconds: float = 0.0) -> None:
        self.ttl_seconds = ttl_seconds
        self.failure_cooldown_seconds = failure_cooldown_seconds
        self.lock = asyncio.Lock()
        self._value: T | None = None
        self._stored_at: float | None = None
        self._failure: BaseException | None = None
        self._failed_at: float | None = None

    def fresh(self) -> T | None:
        """The cached value if it hasn't aged out, else ``None``.

        A non-positive TTL disables caching outright, so this is always
        ``None`` and every call re-fetches.
        """
        if self._value is None or self._stored_at is None:
            return None
        if self.ttl_seconds <= 0:
            return None
        if time.monotonic() - self._stored_at >= self.ttl_seconds:
            return None
        return self._value

    def store(self, value: T) -> None:
        self._value = value
        self._stored_at = time.monotonic()
        self._failure = None
        self._failed_at = None

    def note_failure(self, error: BaseException) -> None:
        """Remember a failed fetch so callers in the cooldown fail fast."""
        self._failure = error
        self._failed_at = time.monotonic()

    def pending_failure(self) -> BaseException | None:
        """The remembered error while its cooldown is open, else ``None``."""
        if self._failure is None or self._failed_at is None:
            return None
        if self.failure_cooldown_seconds <= 0:
            return None
        if time.monotonic() - self._failed_at >= self.failure_cooldown_seconds:
            return None
        return self._failure

    def invalidate(self) -> None:
        self._value = None
        self._stored_at = None
        self._failure = None
        self._failed_at = None

    async def get(self, fetch: Callable[[], Awaitable[T]]) -> T:
        """Return the cached value, fetching it under the lock if stale."""
        async with self.lock:
            cached = self.fresh()
            if cached is not None:
                return cached
            recent = self.pending_failure()
            if recent is not None:
                raise recent
            try:
                value = await fetch()
            except Exception as e:
                self.note_failure(e)
                raise
            self.store(value)
            return value
