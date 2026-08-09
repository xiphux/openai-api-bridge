"""A small single-value cache with a TTL, for upstream model catalogues.

``GET /v1/models`` fans out to every configured provider on every request, so
an uncached backend pays an upstream round trip each time a client refreshes
its model picker — and Venice pays two, since its catalogue is split across
listings.

A TTL rather than a permanent cache: a model added upstream should appear
without restarting the bridge. Fetches run under the lock, so a burst of
concurrent callers collapses into one fetch that they all wait for, rather than
each firing its own.

:meth:`get` is the whole pattern. Its ``ttl_for`` hook lets a caller pick the
window from the value it just fetched, for a result that is usable but
incomplete — Venice serves a listing whose second half failed, but keeps it
only briefly so the missing half is re-attempted soon. An override may only
shorten the window, never create or extend one, so ``ttl_seconds = 0`` always
means "no caching".

:attr:`lock`, :meth:`fresh`, :meth:`store` and :meth:`note_failure` are exposed
for a caller that needs to drive the sequence itself. The fal backend does:
a failed catalogue fetch there degrades to the explicitly configured models
rather than raising, which :meth:`get` can't express — but the TTL, the failure
cooldown and the single-flight lock are the same concerns, and were hand-rolled
there before.

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

from ..errors import UpstreamAuthError

# How long a rejected credential is remembered, when that's longer than the
# provider's configured failure cooldown. Long enough that a genuinely bad key
# stops generating traffic; short enough that a 403 which wasn't really about
# our credential heals without a restart.
AUTH_FAILURE_COOLDOWN_SECONDS = 300.0


class AsyncTTLCache[T]:
    def __init__(self, ttl_seconds: float, failure_cooldown_seconds: float = 0.0) -> None:
        self.ttl_seconds = ttl_seconds
        self.failure_cooldown_seconds = failure_cooldown_seconds
        self.lock = asyncio.Lock()
        self._value: T | None = None
        self._stored_at: float | None = None
        self._stored_ttl: float = ttl_seconds
        self._failure: BaseException | None = None
        self._failed_at: float | None = None

    def fresh(self) -> T | None:
        """The cached value if it hasn't aged out, else ``None``.

        A non-positive TTL disables caching outright, so this is always
        ``None`` and every call re-fetches.
        """
        if self._value is None or self._stored_at is None:
            return None
        if self._stored_ttl <= 0:
            return None
        if time.monotonic() - self._stored_at >= self._stored_ttl:
            return None
        return self._value

    def store(self, value: T, *, ttl_seconds: float | None = None) -> None:
        """Cache a value, optionally for less than the configured TTL.

        A shorter TTL is for a result that's usable but incomplete — worth
        serving, but worth re-attempting sooner than a healthy one. The request
        is clamped to the configured TTL so that an override can only ever
        shorten it: otherwise ``ttl_seconds = 0`` would stop disabling caching
        the moment a caller passed an override, and an override larger than the
        TTL would hold an incomplete result *longer* than a healthy one.
        """
        requested = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        self._value = value
        self._stored_at = time.monotonic()
        self._stored_ttl = min(requested, self.ttl_seconds)
        self._failure = None
        self._failed_at = None

    def note_failure(self, error: BaseException) -> None:
        """Remember a failed fetch so callers in the cooldown fail fast."""
        self._failure = error
        self._failed_at = time.monotonic()

    def _cooldown_for(self, error: BaseException) -> float:
        """How long this particular failure is worth remembering.

        A rejected credential gets a longer window than an ordinary blip:
        provider tokens are read from the environment at startup, so a genuine
        ``UpstreamAuthError`` won't start working again without a restart, and
        re-asking every 30s is pure noise against a key that cannot change.

        It is a longer window and not a permanent latch, deliberately. The
        error is only as reliable as the status that produced it, and 403 in
        particular is routinely *not* about our credential — a WAF or
        Cloudflare interstitial, a geo/IP block, an org-level quota. Latching
        on one of those would remove the provider from ``/v1/models`` for the
        life of the process, and because the models endpoint drops failing
        providers silently, it would vanish with no client-visible error. The
        asymmetry is stark: recovering costs one extra fetch per window, while
        latching wrongly costs a restart to notice. README.md ("Model
        catalogue caching") promises the provider "recovers on its own", and
        that stays true.
        """
        if isinstance(error, UpstreamAuthError):
            return max(self.failure_cooldown_seconds, AUTH_FAILURE_COOLDOWN_SECONDS)
        return self.failure_cooldown_seconds

    def pending_failure(self) -> BaseException | None:
        """The remembered error while its cooldown is open, else ``None``."""
        if self._failure is None or self._failed_at is None:
            return None
        # A cooldown of zero disables failure memory outright, as documented
        # for catalog_retry_seconds ("0 retries immediately"). An auth failure
        # doesn't get to override an operator's explicit "don't cache".
        if self.failure_cooldown_seconds <= 0:
            return None
        if time.monotonic() - self._failed_at >= self._cooldown_for(self._failure):
            return None
        return self._failure

    async def get(
        self,
        fetch: Callable[[], Awaitable[T]],
        *,
        ttl_for: Callable[[T], float] | None = None,
    ) -> T:
        """Return the cached value, fetching it under the lock if stale.

        ``ttl_for`` lets a caller pick the TTL from the fetched value — used to
        cache an incomplete result briefly rather than for the full window.
        """
        async with self.lock:
            cached = self.fresh()
            if cached is not None:
                return cached
            recent = self.pending_failure()
            if recent is not None:
                # Clear the accumulated traceback first: this is one shared
                # object re-raised for every caller in the window, and Python
                # appends a frame each time without resetting. Left alone the
                # chain grows unboundedly within a window, pinning each
                # request's locals — cheap while the error is only logged by
                # message, ruinous the moment something calls log.exception on
                # it.
                raise recent.with_traceback(None)
            try:
                value = await fetch()
            except Exception as e:
                self.note_failure(e)
                raise
            self.store(value, ttl_seconds=ttl_for(value) if ttl_for else None)
            return value
