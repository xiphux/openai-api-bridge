"""``GET /v1/models`` aggregation behaviour.

Covers the properties the endpoint promises that aren't visible from a
single-provider test: providers are queried concurrently, one provider
failing doesn't take the listing down with it, and one provider being *slow*
doesn't either.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from openai_api_bridge.api.models import _entries_for, list_models
from openai_api_bridge.backends.base import Backend, ModelEntry
from openai_api_bridge.errors import UpstreamError


class _BarrierBackend(Backend):
    """Blocks in list_models until every provider has reached the barrier.

    If the endpoint fans out sequentially the first backend waits forever,
    so the barrier times out and the test fails loudly rather than slowly.
    """

    def __init__(self, barrier: asyncio.Barrier, entries: list[ModelEntry]) -> None:
        self._barrier = barrier
        self._entries = entries

    async def list_models(self) -> list[ModelEntry]:
        await asyncio.wait_for(self._barrier.wait(), timeout=5.0)
        return self._entries


class _FailingBackend(Backend):
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def list_models(self) -> list[ModelEntry]:
        raise self._error


def _request_with(providers: list[tuple[str, Backend]], *, budget: float = 5.0) -> Any:
    dispatcher = SimpleNamespace(all_providers=lambda: providers)
    settings = SimpleNamespace(models_timeout_seconds=budget)
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(dispatcher=dispatcher, settings=settings))
    )


async def test_providers_are_queried_concurrently() -> None:
    barrier = asyncio.Barrier(3)
    providers: list[tuple[str, Backend]] = [
        (f"p{i}", _BarrierBackend(barrier, [ModelEntry(id=f"m{i}", kind="image")]))
        for i in range(3)
    ]

    body = await list_models(_request_with(providers))

    assert [row["id"] for row in body["data"]] == ["p0/m0", "p1/m1", "p2/m2"]


async def test_listing_order_follows_provider_order() -> None:
    """Concurrency must not reorder the catalogue — clients see a stable list."""
    barrier = asyncio.Barrier(2)
    providers: list[tuple[str, Backend]] = [
        ("alpha", _BarrierBackend(barrier, [ModelEntry(id="one", kind="image")])),
        ("beta", _BarrierBackend(barrier, [ModelEntry(id="two", kind="video")])),
    ]

    body = await list_models(_request_with(providers))

    assert [row["id"] for row in body["data"]] == ["alpha/one", "beta/two"]


@pytest.mark.parametrize(
    "error",
    [
        UpstreamError("catalogue is down"),
        httpx.ConnectError("connection refused"),
        KeyError("architecture"),
        ValueError("unexpected catalogue shape"),
    ],
    ids=["bridge-error", "raw-httpx", "key-error", "value-error"],
)
async def test_any_provider_failure_is_contained(error: Exception) -> None:
    """Containment can't depend on adapters remembering to wrap their errors.

    A bare httpx error or an unexpected catalogue shape previously escaped
    the BridgeError-only handler and 500'd the whole endpoint, taking every
    healthy provider's models with it.
    """
    providers: list[tuple[str, Backend]] = [
        ("good", _StaticBackend([ModelEntry(id="m", kind="image")])),
        ("bad", _FailingBackend(error)),
    ]

    body = await list_models(_request_with(providers))

    assert [row["id"] for row in body["data"]] == ["good/m"]


class _StaticBackend(Backend):
    def __init__(self, entries: list[ModelEntry]) -> None:
        self._entries = entries

    async def list_models(self) -> list[ModelEntry]:
        return self._entries


class _SlowBackend(Backend):
    """Blocks until released, recording whether its fetch ran to completion."""

    def __init__(self, entries: list[ModelEntry]) -> None:
        self._entries = entries
        self.release = asyncio.Event()
        self.finished = False

    async def list_models(self) -> list[ModelEntry]:
        await self.release.wait()
        self.finished = True
        return self._entries


async def test_slow_provider_does_not_stall_the_listing() -> None:
    """A wedged upstream must not hold every healthy provider hostage."""
    slow = _SlowBackend([ModelEntry(id="slow", kind="image")])
    providers: list[tuple[str, Backend]] = [
        ("fast", _StaticBackend([ModelEntry(id="quick", kind="image")])),
        ("slow", slow),
    ]

    body = await asyncio.wait_for(list_models(_request_with(providers, budget=0.05)), timeout=5.0)

    assert [row["id"] for row in body["data"]] == ["fast/quick"]
    slow.release.set()


async def test_slow_provider_fetch_is_left_running() -> None:
    """The bound omits a provider from one listing; it must not cancel its fetch.

    Every backend caches its catalogue, and a cancelled fetch caches nothing —
    so cancelling here would drop a merely-slow provider from every listing
    forever, each request killing the fetch that would have served the next.
    """
    slow = _SlowBackend([ModelEntry(id="slow", kind="image")])

    body = await list_models(_request_with([("slow", slow)], budget=0.05))
    assert body["data"] == []

    slow.release.set()
    await asyncio.sleep(0)  # let the orphaned fetch resume
    await asyncio.sleep(0)
    assert slow.finished, "the shielded fetch should have run to completion"


async def test_zero_budget_disables_the_bound() -> None:
    """An operator who would rather wait than lose a provider can opt out."""
    slow = _SlowBackend([ModelEntry(id="slow", kind="image")])

    async def release_shortly() -> None:
        await asyncio.sleep(0.05)
        slow.release.set()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(release_shortly())
        listing = tg.create_task(list_models(_request_with([("slow", slow)], budget=0.0)))

    assert [row["id"] for row in listing.result()["data"]] == ["slow/slow"]


@pytest.mark.parametrize("kind", ["image", "video"])
async def test_kind_is_surfaced(kind: str) -> None:
    providers: list[tuple[str, Backend]] = [
        ("p", _StaticBackend([ModelEntry(id="m", kind=kind)])),
    ]

    body = await list_models(_request_with(providers))

    assert body["data"][0]["kind"] == kind


def _fresh_lingering(monkeypatch: Any) -> Any:
    """Swap in an empty registry so cases don't inherit each other's tasks."""
    from openai_api_bridge.api import models as models_api
    from openai_api_bridge.infra.tasks import SingleFlight

    registry: SingleFlight[list[ModelEntry]] = SingleFlight()
    monkeypatch.setattr(models_api, "_lingering", registry)
    return registry


def _live_fetch_tasks() -> list[asyncio.Task[Any]]:
    return [
        t
        for t in asyncio.all_tasks()
        if (t.get_name() or "").startswith("list-models-") and not t.done()
    ]


async def test_concurrent_requests_share_one_fetch(monkeypatch: Any) -> None:
    """The registry must be authoritative from the task's birth, not from the
    moment the first caller gives up.

    Registering only after the budget expires leaves the whole budget window
    open: every concurrent request sees an empty registry and starts its own
    fetch. The survivors look fine, which is what makes it subtle — it's the
    *untracked* task that ends up holding the upstream request, so a shutdown
    drain awaits the wrong one.
    """
    registry = _fresh_lingering(monkeypatch)
    slow = _SlowBackend([ModelEntry(id="slow", kind="image")])
    providers: list[tuple[str, Backend]] = [("slow", slow)]

    bodies = await asyncio.gather(
        list_models(_request_with(providers, budget=0.05)),
        list_models(_request_with(providers, budget=0.05)),
        list_models(_request_with(providers, budget=0.05)),
    )

    assert all(b["data"] == [] for b in bodies)
    live = _live_fetch_tasks()
    assert len(live) == 1, f"{len(live)} concurrent fetches started; expected 1"
    assert registry.keys() == ["slow"]
    assert live[0] is registry.join_or_start("slow", lambda: _entries_for("slow", slow))

    slow.release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)


async def test_a_finished_fetch_does_not_evict_its_successor(monkeypatch: Any) -> None:
    """Removal is by identity, not by key.

    A key outlives the task occupying it. Popping by key lets a finishing task
    delete whatever is in the slot now — untracking a live fetch, which is the
    one thing this registry exists to prevent.
    """
    from openai_api_bridge.infra.tasks import SingleFlight

    registry: SingleFlight[str] = SingleFlight()
    first_done = asyncio.Event()
    second_done = asyncio.Event()

    async def first() -> str:
        await first_done.wait()
        return "first"

    async def second() -> str:
        await second_done.wait()
        return "second"

    task_one = registry.join_or_start("p", first)
    # Force the successor into the slot, as an overwrite would have done.
    registry._tasks["p"] = task_two = asyncio.create_task(second())

    first_done.set()
    await task_one
    await asyncio.sleep(0)

    assert registry.keys() == ["p"], "a finished task evicted its live successor"
    assert registry._tasks["p"] is task_two

    second_done.set()
    await task_two


async def test_a_cancelled_request_leaves_its_fetch_tracked(monkeypatch: Any) -> None:
    """`shield` keeps the fetch alive, so the registry must still hold it."""
    from openai_api_bridge.api import models as models_api

    registry = _fresh_lingering(monkeypatch)
    slow = _SlowBackend([ModelEntry(id="slow", kind="image")])

    task = asyncio.create_task(models_api._entries_within("slow", slow, 30.0))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert registry.keys() == ["slow"], "the shielded fetch lost its only reference"

    slow.release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)


async def test_drain_cancels_immediately_rather_than_waiting(monkeypatch: Any) -> None:
    """Everything in the registry is work a request already gave up on, warming
    a cache inside a backend about to be closed. Waiting first would add its
    grace period to every restart for a result that is discarded."""
    from openai_api_bridge.api import models as models_api

    registry = _fresh_lingering(monkeypatch)
    slow = _SlowBackend([ModelEntry(id="slow", kind="image")])

    await list_models(_request_with([("slow", slow)], budget=0.05))
    assert registry.keys() == ["slow"]

    started = time.monotonic()
    await models_api.drain_lingering()
    elapsed = time.monotonic() - started

    assert elapsed < 0.5, f"drain waited {elapsed:.2f}s on a fetch nobody needs"
    assert registry.keys() == []
    assert _live_fetch_tasks() == []


async def test_drain_is_a_no_op_when_nothing_is_lingering(monkeypatch: Any) -> None:
    from openai_api_bridge.api import models as models_api

    _fresh_lingering(monkeypatch)
    await models_api.drain_lingering()
