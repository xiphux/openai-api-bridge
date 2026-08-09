"""``GET /v1/models`` aggregation behaviour.

Covers the properties the endpoint promises that aren't visible from a
single-provider test: providers are queried concurrently, one provider
failing doesn't take the listing down with it, and one provider being *slow*
doesn't either.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from openai_api_bridge.api.models import list_models
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
