"""``GET /v1/models`` aggregation behaviour.

Covers the two properties the endpoint promises that aren't visible from a
single-provider test: providers are queried concurrently, and one provider
failing doesn't take the listing down with it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

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


def _request_with(providers: list[tuple[str, Backend]]) -> Any:
    dispatcher = SimpleNamespace(all_providers=lambda: providers)
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(dispatcher=dispatcher)))


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


async def test_one_failing_provider_does_not_break_the_listing() -> None:
    providers: list[tuple[str, Backend]] = [
        ("good", _StaticBackend([ModelEntry(id="m", kind="image")])),
        ("bad", _FailingBackend(UpstreamError("catalogue is down"))),
    ]

    body = await list_models(_request_with(providers))

    assert [row["id"] for row in body["data"]] == ["good/m"]


class _StaticBackend(Backend):
    def __init__(self, entries: list[ModelEntry]) -> None:
        self._entries = entries

    async def list_models(self) -> list[ModelEntry]:
        return self._entries


@pytest.mark.parametrize("kind", ["image", "video"])
async def test_kind_is_surfaced(kind: str) -> None:
    providers: list[tuple[str, Backend]] = [
        ("p", _StaticBackend([ModelEntry(id="m", kind=kind)])),
    ]

    body = await list_models(_request_with(providers))

    assert body["data"][0]["kind"] == kind
