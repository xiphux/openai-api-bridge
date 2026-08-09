"""``n > 1`` on providers with no server-side batch parameter.

Venice, ImageRouter and OpenRouter each return a single image per call, so
honouring OpenAI's ``n`` means issuing n requests. Run end-to-end one at a
time, the caller waits for the *sum* of n generations inside one synchronous
POST — at the permitted n=4 that lands well past most clients' timeouts. These
tests pin the requests down as overlapping, and pin the failure behaviour that
makes overlapping safe on a *paid* provider: a sibling must not be left
running (and billing) after another has failed.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from openai_api_bridge.backends.base import Backend
from openai_api_bridge.backends.imagerouter.adapter import ImageRouterBackend
from openai_api_bridge.backends.openrouter.adapter import OpenRouterBackend
from openai_api_bridge.backends.venice.adapter import VeniceBackend
from openai_api_bridge.config import (
    ImageRouterProviderConfig,
    OpenRouterProviderConfig,
    VeniceProviderConfig,
)
from openai_api_bridge.util.concurrency import run_all

_PNG = b"\x89PNG\r\n\x1a\n"
_DATA_URL = "data:image/png;base64,iVBORw0KGgo="

VENICE = "https://api.venice.ai"
IMAGEROUTER = "https://api.imagerouter.io"
OPENROUTER = "https://openrouter.ai/api"


class _Overlap:
    """Counts how many stubbed upstream calls are in flight at once."""

    def __init__(self) -> None:
        self.now = 0
        self.peak = 0

    async def hold(self) -> None:
        self.now += 1
        self.peak = max(self.peak, self.now)
        await asyncio.sleep(0.05)
        self.now -= 1


def _venice(monkeypatch: pytest.MonkeyPatch) -> Backend:
    monkeypatch.setenv("TEST_TOKEN", "secret")
    return VeniceBackend(
        VeniceProviderConfig(backend="venice", id="vn", api_token_env="TEST_TOKEN")
    )


def _imagerouter(monkeypatch: pytest.MonkeyPatch) -> Backend:
    monkeypatch.setenv("TEST_TOKEN", "secret")
    return ImageRouterBackend(
        ImageRouterProviderConfig(backend="imagerouter", id="ir", api_token_env="TEST_TOKEN")
    )


def _openrouter(monkeypatch: pytest.MonkeyPatch) -> Backend:
    monkeypatch.setenv("TEST_TOKEN", "secret")
    return OpenRouterBackend(
        OpenRouterProviderConfig(backend="openrouter", id="or", api_token_env="TEST_TOKEN")
    )


@respx.mock
async def test_venice_runs_n_generations_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    overlap = _Overlap()

    async def responder(request: httpx.Request) -> httpx.Response:
        await overlap.hold()
        return httpx.Response(200, json={"images": ["aGk="]})

    respx.post(f"{VENICE}/api/v1/image/generate").mock(side_effect=responder)

    backend = _venice(monkeypatch)
    try:
        assets = await backend.generate_image(model_slug="flux", prompt="a cat", n=4)
    finally:
        await backend.aclose()

    assert len(assets) == 4
    assert overlap.peak == 4, f"generations ran {overlap.peak}-at-a-time, expected 4"


@respx.mock
async def test_imagerouter_runs_n_generations_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlap = _Overlap()

    async def responder(request: httpx.Request) -> httpx.Response:
        await overlap.hold()
        return httpx.Response(200, json={"data": [{"url": "https://cdn.example/a.png"}]})

    respx.post(f"{IMAGEROUTER}/v1/openai/images/generations").mock(side_effect=responder)
    respx.get("https://cdn.example/a.png").mock(
        return_value=httpx.Response(200, content=_PNG, headers={"content-type": "image/png"})
    )

    backend = _imagerouter(monkeypatch)
    try:
        assets = await backend.generate_image(model_slug="m", prompt="a cat", n=4)
    finally:
        await backend.aclose()

    assert len(assets) == 4
    assert overlap.peak == 4, f"generations ran {overlap.peak}-at-a-time, expected 4"


@respx.mock
async def test_openrouter_runs_n_generations_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlap = _Overlap()

    async def responder(request: httpx.Request) -> httpx.Response:
        await overlap.hold()
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"images": [{"image_url": {"url": _DATA_URL}}]}}],
            },
        )

    respx.post(f"{OPENROUTER}/v1/chat/completions").mock(side_effect=responder)

    backend = _openrouter(monkeypatch)
    try:
        assets = await backend.generate_image(model_slug="m", prompt="a cat", n=4)
    finally:
        await backend.aclose()

    assert len(assets) == 4
    assert overlap.peak == 4, f"generations ran {overlap.peak}-at-a-time, expected 4"


@respx.mock
async def test_order_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrency must not reorder the response array.

    The slowest call is stubbed first, so a naive completion-ordered collect
    would put it last.
    """
    # Paired at *request* time, so the first call issued is also the slowest
    # to answer — a naive completion-ordered collect would put it last.
    schedule = iter([(0.06, b"one"), (0.04, b"two"), (0.02, b"three"), (0.0, b"four")])

    async def responder(request: httpx.Request) -> httpx.Response:
        delay, body = next(schedule)
        await asyncio.sleep(delay)
        return httpx.Response(200, content=body, headers={"content-type": "image/png"})

    respx.post(f"{VENICE}/api/v1/image/edit").mock(side_effect=responder)
    respx.get(f"{VENICE}/api/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))

    from openai_api_bridge.backends.base import InputImage

    backend = _venice(monkeypatch)
    try:
        assets = await backend.edit_image(
            model_slug="m",
            prompt="p",
            images=[InputImage(data=_PNG, content_type="image/png")],
            n=4,
        )
    finally:
        await backend.aclose()

    assert [a.data for a in assets] == [b"one", b"two", b"three", b"four"]


async def test_run_all_cancels_siblings_on_failure() -> None:
    """A failed sibling must not leave the others generating.

    Bare ``gather`` surfaces the first error immediately but leaves its
    siblings running — on a paid provider that keeps costing money, and keeps
    buffering assets into memory, for a request that has already failed.
    """
    started = asyncio.Event()
    cancelled: list[str] = []

    async def slow() -> str:
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append("slow")
            raise
        return "slow"

    async def boom() -> str:
        await started.wait()
        raise RuntimeError("upstream rejected the prompt")

    with pytest.raises(RuntimeError, match="upstream rejected"):
        await run_all([slow, boom])

    assert cancelled == ["slow"], "the surviving sibling should have been cancelled"


async def test_run_all_preserves_order_and_returns_every_result() -> None:
    async def make(i: int) -> int:
        await asyncio.sleep((5 - i) * 0.01)
        return i

    assert await run_all([lambda i=i: make(i) for i in range(5)]) == [0, 1, 2, 3, 4]
