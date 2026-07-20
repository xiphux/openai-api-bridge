"""The one place an upstream HTTP status becomes a bridge error.

Every adapter routes through ``raise_for_upstream_status``; these tests pin
the mapping so it can't drift back to being re-decided per adapter.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from openai_api_bridge.errors import (
    InvalidRequest,
    RateLimited,
    UnsupportedOperation,
    UpstreamAuthError,
    UpstreamError,
)
from openai_api_bridge.util import cache as cache_module
from openai_api_bridge.util.cache import AsyncTTLCache
from openai_api_bridge.util.http import raise_for_upstream_status


def _raise(status: int, body: str = "upstream said no"):
    raise_for_upstream_status(status=status, body=body, provider="TestProvider", endpoint="/thing")


@pytest.mark.parametrize(
    ("status", "expected", "http_status"),
    [
        (400, InvalidRequest, 400),
        (404, InvalidRequest, 400),
        (405, UnsupportedOperation, 400),
        (422, InvalidRequest, 400),
        (429, RateLimited, 429),
        (500, UpstreamError, 502),
        (503, UpstreamError, 502),
    ],
)
def test_status_maps_to_expected_error(
    status: int, expected: type[Exception], http_status: int
) -> None:
    with pytest.raises(expected) as exc:
        _raise(status)
    assert exc.value.status_code == http_status  # type: ignore[attr-defined]


@pytest.mark.parametrize("status", [400, 422])
def test_client_errors_do_not_become_5xx(status: int) -> None:
    """A rejection the client caused must not look retriable.

    Venice and ImageRouter previously routed every status to UpstreamError,
    so a malformed prompt surfaced as a 502 and clients with retry-on-5xx
    logic retried a request that could never succeed — against providers
    that may bill for it.
    """
    with pytest.raises(InvalidRequest) as exc:
        _raise(status)
    assert exc.value.status_code == 400


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_never_echo_the_upstream_body(status: int) -> None:
    """The upstream's complaint concerns the credential we sent, not the client's."""
    secret = "token sk-abcdef123456 is not valid"
    with pytest.raises(UpstreamError) as exc:
        _raise(status, body=secret)
    assert "sk-abcdef123456" not in exc.value.message
    assert "TestProvider" in exc.value.message
    assert str(status) in exc.value.message


@pytest.mark.parametrize("status", [400, 500])
def test_non_auth_errors_do_surface_the_body(status: int) -> None:
    """Diagnostics still reach the client where there's no credential in play."""
    with pytest.raises((InvalidRequest, UpstreamError)) as exc:
        _raise(status, body="prompt was rejected by the safety filter")
    assert "safety filter" in exc.value.message  # type: ignore[attr-defined]


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_get_their_own_error_type(status: int) -> None:
    """Every adapter, not just fal, must distinguish a rejected key.

    Provider tokens are read from the environment at startup, so a rejected
    credential is unlikely to fix itself; callers back off harder on this
    type than on a generic upstream blip.
    """
    with pytest.raises(UpstreamAuthError) as exc:
        _raise(status)
    # Still a 502 to the client — only the type changed, not the status.
    assert exc.value.status_code == 502
    assert isinstance(exc.value, UpstreamError)


@pytest.mark.parametrize("status", [400, 404, 429, 500])
def test_non_auth_failures_do_not_get_the_auth_type(status: int) -> None:
    with pytest.raises(Exception) as exc:
        _raise(status)
    assert not isinstance(exc.value, UpstreamAuthError)


async def test_cache_backs_off_harder_on_a_rejected_credential() -> None:
    """A bad key gets a longer window than an ordinary blip — but not forever."""
    cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60.0, failure_cooldown_seconds=0.01)
    calls = 0

    async def fetch() -> str:
        nonlocal calls
        calls += 1
        raise UpstreamAuthError("provider rejected our credentials (401)")

    with pytest.raises(UpstreamAuthError):
        await cache.get(fetch)
    assert calls == 1

    await asyncio.sleep(0.05)  # past the ordinary cooldown, inside the auth one

    with pytest.raises(UpstreamAuthError):
        await cache.get(fetch)
    assert calls == 1, "auth failure should still be remembered here"


async def test_a_rejected_credential_is_not_latched_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider must recover on its own, as README 'catalogue caching' promises.

    403 is routinely not about our credential at all — a WAF interstitial, a
    geo block, an org quota. Latching on one of those would drop the provider
    from /v1/models for the life of the process, silently, since the models
    endpoint omits failing providers rather than erroring.
    """
    monkeypatch.setattr(cache_module, "AUTH_FAILURE_COOLDOWN_SECONDS", 0.02)
    cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60.0, failure_cooldown_seconds=0.01)
    calls = 0

    async def fetch() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise UpstreamAuthError("provider returned 403 behind a WAF")
        return "recovered"

    with pytest.raises(UpstreamAuthError):
        await cache.get(fetch)

    await asyncio.sleep(0.05)  # past the auth cooldown too

    assert await cache.get(fetch) == "recovered"
    assert calls == 2


async def test_zero_cooldown_disables_failure_memory_even_for_auth() -> None:
    """catalog_retry_seconds = 0 is documented as "retries immediately"."""
    cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60.0, failure_cooldown_seconds=0.0)
    calls = 0

    async def fetch() -> str:
        nonlocal calls
        calls += 1
        raise UpstreamAuthError("provider rejected our credentials (401)")

    for _ in range(3):
        with pytest.raises(UpstreamAuthError):
            await cache.get(fetch)

    assert calls == 3, "an explicit no-cache setting must not be overridden"


async def test_cache_does_retry_a_transient_failure_after_its_cooldown() -> None:
    """The permanence is specific to auth — ordinary blips still recover."""
    cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60.0, failure_cooldown_seconds=0.01)
    calls = 0

    async def fetch() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise UpstreamError("upstream hiccup")
        return "recovered"

    with pytest.raises(UpstreamError):
        await cache.get(fetch)

    await asyncio.sleep(0.05)

    assert await cache.get(fetch) == "recovered"
    assert calls == 2


@pytest.mark.parametrize("status", [401, 403])
def test_fal_auth_error_does_not_echo_the_upstream_body(status: int) -> None:
    """fal's own mapper had the same leak the shared one was fixed for."""
    from openai_api_bridge.backends.fal.client import _status_error

    response = httpx.Response(
        status,
        text="invalid credentials for key fal-key-SECRET123",
        request=httpx.Request("GET", "https://fal.run/x"),
    )
    err = _status_error(
        httpx.HTTPStatusError("boom", request=response.request, response=response), "/x"
    )

    assert isinstance(err, UpstreamAuthError)
    assert "fal-key-SECRET123" not in err.message


def test_fal_non_auth_error_still_carries_the_body() -> None:
    from openai_api_bridge.backends.fal.client import _status_error

    response = httpx.Response(
        500, text="internal explosion", request=httpx.Request("GET", "https://fal.run/x")
    )
    err = _status_error(
        httpx.HTTPStatusError("boom", request=response.request, response=response), "/x"
    )

    assert not isinstance(err, UpstreamAuthError)
    assert "internal explosion" in err.message


async def test_asset_fetch_enforces_the_size_cap() -> None:
    """The memory bound is a parameter of the shared helper, not one adapter's."""
    import respx

    from openai_api_bridge.util.http import fetch_asset_with_retry

    async with respx.mock(assert_all_called=False) as mock:
        mock.get("https://cdn.example/big.png").mock(
            return_value=httpx.Response(
                200, content=b"x" * 5000, headers={"content-type": "image/png"}
            )
        )
        with pytest.raises(UpstreamError, match="exceeded size cap"):
            await fetch_asset_with_retry(
                "https://cdn.example/big.png", provider_label="Test", max_bytes=1000
            )


async def test_asset_fetch_without_a_cap_allows_large_payloads() -> None:
    """Video providers stay uncapped — their outputs are legitimately huge."""
    import respx

    from openai_api_bridge.util.http import fetch_asset_with_retry

    async with respx.mock(assert_all_called=False) as mock:
        mock.get("https://cdn.example/big.mp4").mock(
            return_value=httpx.Response(
                200, content=b"x" * 5000, headers={"content-type": "video/mp4"}
            )
        )
        data, content_type = await fetch_asset_with_retry(
            "https://cdn.example/big.mp4", provider_label="Test"
        )
    assert len(data) == 5000
    assert content_type == "video/mp4"


def test_rate_limit_is_retriable_not_a_client_mistake() -> None:
    """429 is the one retriable 4xx; OpenAI SDKs branch on this.

    Sweeping it into the generic 4xx handling reported a rate limit as
    invalid_request_error, telling clients not to retry precisely when
    backing off and retrying is the correct response.
    """
    with pytest.raises(RateLimited) as exc:
        _raise(429, body="slow down")

    assert exc.value.status_code == 429
    assert exc.value.error_type == "rate_limit_error"
    assert not isinstance(exc.value, InvalidRequest)


def test_fal_maps_429_to_a_rate_limit_too() -> None:
    """fal keeps its own return-style mapper; its status semantics must not drift."""
    from openai_api_bridge.backends.fal.client import _status_error

    response = httpx.Response(
        429, text="too many requests", request=httpx.Request("GET", "https://fal.run/x")
    )
    err = _status_error(
        httpx.HTTPStatusError("boom", request=response.request, response=response), "/x"
    )

    assert isinstance(err, RateLimited)
    assert err.status_code == 429


def test_rate_limited_is_transient_for_adapter_retry_loops() -> None:
    """fal's queue poller retries on UpstreamError; a rate limit must qualify.

    If RateLimited were a sibling of UpstreamError rather than a subclass, a
    429 during status polling would skip those handlers and abort a video job
    that was merely being throttled.
    """
    assert issubclass(RateLimited, UpstreamError)
    assert isinstance(RateLimited("throttled"), UpstreamError)
