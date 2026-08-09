"""The one place an upstream HTTP status becomes a bridge error.

Every adapter routes through ``raise_for_upstream_status``; these tests pin
the mapping so it can't drift back to being re-decided per adapter.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

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
        with pytest.raises(UpstreamError, match="exceeded the 1000 byte cap"):
            await fetch_asset_with_retry(
                "https://cdn.example/big.png", provider_label="Test", max_bytes=1000
            )


async def test_asset_fetch_refuses_on_declared_length_without_reading() -> None:
    """A truthful Content-Length is refused before the body is touched.

    The cap used to be applied to ``response.content`` — after the whole body
    was already resident, the one moment it could no longer help.
    """
    import respx

    from openai_api_bridge.util.http import fetch_asset_with_retry

    async with respx.mock(assert_all_called=False) as mock:
        mock.get("https://cdn.example/huge.mp4").mock(
            return_value=httpx.Response(
                200,
                content=b"x" * 5000,
                headers={"content-type": "video/mp4", "content-length": "5000"},
            )
        )
        with pytest.raises(UpstreamError, match=r"exceeded the 1000 byte cap \(5000 bytes\)"):
            await fetch_asset_with_retry(
                "https://cdn.example/huge.mp4", provider_label="Test", max_bytes=1000
            )


async def test_asset_fetch_stops_reading_a_body_that_understates_its_length() -> None:
    """An absent or lying Content-Length leaves the running byte count as the
    only thing standing between a hostile CDN and the process's memory."""
    import respx

    from openai_api_bridge.util.http import fetch_asset_with_retry

    delivered = 0

    async def endless() -> AsyncIterator[bytes]:
        # Far more than the cap, delivered in chunks. Reaching the end would
        # mean the counter never fired.
        nonlocal delivered
        for _ in range(1000):
            delivered += 1024
            yield b"x" * 1024

    async with respx.mock(assert_all_called=False) as mock:
        # An iterator body: httpx sends it without a Content-Length, so the
        # declared-length check above cannot fire and the counter must.
        mock.get("https://cdn.example/lies.mp4").mock(
            return_value=httpx.Response(
                200,
                content=endless(),
                headers={"content-type": "video/mp4"},
            )
        )
        with pytest.raises(UpstreamError, match="exceeded the 4096 byte cap"):
            await fetch_asset_with_retry(
                "https://cdn.example/lies.mp4", provider_label="Test", max_bytes=4096
            )

    # The point of streaming: it gave up near the cap instead of consuming the
    # ~1MB the far end was willing to send.
    assert delivered <= 4096 + 1024


async def test_asset_fetch_without_a_cap_allows_large_payloads() -> None:
    """``max_bytes=None`` still means unbounded, for a caller that asks."""
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


async def test_asset_fetch_under_the_cap_is_returned_whole() -> None:
    import respx

    from openai_api_bridge.util.http import fetch_asset_with_retry

    async with respx.mock(assert_all_called=False) as mock:
        mock.get("https://cdn.example/ok.png").mock(
            return_value=httpx.Response(
                200, content=b"x" * 900, headers={"content-type": "image/png; charset=binary"}
            )
        )
        data, content_type = await fetch_asset_with_retry(
            "https://cdn.example/ok.png", provider_label="Test", max_bytes=1000
        )
    assert len(data) == 900
    # Charset suffix still stripped for clean storage.
    assert content_type == "image/png"


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


@pytest.mark.parametrize(
    ("status", "expected_attempts"),
    [
        # Worth waiting out.
        (401, 3),  # storage race on a just-minted URL
        (404, 3),  # same
        (408, 3),  # request timeout
        (425, 3),  # "too early" behind a CDN
        (429, 3),  # throttled — the expensive one to get wrong
        (500, 3),
        (502, 3),
        (503, 3),
        (504, 3),
        # Hopeless: re-requesting cannot change the answer.
        (400, 1),
        (402, 1),
        (403, 1),
        (405, 1),
        (409, 1),
        (410, 1),
        (422, 1),
        (501, 1),  # the origin doesn't implement it and won't start
    ],
)
async def test_asset_fetch_retries_exactly_the_recoverable_statuses(
    status: int, expected_attempts: int
) -> None:
    """The whole status space, not just the cases that came to mind.

    Getting this set wrong is silent in both directions. Too broad and a
    hopeless 400 costs three calls and two sleeps; too narrow and a finished,
    already-billed render is thrown away over a momentary throttle — 429 was
    swept in with the hopeless statuses exactly that way, and the tests
    written alongside that change guarded only 404 and 503, so they passed.
    """
    import respx

    from openai_api_bridge.util.http import fetch_asset_with_retry

    async with respx.mock(assert_all_called=False) as mock:
        route = mock.get("https://cdn.example/a.png").mock(
            return_value=httpx.Response(status, text="nope")
        )
        with pytest.raises(UpstreamError):
            await fetch_asset_with_retry(
                "https://cdn.example/a.png", provider_label="Test", base_delay=0.001
            )

    assert route.call_count == expected_attempts


async def test_asset_fetch_reports_how_many_attempts_it_made() -> None:
    """Logs and errors should distinguish "hopeless once" from "retried out"."""
    import respx

    from openai_api_bridge.util.http import fetch_asset_with_retry

    async with respx.mock(assert_all_called=False) as mock:
        mock.get("https://cdn.example/a.png").mock(return_value=httpx.Response(400, text="no"))
        with pytest.raises(UpstreamError, match="after 1 attempt"):
            await fetch_asset_with_retry(
                "https://cdn.example/a.png", provider_label="Test", base_delay=0.001
            )

    async with respx.mock(assert_all_called=False) as mock:
        mock.get("https://cdn.example/b.png").mock(return_value=httpx.Response(503, text="busy"))
        with pytest.raises(UpstreamError, match="after 3 attempt"):
            await fetch_asset_with_retry(
                "https://cdn.example/b.png", provider_label="Test", base_delay=0.001
            )


async def test_asset_fetch_does_not_retry_a_hopeless_status() -> None:
    """400 can never succeed; retrying spent 3 calls and 2 sleeps to learn that."""
    import respx

    from openai_api_bridge.util.http import fetch_asset_with_retry

    async with respx.mock(assert_all_called=False) as mock:
        route = mock.get("https://cdn.example/nope.png").mock(
            return_value=httpx.Response(400, text="bad request")
        )
        with pytest.raises(UpstreamError):
            await fetch_asset_with_retry(
                "https://cdn.example/nope.png", provider_label="Test", base_delay=0.01
            )

    assert route.call_count == 1, f"retried a hopeless 400 {route.call_count} times"


async def test_asset_fetch_still_retries_the_storage_race() -> None:
    """A just-minted asset URL can 404 briefly before storage catches up."""
    import respx

    from openai_api_bridge.util.http import fetch_asset_with_retry

    async with respx.mock(assert_all_called=False) as mock:
        route = mock.get("https://cdn.example/soon.png").mock(
            side_effect=[
                httpx.Response(404, text="not yet"),
                httpx.Response(200, content=b"ok", headers={"content-type": "image/png"}),
            ]
        )
        data, _ = await fetch_asset_with_retry(
            "https://cdn.example/soon.png", provider_label="Test", base_delay=0.01
        )

    assert data == b"ok"
    assert route.call_count == 2


async def test_asset_fetch_still_retries_5xx() -> None:
    import respx

    from openai_api_bridge.util.http import fetch_asset_with_retry

    async with respx.mock(assert_all_called=False) as mock:
        route = mock.get("https://cdn.example/flaky.png").mock(
            side_effect=[
                httpx.Response(503, text="busy"),
                httpx.Response(200, content=b"ok", headers={"content-type": "image/png"}),
            ]
        )
        data, _ = await fetch_asset_with_retry(
            "https://cdn.example/flaky.png", provider_label="Test", base_delay=0.01
        )

    assert data == b"ok"
    assert route.call_count == 2


async def test_throttled_asset_fetch_keeps_its_type_after_exhausting_retries() -> None:
    """A 429 that outlasts our attempts is still a rate limit, not a generic fault.

    fal's _fetch_asset branches on this to skip its "did the asset expire?"
    hint. Untyped, that guard could never fire, and a message plainly saying
    429 got expiry advice appended — the exact misattribution the guard
    exists to prevent.
    """
    import respx

    from openai_api_bridge.util.http import fetch_asset_with_retry

    async with respx.mock(assert_all_called=False) as mock:
        mock.get("https://cdn.example/throttled.png").mock(
            return_value=httpx.Response(429, text="slow down")
        )
        with pytest.raises(RateLimited):
            await fetch_asset_with_retry(
                "https://cdn.example/throttled.png", provider_label="Test", base_delay=0.001
            )


async def test_fal_does_not_blame_expiry_for_a_throttled_fetch() -> None:
    """The guard at fal's _fetch_asset is live now, not decorative."""
    import os

    import respx

    from openai_api_bridge.backends.fal.adapter import FalBackend
    from openai_api_bridge.config import FalProviderConfig

    os.environ["FAL_KEY_FOR_TEST"] = "k"
    cfg = FalProviderConfig(
        backend="fal",
        id="f",
        api_token_env="FAL_KEY_FOR_TEST",
        output_expiration_seconds=60,
    )
    backend = FalBackend(cfg)
    try:
        async with respx.mock(assert_all_called=False) as mock:
            mock.get("https://cdn.example/throttled.png").mock(
                return_value=httpx.Response(429, text="slow down")
            )
            with pytest.raises(RateLimited) as exc:
                await backend._fetch_asset("https://cdn.example/throttled.png")
    finally:
        await backend.aclose()

    assert "output_expiration_seconds" not in exc.value.message


async def test_asset_fetches_share_one_connection_pool() -> None:
    """A client per fetch meant a TLS handshake per image.

    The shared client is what turns n concurrent asset downloads into n
    requests on one pool rather than n pools of one request.
    """
    import respx

    from openai_api_bridge.util.http import _asset_fetch_client, fetch_asset_with_retry

    async with respx.mock(assert_all_called=False) as mock:
        mock.get("https://cdn.example/a.png").mock(
            return_value=httpx.Response(200, content=b"a", headers={"content-type": "image/png"})
        )
        await fetch_asset_with_retry("https://cdn.example/a.png", provider_label="Test")
        first = _asset_fetch_client()
        await fetch_asset_with_retry("https://cdn.example/a.png", provider_label="Test")
        assert _asset_fetch_client() is first


async def test_asset_client_carries_no_authorization_header() -> None:
    """Asset URLs are public CDNs; the bridge's upstream credential must not ride along."""
    from openai_api_bridge.util.http import _asset_fetch_client

    assert "authorization" not in {k.lower() for k in _asset_fetch_client().headers}


async def test_asset_client_is_rebuilt_for_a_new_event_loop() -> None:
    """A pool's connections belong to the loop that opened them.

    The bridge has exactly one loop, so in production this caches a single
    client for the process — but a client carried into a different loop would
    hand out connections attached to a closed one.
    """
    import asyncio

    from openai_api_bridge.util.http import _asset_fetch_client

    outer = _asset_fetch_client()
    inner = await asyncio.to_thread(lambda: asyncio.run(_in_fresh_loop()))
    assert inner is not outer


async def _in_fresh_loop() -> object:
    from openai_api_bridge.util.http import _asset_fetch_client

    return _asset_fetch_client()


# --- signed asset URLs in client-facing messages ----------------------------


def test_redact_url_drops_a_signature_query() -> None:
    from openai_api_bridge.util.http import redact_url

    signed = "https://v3.fal.media/files/abc/out.mp4?Signature=SECRETSIG&Expires=1799999999"
    assert redact_url(signed) == "https://v3.fal.media/files/abc/out.mp4"


def test_redact_url_leaves_an_unsigned_url_alone() -> None:
    from openai_api_bridge.util.http import redact_url

    plain = "https://storage.imagerouter.io/a/b.png"
    assert redact_url(plain) == plain


async def test_fetch_error_does_not_echo_a_signed_asset_url() -> None:
    """Asset URLs are routinely pre-signed, and these messages are rendered
    into the error envelope and sent over the wire."""
    import respx

    from openai_api_bridge.util.http import fetch_asset_with_retry

    signed = "https://cdn.example/out.mp4?Signature=SECRETSIG&Expires=1799999999"
    async with respx.mock(assert_all_called=False) as mock:
        mock.get(url=signed).mock(return_value=httpx.Response(403))
        with pytest.raises(UpstreamError) as exc:
            await fetch_asset_with_retry(signed, provider_label="Test")

    assert "SECRETSIG" not in exc.value.message
    assert "https://cdn.example/out.mp4" in exc.value.message


async def test_size_cap_error_does_not_echo_a_signed_asset_url() -> None:
    import respx

    from openai_api_bridge.util.http import fetch_asset_with_retry

    signed = "https://cdn.example/big.mp4?Signature=SECRETSIG"
    async with respx.mock(assert_all_called=False) as mock:
        mock.get(url=signed).mock(
            return_value=httpx.Response(
                200, content=b"x" * 5000, headers={"content-type": "video/mp4"}
            )
        )
        with pytest.raises(UpstreamError) as exc:
            await fetch_asset_with_retry(signed, provider_label="Test", max_bytes=1000)

    assert "SECRETSIG" not in exc.value.message
