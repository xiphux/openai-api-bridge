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
    UnsupportedOperation,
    UpstreamAuthError,
    UpstreamError,
)
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
        (429, InvalidRequest, 400),
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


@pytest.mark.parametrize("status", [400, 422, 429])
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
def test_auth_failures_are_marked_permanent(status: int) -> None:
    """Every adapter, not just fal, must report a rejected key as permanent.

    Provider tokens are read from the environment at startup, so a rejected
    credential cannot start working again without a restart. Callers that
    retry or re-attempt on a cooldown are meant to stop on this type.
    """
    with pytest.raises(UpstreamAuthError) as exc:
        _raise(status)
    # Still a 502 to the client — only the type changed, not the status.
    assert exc.value.status_code == 502
    assert isinstance(exc.value, UpstreamError)


@pytest.mark.parametrize("status", [400, 404, 429, 500])
def test_non_auth_failures_are_not_marked_permanent(status: int) -> None:
    with pytest.raises(Exception) as exc:
        _raise(status)
    assert not isinstance(exc.value, UpstreamAuthError)


async def test_cache_remembers_a_rejected_credential_past_its_cooldown() -> None:
    """A bad key must not be re-asked once the failure cooldown lapses."""
    cache: AsyncTTLCache[str] = AsyncTTLCache(ttl_seconds=60.0, failure_cooldown_seconds=0.01)
    calls = 0

    async def fetch() -> str:
        nonlocal calls
        calls += 1
        raise UpstreamAuthError("provider rejected our credentials (401)")

    with pytest.raises(UpstreamAuthError):
        await cache.get(fetch)
    assert calls == 1

    await asyncio.sleep(0.05)  # well past the cooldown

    with pytest.raises(UpstreamAuthError):
        await cache.get(fetch)
    assert calls == 1, "a permanently-bad key was re-attempted after the cooldown"


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
