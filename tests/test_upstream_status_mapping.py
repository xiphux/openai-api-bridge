"""The one place an upstream HTTP status becomes a bridge error.

Every adapter routes through ``raise_for_upstream_status``; these tests pin
the mapping so it can't drift back to being re-decided per adapter.
"""

from __future__ import annotations

import pytest

from openai_api_bridge.errors import (
    InvalidRequest,
    UnsupportedOperation,
    UpstreamError,
)
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
