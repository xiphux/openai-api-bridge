"""Request-body size ceiling (``BodySizeLimitMiddleware``).

The bridge reads bodies whole on a single uvicorn worker, so an oversized
request is felt by every other client. These cover both enforcement paths —
the declared ``Content-Length`` and the bytes actually received — plus the
cases that must keep working untouched.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openai_api_bridge.config import reset_caches_for_tests

# Small enough to exercise cheaply; the production default is 100MB.
_LIMIT_MB = 1
_LIMIT_BYTES = _LIMIT_MB * 1024**2


@pytest.fixture
def small_limit_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    files_dir: Path,
    sqlite_path: Path,
) -> Iterator[TestClient]:
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent("""
        [defaults]
        cache_workflows = true
    """)
    )
    monkeypatch.setenv("BRIDGE_API_KEY", "test-bridge-api-key")
    monkeypatch.setenv("BRIDGE_CONFIG_PATH", str(config))
    monkeypatch.setenv("FILES_DIR", str(files_dir))
    monkeypatch.setenv("SQLITE_PATH", str(sqlite_path))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("BRIDGE_MAX_REQUEST_MB", str(_LIMIT_MB))
    reset_caches_for_tests()

    from openai_api_bridge.main import create_app

    with TestClient(create_app()) as c:
        yield c
    reset_caches_for_tests()


def test_oversized_json_body_rejected(
    small_limit_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    r = small_limit_client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={"model": "ghost/x", "messages": [{"role": "user", "content": "x" * _LIMIT_BYTES}]},
    )
    assert r.status_code == 413
    body = r.json()["error"]
    assert body["code"] == "request_too_large"
    assert body["type"] == "invalid_request_error"
    # Refused on the declared Content-Length, i.e. without buffering the body.
    assert "exceeds" in body["message"]


def test_oversized_multipart_upload_rejected(
    small_limit_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """The path that motivated the cap: /v1/images/edits reads every upload
    into memory, and Starlette spools each part past 1MB to a temp file on the
    way there."""
    r = small_limit_client.post(
        "/v1/images/edits",
        headers=auth_headers,
        files={"image": ("big.png", b"\x00" * (_LIMIT_BYTES + 1), "image/png")},
        data={"model": "ghost/x", "prompt": "x"},
    )
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "request_too_large"


def test_oversized_video_input_reference_rejected(
    small_limit_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    r = small_limit_client.post(
        "/v1/videos",
        headers=auth_headers,
        files={"input_reference": ("big.png", b"\x00" * (_LIMIT_BYTES + 1), "image/png")},
        data={"model": "ghost/x", "prompt": "x"},
    )
    assert r.status_code == 413


def test_chunked_body_without_content_length_rejected(
    small_limit_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """A generator body makes httpx send chunked, with no Content-Length to
    check — so the declared-length fast path can't fire and the received-bytes
    counter has to catch it."""

    def oversized() -> Iterator[bytes]:
        for _ in range(_LIMIT_BYTES // 1024 + 2):
            yield b"\x00" * 1024

    r = small_limit_client.post(
        "/v1/embeddings",
        headers={**auth_headers, "Content-Type": "application/json"},
        content=oversized(),
    )
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "request_too_large"
    # Specifically the received-bytes counter, not the header check — pinned so
    # a refactor that drops the streaming half fails here rather than silently
    # leaving chunked requests uncapped.
    assert "while being received" in r.json()["error"]["message"]


def test_understated_content_length_still_rejected(
    small_limit_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """A body that lies about its size in the header is still counted."""

    def oversized() -> Iterator[bytes]:
        for _ in range(_LIMIT_BYTES // 1024 + 2):
            yield b"\x00" * 1024

    r = small_limit_client.post(
        "/v1/embeddings",
        headers={**auth_headers, "Content-Type": "application/json", "Content-Length": "10"},
        content=oversized(),
    )
    assert r.status_code == 413
    assert "while being received" in r.json()["error"]["message"]


# The counter path on FastAPI-parameter-bound routes.
#
# These are the routes the cap exists for — /v1/images/edits reads up to 16
# uploads, /v1/videos an input_reference — and they are exactly the ones where
# the counter's verdict used to be lost: FastAPI reads a bound body inside its
# own handler, which wraps that read in a catch-all `except Exception` and
# rewrites it to `HTTPException(400, "There was an error parsing the body")`.
# The result was a 400 in FastAPI's `{"detail": ...}` shape instead of the
# bridge's 413 envelope, with no Connection: close and no log line.
#
# The multipart tests above cannot cover this: httpx `files=` builds the body
# in memory and sends an honest Content-Length, so they take the fast path and
# never reach the counter. A chunked body is what forces it.
_BOUNDARY = "----bridgetest"


def _oversized_multipart(field: str) -> Iterator[bytes]:
    """A structurally VALID multipart body whose file part is over the cap.

    Valid matters. A stream of null bytes under a multipart content-type is
    rejected by the parser as malformed before the counter ever trips, which
    yields a 400 for an entirely different reason and would make this test
    pass against a broken middleware.
    """
    for name, value in (("model", "ghost/x"), ("prompt", "x")):
        yield (
            f'--{_BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        ).encode()
    yield (
        f'--{_BOUNDARY}\r\nContent-Disposition: form-data; name="{field}"; '
        f'filename="big.png"\r\nContent-Type: image/png\r\n\r\n'
    ).encode()
    for _ in range(_LIMIT_BYTES // 1024 + 2):
        yield b"\x00" * 1024
    yield f"\r\n--{_BOUNDARY}--\r\n".encode()


def _oversized_json() -> Iterator[bytes]:
    yield b'{"model":"ghost/x","prompt":"'
    for _ in range(_LIMIT_BYTES // 1024 + 2):
        yield b"x" * 1024
    yield b'"}'


@pytest.mark.parametrize(
    ("path", "content_type", "make_body"),
    [
        (
            "/v1/images/edits",
            f"multipart/form-data; boundary={_BOUNDARY}",
            lambda: _oversized_multipart("image"),
        ),
        (
            "/v1/videos",
            f"multipart/form-data; boundary={_BOUNDARY}",
            lambda: _oversized_multipart("input_reference"),
        ),
        ("/v1/images/generations", "application/json", _oversized_json),
    ],
)
def test_counter_path_answers_413_on_parameter_bound_routes(
    small_limit_client: TestClient,
    auth_headers: dict[str, str],
    path: str,
    content_type: str,
    make_body: Callable[[], Iterator[bytes]],
) -> None:
    r = small_limit_client.post(
        path,
        headers={**auth_headers, "Content-Type": content_type},
        content=make_body(),
    )

    assert r.status_code == 413, f"{path} answered {r.status_code}: {r.text[:200]}"
    payload = r.json()
    # The OpenAI envelope, not FastAPI's {"detail": ...}.
    assert "error" in payload, f"{path} returned a non-bridge error shape: {payload}"
    assert payload["error"]["code"] == "request_too_large"
    assert payload["error"]["type"] == "invalid_request_error"
    assert "while being received" in payload["error"]["message"]
    assert r.headers.get("connection") == "close"


def test_body_under_the_cap_passes_through(
    small_limit_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """Reaching the router (404 for an unconfigured provider) proves the
    middleware forwarded the request rather than short-circuiting it."""
    r = small_limit_client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={"model": "ghost/x", "messages": [{"role": "user", "content": "x" * 1024}]},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "provider_not_found"


def test_limit_runs_before_auth(small_limit_client: TestClient) -> None:
    """Outermost middleware, so an oversized body is refused without a
    credential — the bytes are the cost, and reading them to find out the
    caller was unauthorised defeats the point."""
    r = small_limit_client.post(
        "/v1/chat/completions",
        json={"model": "ghost/x", "messages": [{"role": "user", "content": "x" * _LIMIT_BYTES}]},
    )
    assert r.status_code == 413


def test_get_requests_unaffected(
    small_limit_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    assert small_limit_client.get("/v1/models", headers=auth_headers).status_code == 200


def test_zero_disables_the_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    files_dir: Path,
    sqlite_path: Path,
    auth_headers: dict[str, str],
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[defaults]\ncache_workflows = true\n")
    monkeypatch.setenv("BRIDGE_API_KEY", "test-bridge-api-key")
    monkeypatch.setenv("BRIDGE_CONFIG_PATH", str(config))
    monkeypatch.setenv("FILES_DIR", str(files_dir))
    monkeypatch.setenv("SQLITE_PATH", str(sqlite_path))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("BRIDGE_MAX_REQUEST_MB", "0")
    reset_caches_for_tests()

    from openai_api_bridge.main import create_app

    with TestClient(create_app()) as c:
        r = c.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "ghost/x",
                "messages": [{"role": "user", "content": "x" * (2 * 1024**2)}],
            },
        )
        # Past the router, not blocked by the (disabled) cap.
        assert r.status_code == 404
    reset_caches_for_tests()
