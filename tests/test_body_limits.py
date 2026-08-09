"""Request-body size ceiling (``BodySizeLimitMiddleware``).

The bridge reads bodies whole on a single uvicorn worker, so an oversized
request is felt by every other client. These cover both enforcement paths —
the declared ``Content-Length`` and the bytes actually received — plus the
cases that must keep working untouched.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
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
