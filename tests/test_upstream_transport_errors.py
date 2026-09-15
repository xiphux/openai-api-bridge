"""Transport failures — nothing answered, or it stopped answering — at every client.

``test_upstream_status_mapping.py`` pins what happens when an upstream *answers*
with an error status. This file pins the other half: a refused connection, a
read timeout, a dropped connection. Every backend client wraps those as
``except httpx2.HTTPError`` → ``UpstreamError``, and until this file none of
those branches ran.

That matters beyond coverage. httpx2 minors auto-merge through Dependabot, and
the one thing these branches depend on is httpx2's exception hierarchy: if a
transport error stopped being an ``httpx2.HTTPError``, it would escape every
adapter as a bare exception and reach clients as a 500 instead of the 502
envelope — with every status-code test still green.

The failures are injected one layer down, as ``httpcore2`` exceptions raised by
the mocked transport. That is where a real failure originates, so httpx2's own
translation of it into ``httpx2.ConnectError`` / ``ReadTimeout`` is exercised
too, rather than assumed by raising the httpx2 type directly.
"""

from __future__ import annotations

import textwrap
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import httpcore2
import pytest
import respx
from fastapi.testclient import TestClient

from openai_api_bridge.backends.base import InputImage
from openai_api_bridge.backends.comfyui.client import ComfyUIClient
from openai_api_bridge.backends.fal.client import FalClient, QueuedRequest
from openai_api_bridge.backends.imagerouter.client import ImageRouterClient
from openai_api_bridge.backends.openai.client import OpenAIClient
from openai_api_bridge.backends.venice.client import VeniceClient
from openai_api_bridge.config import reset_caches_for_tests
from openai_api_bridge.errors import UpstreamError

UPSTREAM = "http://upstream.test"

# The transport failures worth distinguishing: nothing listening, a hung read,
# and a peer that hangs up mid-exchange.
TRANSPORT_FAILURES = [
    pytest.param(httpcore2.ConnectError("connection refused"), id="connect-refused"),
    pytest.param(httpcore2.ReadTimeout("timed out"), id="read-timeout"),
    pytest.param(httpcore2.RemoteProtocolError("peer closed connection"), id="dropped"),
]


def _openai() -> OpenAIClient:
    return OpenAIClient(base_url=UPSTREAM, api_token="t", request_timeout_seconds=5.0)


def _venice() -> VeniceClient:
    return VeniceClient(base_url=UPSTREAM, api_token="t")


def _imagerouter() -> ImageRouterClient:
    return ImageRouterClient(base_url=UPSTREAM, api_token="t")


def _fal() -> FalClient:
    return FalClient(
        base_url=UPSTREAM,
        api_token="t",
        request_timeout_seconds=5.0,
        queue_base_url=UPSTREAM,
    )


def _comfyui() -> ComfyUIClient:
    return ComfyUIClient(base_url=UPSTREAM)


_JOB = QueuedRequest(
    request_id="r1",
    status_url=f"{UPSTREAM}/requests/r1/status",
    response_url=f"{UPSTREAM}/requests/r1",
    cancel_url=f"{UPSTREAM}/requests/r1/cancel",
)
_PNG = b"\x89PNG\r\n\x1a\n"
_HISTORY = {"outputs": {"9": {"images": [{"filename": "out.png", "type": "output"}]}}}


# (id, client factory, the call under test). Each call is one public client
# method whose only network step is the request the failure is injected into.
CASES: list[tuple[str, Callable[[], Any], Callable[[Any], Awaitable[Any]]]] = [
    ("openai.list_models", _openai, lambda c: c.list_models()),
    (
        "openai.chat_completion",
        _openai,
        lambda c: c.chat_completion({"model": "m", "messages": []}),
    ),
    (
        "openai.chat_completion_stream",
        _openai,
        lambda c: c.chat_completion_stream({"model": "m", "messages": []}),
    ),
    (
        "openai.create_embedding",
        _openai,
        lambda c: c.create_embedding({"model": "m", "input": "x"}),
    ),
    ("venice.list_image_models", _venice, lambda c: c.list_image_models()),
    (
        "venice.generate_image",
        _venice,
        lambda c: c.generate_image(
            model="m", prompt="p", width=64, height=64, steps=1, cfg_scale=1.0
        ),
    ),
    (
        "venice.edit_image",
        _venice,
        lambda c: c.edit_image(model="m", prompt="p", image=_PNG, image_content_type="image/png"),
    ),
    ("imagerouter.list_models", _imagerouter, lambda c: c.list_models()),
    (
        "imagerouter.generate_image_url",
        _imagerouter,
        lambda c: c.generate_image_url(model="m", prompt="p"),
    ),
    (
        "imagerouter.edit_image_url",
        _imagerouter,
        lambda c: c.edit_image_url(
            model="m", prompt="p", images=[InputImage(data=_PNG, content_type="image/png")]
        ),
    ),
    (
        "imagerouter.generate_video_url",
        _imagerouter,
        lambda c: c.generate_video_url(model="m", prompt="p"),
    ),
    ("fal.run_image", _fal, lambda c: c.run_image("fal-ai/m", {"prompt": "p"})),
    ("fal.submit_queued", _fal, lambda c: c.submit_queued("fal-ai/m", {"prompt": "p"})),
    ("fal.poll_queued", _fal, lambda c: c.poll_queued(_JOB, model_id="fal-ai/m")),
    (
        "fal.fetch_queued_result",
        _fal,
        lambda c: c.fetch_queued_result(_JOB, model_id="fal-ai/m"),
    ),
    ("fal.cancel_queued", _fal, lambda c: c.cancel_queued(_JOB, model_id="fal-ai/m")),
    ("comfyui.upload_image", _comfyui, lambda c: c.upload_image(_PNG, "image/png")),
    ("comfyui.submit_prompt", _comfyui, lambda c: c.submit_prompt({"1": {}})),
    (
        "comfyui.retrieve_media",
        _comfyui,
        lambda c: c.retrieve_media(_HISTORY, output_type="image"),
    ),
]


@pytest.mark.parametrize("failure", TRANSPORT_FAILURES)
@pytest.mark.parametrize(("name", "factory", "call"), CASES, ids=[c[0] for c in CASES])
async def test_transport_failure_becomes_an_upstream_error(
    name: str,
    factory: Callable[[], Any],
    call: Callable[[Any], Awaitable[Any]],
    failure: Exception,
) -> None:
    client = factory()
    try:
        async with respx.mock(assert_all_called=True) as mock:
            # Every request the client makes fails the same way; the catch-all
            # keeps each case independent of the exact endpoint path.
            mock.route().mock(side_effect=failure)
            with pytest.raises(UpstreamError) as exc:
                await call(client)
    finally:
        await client.aclose()

    # The typed bridge error, rendered as a 502 — never the raw httpx2
    # exception, which the app would turn into a 500.
    assert exc.value.status_code == 502
    assert exc.value.error_type == "api_error"


async def test_asset_fetch_retries_a_transport_failure_then_succeeds() -> None:
    """A dropped CDN connection is worth another attempt, like a 5xx is.

    The status-code retries are pinned in test_upstream_status_mapping.py; the
    ``except httpx2.HTTPError`` retry beside them had no test at all.
    """
    import httpx  # respx authors responses in httpx v1 types; see conftest.py

    from openai_api_bridge.util.http import fetch_asset_with_retry

    async with respx.mock(assert_all_called=False) as mock:
        route = mock.get("https://cdn.example/flaky.png").mock(
            side_effect=[
                httpcore2.ConnectError("reset"),
                httpx.Response(200, content=b"ok", headers={"content-type": "image/png"}),
            ]
        )
        data, content_type = await fetch_asset_with_retry(
            "https://cdn.example/flaky.png", provider_label="Test", base_delay=0.001
        )

    assert (data, content_type) == (b"ok", "image/png")
    assert route.call_count == 2


async def test_asset_fetch_gives_up_on_a_persistent_transport_failure() -> None:
    from openai_api_bridge.util.http import fetch_asset_with_retry

    signed = "https://cdn.example/down.png?Signature=SECRETSIG"
    async with respx.mock(assert_all_called=False) as mock:
        route = mock.get(url=signed).mock(side_effect=httpcore2.ConnectError("refused"))
        with pytest.raises(UpstreamError, match="after 3 attempts") as exc:
            await fetch_asset_with_retry(
                signed, provider_label="Test", max_attempts=3, base_delay=0.001
            )

    assert route.call_count == 3
    assert exc.value.status_code == 502
    # Same redaction the status path guarantees: the message is sent to clients.
    assert "SECRETSIG" not in exc.value.message


# --- through the app: the envelope a client actually receives ---------------


@pytest.fixture
def client_with_openai(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(f"""
        [[providers]]
        id = "llama"
        backend = "openai"
        base_url = "{UPSTREAM}"
    """)
    )
    monkeypatch.setenv("BRIDGE_API_KEY", "test-bridge-api-key")
    monkeypatch.setenv("BRIDGE_CONFIG_PATH", str(config))
    monkeypatch.setenv("FILES_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "state.db"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    reset_caches_for_tests()

    from openai_api_bridge.main import create_app

    with TestClient(create_app()) as c:
        yield c
    reset_caches_for_tests()


HEADERS = {"Authorization": "Bearer test-bridge-api-key"}


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize("failure", TRANSPORT_FAILURES)
def test_unreachable_upstream_reaches_the_client_as_a_502_envelope(
    client_with_openai: TestClient, failure: Exception, stream: bool
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.post(f"{UPSTREAM}/v1/chat/completions").mock(side_effect=failure)
        r = client_with_openai.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "llama/m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": stream,
            },
        )

    assert r.status_code == 502
    error = r.json()["error"]
    assert error["type"] == "api_error"
    assert error["code"] == "upstream_error"
