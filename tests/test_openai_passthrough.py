"""End-to-end tests for the OpenAI-passthrough provider.

Stubs an upstream OpenAI-compat server with respx and verifies:
  * /v1/models aggregates upstream models, prefixed by provider id
  * /v1/chat/completions (non-streaming) forwards body + returns response
  * /v1/chat/completions (streaming) forwards SSE chunks byte-for-byte
  * /v1/embeddings forwards request + returns response
  * Backend rejection: comfyui or venice models routed to chat fails cleanly
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from openai_api_bridge.backends.openai.adapter import _extract_context_window
from openai_api_bridge.config import reset_caches_for_tests

UPSTREAM = "http://upstream.test"


def test_extract_context_window_sources() -> None:
    # Normalized field wins.
    assert _extract_context_window({"context_window": 32768}) == 32768
    # llama.cpp loaded model.
    assert _extract_context_window({"meta": {"n_ctx": 40960}}) == 40960
    # n_ctx_train is the trained ceiling, not the configured window — ignored.
    assert _extract_context_window({"meta": {"n_ctx_train": 262144}}) is None
    # vLLM.
    assert _extract_context_window({"max_model_len": 16384}) == 16384
    # llama.cpp router lists the child's argv even while cold.
    assert _extract_context_window({"status": {"args": ["--ctx-size", "65536"]}}) == 65536
    assert _extract_context_window({"status": {"args": ["-c", "8192"]}}) == 8192
    assert _extract_context_window({"status": {"args": ["--ctx-size=4096"]}}) == 4096
    # A clean numeric field beats the argv parse.
    assert (
        _extract_context_window(
            {"meta": {"n_ctx": 40960}, "status": {"args": ["--ctx-size", "65536"]}}
        )
        == 40960
    )
    # Non-positive / non-numeric / absent → None. (bool must not count as int.)
    assert _extract_context_window({"meta": {"n_ctx": 0}}) is None
    assert _extract_context_window({"context_window": True}) is None
    assert _extract_context_window({"status": {"args": ["--ctx-size", "nope"]}}) is None
    assert _extract_context_window({}) is None


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

    app = create_app()
    with TestClient(app) as c:
        yield c
    reset_caches_for_tests()


HEADERS = {"Authorization": "Bearer test-bridge-api-key"}


@respx.mock
def test_models_lists_upstream_models_with_prefix(
    client_with_openai: TestClient,
) -> None:
    respx.get(f"{UPSTREAM}/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    # A loaded llama.cpp model exposes meta.n_ctx (the configured
                    # context window) — we surface it as context_window.
                    {"id": "llama-3.1-8b", "object": "model", "meta": {"n_ctx": 40960}},
                    {"id": "text-embedding-3-large", "object": "model"},
                ],
            },
        )
    )
    r = client_with_openai.get("/v1/models", headers=HEADERS)
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()["data"]}
    assert "llama/llama-3.1-8b" in by_id
    assert "llama/text-embedding-3-large" in by_id
    assert by_id["llama/llama-3.1-8b"]["owned_by"] == "llama"
    # We don't classify openai-passthrough models — kind is None so the pipe
    # doesn't accidentally surface them as image/video.
    assert by_id["llama/llama-3.1-8b"]["kind"] is None
    # supports_tools is also omitted — this backend multiplexes for any
    # OpenAI-compatible upstream (real OpenAI, local llama, vLLM, etc.)
    # where per-model tool support varies wildly. The bridge can't tell
    # from the catalog; the client's per-endpoint fallback decides.
    assert "supports_tools" not in by_id["llama/llama-3.1-8b"]
    # context_window IS surfaced when the upstream exposed it (the bridge
    # otherwise strips the meta block that carries it). Omitted when unknown.
    assert by_id["llama/llama-3.1-8b"]["context_window"] == 40960
    assert "context_window" not in by_id["llama/text-embedding-3-large"]


@respx.mock
def test_models_catalogue_is_cached(client_with_openai: TestClient) -> None:
    """The catalogue is reused, like every other backend's.

    This backend was the one without a cache, which made a slow upstream cost
    a round trip on every single model-picker refresh rather than one per TTL
    window.
    """
    route = respx.get(f"{UPSTREAM}/v1/models").mock(
        return_value=httpx.Response(
            200, json={"object": "list", "data": [{"id": "m", "object": "model"}]}
        )
    )

    for _ in range(3):
        assert client_with_openai.get("/v1/models", headers=HEADERS).status_code == 200

    assert route.call_count == 1


@respx.mock
def test_chat_completion_sync_forwards_body_and_strips_prefix(
    client_with_openai: TestClient,
) -> None:
    captured = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "cmpl-123",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi!"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(side_effect=_capture)

    r = client_with_openai.post(
        "/v1/chat/completions",
        headers=HEADERS,
        json={
            "model": "llama/llama-3.1-8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Hi!"

    # The forwarded model id has the "llama/" prefix stripped — upstream
    # never sees our naming scheme.
    assert captured["body"]["model"] == "llama-3.1-8b"
    # All other fields passed through unchanged.
    assert captured["body"]["messages"] == [{"role": "user", "content": "Hello"}]
    assert captured["body"]["temperature"] == 0.7


@respx.mock
def test_chat_completion_stream_forwards_sse_chunks(
    client_with_openai: TestClient,
) -> None:
    # Simulate an SSE stream: one chunk per token, then [DONE].
    sse_body = (
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=sse_body, headers={"content-type": "text/event-stream"}
        )
    )

    with client_with_openai.stream(
        "POST",
        "/v1/chat/completions",
        headers=HEADERS,
        json={
            "model": "llama/llama-3.1-8b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        received = b"".join(r.iter_bytes())

    assert b"Hello" in received
    assert b"world" in received
    assert b"[DONE]" in received


@respx.mock
def test_chat_completion_unknown_provider_returns_404(
    client_with_openai: TestClient,
) -> None:
    r = client_with_openai.post(
        "/v1/chat/completions",
        headers=HEADERS,
        json={"model": "ghost/foo", "messages": []},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "provider_not_found"


@respx.mock
def test_chat_completion_missing_model_returns_400(
    client_with_openai: TestClient,
) -> None:
    r = client_with_openai.post(
        "/v1/chat/completions",
        headers=HEADERS,
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "model"


@respx.mock
def test_embeddings_forwards_body_and_strips_prefix(
    client_with_openai: TestClient,
) -> None:
    captured = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": "text-embedding-3-large",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        )

    respx.post(f"{UPSTREAM}/v1/embeddings").mock(side_effect=_capture)

    r = client_with_openai.post(
        "/v1/embeddings",
        headers=HEADERS,
        json={
            "model": "llama/text-embedding-3-large",
            "input": "hello world",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["data"][0]["embedding"] == [0.1, 0.2]
    assert captured["body"]["model"] == "text-embedding-3-large"
    assert captured["body"]["input"] == "hello world"


@respx.mock
def test_chat_against_comfyui_provider_returns_unsupported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A user picking a ComfyUI workflow id for chat completions gets a clean
    400 instead of an internal traceback."""
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(f"""
        [[providers]]
        id = "comfyui"
        backend = "comfyui"
        url = "http://127.0.0.1:8188"
        workflows_dir = "{tmp_path}"
    """)
    )

    monkeypatch.setenv("BRIDGE_API_KEY", "test-bridge-api-key")
    monkeypatch.setenv("BRIDGE_CONFIG_PATH", str(config))
    monkeypatch.setenv("FILES_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "state.db"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    reset_caches_for_tests()

    from openai_api_bridge.main import create_app

    app = create_app()
    try:
        with TestClient(app) as c:
            r = c.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test-bridge-api-key"},
                json={"model": "comfyui/anything", "messages": []},
            )
            assert r.status_code == 400
            assert r.json()["error"]["code"] == "unsupported_operation"
    finally:
        reset_caches_for_tests()


@respx.mock
def test_upstream_5xx_propagates_as_502(
    client_with_openai: TestClient,
) -> None:
    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(503, text="overloaded")
    )
    r = client_with_openai.post(
        "/v1/chat/completions",
        headers=HEADERS,
        json={"model": "llama/foo", "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "upstream_error"


@respx.mock
def test_upstream_4xx_propagates_as_400(
    client_with_openai: TestClient,
) -> None:
    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad model"}})
    )
    r = client_with_openai.post(
        "/v1/chat/completions",
        headers=HEADERS,
        json={"model": "llama/foo", "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 400


@respx.mock
def test_embeddings_body_is_forwarded_byte_for_byte(
    client_with_openai: TestClient,
) -> None:
    """Passthrough means passthrough.

    Parsing an ingestion batch into Python objects only to re-serialise it is
    event-loop time every other client waits through, for a byte-identical
    result — and a round trip through Python floats is not guaranteed to be
    byte-identical anyway.
    """
    # Formatting a real upstream would never produce, and float spellings that
    # a parse-then-dump round trip would normalise away.
    raw = b'{ "object":"list",\n  "data":[{"embedding":[1.0000000000000002,1e-9,-0.0]}] }'
    respx.post(f"{UPSTREAM}/v1/embeddings").mock(
        return_value=httpx.Response(200, content=raw, headers={"content-type": "application/json"})
    )

    r = client_with_openai.post(
        "/v1/embeddings", headers=HEADERS, json={"model": "llama/embed", "input": "hi"}
    )

    assert r.status_code == 200
    assert r.content == raw
    assert r.headers["content-type"].startswith("application/json")


@respx.mock
def test_chat_completion_body_is_forwarded_byte_for_byte(
    client_with_openai: TestClient,
) -> None:
    raw = b'{"id":"c1",\n "choices":[{"message":{"content":"hi","role":"assistant"}}]}'
    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=raw, headers={"content-type": "application/json"})
    )

    r = client_with_openai.post(
        "/v1/chat/completions",
        headers=HEADERS,
        json={"model": "llama/m", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 200
    assert r.content == raw
    assert r.json()["choices"][0]["message"]["content"] == "hi"


@respx.mock
@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (b"<html><head><title>502 Bad Gateway</title></head><body>cf</body></html>", "text/html"),
        (b"", "application/json"),
        (b"   \n", "application/json"),
        (b"[1, 2, 3]", "application/json"),
    ],
    ids=["html-interstitial", "empty", "whitespace-only", "json-array"],
)
def test_non_json_200_surfaces_as_an_upstream_error(
    client_with_openai: TestClient, body: bytes, content_type: str
) -> None:
    """Forwarding the body unexamined is not the same as forwarding it unchecked.

    A captive portal, CDN error page or WAF interstitial answering 200 with
    HTML — or a proxy that committed the status line and sent nothing — would
    otherwise reach the client as a 200 labelled application/json: the SDK
    raises an opaque decode error naming no provider, retry-on-5xx never
    fires, and an empty body reads as a successful zero-length result.
    """
    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=body, headers={"content-type": content_type})
    )

    r = client_with_openai.post(
        "/v1/chat/completions",
        headers=HEADERS,
        json={"model": "llama/m", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 502
    assert "non-JSON 200" in r.json()["error"]["message"]


@respx.mock
def test_non_json_200_on_embeddings_surfaces_as_an_upstream_error(
    client_with_openai: TestClient,
) -> None:
    """The ingestion path especially: an empty 200 must not read as success."""
    respx.post(f"{UPSTREAM}/v1/embeddings").mock(
        return_value=httpx.Response(200, content=b"", headers={"content-type": "application/json"})
    )

    r = client_with_openai.post(
        "/v1/embeddings", headers=HEADERS, json={"model": "llama/e", "input": "hi"}
    )

    assert r.status_code == 502
    assert "non-JSON 200" in r.json()["error"]["message"]


@respx.mock
def test_leading_whitespace_before_a_json_object_is_still_accepted(
    client_with_openai: TestClient,
) -> None:
    """The check is a shape sniff, not a parse — it must not reject valid JSON."""
    raw = b'\n  {"id":"c1","choices":[]}'
    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=raw, headers={"content-type": "application/json"})
    )

    r = client_with_openai.post(
        "/v1/chat/completions",
        headers=HEADERS,
        json={"model": "llama/m", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 200
    assert r.content == raw


@respx.mock
@pytest.mark.parametrize(
    ("body", "label"),
    [
        (b'\xef\xbb\xbf{"id":"c1"}', "utf-8-sig"),
        ('{"id":"c1"}'.encode("utf-16"), "utf-16"),
        ('{"id":"c1"}'.encode("utf-32"), "utf-32"),
    ],
    ids=["utf-8-bom", "utf-16", "utf-32"],
)
def test_an_encoding_preamble_is_not_mistaken_for_a_non_json_body(
    client_with_openai: TestClient, body: bytes, label: str
) -> None:
    """A BOM means "text in a declared encoding", not "an HTML error page".

    json.loads on bytes sniffs the encoding, so these parsed fine before the
    shape check existed — and the client's own parser handles them too.
    Rejecting them would turn a working upstream into a 502 whose message
    claims the body isn't JSON when it demonstrably is.
    """
    # Guard the premise: these really are JSON as far as a parser is concerned.
    assert json.loads(body)["id"] == "c1"

    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=body, headers={"content-type": "application/json"})
    )

    r = client_with_openai.post(
        "/v1/chat/completions",
        headers=HEADERS,
        json={"model": "llama/m", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 200, f"{label} body rejected: {r.text[:200]}"
    assert r.content == body


@respx.mock
@pytest.mark.parametrize(
    ("body", "why"),
    [
        (b'{"id": "abc", "choices": [', "truncated mid-array"),
        (b'{"id": "abc"}{"id": "def"}', "two concatenated objects"),
        (b"\xef\xbb\xbf<html><body>WAF</body></html>", "BOM-prefixed HTML"),
        (b"\xef\xbb\xbf", "bare BOM, no payload"),
        (b'[{"id": "abc"}]', "a JSON array, not an object"),
        (b'{"id": "abc",}', "trailing comma"),
    ],
    ids=["truncated", "concatenated", "bom-html", "bare-bom", "array", "trailing-comma"],
)
def test_a_200_that_is_not_a_json_object_is_rejected(
    client_with_openai: TestClient, body: bytes, why: str
) -> None:
    """A first-byte sniff cannot tell these from valid JSON; a parse can.

    Every one of these starts with `{` or a BOM, so a shape check waves them
    through — and each then reaches the client as a 200 labelled
    application/json, which is the exact failure this guard exists to stop.
    """
    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=body, headers={"content-type": "application/json"})
    )

    r = client_with_openai.post(
        "/v1/chat/completions",
        headers=HEADERS,
        json={"model": "llama/m", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 502, f"{why} was accepted"
    assert "non-JSON 200" in r.json()["error"]["message"]


@respx.mock
def test_a_body_past_the_validation_gate_skips_the_parse(
    client_with_openai: TestClient,
) -> None:
    """The gate is what keeps a multi-MB embedding batch off the event loop.

    Past it the body is only shape-checked, so a large malformed one is
    forwarded — a deliberate, documented gap, pinned here so it can't change
    silently.
    """
    from openai_api_bridge.backends.openai.client import _MAX_VALIDATED_BODY

    oversized = b'{"data": [' + b"0" * (_MAX_VALIDATED_BODY + 1)
    assert len(oversized) > _MAX_VALIDATED_BODY
    respx.post(f"{UPSTREAM}/v1/embeddings").mock(
        return_value=httpx.Response(
            200, content=oversized, headers={"content-type": "application/json"}
        )
    )

    r = client_with_openai.post(
        "/v1/embeddings", headers=HEADERS, json={"model": "llama/e", "input": "hi"}
    )

    assert r.status_code == 200
    assert r.content == oversized
