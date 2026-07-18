"""End-to-end tests for the ImageRouter provider.

Stubs the upstream imagerouter.io HTTP surface with respx and verifies:
  * /v1/models aggregates the v2/models catalog with image+video kinds
  * /v1/images/generations returns image bytes via fetch-by-url
  * /v1/images/edits forwards the multipart upload
  * /v1/videos round-trip: queued → in_progress → completed → /content
  * Filtering: text-only and audio models are dropped from the catalog
"""

from __future__ import annotations

import textwrap
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from openai_api_bridge.config import reset_caches_for_tests

UPSTREAM = "https://api.imagerouter.io"


@pytest.fixture
def client_with_imagerouter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[TestClient]:
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent("""
		[[providers]]
		id = "ir"
		backend = "imagerouter"
		api_token_env = "TEST_IR_TOKEN"
	""")
    )

    monkeypatch.setenv("BRIDGE_API_KEY", "test-bridge-key")
    monkeypatch.setenv("BRIDGE_CONFIG_PATH", str(config))
    monkeypatch.setenv("FILES_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "state.db"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("TEST_IR_TOKEN", "ir-secret")
    reset_caches_for_tests()

    from openai_api_bridge.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    reset_caches_for_tests()


HEADERS = {"Authorization": "Bearer test-bridge-key"}


# Realistic v2/models response shape — array of model objects, each with
# id + output array. Mixed modalities so we can verify our image+video
# filter does the right thing.
_MODELS_FIXTURE = [
    # Real shape: ImageRouter states accepted inputs per model, and they differ
    # — FLUX-1.1-pro is text-only while schnell also takes a reference image.
    {
        "id": "black-forest-labs/FLUX-1.1-pro",
        "output": ["image"],
        "inputs": {"text": True, "image": False, "mask": False},
    },
    {
        "id": "black-forest-labs/FLUX-1-schnell",
        "output": ["image"],
        "inputs": {"text": True, "image": True, "mask": True},
    },
    {"id": "openai/gpt-image-1", "output": ["image"]},
    {"id": "stability-ai/stable-video-3d", "output": ["video"]},
    # These should be filtered out — bridge only surfaces image+video.
    {"id": "openai/gpt-4o", "output": ["text"]},
    {"id": "elevenlabs/eleven-multilingual-v2", "output": ["audio"]},
    # Defensive: malformed entries should be skipped, not crash.
    {"id": "broken-no-output"},
    {"output": ["image"]},  # missing id
    "not-even-an-object",
]


@respx.mock
def test_models_lists_image_and_video_only(
    client_with_imagerouter: TestClient,
) -> None:
    respx.get(f"{UPSTREAM}/v2/models").mock(return_value=httpx.Response(200, json=_MODELS_FIXTURE))
    r = client_with_imagerouter.get("/v1/models", headers=HEADERS)
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()["data"]}
    assert "ir/black-forest-labs/FLUX-1.1-pro" in by_id
    assert "ir/openai/gpt-image-1" in by_id
    assert "ir/stability-ai/stable-video-3d" in by_id
    # Filtered out
    assert "ir/openai/gpt-4o" not in by_id
    assert "ir/elevenlabs/eleven-multilingual-v2" not in by_id
    # Kind hints surface from the upstream's `output` array.
    assert by_id["ir/black-forest-labs/FLUX-1.1-pro"]["kind"] == "image"
    assert by_id["ir/stability-ai/stable-video-3d"]["kind"] == "video"
    # Capabilities come from the catalogue's own `inputs` map, so a text-only
    # model is distinguishable from one that accepts a reference image.
    assert by_id["ir/black-forest-labs/FLUX-1.1-pro"]["capabilities"] == ["text-to-image"]
    assert by_id["ir/black-forest-labs/FLUX-1-schnell"]["capabilities"] == [
        "text-to-image",
        "image-to-image",
    ]
    # Omitted, not guessed, when the upstream didn't say.
    assert "capabilities" not in by_id["ir/openai/gpt-image-1"]


@respx.mock
def test_images_generations_round_trip(
    client_with_imagerouter: TestClient,
) -> None:
    """Bridge POSTs to /v1/openai/images/generations, gets back a URL,
    fetches the bytes, persists via the FileStore, returns an OpenAI-shaped
    envelope to the client."""
    gen_route = respx.post(f"{UPSTREAM}/v1/openai/images/generations").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"url": "https://cdn.example/imgs/abc.png"}]},
        )
    )
    img_bytes = b"\x89PNG\r\n\x1a\nfake-png-bytes"
    asset_route = respx.get("https://cdn.example/imgs/abc.png").mock(
        return_value=httpx.Response(200, content=img_bytes, headers={"content-type": "image/png"})
    )

    r = client_with_imagerouter.post(
        "/v1/images/generations",
        headers=HEADERS,
        json={
            "model": "ir/openai/gpt-image-1",
            "prompt": "a red panda",
            "size": "1024x1024",
        },
    )
    assert r.status_code == 200
    body = r.json()
    # OpenAI envelope shape — { data: [{ url: ... }] } where url points back
    # at the bridge's /v1/files/.../content surface.
    assert "data" in body and len(body["data"]) == 1
    url = body["data"][0]["url"]
    assert "/v1/files/" in url and url.endswith("/content")

    # The bridge's outgoing call: prompt + model + size were forwarded; the
    # upstream model id had its provider prefix stripped by parse_model_id.
    assert gen_route.called
    sent_payload = gen_route.calls.last.request.read()
    import json

    sent = json.loads(sent_payload)
    assert sent["model"] == "openai/gpt-image-1"
    assert sent["prompt"] == "a red panda"
    assert sent["size"] == "1024x1024"
    assert sent["response_format"] == "url"

    # Asset-fetch must NOT carry the Bearer auth — ImageRouter's storage
    # host (storage.imagerouter.io) is publicly accessible per their docs,
    # and sending auth headers can cause issues. The bridge fetches assets
    # without authentication to avoid any cross-origin header-stripping
    # behavior and to match the documented public access model.
    assert asset_route.called
    asset_req = asset_route.calls.last.request
    assert asset_req.headers.get("authorization") is None


@respx.mock
def test_images_edits_round_trip(
    client_with_imagerouter: TestClient,
) -> None:
    """Bridge forwards multipart upload to /v1/openai/images/edits, fetches
    the result by url, returns it to the client."""
    edit_route = respx.post(f"{UPSTREAM}/v1/openai/images/edits").mock(
        return_value=httpx.Response(
            200, json={"data": [{"url": "https://cdn.example/imgs/edited.png"}]}
        )
    )
    respx.get("https://cdn.example/imgs/edited.png").mock(
        return_value=httpx.Response(
            200, content=b"edited-bytes", headers={"content-type": "image/png"}
        )
    )

    r = client_with_imagerouter.post(
        "/v1/images/edits",
        headers=HEADERS,
        files={"image": ("input.png", b"fake-input-png", "image/png")},
        data={
            "model": "ir/openai/gpt-image-1",
            "prompt": "make her hair blue",
        },
    )
    assert r.status_code == 200
    url = r.json()["data"][0]["url"]
    assert "/v1/files/" in url

    # Verify the multipart was forwarded with the expected form fields.
    assert edit_route.called
    sent = edit_route.calls.last.request
    body_text = sent.content.decode("utf-8", errors="replace")
    assert "make her hair blue" in body_text
    assert "openai/gpt-image-1" in body_text
    # image[] field name (not image) — multi-image-friendly
    assert 'name="image[]"' in body_text


@respx.mock
def test_images_edits_forwards_multiple_references(
    client_with_imagerouter: TestClient,
) -> None:
    """Two reference images sent as a repeated ``image`` field both survive
    to the outgoing ``image[]`` multipart, in order — guards the bug where a
    single declared UploadFile dropped all but the last reference image."""
    edit_route = respx.post(f"{UPSTREAM}/v1/openai/images/edits").mock(
        return_value=httpx.Response(
            200, json={"data": [{"url": "https://cdn.example/imgs/edited.png"}]}
        )
    )
    respx.get("https://cdn.example/imgs/edited.png").mock(
        return_value=httpx.Response(
            200, content=b"edited-bytes", headers={"content-type": "image/png"}
        )
    )

    r = client_with_imagerouter.post(
        "/v1/images/edits",
        headers=HEADERS,
        # Repeated ``image`` field — the shape GlyphStream sends for multi-ref.
        files=[
            ("image", ("first.png", b"FIRST-REFERENCE-BYTES", "image/png")),
            ("image", ("second.png", b"SECOND-REFERENCE-BYTES", "image/png")),
        ],
        data={"model": "ir/openai/gpt-image-1", "prompt": "combine these"},
    )
    assert r.status_code == 200, r.text

    sent = edit_route.calls.last.request
    body_bytes = sent.content
    # Both reference payloads forwarded, each under image[].
    assert body_bytes.count(b'name="image[]"') == 2
    assert b"FIRST-REFERENCE-BYTES" in body_bytes
    assert b"SECOND-REFERENCE-BYTES" in body_bytes
    # Order preserved: first reference precedes the second.
    assert body_bytes.index(b"FIRST-REFERENCE-BYTES") < body_bytes.index(b"SECOND-REFERENCE-BYTES")


@respx.mock
def test_images_edits_accepts_bracket_field(
    client_with_imagerouter: TestClient,
) -> None:
    """The ``image[]`` array convention is accepted alongside plain ``image``."""
    respx.post(f"{UPSTREAM}/v1/openai/images/edits").mock(
        return_value=httpx.Response(
            200, json={"data": [{"url": "https://cdn.example/imgs/edited.png"}]}
        )
    )
    respx.get("https://cdn.example/imgs/edited.png").mock(
        return_value=httpx.Response(
            200, content=b"edited-bytes", headers={"content-type": "image/png"}
        )
    )

    r = client_with_imagerouter.post(
        "/v1/images/edits",
        headers=HEADERS,
        files=[
            ("image[]", ("a.png", b"A-BYTES", "image/png")),
            ("image[]", ("b.png", b"B-BYTES", "image/png")),
        ],
        data={"model": "ir/openai/gpt-image-1", "prompt": "combine these"},
    )
    assert r.status_code == 200, r.text


@respx.mock
def test_videos_round_trip_t2v(
    client_with_imagerouter: TestClient,
) -> None:
    """POST /v1/videos → queued → poll → completed → fetch /content."""
    respx.post(f"{UPSTREAM}/v1/openai/videos/generations").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"url": "https://cdn.example/vids/clip.mp4"}]},
        )
    )
    video_bytes = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100
    respx.get("https://cdn.example/vids/clip.mp4").mock(
        return_value=httpx.Response(200, content=video_bytes, headers={"content-type": "video/mp4"})
    )

    r = client_with_imagerouter.post(
        "/v1/videos",
        headers=HEADERS,
        data={"model": "ir/stability-ai/stable-video-3d", "prompt": "a soaring eagle"},
    )
    assert r.status_code == 200
    job = r.json()
    assert "id" in job
    assert job["status"] in ("queued", "in_progress")

    # Poll until the runner finishes. With respx-stubbed sync upstream this
    # is bounded — should finish in a few hundred ms.
    job_id = job["id"]
    deadline = time.time() + 5.0
    state = {}
    while time.time() < deadline:
        r = client_with_imagerouter.get(f"/v1/videos/{job_id}", headers=HEADERS)
        assert r.status_code == 200
        state = r.json()
        if state["status"] == "completed":
            break
        if state["status"] == "failed":
            pytest.fail(f"Video job failed: {state.get('error', {}).get('message')}")
        time.sleep(0.05)
    else:
        pytest.fail(
            f"Video job didn't complete in time (last status={state.get('status', 'unknown')})"
        )

    r = client_with_imagerouter.get(f"/v1/videos/{job_id}/content", headers=HEADERS)
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert r.content == video_bytes


@respx.mock
def test_models_upstream_error_surfaces(
    client_with_imagerouter: TestClient,
) -> None:
    respx.get(f"{UPSTREAM}/v2/models").mock(
        return_value=httpx.Response(500, json={"error": {"message": "imagerouter is on fire"}})
    )
    r = client_with_imagerouter.get("/v1/models", headers=HEADERS)
    # The /v1/models endpoint aggregates across providers; a single backend
    # failing shouldn't 500 the whole listing — it should be visible
    # either via an absent entry or via the per-provider error envelope.
    # (Either way: we don't crash.)
    assert r.status_code in (200, 502, 503)


@respx.mock
def test_images_generations_upstream_error_surfaces(
    client_with_imagerouter: TestClient,
) -> None:
    respx.post(f"{UPSTREAM}/v1/openai/images/generations").mock(
        return_value=httpx.Response(
            429,
            json={"error": {"message": "rate limited"}},
        )
    )
    r = client_with_imagerouter.post(
        "/v1/images/generations",
        headers=HEADERS,
        json={"model": "ir/openai/gpt-image-1", "prompt": "x"},
    )
    # Bridge should surface the upstream failure as something non-2xx,
    # not crash.
    assert r.status_code >= 400
    assert "rate limited" in r.text.lower() or "500" not in r.text


@respx.mock
def test_model_catalog_is_cached(client_with_imagerouter: TestClient) -> None:
    """/v1/models fans out to every provider on each request, so repeating it
    shouldn't keep costing an upstream round trip."""
    route = respx.get(f"{UPSTREAM}/v2/models").mock(
        return_value=httpx.Response(200, json=_MODELS_FIXTURE)
    )
    for _ in range(3):
        assert client_with_imagerouter.get("/v1/models", headers=HEADERS).status_code == 200
    assert route.call_count == 1


@respx.mock
def test_failed_catalog_fetch_is_not_retried_during_the_cooldown(
    client_with_imagerouter: TestClient,
) -> None:
    """The fetch runs under a lock, so a remembered failure is what stops a
    burst during an upstream hang from queueing up — each waiter otherwise
    starts its own fetch once the previous one gives up."""
    calls = {"n": 0}

    def responder(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": "down"})

    respx.get(f"{UPSTREAM}/v2/models").mock(side_effect=responder)

    for _ in range(3):
        r = client_with_imagerouter.get("/v1/models", headers=HEADERS)
        # The listing itself still works; this provider is just absent.
        assert r.status_code == 200
        assert not [m for m in r.json().get("data", []) if m["id"].startswith("ir/")]
    assert calls["n"] == 1, f"one attempt expected during the cooldown, got {calls['n']}"


@respx.mock
def test_failed_catalog_fetch_recovers_after_the_cooldown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failure is remembered, not latched — the provider returns on its own."""
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent("""
        [[providers]]
        id = "ir"
        backend = "imagerouter"
        api_token_env = "TEST_IR_TOKEN"
        catalog_retry_seconds = 0
    """)
    )
    monkeypatch.setenv("BRIDGE_API_KEY", "test-bridge-key")
    monkeypatch.setenv("BRIDGE_CONFIG_PATH", str(config))
    monkeypatch.setenv("FILES_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "state.db"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("TEST_IR_TOKEN", "ir-secret")
    reset_caches_for_tests()

    from openai_api_bridge.main import create_app

    calls = {"n": 0}

    def responder(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": "down"})
        return httpx.Response(200, json=_MODELS_FIXTURE)

    respx.get(f"{UPSTREAM}/v2/models").mock(side_effect=responder)

    with TestClient(create_app()) as client:
        first = client.get("/v1/models", headers=HEADERS)
        assert not [m for m in first.json().get("data", []) if m["id"].startswith("ir/")]
        second = client.get("/v1/models", headers=HEADERS)
        assert [m for m in second.json()["data"] if m["id"].startswith("ir/")]
    reset_caches_for_tests()


@respx.mock
def test_non_json_200_is_an_upstream_error_not_a_crash(
    client_with_imagerouter: TestClient,
) -> None:
    """A 200 carrying HTML — captive portal, CDN error page, WAF interstitial —
    used to raise JSONDecodeError, which isn't a BridgeError, so it escaped the
    per-provider handler as an unhandled 500 with a full traceback."""
    respx.get(f"{UPSTREAM}/v2/models").mock(
        return_value=httpx.Response(
            200,
            text="<html><body>Gateway Timeout</body></html>",
            headers={"content-type": "text/html"},
        )
    )
    r = client_with_imagerouter.get("/v1/models", headers=HEADERS)
    # Handled per-provider: the listing survives, this provider is just absent.
    assert r.status_code == 200
    assert not [m for m in r.json().get("data", []) if m["id"].startswith("ir/")]


@respx.mock
def test_non_json_200_on_generation_surfaces_as_502(
    client_with_imagerouter: TestClient,
) -> None:
    respx.post(f"{UPSTREAM}/v1/openai/images/generations").mock(
        return_value=httpx.Response(200, text="<html>nope</html>")
    )
    r = client_with_imagerouter.post(
        "/v1/images/generations",
        headers=HEADERS,
        json={"model": "ir/openai/gpt-image-1", "prompt": "x"},
    )
    assert r.status_code == 502, r.text
    assert "non-JSON" in r.json()["error"]["message"]
