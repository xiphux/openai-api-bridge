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
    {"id": "black-forest-labs/FLUX-1.1-pro", "output": ["image"]},
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

    # Asset-fetch must carry the Bearer auth — ImageRouter's storage host
    # is auth-gated and returns 401 without it. Regression guard for the
    # original "auth not needed for CDN URLs" misconception that caused
    # real production breakage; respx doesn't filter by headers, so the
    # previous bug ran green here even though the asset fetch was
    # unauthenticated in production.
    assert asset_route.called
    asset_req = asset_route.calls.last.request
    assert asset_req.headers.get("authorization") == "Bearer ir-secret"


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
        pytest.fail(f"Video job didn't complete in time (last status={state['status']})")

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
