"""End-to-end ComfyUI generation flow with respx-stubbed upstream.

Covers the full image-generation happy path:
  1. ``POST /v1/images/generations`` with model="comfyui/<slug>"
  2. Workflow loaded, prompt injected, submitted to fake ComfyUI
  3. /history poll returns a completed entry
  4. /view fetches the output bytes
  5. FileStore persists; response carries a /v1/files/{id}/content URL
  6. Hitting that URL streams the bytes back
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

from openai_api_bridge.config import reset_caches_for_tests


@pytest.fixture
def comfyui_workflows_dir(tmp_path: Path) -> Path:
    d = tmp_path / "workflows"
    d.mkdir()
    # Minimal valid workflow: a CLIPTextEncode + a SaveImage so output_type=image.
    (d / "tiny-t2i.json").write_text(
        json.dumps(
            {
                "1": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": "placeholder"},
                },
                "2": {"class_type": "SaveImage", "inputs": {}},
            }
        )
    )
    (d / "tiny-t2i.meta.json").write_text(
        json.dumps(
            {
                "positive_prompt_node": "1",
                "display_name": "Tiny T2I",
                "prompt_style": "booru-tags",
                "prompt_hint": "prefix: masterpiece, best quality",
            }
        )
    )
    return d


@pytest.fixture
def client_with_comfyui(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    comfyui_workflows_dir: Path,
) -> Iterator[TestClient]:
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(f"""
        [[providers]]
        id = "comfyui"
        backend = "comfyui"
        url = "http://127.0.0.1:8188"
        workflows_dir = "{comfyui_workflows_dir}"
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


COMFY = "http://127.0.0.1:8188"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20


@respx.mock
def test_models_lists_comfyui_workflow(
    client_with_comfyui: TestClient,
) -> None:
    r = client_with_comfyui.get(
        "/v1/models", headers={"Authorization": "Bearer test-bridge-api-key"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    by_id = {m["id"]: m for m in body["data"]}
    assert "comfyui/tiny-t2i" in by_id
    entry = by_id["comfyui/tiny-t2i"]
    assert entry["owned_by"] == "comfyui"
    # The non-standard display_name carries the human-readable label from meta.json.
    assert entry["display_name"] == "Tiny T2I"
    # `kind` lets gateway-aware clients route to the right endpoint.
    assert entry["kind"] == "image"
    # prompt_style / prompt_hint flow from meta.json for an image model's
    # prompt-enhancement pass.
    assert entry["prompt_style"] == "booru-tags"
    assert entry["prompt_hint"] == "prefix: masterpiece, best quality"
    # This workflow declares no image_inputs, so it's text-only — the same
    # declaration edit_image gates on, surfaced so a client knows up front
    # instead of discovering it from a rejected request.
    assert entry["capabilities"] == ["text-to-image"]


@respx.mock
def test_models_capabilities_follow_declared_image_inputs(
    client_with_comfyui: TestClient,
    comfyui_workflows_dir: Path,
) -> None:
    """A workflow that declares image_inputs accepts img2img, and says so."""
    (comfyui_workflows_dir / "i2i.json").write_text(
        (comfyui_workflows_dir / "tiny-t2i.json").read_text()
    )
    (comfyui_workflows_dir / "i2i.meta.json").write_text(
        json.dumps(
            {
                "positive_prompt_node": "1",
                "display_name": "Tiny I2I",
                "image_inputs": [{"node": "2"}],
            }
        )
    )
    r = client_with_comfyui.get(
        "/v1/models", headers={"Authorization": "Bearer test-bridge-api-key"}
    )
    by_id = {m["id"]: m for m in r.json()["data"]}
    assert by_id["comfyui/i2i"]["capabilities"] == ["text-to-image", "image-to-image"]
    assert by_id["comfyui/tiny-t2i"]["capabilities"] == ["text-to-image"]


@respx.mock
def test_models_display_name_falls_back_to_slug_when_meta_lacks_one(
    client_with_comfyui: TestClient,
    comfyui_workflows_dir: Path,
) -> None:
    # Drop a second workflow whose meta.json doesn't set display_name.
    (comfyui_workflows_dir / "no-name.json").write_text(
        json.dumps(
            {
                "1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
                "2": {"class_type": "SaveImage", "inputs": {}},
            }
        )
    )
    (comfyui_workflows_dir / "no-name.meta.json").write_text(
        json.dumps({"positive_prompt_node": "1"})
    )
    r = client_with_comfyui.get(
        "/v1/models", headers={"Authorization": "Bearer test-bridge-api-key"}
    )
    by_id = {m["id"]: m for m in r.json()["data"]}
    # Without display_name in meta, the bridge falls back to the source filename.
    assert by_id["comfyui/no-name"]["display_name"] == "no-name"
    # A workflow that doesn't declare prompt_style/prompt_hint omits them
    # entirely (not null) so clients apply their own fallback.
    assert "prompt_style" not in by_id["comfyui/no-name"]
    assert "prompt_hint" not in by_id["comfyui/no-name"]


@respx.mock
def test_image_generation_full_flow(
    client_with_comfyui: TestClient,
) -> None:
    headers = {"Authorization": "Bearer test-bridge-api-key"}

    # ComfyUI accepts the prompt and returns a prompt_id.
    submit_route = respx.post(f"{COMFY}/prompt").mock(
        return_value=httpx.Response(200, json={"prompt_id": "abc-123"})
    )
    # First poll: empty (job not complete).
    # Second poll: returns the prompt_id with a SaveImage output.
    history_responses = [
        httpx.Response(200, json={}),
        httpx.Response(
            200,
            json={
                "abc-123": {
                    "outputs": {
                        "2": {
                            "images": [
                                {
                                    "filename": "ComfyUI_00001_.png",
                                    "subfolder": "",
                                    "type": "output",
                                }
                            ]
                        }
                    }
                }
            },
        ),
    ]
    respx.get(f"{COMFY}/history/abc-123").mock(side_effect=history_responses)
    # /view returns the PNG bytes.
    respx.get(f"{COMFY}/view").mock(
        return_value=httpx.Response(
            200,
            content=PNG_MAGIC,
            headers={"content-type": "image/png"},
        )
    )

    r = client_with_comfyui.post(
        "/v1/images/generations",
        headers=headers,
        json={
            "model": "comfyui/tiny-t2i",
            "prompt": "a tiny test image",
            "size": "512x512",
            "n": 1,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "data" in body and len(body["data"]) == 1
    url = body["data"][0]["url"]
    assert url.startswith("/v1/files/") and url.endswith("/content")

    # ComfyUI was called with our prompt + a client_id (last_node_id workaround).
    submitted = json.loads(submit_route.calls[0].request.content)
    assert "client_id" in submitted
    assert submitted["prompt"]["1"]["inputs"]["text"] == "a tiny test image"

    # Now hit the bridge's own URL — should stream back the bytes we stubbed.
    r2 = client_with_comfyui.get(url, headers=headers)
    assert r2.status_code == 200
    assert r2.content == PNG_MAGIC
    assert r2.headers["content-type"] == "image/png"

    # A generated asset never changes, so a client holding it should be able
    # to skip the transfer entirely on the next render rather than
    # re-downloading the whole file.
    assert r2.headers["cache-control"] == "private, max-age=31536000, immutable"
    r3 = client_with_comfyui.get(url, headers={**headers, "If-None-Match": r2.headers["etag"]})
    assert r3.status_code == 304
    assert r3.content == b""


@respx.mock
def test_image_generation_b64_json_response_format(
    client_with_comfyui: TestClient,
) -> None:
    headers = {"Authorization": "Bearer test-bridge-api-key"}
    respx.post(f"{COMFY}/prompt").mock(return_value=httpx.Response(200, json={"prompt_id": "p"}))
    respx.get(f"{COMFY}/history/p").mock(
        return_value=httpx.Response(
            200,
            json={
                "p": {
                    "outputs": {
                        "2": {"images": [{"filename": "x.png", "subfolder": "", "type": "output"}]}
                    }
                }
            },
        )
    )
    respx.get(f"{COMFY}/view").mock(
        return_value=httpx.Response(200, content=PNG_MAGIC, headers={"content-type": "image/png"})
    )

    r = client_with_comfyui.post(
        "/v1/images/generations",
        headers=headers,
        json={
            "model": "comfyui/tiny-t2i",
            "prompt": "x",
            "response_format": "b64_json",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["data"][0].get("b64_json")
    assert body["data"][0].get("url") is None


@respx.mock
def test_workflow_invalid_returns_400(
    client_with_comfyui: TestClient,
) -> None:
    headers = {"Authorization": "Bearer test-bridge-api-key"}
    respx.post(f"{COMFY}/prompt").mock(
        return_value=httpx.Response(
            400,
            text='{"error":"node 99 not found"}',
        )
    )
    r = client_with_comfyui.post(
        "/v1/images/generations",
        headers=headers,
        json={"model": "comfyui/tiny-t2i", "prompt": "x"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "workflow_invalid"


@respx.mock
def test_video_endpoint_rejects_image_workflow(
    client_with_comfyui: TestClient,
) -> None:
    headers = {"Authorization": "Bearer test-bridge-api-key"}
    # The runner runs in the background — we wait on it via a polled GET
    # that flips to 'failed' once the upstream rejects.
    respx.post(f"{COMFY}/prompt")  # not actually called (rejected before submit)

    r = client_with_comfyui.post(
        "/v1/videos",
        headers=headers,
        data={"model": "comfyui/tiny-t2i", "prompt": "x"},
    )
    # Job creation succeeds; runner will then mark it failed.
    assert r.status_code == 200
    job_id = r.json()["id"]

    # Poll briefly until the runner sets status=failed.
    import time

    for _ in range(20):
        r = client_with_comfyui.get(f"/v1/videos/{job_id}", headers=headers)
        body = r.json()
        if body["status"] == "failed":
            assert "produces image" in (body["error"] or {}).get("message", "")
            return
        time.sleep(0.05)
    raise AssertionError("Runner never moved job to failed state")
