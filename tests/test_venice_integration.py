"""End-to-end tests for the Venice provider.

Stubs the upstream Venice HTTP surface with respx and verifies:
  * /v1/images/edits forwards the multipart upload to /api/v1/image/edit and
    serves the (binary) edited image back through /v1/files/{id}/content
  * Multiple reference images are rejected (Venice edits are single-image)
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from openai_api_bridge.config import reset_caches_for_tests

UPSTREAM = "https://api.venice.ai"


@pytest.fixture
def client_with_venice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent("""
		[[providers]]
		id = "vn"
		backend = "venice"
		api_token_env = "TEST_VN_TOKEN"
	""")
    )

    monkeypatch.setenv("BRIDGE_API_KEY", "test-bridge-key")
    monkeypatch.setenv("BRIDGE_CONFIG_PATH", str(config))
    monkeypatch.setenv("FILES_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "state.db"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("TEST_VN_TOKEN", "vn-secret")
    reset_caches_for_tests()

    from openai_api_bridge.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    reset_caches_for_tests()


HEADERS = {"Authorization": "Bearer test-bridge-key"}


@respx.mock
def test_images_edits_round_trip(
    client_with_venice: TestClient,
) -> None:
    """Bridge forwards the multipart upload to /api/v1/image/edit, then serves
    the returned binary image back via /v1/files/{id}/content."""
    edited_png = b"\x89PNG\r\n\x1a\nedited-bytes"
    edit_route = respx.post(f"{UPSTREAM}/api/v1/image/edit").mock(
        return_value=httpx.Response(200, content=edited_png, headers={"content-type": "image/png"})
    )

    r = client_with_venice.post(
        "/v1/images/edits",
        headers=HEADERS,
        files={"image": ("input.png", b"fake-input-png", "image/png")},
        data={"model": "vn/firered-image-edit", "prompt": "make her hair blue"},
    )
    assert r.status_code == 200, r.text
    url = r.json()["data"][0]["url"]
    assert "/v1/files/" in url

    # Outgoing multipart carried prompt, model (slug stripped of the vn/ prefix),
    # and the input image.
    assert edit_route.called
    sent = edit_route.calls.last.request
    body_text = sent.content.decode("utf-8", errors="replace")
    assert "make her hair blue" in body_text
    assert "firered-image-edit" in body_text
    assert "fake-input-png" in body_text
    assert 'name="image"' in body_text

    # The stored asset round-trips back to the bytes Venice returned.
    file_resp = client_with_venice.get(url, headers=HEADERS)
    assert file_resp.status_code == 200
    assert file_resp.content == edited_png
    assert file_resp.headers["content-type"].startswith("image/png")


@respx.mock
def test_images_edits_rejects_multiple_references(
    client_with_venice: TestClient,
) -> None:
    """Venice edits are single-image; supplying two surfaces a 400 rather than
    silently dropping one."""
    edit_route = respx.post(f"{UPSTREAM}/api/v1/image/edit").mock(
        return_value=httpx.Response(200, content=b"x", headers={"content-type": "image/png"})
    )

    r = client_with_venice.post(
        "/v1/images/edits",
        headers=HEADERS,
        files=[
            ("image", ("a.png", b"first", "image/png")),
            ("image", ("b.png", b"second", "image/png")),
        ],
        data={"model": "vn/firered-image-edit", "prompt": "combine"},
    )
    assert r.status_code == 400
    body = r.json()["error"]
    assert body["code"] == "invalid_request"
    assert "image" in body.get("param", "")
    # Never reached the upstream.
    assert not edit_route.called


# --- edit-model pairing ----------------------------------------------------
#
# Venice files text-to-image under type=image and image-to-image under
# type=inpaint, naming the latter by suffixing the base id (`gpt-image-2` ->
# `gpt-image-2-edit`). Its edit endpoint only accepts the `-edit` ids, so the
# bridge lists the base and routes edits to the counterpart.


def _stub_venice_catalog(image: list[str], inpaint: list[str]) -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        model_type = request.url.params.get("type")
        ids = image if model_type == "image" else inpaint if model_type == "inpaint" else []
        return httpx.Response(200, json={"data": [{"id": i, "type": model_type} for i in ids]})

    respx.get(f"{UPSTREAM}/api/v1/models").mock(side_effect=responder)


@respx.mock
def test_models_collapse_edit_pairs_and_expose_capabilities(
    client_with_venice: TestClient,
) -> None:
    _stub_venice_catalog(
        image=["gpt-image-2", "flux-2-pro"],
        inpaint=["gpt-image-2-edit", "qwen-edit-uncensored"],
    )
    r = client_with_venice.get("/v1/models", headers=HEADERS)
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()["data"]}

    # Paired: one entry, advertising both halves.
    assert "vn/gpt-image-2" in by_id
    assert "vn/gpt-image-2-edit" not in by_id
    assert by_id["vn/gpt-image-2"]["capabilities"] == ["text-to-image", "image-to-image"]
    # Unpaired generate model stays text-only.
    assert by_id["vn/flux-2-pro"]["capabilities"] == ["text-to-image"]
    # Edit-only model is still listed — it just can't do text-to-image. It was
    # previously absent entirely, since only type=image was fetched.
    assert by_id["vn/qwen-edit-uncensored"]["capabilities"] == ["image-to-image"]


@respx.mock
def test_edit_is_routed_to_the_suffixed_model(client_with_venice: TestClient) -> None:
    """Venice's edit endpoint rejects the base id, so an edit naming
    `gpt-image-2` has to go out as `gpt-image-2-edit`."""
    _stub_venice_catalog(image=["gpt-image-2"], inpaint=["gpt-image-2-edit"])
    edit_route = respx.post(f"{UPSTREAM}/api/v1/image/edit").mock(
        return_value=httpx.Response(
            200, content=b"\x89PNGedited", headers={"content-type": "image/png"}
        )
    )

    r = client_with_venice.post(
        "/v1/images/edits",
        headers=HEADERS,
        files={"image": ("input.png", b"png", "image/png")},
        data={"model": "vn/gpt-image-2", "prompt": "make it blue"},
    )
    assert r.status_code == 200, r.text
    sent = edit_route.calls.last.request.content.decode("utf-8", errors="replace")
    assert "gpt-image-2-edit" in sent


@respx.mock
def test_edit_routing_works_without_a_prior_models_call(
    client_with_venice: TestClient,
) -> None:
    """Routing loads the catalogue itself, so the same request can't behave
    differently depending on unrelated earlier traffic."""
    _stub_venice_catalog(image=["gpt-image-2"], inpaint=["gpt-image-2-edit"])
    edit_route = respx.post(f"{UPSTREAM}/api/v1/image/edit").mock(
        return_value=httpx.Response(
            200, content=b"\x89PNGedited", headers={"content-type": "image/png"}
        )
    )
    r = client_with_venice.post(
        "/v1/images/edits",
        headers=HEADERS,
        files={"image": ("input.png", b"png", "image/png")},
        data={"model": "vn/gpt-image-2", "prompt": "x"},
    )
    assert r.status_code == 200, r.text
    assert "gpt-image-2-edit" in edit_route.calls.last.request.content.decode(
        "utf-8", errors="replace"
    )
