"""End-to-end tests for the fal.ai provider.

Stubs the upstream fal.run HTTP surface with respx and verifies:
  * /v1/models reflects the configured model list (no upstream catalog call)
  * /v1/images/generations injects the loosest per-family moderation knob
    (Seedream -> enable_safety_checker=false, Nano Banana -> safety_tolerance=6)
  * families the bridge doesn't recognise (gpt-image) get nothing injected
  * disable_safety=false and per-model `params` overrides behave correctly
  * size maps to image_size for Seedream but is dropped for Nano Banana
  * /v1/images/edits forwards reference images as image_urls data URIs
  * the generation request carries `Authorization: Key ...`, asset fetch doesn't
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

FAL = "https://fal.run"

SEEDREAM_T2I = "fal-ai/bytedance/seedream/v4/text-to-image"
SEEDREAM_EDIT = "fal-ai/bytedance/seedream/v4/edit"
NANO = "fal-ai/nano-banana-2"
NANO_PRO = "fal-ai/nano-banana-pro"
GPT_IMAGE = "fal-ai/gpt-image-2"
PLAIN = "fal-ai/some/plain-model"


@pytest.fixture
def client_with_fal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(f"""
        [[providers]]
        id = "fal"
        backend = "fal"
        api_token_env = "TEST_FAL_TOKEN"

        [[providers.models]]
        id = "{SEEDREAM_T2I}"
        kind = "image"
        display_name = "Seedream 4"

        [[providers.models]]
        id = "{SEEDREAM_EDIT}"
        kind = "image"

        [[providers.models]]
        id = "{NANO}"
        kind = "image"

        [[providers.models]]
        id = "{GPT_IMAGE}"
        kind = "image"

        # disable_safety off — leave upstream defaults untouched.
        [[providers.models]]
        id = "{PLAIN}"
        kind = "image"
        disable_safety = false

        # Per-model params override the built-in Nano Banana default.
        [[providers.models]]
        id = "{NANO_PRO}"
        kind = "image"
        [providers.models.params]
        safety_tolerance = "3"
    """)
    )

    monkeypatch.setenv("BRIDGE_API_KEY", "test-bridge-key")
    monkeypatch.setenv("BRIDGE_CONFIG_PATH", str(config))
    monkeypatch.setenv("FILES_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "state.db"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("TEST_FAL_TOKEN", "fal-secret")
    reset_caches_for_tests()

    from openai_api_bridge.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    reset_caches_for_tests()


HEADERS = {"Authorization": "Bearer test-bridge-key"}

_PNG = b"\x89PNG\r\n\x1a\nfake-png-bytes"


def _mock_generation(model_id: str, url: str = "https://v3.fal.media/files/out.png") -> respx.Route:
    """Stub a fal image call + its asset URL. Returns the generation route so
    the test can inspect the outgoing body."""
    gen = respx.post(f"{FAL}/{model_id}").mock(
        return_value=httpx.Response(200, json={"images": [{"url": url}], "seed": 42})
    )
    respx.get(url).mock(
        return_value=httpx.Response(200, content=_PNG, headers={"content-type": "image/png"})
    )
    return gen


def _sent_body(route: respx.Route) -> dict:
    return json.loads(route.calls.last.request.read())


def _generate(client: TestClient, model_id: str, **extra: object) -> httpx.Response:
    body: dict[str, object] = {"model": f"fal/{model_id}", "prompt": "a red panda"}
    body.update(extra)
    return client.post("/v1/images/generations", headers=HEADERS, json=body)


# --- catalog ---------------------------------------------------------------


def test_models_reflect_config(client_with_fal: TestClient) -> None:
    r = client_with_fal.get("/v1/models", headers=HEADERS)
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()["data"]}
    assert f"fal/{SEEDREAM_T2I}" in by_id
    assert by_id[f"fal/{SEEDREAM_T2I}"]["kind"] == "image"
    assert by_id[f"fal/{SEEDREAM_T2I}"]["display_name"] == "Seedream 4"
    assert f"fal/{NANO}" in by_id
    assert f"fal/{GPT_IMAGE}" in by_id


# --- per-family moderation injection ---------------------------------------


@respx.mock
def test_seedream_injects_safety_checker_and_size(client_with_fal: TestClient) -> None:
    gen = _mock_generation(SEEDREAM_T2I)
    r = _generate(client_with_fal, SEEDREAM_T2I, size="1024x768")
    assert r.status_code == 200, r.text
    url = r.json()["data"][0]["url"]
    assert "/v1/files/" in url and url.endswith("/content")

    body = _sent_body(gen)
    assert body["prompt"] == "a red panda"
    assert body["num_images"] == 1
    # Seedream's boolean checker, turned off.
    assert body["enable_safety_checker"] is False
    # size maps to an image_size object for this family.
    assert body["image_size"] == {"width": 1024, "height": 768}


@respx.mock
def test_nano_banana_injects_safety_tolerance_and_drops_size(
    client_with_fal: TestClient,
) -> None:
    gen = _mock_generation(NANO)
    r = _generate(client_with_fal, NANO, size="1024x1024")
    assert r.status_code == 200, r.text
    body = _sent_body(gen)
    # Nano Banana's string enum, set to loosest.
    assert body["safety_tolerance"] == "6"
    # image_size is unsupported by this family — must not be forwarded.
    assert "image_size" not in body


@respx.mock
def test_gpt_image_gets_no_safety_field(client_with_fal: TestClient) -> None:
    # fal's gpt-image wrapper exposes no moderation param; the bridge must not
    # guess one (an unknown field would 422 upstream).
    gen = _mock_generation(GPT_IMAGE)
    r = _generate(client_with_fal, GPT_IMAGE)
    assert r.status_code == 200, r.text
    body = _sent_body(gen)
    assert "safety_tolerance" not in body
    assert "enable_safety_checker" not in body


@respx.mock
def test_disable_safety_false_injects_nothing(client_with_fal: TestClient) -> None:
    gen = _mock_generation(PLAIN)
    r = _generate(client_with_fal, PLAIN)
    assert r.status_code == 200, r.text
    body = _sent_body(gen)
    assert "safety_tolerance" not in body
    assert "enable_safety_checker" not in body


@respx.mock
def test_multiple_images_from_single_call(client_with_fal: TestClient) -> None:
    # fal honours num_images server-side: one generation call returns all n
    # image URLs, which the bridge fetches concurrently.
    u1 = "https://v3.fal.media/files/one.png"
    u2 = "https://v3.fal.media/files/two.png"
    gen = respx.post(f"{FAL}/{SEEDREAM_T2I}").mock(
        return_value=httpx.Response(200, json={"images": [{"url": u1}, {"url": u2}], "seed": 7})
    )
    for u in (u1, u2):
        respx.get(u).mock(
            return_value=httpx.Response(200, content=_PNG, headers={"content-type": "image/png"})
        )
    r = _generate(client_with_fal, SEEDREAM_T2I, n=2)
    assert r.status_code == 200, r.text
    assert len(r.json()["data"]) == 2
    # A single upstream generation call, num_images forwarded.
    assert gen.call_count == 1
    assert _sent_body(gen)["num_images"] == 2


@respx.mock
def test_params_override_wins_over_family_default(client_with_fal: TestClient) -> None:
    # NANO_PRO is Nano-Banana family (default safety_tolerance="6") but config
    # pins it to "3" — the explicit params must win.
    gen = _mock_generation(NANO_PRO)
    r = _generate(client_with_fal, NANO_PRO)
    assert r.status_code == 200, r.text
    body = _sent_body(gen)
    assert body["safety_tolerance"] == "3"


# --- auth / asset fetch ----------------------------------------------------


@respx.mock
def test_generation_uses_key_auth_and_asset_fetch_is_unauthenticated(
    client_with_fal: TestClient,
) -> None:
    gen = _mock_generation(SEEDREAM_T2I)
    asset = respx.routes[1] if len(respx.routes) > 1 else None
    r = _generate(client_with_fal, SEEDREAM_T2I)
    assert r.status_code == 200, r.text
    # fal REST auth is "Key <token>", not "Bearer".
    assert gen.calls.last.request.headers.get("authorization") == "Key fal-secret"
    # Asset fetch must not carry our fal key.
    assert asset is not None and asset.called
    assert asset.calls.last.request.headers.get("authorization") is None


# --- edits -----------------------------------------------------------------


@respx.mock
def test_edit_forwards_reference_images_as_data_uris(client_with_fal: TestClient) -> None:
    gen = _mock_generation(SEEDREAM_EDIT)
    r = client_with_fal.post(
        "/v1/images/edits",
        headers=HEADERS,
        files=[
            ("image", ("a.png", b"FIRST-REF", "image/png")),
            ("image", ("b.png", b"SECOND-REF", "image/png")),
        ],
        data={"model": f"fal/{SEEDREAM_EDIT}", "prompt": "combine these"},
    )
    assert r.status_code == 200, r.text
    body = _sent_body(gen)
    assert body["prompt"] == "combine these"
    urls = body["image_urls"]
    assert len(urls) == 2
    assert all(u.startswith("data:image/png;base64,") for u in urls)
    assert body["enable_safety_checker"] is False


# --- error surfacing -------------------------------------------------------


@respx.mock
def test_upstream_error_surfaces(client_with_fal: TestClient) -> None:
    respx.post(f"{FAL}/{SEEDREAM_T2I}").mock(
        return_value=httpx.Response(422, json={"detail": "safety_tolerance is invalid"})
    )
    r = _generate(client_with_fal, SEEDREAM_T2I)
    assert r.status_code >= 400
    assert "500" not in r.text or "safety_tolerance" in r.text.lower()


def test_unconfigured_model_is_model_not_found(client_with_fal: TestClient) -> None:
    r = _generate(client_with_fal, "fal-ai/not-configured")
    assert r.status_code == 404
