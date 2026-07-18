"""End-to-end tests for the fal.ai provider.

Stubs the upstream fal.run HTTP surface with respx and verifies:
  * /v1/models reflects the configured model list (no upstream catalog call)
  * /v1/images/generations injects the loosest moderation knob
  * models exposing no knob (gpt-image) get nothing injected
  * disable_safety=false and per-model `params` overrides behave correctly
  * size maps to image_size for Seedream but is dropped for Nano Banana
  * /v1/images/edits forwards reference images as image_urls data URIs
  * the generation request carries `Authorization: Key ...`, asset fetch doesn't

Moderation settings come from schema introspection against fal's model API.
Tests that leave that API unstubbed exercise the *fallback* static map; the
"schema introspection" section below stubs it to exercise the derived path.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from openai_api_bridge.backends.fal.adapter import SUPPORTED_CATEGORIES, FalBackend
from openai_api_bridge.config import FalModelConfig, FalProviderConfig, reset_caches_for_tests
from openai_api_bridge.errors import BridgeError

FAL = "https://fal.run"

SEEDREAM_T2I = "fal-ai/bytedance/seedream/v4/text-to-image"
SEEDREAM_EDIT = "fal-ai/bytedance/seedream/v4/edit"
NANO = "fal-ai/nano-banana-2"
NANO_PRO = "fal-ai/nano-banana-pro"
GPT_IMAGE = "openai/gpt-image-2"
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
        # These tests cover moderation and request shaping, not discovery.
        discover_models = false

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


# --- schema introspection --------------------------------------------------
#
# The tests above exercise the *fallback* path: respx leaves fal's model API
# unstubbed, so introspection fails and the static map applies. These stub the
# model API so the schema-derived path runs.

MODELS_API = "https://api.fal.ai/v1/models"


def _openapi(model_title: str, props: dict) -> dict:
    """Minimal fal-shaped OpenAPI doc with the given input properties."""
    return {
        "openapi": "3.0.4",
        "paths": {
            f"/{model_title}": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": f"#/components/schemas/{model_title}Input"}
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                f"{model_title}Input": {"properties": {"prompt": {"type": "string"}, **props}},
                # Output side carries has_nsfw_concepts — must never be picked up.
                f"{model_title}Output": {"properties": {"has_nsfw_concepts": {"type": "array"}}},
            }
        },
    }


def _stub_models_api(schemas: dict[str, dict]) -> None:
    respx.get(MODELS_API).mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [{"endpoint_id": mid, "openapi": spec} for mid, spec in schemas.items()],
                "has_more": False,
            },
        )
    )


@respx.mock
def test_schema_derived_enum_uses_model_specific_ceiling(client_with_fal: TestClient) -> None:
    """The whole point of introspection: most models cap safety_tolerance at
    "6", but some (flux-2-flex) cap at "5" — a hardcoded "6" would be rejected.
    The value must come from the model's own enum."""
    _stub_models_api(
        {
            NANO: _openapi(
                "Nano",
                {"safety_tolerance": {"enum": ["1", "2", "3", "4", "5", "6"], "default": "4"}},
            )
        }
    )
    gen = _mock_generation(NANO)
    assert _generate(client_with_fal, NANO).status_code == 200
    assert _sent_body(gen)["safety_tolerance"] == "6"


@respx.mock
def test_schema_derived_respects_narrower_enum(client_with_fal: TestClient) -> None:
    """A model whose enum tops out at "5" (as fal-ai/flux-2-flex really does)
    must get "5" — the hardcoded "6" this replaced would be rejected."""
    _stub_models_api(
        {
            SEEDREAM_T2I: _openapi(
                "Narrow",
                {"safety_tolerance": {"enum": ["1", "2", "3", "4", "5"], "default": "2"}},
            )
        }
    )
    gen = _mock_generation(SEEDREAM_T2I)
    assert _generate(client_with_fal, SEEDREAM_T2I).status_code == 200
    body = _sent_body(gen)
    # Derived from the schema (5), not the hardcoded family default (6), and
    # not the seedream fallback (enable_safety_checker).
    assert body["safety_tolerance"] == "5"
    assert "enable_safety_checker" not in body


@respx.mock
def test_schema_derived_boolean_checker(client_with_fal: TestClient) -> None:
    _stub_models_api(
        {
            SEEDREAM_T2I: _openapi(
                "Seedream", {"enable_safety_checker": {"type": "boolean", "default": True}}
            )
        }
    )
    gen = _mock_generation(SEEDREAM_T2I)
    assert _generate(client_with_fal, SEEDREAM_T2I).status_code == 200
    assert _sent_body(gen)["enable_safety_checker"] is False


@respx.mock
def test_output_only_and_decoy_fields_are_ignored(client_with_fal: TestClient) -> None:
    """has_nsfw_concepts is an output field; safety_checker_version selects
    WHICH checker runs, not how strict — neither may be injected."""
    _stub_models_api(
        {
            SEEDREAM_T2I: _openapi(
                "Decoy",
                {
                    "safety_checker_version": {"enum": ["v1", "v2"], "default": "v1"},
                    "enable_safety_checker": {"type": "boolean", "default": True},
                },
            )
        }
    )
    gen = _mock_generation(SEEDREAM_T2I)
    assert _generate(client_with_fal, SEEDREAM_T2I).status_code == 200
    body = _sent_body(gen)
    assert body["enable_safety_checker"] is False
    assert "safety_checker_version" not in body
    assert "has_nsfw_concepts" not in body


@respx.mock
def test_model_with_no_knob_gets_nothing_injected(client_with_fal: TestClient) -> None:
    # fal's gpt-image wrapper exposes no moderation field at all.
    _stub_models_api({GPT_IMAGE: _openapi("GptImage", {"quality": {"type": "string"}})})
    gen = _mock_generation(GPT_IMAGE)
    assert _generate(client_with_fal, GPT_IMAGE).status_code == 200
    body = _sent_body(gen)
    assert "safety_tolerance" not in body
    assert "enable_safety_checker" not in body


@respx.mock
def test_schemas_fetched_once_and_cached(client_with_fal: TestClient) -> None:
    # Every configured model answered in the one batch, so no per-model retry
    # pass is triggered — this isolates the caching behaviour.
    spec = _openapi("S", {"enable_safety_checker": {"type": "boolean", "default": True}})
    route = respx.get(MODELS_API).mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [
                    {"endpoint_id": mid, "openapi": spec}
                    for mid in (
                        SEEDREAM_T2I,
                        SEEDREAM_EDIT,
                        NANO,
                        NANO_PRO,
                        GPT_IMAGE,
                        PLAIN,
                    )
                ],
                "has_more": False,
            },
        )
    )
    _mock_generation(SEEDREAM_T2I)
    for _ in range(3):
        assert _generate(client_with_fal, SEEDREAM_T2I).status_code == 200
    # One batched lookup covers every configured model, for the process.
    assert route.call_count == 1


@respx.mock
def test_truncated_batch_is_retried_per_model(client_with_fal: TestClient) -> None:
    """fal silently truncates expanded-schema responses, so a model missing
    from a batch must be re-fetched individually rather than silently losing
    its moderation settings."""
    seedream_spec = _openapi("S", {"enable_safety_checker": {"type": "boolean", "default": True}})

    def responder(request: httpx.Request) -> httpx.Response:
        asked = request.url.params.get_list("endpoint_id")
        # Batch call: pretend the response got truncated to a single unrelated
        # model, omitting the one we care about. Single-id call: answer fully.
        if len(asked) > 1:
            return httpx.Response(200, json={"models": [{"endpoint_id": NANO, "openapi": {}}]})
        if asked and asked[0] == SEEDREAM_T2I:
            return httpx.Response(
                200, json={"models": [{"endpoint_id": SEEDREAM_T2I, "openapi": seedream_spec}]}
            )
        return httpx.Response(200, json={"models": []})

    respx.get(MODELS_API).mock(side_effect=responder)
    gen = _mock_generation(SEEDREAM_T2I)
    assert _generate(client_with_fal, SEEDREAM_T2I).status_code == 200
    # Recovered via the per-model retry, not the static fallback.
    assert _sent_body(gen)["enable_safety_checker"] is False


@respx.mock
async def test_concurrent_first_requests_share_one_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A burst of simultaneous generations (e.g. a multi-model fan-out from the
    UI) must collapse into a single schema lookup, with every request waiting
    for it — not one racing ahead un-loosened while N fetches fire in parallel.
    """
    monkeypatch.setenv("TEST_FAL_TOKEN", "fal-secret")
    ids = [SEEDREAM_T2I, NANO, GPT_IMAGE]
    spec = _openapi("S", {"enable_safety_checker": {"type": "boolean", "default": True}})

    async def slow_responder(request: httpx.Request) -> httpx.Response:
        # Latency widens the window in which a broken lock would let other
        # requests start their own fetch.
        await asyncio.sleep(0.05)
        return httpx.Response(
            200,
            json={"models": [{"endpoint_id": m, "openapi": spec} for m in ids]},
        )

    route = respx.get(MODELS_API).mock(side_effect=slow_responder)

    cfg = FalProviderConfig(
        backend="fal",
        id="fal",
        api_token_env="TEST_FAL_TOKEN",
        models=[FalModelConfig(id=m) for m in ids],
    )
    backend = FalBackend(cfg)
    try:
        results = await asyncio.gather(
            *(_safety(backend, cfg.models[i % len(ids)]) for i in range(8))
        )
    finally:
        await backend.aclose()

    # 8 concurrent callers over 3 distinct models -> one lookup per model, not
    # one per caller (without the per-model lock this would be 8).
    assert route.call_count == len(ids)
    # And none of them slipped through before the cache was populated.
    assert all(r == {"enable_safety_checker": False} for r in results)


async def _safety(backend: FalBackend, mcfg: FalModelConfig) -> dict:
    """Moderation params for a model whose endpoint is its own id (i.e. not a
    routed edit, which resolves against the sibling's schema instead)."""
    return await backend._safety_params(mcfg, mcfg.id)


def _direct_backend(retry_seconds: float, model_ids: list[str]) -> FalBackend:
    """A FalBackend built straight from config, for tests that need to poke
    introspection behaviour without going through the HTTP layer."""
    cfg = FalProviderConfig(
        backend="fal",
        id="fal",
        api_token_env="TEST_FAL_TOKEN",
        models=[FalModelConfig(id=m) for m in model_ids],
        introspect_retry_seconds=retry_seconds,
    )
    return FalBackend(cfg)


# A model id the static fallback map does NOT match, so schema-derived params
# ({"enable_safety_checker": False}) are distinguishable from fallback ({}).
UNMAPPED = "fal-ai/some-unmapped-model"


@respx.mock
async def test_failed_introspection_is_not_retried_during_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_FAL_TOKEN", "fal-secret")
    route = respx.get(MODELS_API).mock(return_value=httpx.Response(503, json={"error": "down"}))

    backend = _direct_backend(retry_seconds=300.0, model_ids=[UNMAPPED])
    try:
        for _ in range(3):
            assert await _safety(backend, backend.cfg.models[0]) == {}
    finally:
        await backend.aclose()
    # One attempt, not one per request — the cooldown suppresses the rest.
    assert route.call_count == 1


@respx.mock
async def test_introspection_retries_after_cooldown_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the cooldown elapses the lookup is retried, and a now-healthy fal
    API upgrades the process from fallback to schema-derived settings without a
    restart."""
    monkeypatch.setenv("TEST_FAL_TOKEN", "fal-secret")
    spec = _openapi("S", {"enable_safety_checker": {"type": "boolean", "default": True}})
    responses = [
        httpx.Response(503, json={"error": "down"}),
        httpx.Response(200, json={"models": [{"endpoint_id": UNMAPPED, "openapi": spec}]}),
    ]
    route = respx.get(MODELS_API).mock(side_effect=responses)

    # retry_seconds=0 -> the cooldown has always elapsed, so the next request retries.
    backend = _direct_backend(retry_seconds=0.0, model_ids=[UNMAPPED])
    try:
        model = backend.cfg.models[0]
        # First attempt fails -> fallback (no static rule for this id).
        assert await _safety(backend, model) == {}
        # Second attempt retries and succeeds -> schema-derived.
        assert await _safety(backend, model) == {"enable_safety_checker": False}
        # Now cached: no further lookups.
        assert await _safety(backend, model) == {"enable_safety_checker": False}
    finally:
        await backend.aclose()
    assert route.call_count == 2


@respx.mock
async def test_empty_but_successful_response_arms_the_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 carrying no schemas must not latch an empty cache. Without arming
    the cooldown the first call would poison the cache and no later call would
    ever retry — the exact degraded-until-restart mode the cooldown prevents,
    reached without an exception."""
    monkeypatch.setenv("TEST_FAL_TOKEN", "fal-secret")
    spec = _openapi("S", {"enable_safety_checker": {"type": "boolean", "default": True}})
    calls: list[list[str]] = []

    def responder(request: httpx.Request) -> httpx.Response:
        asked = request.url.params.get_list("endpoint_id")
        calls.append(asked)
        # Schemas resolve one model at a time, so a round is a single call.
        if len(calls) <= 1:
            return httpx.Response(200, json={"models": []})
        return httpx.Response(200, json={"models": [{"endpoint_id": UNMAPPED, "openapi": spec}]})

    respx.get(MODELS_API).mock(side_effect=responder)

    backend = _direct_backend(retry_seconds=0.0, model_ids=[UNMAPPED])
    try:
        model = backend.cfg.models[0]
        assert await _safety(backend, model) == {}
        # Retried rather than serving a latched empty cache forever.
        assert await _safety(backend, model) == {"enable_safety_checker": False}
    finally:
        await backend.aclose()


@respx.mock
async def test_models_resolve_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    """Schemas are resolved per model: one model failing doesn't hold back
    another, and a model that already resolved isn't re-fetched when its
    neighbour retries."""
    monkeypatch.setenv("TEST_FAL_TOKEN", "fal-secret")
    spec = _openapi("S", {"enable_safety_checker": {"type": "boolean", "default": True}})
    other = "fal-ai/another-unmapped-model"
    asked: list[str] = []

    def responder(request: httpx.Request) -> httpx.Response:
        ids = request.url.params.get_list("endpoint_id")
        asked.extend(ids)
        # `other` stays unresolved on its first attempt, resolves on the retry.
        served = [m for m in ids if m == UNMAPPED or asked.count(m) > 1]
        return httpx.Response(
            200, json={"models": [{"endpoint_id": m, "openapi": spec} for m in served]}
        )

    respx.get(MODELS_API).mock(side_effect=responder)

    backend = _direct_backend(retry_seconds=0.0, model_ids=[UNMAPPED, other])
    try:
        resolved_model, retried_model = backend.cfg.models
        assert await _safety(backend, resolved_model) == {"enable_safety_checker": False}
        # Unresolved on the first attempt -> fallback for this request.
        assert await _safety(backend, retried_model) == {}
        # Retried independently, and succeeds.
        assert await _safety(backend, retried_model) == {"enable_safety_checker": False}
        # The one that already resolved is served from cache throughout.
        assert await _safety(backend, resolved_model) == {"enable_safety_checker": False}
    finally:
        await backend.aclose()

    assert asked.count(UNMAPPED) == 1, "a cached model must not be re-fetched"
    assert asked.count(other) == 2, "the failing model is retried on its own"


@respx.mock
async def test_persistently_missing_model_is_throttled_by_the_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model the catalog never returns (a typo'd id, say) must not cost a
    lookup on every single request — the cooldown throttles the retry."""
    monkeypatch.setenv("TEST_FAL_TOKEN", "fal-secret")
    spec = _openapi("S", {"enable_safety_checker": {"type": "boolean", "default": True}})
    other = "fal-ai/never-returned"

    def responder(request: httpx.Request) -> httpx.Response:
        asked = request.url.params.get_list("endpoint_id")
        served = [m for m in asked if m == UNMAPPED]
        return httpx.Response(
            200, json={"models": [{"endpoint_id": m, "openapi": spec} for m in served]}
        )

    route = respx.get(MODELS_API).mock(side_effect=responder)

    backend = _direct_backend(retry_seconds=300.0, model_ids=[UNMAPPED, other])
    try:
        missing_model = backend.cfg.models[1]
        for _ in range(3):
            assert await _safety(backend, missing_model) == {}
    finally:
        await backend.aclose()

    # One attempt only, rather than a fresh lookup for each of the three
    # requests — the cooldown throttles it.
    assert route.call_count == 1


@respx.mock
def test_introspection_failure_falls_back_to_static_map(client_with_fal: TestClient) -> None:
    respx.get(MODELS_API).mock(return_value=httpx.Response(503, json={"error": "down"}))
    gen = _mock_generation(SEEDREAM_T2I)
    assert _generate(client_with_fal, SEEDREAM_T2I).status_code == 200
    # Static fallback still loosens the seedream family.
    assert _sent_body(gen)["enable_safety_checker"] is False


# --- model discovery -------------------------------------------------------


@pytest.fixture
def discovering_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """A fal provider using catalogue discovery (the default), with one
    [[providers.models]] entry present purely as an override."""
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(f"""
        [[providers]]
        id = "fal"
        backend = "fal"
        api_token_env = "TEST_FAL_TOKEN"
        video_poll_interval_seconds = 0.01

        [[providers.models]]
        id = "{NANO}"
        display_name = "Nano (renamed locally)"
        prompt_style = "natural-language"
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


def _catalog_page(models: list[tuple[str, str]], has_more: bool = False) -> dict:
    return {
        "models": [
            {"endpoint_id": mid, "metadata": {"display_name": name, "category": "text-to-image"}}
            for mid, name in models
        ],
        "next_cursor": "next" if has_more else None,
        "has_more": has_more,
    }


def _stub_catalog(pages_by_category: dict[str, list[dict]]) -> respx.Route:
    """Serve catalogue pages per category, honouring the cursor for pagination."""
    state: dict[str, int] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        category = request.url.params.get("category")
        # `expand` marks a schema lookup, not a listing — leave those alone.
        if request.url.params.get("expand"):
            return httpx.Response(200, json={"models": []})
        pages = pages_by_category.get(category or "", [])
        idx = state.get(category or "", 0)
        state[category or ""] = idx + 1
        if idx >= len(pages):
            return httpx.Response(200, json={"models": [], "has_more": False})
        return httpx.Response(200, json=pages[idx])

    return respx.get(MODELS_API).mock(side_effect=responder)


@respx.mock
def test_discovery_lists_catalog_filtered_to_supported_categories(
    discovering_client: TestClient,
) -> None:
    _stub_catalog(
        {
            "text-to-image": [_catalog_page([(NANO, "Nano Banana 2"), (GPT_IMAGE, "GPT Image 2")])],
            "image-to-image": [_catalog_page([(SEEDREAM_EDIT, "Seedream 4 Edit")])],
        }
    )
    r = discovering_client.get("/v1/models", headers=HEADERS)
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()["data"]}
    # Everything the catalogue returned for the supported categories is served,
    # not just what's in config.
    assert f"fal/{GPT_IMAGE}" in by_id
    assert f"fal/{SEEDREAM_EDIT}" in by_id
    assert by_id[f"fal/{GPT_IMAGE}"]["display_name"] == "GPT Image 2"
    assert by_id[f"fal/{GPT_IMAGE}"]["kind"] == "image"
    # A [[providers.models]] entry enriches its match rather than restricting.
    assert by_id[f"fal/{NANO}"]["display_name"] == "Nano (renamed locally)"
    assert by_id[f"fal/{NANO}"]["prompt_style"] == "natural-language"


@respx.mock
def test_discovery_only_requests_supported_categories(
    discovering_client: TestClient,
) -> None:
    route = _stub_catalog({"text-to-image": [_catalog_page([(NANO, "Nano")])]})
    assert discovering_client.get("/v1/models", headers=HEADERS).status_code == 200
    asked = {c.request.url.params.get("category") for c in route.calls}
    assert asked == {"text-to-image", "image-to-image", "text-to-video", "image-to-video"}
    # Audio/3d have no code path in this backend, so they're never advertised.
    assert "text-to-audio" not in asked
    assert "image-to-3d" not in asked
    # Deprecated models are excluded from the listing.
    assert all(c.request.url.params.get("status") == "active" for c in route.calls)


@respx.mock
def test_discovery_paginates(discovering_client: TestClient) -> None:
    _stub_catalog(
        {
            "text-to-image": [
                _catalog_page([(NANO, "Nano")], has_more=True),
                _catalog_page([(GPT_IMAGE, "GPT Image 2")]),
            ]
        }
    )
    r = discovering_client.get("/v1/models", headers=HEADERS)
    by_id = {m["id"] for m in r.json()["data"]}
    assert f"fal/{NANO}" in by_id and f"fal/{GPT_IMAGE}" in by_id


@respx.mock
def test_discovered_model_generates_without_being_configured(
    discovering_client: TestClient,
) -> None:
    """A model that exists only in the catalogue is usable — no 404 — and still
    gets its moderation knob derived from its schema."""
    spec = _openapi("S", {"enable_safety_checker": {"type": "boolean", "default": True}})

    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("expand"):
            return httpx.Response(
                200, json={"models": [{"endpoint_id": SEEDREAM_T2I, "openapi": spec}]}
            )
        return httpx.Response(200, json={"models": [], "has_more": False})

    respx.get(MODELS_API).mock(side_effect=responder)
    gen = _mock_generation(SEEDREAM_T2I)

    r = _generate(discovering_client, SEEDREAM_T2I)
    assert r.status_code == 200, r.text
    assert _sent_body(gen)["enable_safety_checker"] is False


@respx.mock
def test_catalog_failure_falls_back_to_configured_models(
    discovering_client: TestClient,
) -> None:
    respx.get(MODELS_API).mock(return_value=httpx.Response(503, json={"error": "down"}))
    r = discovering_client.get("/v1/models", headers=HEADERS)
    assert r.status_code == 200
    by_id = {m["id"] for m in r.json()["data"]}
    # Degrades to the explicitly configured entry rather than serving nothing.
    assert by_id == {f"fal/{NANO}"}


@respx.mock
def test_catalog_is_cached(discovering_client: TestClient) -> None:
    route = _stub_catalog({"text-to-image": [_catalog_page([(NANO, "Nano")])]})
    for _ in range(3):
        assert discovering_client.get("/v1/models", headers=HEADERS).status_code == 200
    # One pass over the categories, not one per /v1/models request.
    assert route.call_count == len(SUPPORTED_CATEGORIES)


# --- rejected credentials --------------------------------------------------
#
# A missing key already fails at startup (api_token_env is required and
# resolve_api_token raises during dispatcher construction). A *wrong* key can't
# be caught that way without putting a network call in the lifespan, so it's
# handled at use: reported once at ERROR and never retried, since the token is
# read from the environment at startup and can't heal at runtime.


@respx.mock
def test_rejected_key_is_not_retried_on_the_catalog(
    discovering_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    route = respx.get(MODELS_API).mock(
        return_value=httpx.Response(
            401, json={"error": {"type": "authorization_error", "message": "Invalid API key"}}
        )
    )
    with caplog.at_level("ERROR"):
        for _ in range(3):
            r = discovering_client.get("/v1/models", headers=HEADERS)
            assert r.status_code == 200
            # Degrades to the configured entry rather than 500-ing the listing.
            assert {m["id"] for m in r.json()["data"]} == {f"fal/{NANO}"}

    # One attempt only — one call per category, since they're walked
    # concurrently — rather than a fresh attempt per request. A bad key cannot
    # start working, so there is no cooldown retry.
    assert route.call_count == len(SUPPORTED_CATEGORIES)
    # And it says so once, actionably — naming the env var to fix.
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1
    assert "TEST_FAL_TOKEN" in errors[0].getMessage()


@respx.mock
async def test_rejected_key_stops_schema_lookups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_FAL_TOKEN", "fal-secret")
    route = respx.get(MODELS_API).mock(return_value=httpx.Response(401, json={"error": "nope"}))

    # retry_seconds=0 would retry every call if this were treated as transient.
    backend = _direct_backend(retry_seconds=0.0, model_ids=[UNMAPPED])
    try:
        for _ in range(3):
            assert await _safety(backend, backend.cfg.models[0]) == {}
    finally:
        await backend.aclose()
    assert route.call_count == 1


@respx.mock
async def test_transient_failure_still_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard: only 401/403 are permanent — a 503 must still retry."""
    monkeypatch.setenv("TEST_FAL_TOKEN", "fal-secret")
    route = respx.get(MODELS_API).mock(return_value=httpx.Response(503, json={"error": "down"}))

    backend = _direct_backend(retry_seconds=0.0, model_ids=[UNMAPPED])
    try:
        for _ in range(3):
            assert await _safety(backend, backend.cfg.models[0]) == {}
    finally:
        await backend.aclose()
    # Each call re-attempts (batch + the client's own straggler retry per round).
    assert route.call_count > 1


@respx.mock
def test_configured_model_absent_from_catalog_is_still_listed(
    discovering_client: TestClient,
) -> None:
    """A [[providers.models]] entry the catalogue doesn't return must survive
    in /v1/models. fal's ids don't always match what operators configured, and
    generation still works for it — so dropping it would leave the listing and
    generation surfaces disagreeing."""
    # NANO is configured but deliberately absent from the stubbed catalogue.
    _stub_catalog({"text-to-image": [_catalog_page([(GPT_IMAGE, "GPT Image 2")])]})
    r = discovering_client.get("/v1/models", headers=HEADERS)
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()["data"]}
    assert f"fal/{GPT_IMAGE}" in by_id
    assert f"fal/{NANO}" in by_id, "configured model vanished from the listing"
    assert by_id[f"fal/{NANO}"]["display_name"] == "Nano (renamed locally)"


@respx.mock
async def test_unknown_slugs_do_not_grow_state_without_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With discovery on, any slug is accepted, so per-model bookkeeping is
    keyed by unvalidated client input. It must stay bounded."""
    from openai_api_bridge.backends.fal.adapter import _MAX_TRACKED_MODELS

    monkeypatch.setenv("TEST_FAL_TOKEN", "fal-secret")
    route = respx.get(MODELS_API).mock(
        return_value=httpx.Response(200, json={"models": [], "has_more": False})
    )

    backend = _direct_backend(retry_seconds=300.0, model_ids=[])
    try:
        for i in range(_MAX_TRACKED_MODELS + 50):
            await _safety(backend, FalModelConfig(id=f"fal-ai/bogus-{i}"))
        assert len(backend._schema_locks) <= _MAX_TRACKED_MODELS
        assert len(backend._introspect_failed_at) <= _MAX_TRACKED_MODELS
    finally:
        await backend.aclose()

    # One upstream call per distinct slug — the redundant single-id straggler
    # re-ask (byte-identical to the call that just failed) is gone.
    assert route.call_count == _MAX_TRACKED_MODELS + 50


@respx.mock
def test_non_image_output_is_a_terminal_client_error(
    discovering_client: TestClient,
) -> None:
    """Pointing the image endpoint at a video model gives a 4xx, not a
    retryable 5xx: fal really runs the job and returns a video envelope."""
    respx.get(MODELS_API).mock(return_value=httpx.Response(200, json={"models": []}))
    respx.post(f"{FAL}/fal-ai/some-video-model").mock(
        return_value=httpx.Response(200, json={"video": {"url": "https://v3.fal.media/v.mp4"}})
    )
    r = _generate(discovering_client, "fal-ai/some-video-model")
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "unsupported_operation"


@respx.mock
def test_rejected_key_on_generation_is_reported(
    discovering_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The 'reported once at ERROR' promise must hold even when nothing touches
    fal before the generation call (introspection disabled, or a model with
    disable_safety = false)."""
    respx.get(MODELS_API).mock(return_value=httpx.Response(200, json={"models": []}))
    respx.post(f"{FAL}/{SEEDREAM_T2I}").mock(
        return_value=httpx.Response(401, json={"error": "Invalid API key"})
    )
    with caplog.at_level("ERROR"):
        r = _generate(discovering_client, SEEDREAM_T2I)
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "upstream_auth_error"
    errors = [rec for rec in caplog.records if rec.levelname == "ERROR"]
    assert errors and "TEST_FAL_TOKEN" in errors[0].getMessage()


@respx.mock
async def test_cancelled_catalog_fetch_does_not_arm_the_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client disconnecting mid-fetch (uvicorn cancels the request task) must
    not be mistaken for a fal outage: arming the cooldown there would degrade
    /v1/models for every other caller for the whole retry window. It must also
    leave no partial state and no held lock."""
    monkeypatch.setenv("TEST_FAL_TOKEN", "fal-secret")

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return httpx.Response(
            200,
            json={"models": [{"endpoint_id": NANO, "metadata": {"display_name": "N"}}]},
        )

    respx.get(MODELS_API).mock(side_effect=slow)
    backend = _direct_backend(retry_seconds=300.0, model_ids=[])
    try:
        task = asyncio.create_task(backend.list_models())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert backend._catalog_failed_at is None, "cancellation must not arm the cooldown"
        assert backend._catalog_cache is None, "no partial catalogue may be cached"
        assert not backend._catalog_lock.locked(), "the lock must be released"

        # And a subsequent request still works — not stuck behind a cooldown.
        assert len(await backend.list_models()) == 1
    finally:
        await backend.aclose()


# --- video -----------------------------------------------------------------

QUEUE = "https://queue.fal.run"
VIDEO_MODEL = "fal-ai/veo3"


def _stub_video_job(
    model_id: str = VIDEO_MODEL,
    *,
    statuses: list[str] | None = None,
    result: dict | None = None,
) -> respx.Route:
    """Stub fal's queue lifecycle: submit -> status -> result -> asset."""
    req = "req-123"
    submit = respx.post(f"{QUEUE}/{model_id}").mock(
        return_value=httpx.Response(
            200,
            json={
                "request_id": req,
                "status_url": f"{QUEUE}/{model_id}/requests/{req}/status",
                "response_url": f"{QUEUE}/{model_id}/requests/{req}",
            },
        )
    )
    pending = list(statuses or ["COMPLETED"])
    respx.get(f"{QUEUE}/{model_id}/requests/{req}/status").mock(
        side_effect=lambda request: httpx.Response(
            200, json={"status": pending.pop(0) if len(pending) > 1 else pending[0]}
        )
    )
    respx.get(f"{QUEUE}/{model_id}/requests/{req}").mock(
        return_value=httpx.Response(
            200, json=result or {"video": {"url": "https://v3.fal.media/out.mp4"}}
        )
    )
    respx.get("https://v3.fal.media/out.mp4").mock(
        return_value=httpx.Response(
            200,
            content=b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64,
            headers={"content-type": "video/mp4"},
        )
    )
    return submit


def _await_video(client: TestClient, job_id: str) -> dict:
    deadline = time.time() + 5.0
    state: dict = {}
    while time.time() < deadline:
        r = client.get(f"/v1/videos/{job_id}", headers=HEADERS)
        assert r.status_code == 200
        state = r.json()
        if state["status"] == "completed":
            return state
        if state["status"] == "failed":
            pytest.fail(f"video job failed: {state.get('error', {}).get('message')}")
        time.sleep(0.05)
    pytest.fail(f"video job never completed (last status {state.get('status')})")


@respx.mock
def test_video_round_trip_through_the_queue(discovering_client: TestClient) -> None:
    """POST /v1/videos -> fal queue submit -> poll -> result -> /content."""
    respx.get(MODELS_API).mock(return_value=httpx.Response(200, json={"models": []}))
    submit = _stub_video_job(statuses=["IN_QUEUE", "IN_PROGRESS", "COMPLETED"])

    r = discovering_client.post(
        "/v1/videos",
        headers=HEADERS,
        data={"model": f"fal/{VIDEO_MODEL}", "prompt": "a soaring eagle"},
    )
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["status"] in ("queued", "in_progress")

    _await_video(discovering_client, job["id"])
    content = discovering_client.get(f"/v1/videos/{job['id']}/content", headers=HEADERS)
    assert content.status_code == 200
    assert content.headers["content-type"] == "video/mp4"
    # Submitted to the queue host, not the synchronous endpoint.
    assert submit.called
    assert json.loads(submit.calls.last.request.read())["prompt"] == "a soaring eagle"


@respx.mock
def test_video_duration_uses_the_models_own_spelling(discovering_client: TestClient) -> None:
    """`seconds` maps onto whatever the model's duration enum accepts: veo3
    wants "4s"/"6s"/"8s", so 5s becomes "6s" — a bare "5" would 422."""
    spec = _openapi("V", {"duration": {"enum": ["4s", "6s", "8s"], "default": "8s"}})

    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("expand"):
            return httpx.Response(
                200, json={"models": [{"endpoint_id": VIDEO_MODEL, "openapi": spec}]}
            )
        return httpx.Response(200, json={"models": [], "has_more": False})

    respx.get(MODELS_API).mock(side_effect=responder)
    submit = _stub_video_job()

    r = discovering_client.post(
        "/v1/videos",
        headers=HEADERS,
        data={"model": f"fal/{VIDEO_MODEL}", "prompt": "x", "seconds": "5"},
    )
    assert r.status_code == 200, r.text
    _await_video(discovering_client, r.json()["id"])
    assert json.loads(submit.calls.last.request.read())["duration"] == "6s"


@respx.mock
def test_video_model_without_duration_gets_none_injected(
    discovering_client: TestClient,
) -> None:
    """`wan` counts frames and has no duration field — sending one would 422."""
    spec = _openapi("V", {"num_frames": {"type": "integer"}})

    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("expand"):
            return httpx.Response(
                200, json={"models": [{"endpoint_id": VIDEO_MODEL, "openapi": spec}]}
            )
        return httpx.Response(200, json={"models": [], "has_more": False})

    respx.get(MODELS_API).mock(side_effect=responder)
    submit = _stub_video_job()

    r = discovering_client.post(
        "/v1/videos",
        headers=HEADERS,
        data={"model": f"fal/{VIDEO_MODEL}", "prompt": "x", "seconds": "5"},
    )
    assert r.status_code == 200, r.text
    _await_video(discovering_client, r.json()["id"])
    assert "duration" not in json.loads(submit.calls.last.request.read())


@respx.mock
def test_image_to_video_forwards_the_still_as_image_url(
    discovering_client: TestClient,
) -> None:
    respx.get(MODELS_API).mock(return_value=httpx.Response(200, json={"models": []}))
    submit = _stub_video_job()

    r = discovering_client.post(
        "/v1/videos",
        headers=HEADERS,
        files={"input_reference": ("still.png", b"PNG-BYTES", "image/png")},
        data={"model": f"fal/{VIDEO_MODEL}", "prompt": "animate this"},
    )
    assert r.status_code == 200, r.text
    _await_video(discovering_client, r.json()["id"])
    body = json.loads(submit.calls.last.request.read())
    assert body["image_url"].startswith("data:image/png;base64,")


@respx.mock
def test_video_models_are_listed_with_video_kind(discovering_client: TestClient) -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("expand"):
            return httpx.Response(200, json={"models": []})
        category = request.url.params.get("category")
        if category == "text-to-video":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "endpoint_id": VIDEO_MODEL,
                            "metadata": {"display_name": "Veo 3", "category": "text-to-video"},
                        }
                    ],
                    "has_more": False,
                },
            )
        return httpx.Response(200, json={"models": [], "has_more": False})

    respx.get(MODELS_API).mock(side_effect=responder)
    r = discovering_client.get("/v1/models", headers=HEADERS)
    by_id = {m["id"]: m for m in r.json()["data"]}
    assert by_id[f"fal/{VIDEO_MODEL}"]["kind"] == "video"


# --- video resilience ------------------------------------------------------
#
# A 30-minute clip issues hundreds of status polls, any one of which can blip
# without the render being in trouble — and the result fetch happens after fal
# has already rendered (and billed for) the video. Neither may be a single
# point of failure.


def _stub_video_job_with(
    status_responder: object, result_responder: object, model_id: str = VIDEO_MODEL
) -> respx.Route:
    req = "req-flaky"
    submit = respx.post(f"{QUEUE}/{model_id}").mock(
        return_value=httpx.Response(
            200,
            json={
                "request_id": req,
                "status_url": f"{QUEUE}/{model_id}/requests/{req}/status",
                "response_url": f"{QUEUE}/{model_id}/requests/{req}",
            },
        )
    )
    respx.get(f"{QUEUE}/{model_id}/requests/{req}/status").mock(side_effect=status_responder)
    respx.get(f"{QUEUE}/{model_id}/requests/{req}").mock(side_effect=result_responder)
    respx.get("https://v3.fal.media/out.mp4").mock(
        return_value=httpx.Response(
            200,
            content=b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64,
            headers={"content-type": "video/mp4"},
        )
    )
    return submit


_OK_RESULT = {"video": {"url": "https://v3.fal.media/out.mp4"}}


@respx.mock
def test_transient_poll_failures_do_not_kill_the_job(discovering_client: TestClient) -> None:
    respx.get(MODELS_API).mock(return_value=httpx.Response(200, json={"models": []}))
    polls = {"n": 0}

    def status(request: httpx.Request) -> httpx.Response:
        polls["n"] += 1
        # A 503 then a 429 mid-flight; the render is fine.
        if polls["n"] in (2, 3):
            return httpx.Response(503, json={"error": "blip"})
        if polls["n"] < 5:
            return httpx.Response(200, json={"status": "IN_PROGRESS"})
        return httpx.Response(200, json={"status": "COMPLETED"})

    _stub_video_job_with(status, lambda request: httpx.Response(200, json=_OK_RESULT))

    r = discovering_client.post(
        "/v1/videos", headers=HEADERS, data={"model": f"fal/{VIDEO_MODEL}", "prompt": "x"}
    )
    assert r.status_code == 200, r.text
    state = _await_video(discovering_client, r.json()["id"])
    assert state["status"] == "completed"


@respx.mock
def test_sustained_poll_failures_do_eventually_fail(discovering_client: TestClient) -> None:
    """Tolerance is bounded — a genuinely broken job must not poll forever."""
    respx.get(MODELS_API).mock(return_value=httpx.Response(200, json={"models": []}))
    _stub_video_job_with(
        lambda request: httpx.Response(503, json={"error": "down"}),
        lambda request: httpx.Response(200, json=_OK_RESULT),
    )

    r = discovering_client.post(
        "/v1/videos", headers=HEADERS, data={"model": f"fal/{VIDEO_MODEL}", "prompt": "x"}
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    deadline = time.time() + 5.0
    while time.time() < deadline:
        state = discovering_client.get(f"/v1/videos/{job_id}", headers=HEADERS).json()
        if state["status"] == "failed":
            return
        time.sleep(0.05)
    pytest.fail("a permanently failing poll should surface as a failed job")


@respx.mock
def test_result_fetch_is_retried_after_the_render_is_paid_for(
    discovering_client: TestClient,
) -> None:
    """fal already rendered and billed for the clip — one hiccup collecting it
    must not throw the result away."""
    respx.get(MODELS_API).mock(return_value=httpx.Response(200, json={"models": []}))
    fetches = {"n": 0}

    def result(request: httpx.Request) -> httpx.Response:
        fetches["n"] += 1
        if fetches["n"] == 1:
            return httpx.Response(502, json={"error": "blip"})
        return httpx.Response(200, json=_OK_RESULT)

    _stub_video_job_with(lambda request: httpx.Response(200, json={"status": "COMPLETED"}), result)

    r = discovering_client.post(
        "/v1/videos", headers=HEADERS, data={"model": f"fal/{VIDEO_MODEL}", "prompt": "x"}
    )
    assert r.status_code == 200, r.text
    state = _await_video(discovering_client, r.json()["id"])
    assert state["status"] == "completed"
    assert fetches["n"] == 2, "the result fetch should have been retried once"


@respx.mock
def test_key_revoked_mid_poll_is_reported(
    discovering_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The 'reported once at ERROR' guarantee must cover the whole submit ->
    poll -> fetch sequence, not just submit."""
    respx.get(MODELS_API).mock(return_value=httpx.Response(200, json={"models": []}))
    _stub_video_job_with(
        lambda request: httpx.Response(401, json={"error": "Invalid API key"}),
        lambda request: httpx.Response(200, json=_OK_RESULT),
    )

    with caplog.at_level("ERROR"):
        r = discovering_client.post(
            "/v1/videos", headers=HEADERS, data={"model": f"fal/{VIDEO_MODEL}", "prompt": "x"}
        )
        assert r.status_code == 200, r.text
        job_id = r.json()["id"]
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if (
                discovering_client.get(f"/v1/videos/{job_id}", headers=HEADERS).json()["status"]
                == "failed"
            ):
                break
            time.sleep(0.05)
        else:
            pytest.fail("a rejected key should fail the job")

    errors = [rec for rec in caplog.records if rec.levelname == "ERROR"]
    assert errors and any("TEST_FAL_TOKEN" in rec.getMessage() for rec in errors)


# --- upstream data retention -----------------------------------------------


@pytest.fixture
def private_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(f"""
        [[providers]]
        id = "fal"
        backend = "fal"
        api_token_env = "TEST_FAL_TOKEN"
        discover_models = false
        store_payloads = false
        output_expiration_seconds = 300

        [[providers.models]]
        id = "{SEEDREAM_T2I}"
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


@respx.mock
def test_retention_headers_ride_on_inference_requests(private_client: TestClient) -> None:
    gen = _mock_generation(SEEDREAM_T2I)
    assert _generate(private_client, SEEDREAM_T2I).status_code == 200
    headers = gen.calls.last.request.headers
    assert headers["x-fal-store-io"] == "0"
    assert json.loads(headers["x-fal-object-lifecycle-preference"]) == {
        "expiration_duration_seconds": 300
    }


@respx.mock
def test_retention_headers_absent_by_default(client_with_fal: TestClient) -> None:
    gen = _mock_generation(SEEDREAM_T2I)
    assert _generate(client_with_fal, SEEDREAM_T2I).status_code == 200
    headers = gen.calls.last.request.headers
    assert "x-fal-store-io" not in headers
    assert "x-fal-object-lifecycle-preference" not in headers


@respx.mock
def test_expired_asset_failure_names_the_setting(private_client: TestClient) -> None:
    """A too-short expiry looks like an ordinary fetch failure; the error must
    point at the knob that caused it."""
    respx.post(f"{FAL}/{SEEDREAM_T2I}").mock(
        return_value=httpx.Response(
            200, json={"images": [{"url": "https://v3.fal.media/gone.png"}]}
        )
    )
    respx.get("https://v3.fal.media/gone.png").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    r = _generate(private_client, SEEDREAM_T2I)
    assert r.status_code >= 400
    assert "output_expiration_seconds" in r.json()["error"]["message"]


def test_expiration_below_the_floor_is_rejected() -> None:
    """A value that could never survive the retrieval budget must fail at
    config load, not silently lose generated assets at runtime."""
    with pytest.raises(Exception) as excinfo:
        FalProviderConfig(
            backend="fal", id="fal", api_token_env="TEST_FAL_TOKEN", output_expiration_seconds=5
        )
    assert "output_expiration_seconds" in str(excinfo.value)


# --- collapsing edit variants ----------------------------------------------


def _cat_entry(model_id: str, category: str, name: str | None = None) -> dict:
    return {
        "endpoint_id": model_id,
        "metadata": {"display_name": name or model_id, "category": category},
    }


def _stub_catalog_entries(entries: list[dict]) -> respx.Route:
    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("expand"):
            return httpx.Response(200, json={"models": []})
        category = request.url.params.get("category")
        return httpx.Response(
            200,
            json={
                "models": [e for e in entries if e["metadata"]["category"] == category],
                "has_more": False,
            },
        )

    return respx.get(MODELS_API).mock(side_effect=responder)


@respx.mock
def test_edit_variant_is_collapsed_into_its_base(discovering_client: TestClient) -> None:
    _stub_catalog_entries(
        [
            _cat_entry(NANO, "text-to-image", "Nano Banana 2"),
            _cat_entry(f"{NANO}/edit", "image-to-image", "Nano Banana 2 Edit"),
        ]
    )
    r = discovering_client.get("/v1/models", headers=HEADERS)
    ids = {m["id"] for m in r.json()["data"]}
    assert f"fal/{NANO}" in ids
    assert f"fal/{NANO}/edit" not in ids, "the edit half should not be listed separately"


@respx.mock
def test_sibling_suffix_pairs_are_collapsed(discovering_client: TestClient) -> None:
    """Seedream splits as `.../text-to-image` and `.../edit` — neither is a
    prefix of the other, so a naive base+suffix rule would miss it."""
    base = "bytedance/seedream/v5/pro"
    _stub_catalog_entries(
        [
            _cat_entry(f"{base}/text-to-image", "text-to-image", "Seedream 5 Pro"),
            _cat_entry(f"{base}/edit", "image-to-image", "Seedream 5 Pro Edit"),
        ]
    )
    ids = {m["id"] for m in discovering_client.get("/v1/models", headers=HEADERS).json()["data"]}
    assert f"fal/{base}/text-to-image" in ids
    assert f"fal/{base}/edit" not in ids


@respx.mock
def test_edit_only_models_are_left_alone(discovering_client: TestClient) -> None:
    """~299 image-to-image models have no text-to-image half — inpainting,
    upscalers, background removal. Those must stay listed."""
    _stub_catalog_entries(
        [
            _cat_entry("fal-ai/image-editing/object-removal", "image-to-image", "Object Removal"),
            _cat_entry("fal-ai/some-upscaler", "image-to-image", "Upscaler"),
        ]
    )
    ids = {m["id"] for m in discovering_client.get("/v1/models", headers=HEADERS).json()["data"]}
    assert "fal/fal-ai/image-editing/object-removal" in ids
    assert "fal/fal-ai/some-upscaler" in ids


@respx.mock
def test_no_collapse_when_the_candidate_is_not_an_edit_model(
    discovering_client: TestClient,
) -> None:
    """A name that merely looks like a sibling isn't enough — the categories
    have to be the two halves we expect."""
    _stub_catalog_entries(
        [
            _cat_entry(NANO, "text-to-image", "Nano"),
            # Same name shape, but it's another text-to-image model.
            _cat_entry(f"{NANO}/edit", "text-to-image", "Confusingly Named"),
        ]
    )
    ids = {m["id"] for m in discovering_client.get("/v1/models", headers=HEADERS).json()["data"]}
    assert f"fal/{NANO}" in ids
    assert f"fal/{NANO}/edit" in ids, "must not collapse on the name alone"


@respx.mock
def test_edit_request_is_routed_to_the_collapsed_sibling(
    discovering_client: TestClient,
) -> None:
    """The point of collapsing: an edit sent to the base model reaches the
    `/edit` endpoint, which is the mistake users would otherwise make."""
    _stub_catalog_entries(
        [
            _cat_entry(NANO, "text-to-image", "Nano Banana 2"),
            _cat_entry(f"{NANO}/edit", "image-to-image", "Nano Banana 2 Edit"),
        ]
    )
    edit_route = _mock_generation(f"{NANO}/edit")
    base_route = respx.post(f"{FAL}/{NANO}")

    r = discovering_client.post(
        "/v1/images/edits",
        headers=HEADERS,
        files={"image": ("in.png", b"PNG", "image/png")},
        data={"model": f"fal/{NANO}", "prompt": "make it blue"},
    )
    assert r.status_code == 200, r.text
    assert edit_route.called, "the edit should have gone to the /edit endpoint"
    assert not base_route.called, "the text-to-image endpoint must not receive edits"


@respx.mock
def test_generation_still_uses_the_base_endpoint(discovering_client: TestClient) -> None:
    _stub_catalog_entries(
        [
            _cat_entry(NANO, "text-to-image", "Nano Banana 2"),
            _cat_entry(f"{NANO}/edit", "image-to-image", "Nano Banana 2 Edit"),
        ]
    )
    base_route = _mock_generation(NANO)
    edit_route = respx.post(f"{FAL}/{NANO}/edit")

    assert _generate(discovering_client, NANO).status_code == 200
    assert base_route.called
    assert not edit_route.called


@respx.mock
def test_edit_routing_does_not_depend_on_prior_traffic(
    discovering_client: TestClient,
) -> None:
    """Routing comes from the catalogue, so an edit issued before anything has
    called /v1/models must still route — otherwise the same request would
    behave differently depending on unrelated earlier traffic."""
    _stub_catalog_entries(
        [
            _cat_entry(NANO, "text-to-image", "Nano Banana 2"),
            _cat_entry(f"{NANO}/edit", "image-to-image", "Nano Banana 2 Edit"),
        ]
    )
    edit_route = _mock_generation(f"{NANO}/edit")

    # No GET /v1/models first — this is the very first request.
    r = discovering_client.post(
        "/v1/images/edits",
        headers=HEADERS,
        files={"image": ("in.png", b"PNG", "image/png")},
        data={"model": f"fal/{NANO}", "prompt": "x"},
    )
    assert r.status_code == 200, r.text
    assert edit_route.called


# --- capability metadata ---------------------------------------------------
#
# Collapsing removes the `/edit` suffix that used to signal a model's modality,
# so the listing has to say it explicitly — otherwise a client can't tell a
# text-only model from one that also takes a reference image, and would learn
# the difference from a failed request.


@respx.mock
def test_collapsed_model_advertises_both_halves(discovering_client: TestClient) -> None:
    _stub_catalog_entries(
        [
            _cat_entry(NANO, "text-to-image", "Nano Banana 2"),
            _cat_entry(f"{NANO}/edit", "image-to-image", "Nano Banana 2 Edit"),
        ]
    )
    by_id = {
        m["id"]: m for m in discovering_client.get("/v1/models", headers=HEADERS).json()["data"]
    }
    assert by_id[f"fal/{NANO}"]["capabilities"] == ["text-to-image", "image-to-image"]


@respx.mock
def test_unpaired_text_model_advertises_text_only(discovering_client: TestClient) -> None:
    """The inverse of the collapsing problem: a text-only model must not look
    like it accepts an image."""
    _stub_catalog_entries([_cat_entry(NANO, "text-to-image", "Nano Banana 2")])
    by_id = {
        m["id"]: m for m in discovering_client.get("/v1/models", headers=HEADERS).json()["data"]
    }
    assert by_id[f"fal/{NANO}"]["capabilities"] == ["text-to-image"]


@respx.mock
def test_reference_only_model_advertises_image_input(discovering_client: TestClient) -> None:
    _stub_catalog_entries([_cat_entry("fal-ai/some-upscaler", "image-to-image", "Upscaler")])
    by_id = {
        m["id"]: m for m in discovering_client.get("/v1/models", headers=HEADERS).json()["data"]
    }
    assert by_id["fal/fal-ai/some-upscaler"]["capabilities"] == ["image-to-image"]


@respx.mock
def test_video_capabilities_and_collapsing(discovering_client: TestClient) -> None:
    """/v1/videos is a single endpoint for both text-to-video and
    image-to-video, so the caller can't express which half it wants — making
    the pairing more load-bearing here than for images."""
    t2v = "fal-ai/veo3.1"
    _stub_catalog_entries(
        [
            _cat_entry(t2v, "text-to-video", "Veo 3.1"),
            _cat_entry(f"{t2v}/image-to-video", "image-to-video", "Veo 3.1 I2V"),
        ]
    )
    by_id = {
        m["id"]: m for m in discovering_client.get("/v1/models", headers=HEADERS).json()["data"]
    }
    assert f"fal/{t2v}/image-to-video" not in by_id
    assert by_id[f"fal/{t2v}"]["kind"] == "video"
    assert by_id[f"fal/{t2v}"]["capabilities"] == ["text-to-video", "image-to-video"]


@respx.mock
def test_video_with_a_still_routes_to_the_image_to_video_half(
    discovering_client: TestClient,
) -> None:
    t2v = "fal-ai/veo3.1"
    _stub_catalog_entries(
        [
            _cat_entry(t2v, "text-to-video", "Veo 3.1"),
            _cat_entry(f"{t2v}/image-to-video", "image-to-video", "Veo 3.1 I2V"),
        ]
    )
    i2v_submit = _stub_video_job(f"{t2v}/image-to-video")
    t2v_submit = respx.post(f"{QUEUE}/{t2v}")

    r = discovering_client.post(
        "/v1/videos",
        headers=HEADERS,
        files={"input_reference": ("still.png", b"PNG", "image/png")},
        data={"model": f"fal/{t2v}", "prompt": "animate"},
    )
    assert r.status_code == 200, r.text
    _await_video(discovering_client, r.json()["id"])
    assert i2v_submit.called, "a still should reach the image-to-video endpoint"
    assert not t2v_submit.called


@respx.mock
def test_video_without_a_still_uses_the_text_half(discovering_client: TestClient) -> None:
    t2v = "fal-ai/veo3.1"
    _stub_catalog_entries(
        [
            _cat_entry(t2v, "text-to-video", "Veo 3.1"),
            _cat_entry(f"{t2v}/image-to-video", "image-to-video", "Veo 3.1 I2V"),
        ]
    )
    t2v_submit = _stub_video_job(t2v)
    i2v_submit = respx.post(f"{QUEUE}/{t2v}/image-to-video")

    r = discovering_client.post(
        "/v1/videos", headers=HEADERS, data={"model": f"fal/{t2v}", "prompt": "a river"}
    )
    assert r.status_code == 200, r.text
    _await_video(discovering_client, r.json()["id"])
    assert t2v_submit.called
    assert not i2v_submit.called


@respx.mock
def test_routed_edit_honours_the_siblings_own_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config on the `/edit` half must govern a routed edit. It's the endpoint
    actually running, so taking the base model's settings would silently ignore
    the block an operator wrote for it."""
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(f"""
        [[providers]]
        id = "fal"
        backend = "fal"
        api_token_env = "TEST_FAL_TOKEN"
        introspect_safety = false

        [[providers.models]]
        id = "{NANO}/edit"
        [providers.models.params]
        marker = "from-the-edit-block"
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

    _stub_catalog_entries(
        [
            _cat_entry(NANO, "text-to-image", "Nano"),
            _cat_entry(f"{NANO}/edit", "image-to-image", "Nano Edit"),
        ]
    )
    gen = _mock_generation(f"{NANO}/edit")

    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/images/edits",
            headers=HEADERS,
            files={"image": ("in.png", b"PNG", "image/png")},
            data={"model": f"fal/{NANO}", "prompt": "x"},
        )
        assert r.status_code == 200, r.text
        assert _sent_body(gen)["marker"] == "from-the-edit-block"
    reset_caches_for_tests()


@respx.mock
def test_sibling_block_does_not_revert_an_explicit_safety_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sibling block layers over the base; it doesn't replace it.

    `disable_safety` defaults to True (inject the loosest moderation params),
    so if a sibling block written for some unrelated reason replaced the base
    wholesale, an operator's explicit `disable_safety = false` would silently
    flip back to permissive on routed edits — and pinned `params` would vanish.
    """
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(f"""
        [[providers]]
        id = "fal"
        backend = "fal"
        api_token_env = "TEST_FAL_TOKEN"
        introspect_safety = false

        [[providers.models]]
        id = "{NANO}"
        disable_safety = false
        [providers.models.params]
        aspect_ratio = "16:9"

        # Written only to rename the edit half — it says nothing about safety.
        [[providers.models]]
        id = "{NANO}/edit"
        display_name = "Nano (edit)"
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

    _stub_catalog_entries(
        [
            _cat_entry(NANO, "text-to-image", "Nano"),
            _cat_entry(f"{NANO}/edit", "image-to-image", "Nano Edit"),
        ]
    )
    gen = _mock_generation(f"{NANO}/edit")

    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/images/edits",
            headers=HEADERS,
            files={"image": ("in.png", b"PNG", "image/png")},
            data={"model": f"fal/{NANO}", "prompt": "x"},
        )
        assert r.status_code == 200, r.text
        body = _sent_body(gen)
        # The base's opt-out survives: no moderation params injected.
        assert "safety_tolerance" not in body
        assert "enable_safety_checker" not in body
        # And its pinned params aren't dropped by the sibling block.
        assert body["aspect_ratio"] == "16:9"
    reset_caches_for_tests()


@respx.mock
def test_unknown_category_publishes_no_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Categories double as capability strings, so an operator pointing
    `categories` at something outside the known set must not publish a value
    that isn't in the documented {input}-to-{output} vocabulary."""
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent("""
        [[providers]]
        id = "fal"
        backend = "fal"
        api_token_env = "TEST_FAL_TOKEN"
        categories = ["upscaling"]
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

    _stub_catalog_entries([_cat_entry("fal-ai/odd", "upscaling", "Odd")])
    with TestClient(create_app()) as client:
        by_id = {m["id"]: m for m in client.get("/v1/models", headers=HEADERS).json()["data"]}
        assert "fal/fal-ai/odd" in by_id, "the configured category should still be listed"
        assert "capabilities" not in by_id["fal/fal-ai/odd"]
    reset_caches_for_tests()


@respx.mock
def test_configured_sibling_block_does_not_re_list_the_collapsed_half(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configuring the `/edit` half must not undo the collapsing.

    The listing unions in configured models the catalogue didn't return — but a
    collapsed sibling is absent *by design*, not because it's uncatalogued, so
    without distinguishing the two a config block puts it straight back. That's
    the shipped example (config.toml.example pairs seedream text-to-image with
    its /edit half), and the README endorses sibling blocks for overriding a
    routed edit, so it's the documented shape rather than a corner case.
    """
    t2v = "fal-ai/veo3.1"
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(f"""
        [[providers]]
        id = "fal"
        backend = "fal"
        api_token_env = "TEST_FAL_TOKEN"
        introspect_safety = false

        [[providers.models]]
        id = "{NANO}/edit"
        display_name = "Nano (edit)"

        [[providers.models]]
        id = "{t2v}/image-to-video"
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

    _stub_catalog_entries(
        [
            _cat_entry(NANO, "text-to-image", "Nano"),
            _cat_entry(f"{NANO}/edit", "image-to-image", "Nano Edit"),
            _cat_entry(t2v, "text-to-video", "Veo 3.1"),
            _cat_entry(f"{t2v}/image-to-video", "image-to-video", "Veo 3.1 I2V"),
        ]
    )

    with TestClient(create_app()) as client:
        by_id = {m["id"]: m for m in client.get("/v1/models", headers=HEADERS).json()["data"]}
        assert f"fal/{NANO}" in by_id
        assert f"fal/{NANO}/edit" not in by_id, "collapsed edit half was re-listed"
        # The video pair is the sharper edge: a re-added entry comes from the
        # config, which carries no capabilities and defaults kind to "image" —
        # so a client filtering on kind would send a video model to /v1/images.
        assert f"fal/{t2v}" in by_id
        assert f"fal/{t2v}/image-to-video" not in by_id, "collapsed video half was re-listed"
    reset_caches_for_tests()


@respx.mock
def test_catalog_categories_are_walked_concurrently(discovering_client: TestClient) -> None:
    """Serially this is ~10-13 round trips on the first /v1/models after boot,
    all of it holding the catalogue lock with every other provider queued
    behind it."""
    inflight = {"now": 0, "max": 0}

    async def responder(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("expand"):
            return httpx.Response(200, json={"models": []})
        inflight["now"] += 1
        inflight["max"] = max(inflight["max"], inflight["now"])
        await asyncio.sleep(0.05)
        inflight["now"] -= 1
        return httpx.Response(200, json={"models": [], "has_more": False})

    respx.get(MODELS_API).mock(side_effect=responder)
    assert discovering_client.get("/v1/models", headers=HEADERS).status_code == 200
    assert inflight["max"] == len(SUPPORTED_CATEGORIES), (
        f"categories should overlap; peak concurrency was {inflight['max']}"
    )


def test_ref_encoded_enums_are_resolved() -> None:
    """fal inlines its enums today, so this is insurance rather than a live
    fix — but the failure mode if that changes is silent: an unresolved enum
    reads as "no values" and the moderation knob is simply omitted."""
    from openai_api_bridge.backends.fal.safety import safety_params_from_schema
    from openai_api_bridge.backends.fal.schema import duration_params, duration_property

    spec = {
        "paths": {
            "/m": {
                "post": {
                    "requestBody": {
                        "content": {"application/json": {"schema": {"$ref": "#/c/s/MInput"}}}
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "MInput": {
                    "properties": {
                        "safety_tolerance": {"allOf": [{"$ref": "#/c/s/Tolerance"}]},
                        "duration": {"allOf": [{"$ref": "#/c/s/Duration"}]},
                    }
                },
                "Tolerance": {"enum": ["1", "2", "3", "4", "5", "6"]},
                "Duration": {"enum": ["4s", "6s", "8s"]},
            }
        },
    }
    # Nested under "c/s" to match the $ref paths above.
    spec["c"] = spec.pop("components")
    spec["c"]["s"] = spec["c"].pop("schemas")

    assert safety_params_from_schema(spec) == {"safety_tolerance": "6"}
    assert duration_params(duration_property(spec), 5.0) == {"duration": "6s"}


# --- abandoned upstream jobs ------------------------------------------------
#
# fal bills a render whether or not anyone collects it, so every way out of the
# poll loop that isn't a finished clip has to tell fal to stop.


def _stub_cancellable_job(model_id: str = VIDEO_MODEL) -> dict[str, respx.Route]:
    req = "req-cancel"
    base = f"{QUEUE}/{model_id}/requests/{req}"
    submit = respx.post(f"{QUEUE}/{model_id}").mock(
        return_value=httpx.Response(
            200,
            json={
                "request_id": req,
                "status_url": f"{base}/status",
                "response_url": base,
                "cancel_url": f"{base}/cancel",
            },
        )
    )
    cancel = respx.put(f"{base}/cancel").mock(
        return_value=httpx.Response(202, json={"status": "CANCELLATION_REQUESTED"})
    )
    return {"submit": submit, "cancel": cancel, "base": base}


@respx.mock
async def test_client_cancellation_stops_the_upstream_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DELETE /v1/videos cancels our task; without this the fal job kept
    rendering — and billing — to completion, unread."""
    monkeypatch.setenv("TEST_FAL_TOKEN", "fal-secret")
    routes = _stub_cancellable_job()
    respx.get(f"{routes['base']}/status").mock(
        return_value=httpx.Response(200, json={"status": "IN_PROGRESS"})
    )

    backend = _direct_backend(retry_seconds=300.0, model_ids=[VIDEO_MODEL])
    try:
        task = asyncio.create_task(backend.generate_video(model_slug=VIDEO_MODEL, prompt="a river"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await backend.aclose()

    assert routes["cancel"].called, "an abandoned job must be cancelled upstream"


@respx.mock
async def test_timeout_stops_the_upstream_render(monkeypatch: pytest.MonkeyPatch) -> None:
    """Giving up locally leaves fal rendering just as surely as a DELETE does."""
    monkeypatch.setenv("TEST_FAL_TOKEN", "fal-secret")
    routes = _stub_cancellable_job()
    respx.get(f"{routes['base']}/status").mock(
        return_value=httpx.Response(200, json={"status": "IN_QUEUE"})
    )

    from openai_api_bridge.backends.fal.adapter import FalBackend
    from openai_api_bridge.config import FalModelConfig, FalProviderConfig

    backend = FalBackend(
        FalProviderConfig(
            backend="fal",
            id="fal",
            api_token_env="TEST_FAL_TOKEN",
            models=[FalModelConfig(id=VIDEO_MODEL)],
            introspect_safety=False,
            video_poll_interval_seconds=0.01,
            video_poll_timeout_seconds=0.05,
        )
    )
    try:
        with pytest.raises(BridgeError):
            await backend.generate_video(model_slug=VIDEO_MODEL, prompt="x")
    finally:
        await backend.aclose()

    assert routes["cancel"].called


@respx.mock
async def test_completed_job_is_not_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only abandonment triggers it — a collected clip must not be cancelled."""
    monkeypatch.setenv("TEST_FAL_TOKEN", "fal-secret")
    routes = _stub_cancellable_job()
    respx.get(f"{routes['base']}/status").mock(
        return_value=httpx.Response(200, json={"status": "COMPLETED"})
    )
    respx.get(routes["base"]).mock(
        return_value=httpx.Response(200, json={"video": {"url": "https://v3.fal.media/o.mp4"}})
    )
    respx.get("https://v3.fal.media/o.mp4").mock(
        return_value=httpx.Response(200, content=b"mp4", headers={"content-type": "video/mp4"})
    )

    backend = _direct_backend(retry_seconds=300.0, model_ids=[VIDEO_MODEL])
    try:
        asset = await backend.generate_video(model_slug=VIDEO_MODEL, prompt="x")
        assert asset.kind == "video"
    finally:
        await backend.aclose()

    assert not routes["cancel"].called


@respx.mock
async def test_a_failing_cancel_does_not_mask_the_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cancel is best-effort; it must never replace the failure that caused
    us to abandon the job."""
    monkeypatch.setenv("TEST_FAL_TOKEN", "fal-secret")
    routes = _stub_cancellable_job()
    respx.put(f"{routes['base']}/cancel").mock(
        return_value=httpx.Response(500, json={"error": "cancel unavailable"})
    )
    respx.get(f"{routes['base']}/status").mock(
        return_value=httpx.Response(200, json={"status": "IN_QUEUE"})
    )

    from openai_api_bridge.backends.fal.adapter import FalBackend
    from openai_api_bridge.config import FalModelConfig, FalProviderConfig

    backend = FalBackend(
        FalProviderConfig(
            backend="fal",
            id="fal",
            api_token_env="TEST_FAL_TOKEN",
            models=[FalModelConfig(id=VIDEO_MODEL)],
            introspect_safety=False,
            video_poll_interval_seconds=0.01,
            video_poll_timeout_seconds=0.05,
        )
    )
    try:
        with pytest.raises(BridgeError) as excinfo:
            await backend.generate_video(model_slug=VIDEO_MODEL, prompt="x")
        # The timeout, not the cancel failure.
        assert "did not finish" in str(excinfo.value)
    finally:
        await backend.aclose()


# --- display-name disambiguation -------------------------------------------


@respx.mock
def test_family_named_endpoints_are_disambiguated(discovering_client: TestClient) -> None:
    """fal titles every endpoint of a family after the family, so a picker
    rendering display_name shows a run of identical rows."""
    _stub_catalog_entries(
        [
            _cat_entry("fal-ai/florence-2-large/object-detection", "image-to-image", "Florence-2"),
            _cat_entry("fal-ai/florence-2-large/ocr-with-region", "image-to-image", "Florence-2"),
            _cat_entry("fal-ai/florence-2-large/region-proposal", "image-to-image", "Florence-2"),
        ]
    )
    r = discovering_client.get("/v1/models", headers=HEADERS)
    assert r.status_code == 200
    names = {m["id"]: m["display_name"] for m in r.json()["data"]}
    assert names["fal/fal-ai/florence-2-large/object-detection"] == "Florence-2 (object-detection)"
    assert names["fal/fal-ai/florence-2-large/ocr-with-region"] == "Florence-2 (ocr-with-region)"
    assert names["fal/fal-ai/florence-2-large/region-proposal"] == "Florence-2 (region-proposal)"


@respx.mock
def test_disambiguation_keeps_only_the_differing_path(discovering_client: TestClient) -> None:
    """The suffix is what actually differs within the group — the shared
    family segments are already visible in the id and would be noise."""
    _stub_catalog_entries(
        [
            _cat_entry("fal-ai/wan/v2.7/text-to-image", "text-to-image", "Wan"),
            _cat_entry("fal-ai/wan/v2.2-a14b/text-to-image", "text-to-image", "Wan"),
        ]
    )
    r = discovering_client.get("/v1/models", headers=HEADERS)
    names = {m["id"]: m["display_name"] for m in r.json()["data"]}
    assert names["fal/fal-ai/wan/v2.7/text-to-image"] == "Wan (v2.7/text-to-image)"
    assert names["fal/fal-ai/wan/v2.2-a14b/text-to-image"] == "Wan (v2.2-a14b/text-to-image)"


@respx.mock
def test_unique_display_names_are_left_alone(discovering_client: TestClient) -> None:
    _stub_catalog_entries(
        [
            _cat_entry(SEEDREAM_T2I, "text-to-image", "Seedream 4"),
            _cat_entry(GPT_IMAGE, "text-to-image", "GPT Image 2"),
        ]
    )
    r = discovering_client.get("/v1/models", headers=HEADERS)
    names = {m["id"]: m["display_name"] for m in r.json()["data"]}
    assert names[f"fal/{SEEDREAM_T2I}"] == "Seedream 4"
    assert names[f"fal/{GPT_IMAGE}"] == "GPT Image 2"


@respx.mock
def test_configured_display_name_is_not_disambiguated(discovering_client: TestClient) -> None:
    """An operator who pinned a name in config asked for that exact string."""
    _stub_catalog_entries(
        [
            _cat_entry(NANO, "text-to-image", "Nano Banana 2"),
            _cat_entry(f"{NANO}/lite", "text-to-image", "Nano (renamed locally)"),
        ]
    )
    r = discovering_client.get("/v1/models", headers=HEADERS)
    names = {m["id"]: m["display_name"] for m in r.json()["data"]}
    # NANO is pinned by the fixture's [[providers.models]] block, so it keeps
    # the operator's exact string.
    assert names[f"fal/{NANO}"] == "Nano (renamed locally)"
    # Its catalogue-named sibling collides with that pin, and is pulled apart
    # from it — honouring the pin must not leave a duplicate standing.
    assert names[f"fal/{NANO}/lite"] == "Nano (renamed locally) (lite)"
