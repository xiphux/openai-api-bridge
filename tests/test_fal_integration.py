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
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from openai_api_bridge.backends.fal.adapter import FalBackend
from openai_api_bridge.config import FalModelConfig, FalProviderConfig, reset_caches_for_tests

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
            *(backend._safety_params(cfg.models[i % len(ids)]) for i in range(8))
        )
    finally:
        await backend.aclose()

    # 8 concurrent callers over 3 distinct models -> one lookup per model, not
    # one per caller (without the per-model lock this would be 8).
    assert route.call_count == len(ids)
    # And none of them slipped through before the cache was populated.
    assert all(r == {"enable_safety_checker": False} for r in results)


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
            assert await backend._safety_params(backend.cfg.models[0]) == {}
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
        assert await backend._safety_params(model) == {}
        # Second attempt retries and succeeds -> schema-derived.
        assert await backend._safety_params(model) == {"enable_safety_checker": False}
        # Now cached: no further lookups.
        assert await backend._safety_params(model) == {"enable_safety_checker": False}
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
        # The client itself retries a missing id once (its truncation guard),
        # so the whole first round — batch + straggler — must come back empty
        # for the adapter to see a successful-but-useless lookup.
        if len(calls) <= 2:
            return httpx.Response(200, json={"models": []})
        return httpx.Response(200, json={"models": [{"endpoint_id": UNMAPPED, "openapi": spec}]})

    respx.get(MODELS_API).mock(side_effect=responder)

    backend = _direct_backend(retry_seconds=0.0, model_ids=[UNMAPPED])
    try:
        model = backend.cfg.models[0]
        assert await backend._safety_params(model) == {}
        # Retried rather than serving a latched empty cache forever.
        assert await backend._safety_params(model) == {"enable_safety_checker": False}
    finally:
        await backend.aclose()


@respx.mock
async def test_partial_response_retries_only_the_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a response resolves some models but omits others, the resolved ones
    are cached and only the stragglers are re-fetched after the cooldown."""
    monkeypatch.setenv("TEST_FAL_TOKEN", "fal-secret")
    spec = _openapi("S", {"enable_safety_checker": {"type": "boolean", "default": True}})
    other = "fal-ai/another-unmapped-model"
    calls: list[list[str]] = []

    def responder(request: httpx.Request) -> httpx.Response:
        asked = request.url.params.get_list("endpoint_id")
        calls.append(asked)
        # Round 1 is the batch plus the client's own straggler retry; `other`
        # stays unanswered through both so the adapter sees a partial result.
        served = [m for m in asked if m == UNMAPPED or len(calls) >= 3]
        return httpx.Response(
            200, json={"models": [{"endpoint_id": m, "openapi": spec} for m in served]}
        )

    respx.get(MODELS_API).mock(side_effect=responder)

    backend = _direct_backend(retry_seconds=0.0, model_ids=[UNMAPPED, other])
    try:
        resolved_model, missing_model = backend.cfg.models
        # The model that did resolve is cached from the partial round...
        assert await backend._safety_params(resolved_model) == {"enable_safety_checker": False}
        # ...and the straggler is retried rather than left on fallback forever.
        assert await backend._safety_params(missing_model) == {"enable_safety_checker": False}
    finally:
        await backend.aclose()

    # That retry asked only for the model still missing, not the whole set.
    assert calls[-1] == [other]


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
            assert await backend._safety_params(missing_model) == {}
    finally:
        await backend.aclose()

    # One round only — the batch plus the client's own straggler retry — rather
    # than a fresh lookup for each of the three requests.
    assert route.call_count == 2


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
    assert asked == {"text-to-image", "image-to-image"}
    # Video models would all fail with UnsupportedOperation, so they're not listed.
    assert "text-to-video" not in asked
    assert "image-to-video" not in asked
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
    assert route.call_count == 2


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

    # One attempt only: a bad key cannot start working, so no cooldown retry.
    assert route.call_count == 1
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
            assert await backend._safety_params(backend.cfg.models[0]) == {}
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
            assert await backend._safety_params(backend.cfg.models[0]) == {}
    finally:
        await backend.aclose()
    # Each call re-attempts (batch + the client's own straggler retry per round).
    assert route.call_count > 1
