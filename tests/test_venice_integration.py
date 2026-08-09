"""End-to-end tests for the Venice provider.

Stubs the upstream Venice HTTP surface with respx and verifies:
  * /v1/images/edits forwards the multipart upload to /api/v1/image/edit and
    serves the (binary) edited image back through /v1/files/{id}/content
  * Multiple reference images are rejected (Venice edits are single-image)
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

    monkeypatch.setenv("BRIDGE_API_KEY", "test-bridge-api-key")
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


HEADERS = {"Authorization": "Bearer test-bridge-api-key"}


@respx.mock
@pytest.mark.parametrize("size", ["-1x100", "1024x0", "garbage", "auto"])
def test_malformed_size_falls_back_to_configured_defaults(
    client_with_venice: TestClient,
    size: str,
) -> None:
    """Venice applies its default per dimension with ``w or default``, and
    ``-1`` is truthy — so a negative width used to arrive upstream as a real
    request parameter rather than being replaced. fal and ComfyUI both guard
    with ``> 0``; this backend was the one that didn't, which is why the fix
    belongs in ``parse_size`` rather than at each call site.
    """
    sent: dict[str, object] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"images": ["aGk="]})

    respx.post(f"{UPSTREAM}/api/v1/image/generate").mock(side_effect=responder)

    r = client_with_venice.post(
        "/v1/images/generations",
        headers=HEADERS,
        json={"model": "vn/flux", "prompt": "a cat", "size": size, "response_format": "b64_json"},
    )
    assert r.status_code == 200
    # The provider's configured defaults, never a number derived from the
    # malformed string.
    assert sent["width"] == 1024
    assert sent["height"] == 1024


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


@respx.mock
def test_inpaint_failure_does_not_drop_the_whole_provider(
    client_with_venice: TestClient,
) -> None:
    """`type=inpaint` is a narrower query than `type=image`; a proxy or older
    API version could reject it outright. That must not take a perfectly
    healthy text-to-image catalogue down with it."""

    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("type") == "inpaint":
            return httpx.Response(400, json={"error": "unknown type"})
        return httpx.Response(200, json={"data": [{"id": "flux-2-pro", "type": "image"}]})

    respx.get(f"{UPSTREAM}/api/v1/models").mock(side_effect=responder)

    r = client_with_venice.get("/v1/models", headers=HEADERS)
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert "vn/flux-2-pro" in ids, "generate models must survive an inpaint failure"


@respx.mock
def test_failed_route_load_is_not_retried_every_request(
    client_with_venice: TestClient,
) -> None:
    """Without a cooldown each edit during an outage re-runs the catalogue
    fetch — and because waiters take the lock in turn rather than sharing one
    result, concurrent edits serialise into sequential attempts."""
    calls = {"n": 0}

    def responder(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": "down"})

    respx.get(f"{UPSTREAM}/api/v1/models").mock(side_effect=responder)
    respx.post(f"{UPSTREAM}/api/v1/image/edit").mock(
        return_value=httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})
    )

    for _ in range(3):
        client_with_venice.post(
            "/v1/images/edits",
            headers=HEADERS,
            files={"image": ("in.png", b"png", "image/png")},
            data={"model": "vn/gpt-image-2", "prompt": "x"},
        )
    # One attempt (both listings issued together), not one per request.
    assert calls["n"] <= 2, f"catalogue re-fetched per request: {calls['n']} calls"


@respx.mock
def test_both_catalog_listings_are_fetched_concurrently(
    client_with_venice: TestClient,
) -> None:
    """Serialising them would double this endpoint's tail latency while every
    other provider waits behind it."""
    inflight = {"now": 0, "max": 0}

    async def responder(request: httpx.Request) -> httpx.Response:
        inflight["now"] += 1
        inflight["max"] = max(inflight["max"], inflight["now"])
        await asyncio.sleep(0.05)
        inflight["now"] -= 1
        return httpx.Response(200, json={"data": []})

    respx.get(f"{UPSTREAM}/api/v1/models").mock(side_effect=responder)
    assert client_with_venice.get("/v1/models", headers=HEADERS).status_code == 200
    assert inflight["max"] == 2, "the two listings should overlap"


@respx.mock
async def test_degraded_listing_leaves_routing_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A listing that served generate models but lost the inpaint half must not
    mark routing as resolved — otherwise an empty route map latches for the
    life of the process and edits never route again, with the listing looking
    perfectly healthy."""
    monkeypatch.setenv("VENICE_API_TOKEN", "venice-secret")
    from openai_api_bridge.backends.venice.adapter import VeniceBackend
    from openai_api_bridge.config import VeniceProviderConfig

    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("type") == "inpaint":
            return httpx.Response(503, json={"error": "down"})
        return httpx.Response(200, json={"data": [{"id": "gpt-image-2", "type": "image"}]})

    respx.get(f"{UPSTREAM}/api/v1/models").mock(side_effect=responder)
    backend = VeniceBackend(
        VeniceProviderConfig(backend="venice", id="vn", api_token_env="VENICE_API_TOKEN")
    )
    try:
        entries = await backend.list_models()
        assert [e.id for e in entries] == ["gpt-image-2"]
        assert backend._routes_loaded is False, "a degraded listing must not latch routing"
        assert backend._routes_failed_at is not None, "and must arm the retry cooldown"
    finally:
        await backend.aclose()


# --- catalogue caching -----------------------------------------------------


def _venice_backend(**overrides: object) -> object:
    from openai_api_bridge.backends.venice.adapter import VeniceBackend
    from openai_api_bridge.config import VeniceProviderConfig

    return VeniceBackend(
        VeniceProviderConfig(
            backend="venice", id="vn", api_token_env="VENICE_API_TOKEN", **overrides
        )
    )


def _counting_catalog(calls: dict[str, int], *, inpaint_ok: bool = True) -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        calls["n"] = calls.get("n", 0) + 1
        model_type = request.url.params.get("type")
        if model_type == "inpaint" and not inpaint_ok:
            return httpx.Response(503, json={"error": "down"})
        ids = ["gpt-image-2"] if model_type == "image" else ["gpt-image-2-edit"]
        return httpx.Response(200, json={"data": [{"id": i, "type": model_type} for i in ids]})

    respx.get(f"{UPSTREAM}/api/v1/models").mock(side_effect=responder)


@respx.mock
async def test_catalog_is_cached_across_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """/v1/models costs two upstream calls on Venice, so repeating it shouldn't
    keep paying for both."""
    monkeypatch.setenv("VENICE_API_TOKEN", "venice-secret")
    calls: dict[str, int] = {}
    _counting_catalog(calls)

    backend = _venice_backend(catalog_ttl_seconds=300.0)
    try:
        for _ in range(4):
            assert len(await backend.list_models()) == 1
    finally:
        await backend.aclose()
    assert calls["n"] == 2, f"expected one image+inpaint pair, got {calls['n']} calls"


@respx.mock
async def test_catalog_cache_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TTL rather than a permanent cache, so models Venice adds show up
    without restarting the bridge."""
    monkeypatch.setenv("VENICE_API_TOKEN", "venice-secret")
    calls: dict[str, int] = {}
    _counting_catalog(calls)

    backend = _venice_backend(catalog_ttl_seconds=0.05)
    try:
        await backend.list_models()
        assert calls["n"] == 2
        await asyncio.sleep(0.1)
        await backend.list_models()
    finally:
        await backend.aclose()
    assert calls["n"] == 4, "the catalogue should be re-read once the TTL lapses"


@respx.mock
async def test_catalog_caching_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VENICE_API_TOKEN", "venice-secret")
    calls: dict[str, int] = {}
    _counting_catalog(calls)

    backend = _venice_backend(catalog_ttl_seconds=0)
    try:
        await backend.list_models()
        await backend.list_models()
    finally:
        await backend.aclose()
    assert calls["n"] == 4


@respx.mock
async def test_degraded_listing_is_cached_only_briefly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A listing whose inpaint half failed is still served — dropping the
    provider over its narrower query would be worse — but it's incomplete, so
    it's cached only for the failure window. Not caching it at all would let a
    burst during an inpaint hang queue behind the lock, each waiter starting
    its own fetch; caching it for the full TTL would pin routing unresolved."""
    monkeypatch.setenv("VENICE_API_TOKEN", "venice-secret")
    calls: dict[str, int] = {}
    _counting_catalog(calls, inpaint_ok=False)

    backend = _venice_backend(catalog_ttl_seconds=300.0, catalog_retry_seconds=0.05)
    try:
        entries = await backend.list_models()
        assert [e.id for e in entries] == ["gpt-image-2"]
        assert calls["n"] == 2

        # Repeats inside the window are served from cache, not re-fetched.
        for _ in range(3):
            await backend.list_models()
        assert calls["n"] == 2, f"degraded listing re-fetched per call: {calls['n']}"

        # Routing stays unresolved — the half it depends on never arrived.
        assert backend._routes_loaded is False

        # And the missing half is re-attempted once the short window lapses.
        await asyncio.sleep(0.1)
        await backend.list_models()
        assert calls["n"] == 4
    finally:
        await backend.aclose()


@respx.mock
async def test_edit_routing_reuses_the_cached_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Routing reads the same catalogue, so an edit after a listing shouldn't
    re-fetch it."""
    monkeypatch.setenv("VENICE_API_TOKEN", "venice-secret")
    calls: dict[str, int] = {}
    _counting_catalog(calls)

    backend = _venice_backend(catalog_ttl_seconds=300.0)
    try:
        await backend.list_models()
        assert calls["n"] == 2
        assert await backend._edit_target("gpt-image-2") == "gpt-image-2-edit"
    finally:
        await backend.aclose()
    assert calls["n"] == 2, "edit routing should hit the cache, not re-fetch"


@respx.mock
async def test_ttl_zero_disables_caching_on_the_degraded_path_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`catalog_ttl_seconds = 0` documents caching as off. The short-TTL
    override for an incomplete listing must not quietly reinstate it — an
    override may only ever shorten the window, never create one."""
    monkeypatch.setenv("VENICE_API_TOKEN", "venice-secret")
    calls: dict[str, int] = {}
    _counting_catalog(calls, inpaint_ok=False)

    backend = _venice_backend(catalog_ttl_seconds=0, catalog_retry_seconds=300.0)
    try:
        for _ in range(3):
            await backend.list_models()
    finally:
        await backend.aclose()
    assert calls["n"] == 6, f"caching should be off entirely, saw {calls['n']} calls"


@respx.mock
async def test_incomplete_listing_is_never_held_longer_than_a_healthy_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry window wider than the TTL must not invert the relationship."""
    monkeypatch.setenv("VENICE_API_TOKEN", "venice-secret")
    calls: dict[str, int] = {}
    _counting_catalog(calls, inpaint_ok=False)

    backend = _venice_backend(catalog_ttl_seconds=0.05, catalog_retry_seconds=300.0)
    try:
        await backend.list_models()
        assert calls["n"] == 2
        await asyncio.sleep(0.1)
        await backend.list_models()
    finally:
        await backend.aclose()
    assert calls["n"] == 4, "the incomplete listing outlived the configured TTL"
