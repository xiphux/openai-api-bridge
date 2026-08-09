"""End-to-end tests for the OpenRouter provider.

Stubs the upstream OpenRouter HTTP surface with respx and verifies:
  * /v1/models classifies by architecture.output_modalities
  * /v1/chat/completions passes through (sync + streaming)
  * /v1/images/generations translates to chat completions + extracts
    the data URL out of the message.images array
  * /v1/images/edits forwards the input image as a base64 data URL
    inside the user message content
  * Empty images array surfaces as an upstream error (no silent drop)
"""

from __future__ import annotations

import base64
import json
import textwrap
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from openai_api_bridge.config import reset_caches_for_tests

UPSTREAM = "https://openrouter.ai/api"


@pytest.fixture
def client_with_openrouter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent("""
		[[providers]]
		id = "or"
		backend = "openrouter"
		api_token_env = "TEST_OR_TOKEN"
	""")
    )

    monkeypatch.setenv("BRIDGE_API_KEY", "test-bridge-api-key")
    monkeypatch.setenv("BRIDGE_CONFIG_PATH", str(config))
    monkeypatch.setenv("FILES_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "state.db"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("TEST_OR_TOKEN", "or-secret")
    reset_caches_for_tests()

    from openai_api_bridge.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    reset_caches_for_tests()


HEADERS = {"Authorization": "Bearer test-bridge-api-key"}


# OpenRouter /v1/models — rich metadata with input/output modalities. The
# bridge classifies image, video, embedding, and chat models from these.
# Audio + unclassifiable entries should be filtered out.
_MODELS_FIXTURE = {
    "data": [
        {
            "id": "openai/gpt-4o",
            "name": "GPT-4o",
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
            },
        },
        {
            "id": "google/gemini-2.5-flash-image",
            "name": "Gemini 2.5 Flash (Image)",
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["image"],
            },
        },
        {
            "id": "openai/text-embedding-3-large",
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["embedding"],
            },
        },
        {
            "id": "some-vendor/video-model",
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["video"],
            },
        },
        # Filtered: audio output
        {
            "id": "audio-provider/tts",
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["audio"],
            },
        },
        # Filtered: missing architecture
        {"id": "no-arch/model"},
    ]
}


@respx.mock
def test_models_classifies_by_output_modalities(
    client_with_openrouter: TestClient,
) -> None:
    respx.get(f"{UPSTREAM}/v1/models").mock(return_value=httpx.Response(200, json=_MODELS_FIXTURE))
    r = client_with_openrouter.get("/v1/models", headers=HEADERS)
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()["data"]}
    # Chat (text output) — vision-capable models with text-only output
    # classify as chat; image *input* doesn't matter for our routing.
    assert by_id["or/openai/gpt-4o"]["kind"] == "chat"
    # Image gen
    assert by_id["or/google/gemini-2.5-flash-image"]["kind"] == "image"
    # Embedding
    assert by_id["or/openai/text-embedding-3-large"]["kind"] == "embedding"
    # Video
    assert by_id["or/some-vendor/video-model"]["kind"] == "video"
    # Filtered: audio + no-arch don't appear
    assert "or/audio-provider/tts" not in by_id
    assert "or/no-arch/model" not in by_id
    # display_name surfaces from the upstream's ``name`` field when present
    assert by_id["or/openai/gpt-4o"]["display_name"] == "GPT-4o"
    # supports_tools omitted from rows where upstream didn't set
    # `supported_parameters` (the bridge doesn't guess).
    assert "supports_tools" not in by_id["or/openai/gpt-4o"]


# Fixture exercising the `supported_parameters` field — OpenRouter's
# per-model capability hint we use to derive supports_tools.
_MODELS_WITH_PARAMS_FIXTURE = {
    "data": [
        {
            "id": "openai/gpt-4o",
            "name": "GPT-4o",
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["tools", "tool_choice", "temperature"],
        },
        {
            "id": "old-vendor/no-tools-model",
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["temperature", "top_p"],
        },
    ]
}


@respx.mock
def test_models_surfaces_supports_tools_from_supported_parameters(
    client_with_openrouter: TestClient,
) -> None:
    respx.get(f"{UPSTREAM}/v1/models").mock(
        return_value=httpx.Response(200, json=_MODELS_WITH_PARAMS_FIXTURE)
    )
    r = client_with_openrouter.get("/v1/models", headers=HEADERS)
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()["data"]}
    # True when "tools" is in supported_parameters
    assert by_id["or/openai/gpt-4o"]["supports_tools"] is True
    # False when supported_parameters is present but lacks "tools"
    assert by_id["or/old-vendor/no-tools-model"]["supports_tools"] is False


@respx.mock
def test_chat_completion_sync_passthrough(
    client_with_openrouter: TestClient,
) -> None:
    chat_route = respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "cmpl-abc",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )
    r = client_with_openrouter.post(
        "/v1/chat/completions",
        headers=HEADERS,
        json={
            "model": "or/openai/gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hi"
    # Outgoing request stripped the "or/" prefix off the model
    sent = json.loads(chat_route.calls.last.request.read())
    assert sent["model"] == "openai/gpt-4o"


@respx.mock
def test_images_generations_translated_via_chat(
    client_with_openrouter: TestClient,
) -> None:
    """The bridge POSTs a chat-completion to OpenRouter; the response carries
    the image as a base64 data URL in message.images; the bridge decodes it
    and serves it back through /v1/files/{id}/content."""
    raw_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
    data_url = "data:image/png;base64," + base64.b64encode(raw_png).decode()
    chat_route = respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "cmpl-img",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Here you go!",
                            "images": [{"image_url": {"url": data_url}}],
                        }
                    }
                ],
            },
        )
    )

    r = client_with_openrouter.post(
        "/v1/images/generations",
        headers=HEADERS,
        json={
            "model": "or/google/gemini-2.5-flash-image",
            "prompt": "a red panda",
        },
    )
    assert r.status_code == 200
    body = r.json()
    url = body["data"][0]["url"]
    assert "/v1/files/" in url and url.endswith("/content")

    # Outgoing chat body: model unprefixed, prompt in a user message, with
    # the modalities hint set so OpenRouter knows we want image output.
    sent = json.loads(chat_route.calls.last.request.read())
    assert sent["model"] == "google/gemini-2.5-flash-image"
    assert sent["messages"][0]["role"] == "user"
    assert sent["messages"][0]["content"] == "a red panda"
    assert "image" in sent["modalities"]

    # Fetch the bytes back through the bridge's files endpoint to confirm
    # the round-trip preserves the PNG payload.
    file_resp = client_with_openrouter.get(url, headers=HEADERS)
    assert file_resp.status_code == 200
    assert file_resp.headers["content-type"].startswith("image/png")
    assert file_resp.content == raw_png


@respx.mock
def test_images_edits_sends_input_as_base64(
    client_with_openrouter: TestClient,
) -> None:
    """Input image is encoded as a base64 data URL inside the user message
    content (the multimodal-message shape OpenRouter expects)."""
    output_png = b"\x89PNG\r\n\x1a\nedit-output"
    out_data_url = "data:image/png;base64," + base64.b64encode(output_png).decode()
    chat_route = respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Done.",
                            "images": [{"image_url": {"url": out_data_url}}],
                        }
                    }
                ]
            },
        )
    )
    input_png = b"\x89PNG\r\n\x1a\noriginal"
    r = client_with_openrouter.post(
        "/v1/images/edits",
        headers=HEADERS,
        files={"image": ("in.png", input_png, "image/png")},
        data={
            "model": "or/google/gemini-2.5-flash-image",
            "prompt": "make her hair blue",
        },
    )
    assert r.status_code == 200, r.text

    # Outgoing chat body should have the input image as a data URL in the
    # message content array, alongside the text prompt.
    sent = json.loads(chat_route.calls.last.request.read())
    content = sent["messages"][0]["content"]
    assert isinstance(content, list)
    # text + image_url parts
    parts_by_type = {p["type"]: p for p in content}
    assert parts_by_type["text"]["text"] == "make her hair blue"
    img_url = parts_by_type["image_url"]["image_url"]["url"]
    assert img_url.startswith("data:image/png;base64,")
    # The base64 payload should round-trip back to the original bytes
    _, _, payload = img_url.partition(",")
    assert base64.b64decode(payload) == input_png


@respx.mock
def test_images_edits_sends_multiple_inputs(
    client_with_openrouter: TestClient,
) -> None:
    """Multiple reference images each become their own image_url part in the
    chat message content, in order — none are dropped."""
    output_png = b"\x89PNG\r\n\x1a\nedit-output"
    out_data_url = "data:image/png;base64," + base64.b64encode(output_png).decode()
    chat_route = respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Done.",
                            "images": [{"image_url": {"url": out_data_url}}],
                        }
                    }
                ]
            },
        )
    )
    first_png = b"\x89PNG\r\n\x1a\nfirst"
    second_png = b"\x89PNG\r\n\x1a\nsecond"
    r = client_with_openrouter.post(
        "/v1/images/edits",
        headers=HEADERS,
        files=[
            ("image", ("first.png", first_png, "image/png")),
            ("image", ("second.png", second_png, "image/png")),
        ],
        data={"model": "or/google/gemini-2.5-flash-image", "prompt": "combine these"},
    )
    assert r.status_code == 200, r.text

    sent = json.loads(chat_route.calls.last.request.read())
    content = sent["messages"][0]["content"]
    image_parts = [p for p in content if p["type"] == "image_url"]
    assert len(image_parts) == 2
    payloads = [base64.b64decode(p["image_url"]["url"].partition(",")[2]) for p in image_parts]
    assert payloads == [first_png, second_png]


@respx.mock
def test_images_generations_empty_images_array_surfaces_error(
    client_with_openrouter: TestClient,
) -> None:
    """When a model returns text only (no images array, or empty), the
    bridge should fail loudly rather than handing back an empty success."""
    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I cannot generate images.",
                        }
                    }
                ]
            },
        )
    )
    r = client_with_openrouter.post(
        "/v1/images/generations",
        headers=HEADERS,
        json={"model": "or/openai/gpt-4o", "prompt": "x"},
    )
    # Bad-shape upstream response surfaces as a non-2xx. The exact code is
    # whatever mapping the bridge's error handler picks; we just don't want
    # a silent success.
    assert r.status_code >= 400
    assert "image" in r.text.lower()


@respx.mock
def test_chat_completion_upstream_error_passes_through(
    client_with_openrouter: TestClient,
) -> None:
    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        return_value=httpx.Response(
            429, json={"error": {"message": "rate limited", "type": "rate_limit"}}
        )
    )
    r = client_with_openrouter.post(
        "/v1/chat/completions",
        headers=HEADERS,
        json={
            "model": "or/openai/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code >= 400


def test_openrouter_provider_loads_from_toml(tmp_path: Path) -> None:
    """Round-trip the TOML schema for an OpenRouter provider block."""
    from openai_api_bridge.config import (
        OpenRouterProviderConfig,
        load_providers,
    )

    cfg = tmp_path / "config.toml"
    cfg.write_text(
        textwrap.dedent("""
		[[providers]]
		id = "or"
		backend = "openrouter"
		api_token_env = "TEST_OR_TOKEN"
	""")
    )
    providers = load_providers(cfg)
    by_id = {p.id: p for p in providers.providers}
    assert isinstance(by_id["or"], OpenRouterProviderConfig)
    assert by_id["or"].base_url == "https://openrouter.ai/api"


@respx.mock
def test_models_expose_capabilities_from_architecture(
    client_with_openrouter: TestClient,
) -> None:
    """OpenRouter states input_modalities alongside the output ones
    classify_kind already reads, so a vision model can be told from a
    text-only one without guessing."""
    respx.get(f"{UPSTREAM}/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "vendor/vision-chat",
                        "architecture": {
                            "input_modalities": ["text", "image"],
                            "output_modalities": ["text"],
                        },
                    },
                    {
                        "id": "vendor/text-chat",
                        "architecture": {
                            "input_modalities": ["text"],
                            "output_modalities": ["text"],
                        },
                    },
                    {
                        "id": "vendor/img",
                        "architecture": {
                            "input_modalities": ["text", "image"],
                            "output_modalities": ["image"],
                        },
                    },
                ]
            },
        )
    )
    r = client_with_openrouter.get("/v1/models", headers=HEADERS)
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()["data"]}
    # A chat model's output modality is text, not its `kind`.
    assert by_id["or/vendor/vision-chat"]["capabilities"] == ["text-to-text", "image-to-text"]
    assert by_id["or/vendor/text-chat"]["capabilities"] == ["text-to-text"]
    assert by_id["or/vendor/img"]["capabilities"] == ["text-to-image", "image-to-image"]


@respx.mock
def test_model_catalog_is_cached(client_with_openrouter: TestClient) -> None:
    route = respx.get(f"{UPSTREAM}/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "vendor/img",
                        "architecture": {
                            "input_modalities": ["text"],
                            "output_modalities": ["image"],
                        },
                    }
                ]
            },
        )
    )
    for _ in range(3):
        assert client_with_openrouter.get("/v1/models", headers=HEADERS).status_code == 200
    assert route.call_count == 1
