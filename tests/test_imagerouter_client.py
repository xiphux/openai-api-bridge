"""ImageRouterClient request shapes and response parsing, at the client.

test_imagerouter_integration.py drives the happy paths through the app; what it
leaves out is how ``size`` / ``seconds`` / a reference image change the request
ImageRouter receives, and what happens when the response envelope is malformed.
Those are pinned here against the client directly, with respx capturing the
outgoing request.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx  # respx authors responses in httpx v1 types; see conftest.py
import pytest
import respx

from openai_api_bridge.backends.base import InputImage
from openai_api_bridge.backends.imagerouter.client import ImageRouterClient
from openai_api_bridge.errors import InvalidRequest, UpstreamError

BASE = "https://api.imagerouter.io"
VIDEOS = f"{BASE}/v1/openai/videos/generations"
EDITS = f"{BASE}/v1/openai/images/edits"
OK = {"data": [{"url": "https://storage.imagerouter.io/out.mp4"}]}


@pytest.fixture
async def client() -> AsyncIterator[ImageRouterClient]:
    c = ImageRouterClient(base_url=BASE, api_token="ir-token")
    yield c
    await c.aclose()


def _form_fields(request: Any) -> dict[str, str]:
    """Plain (non-file) fields of a multipart body, by name."""
    fields: dict[str, str] = {}
    for part in request.content.split(
        b"--" + request.headers["content-type"].split("boundary=")[1].encode()
    ):
        head, _, value = part.partition(b"\r\n\r\n")
        if b"filename=" in head or b'name="' not in head:
            continue
        name = head.split(b'name="', 1)[1].split(b'"', 1)[0].decode()
        fields[name] = value.rstrip(b"\r\n").decode()
    return fields


@respx.mock
async def test_text_to_video_sends_json_with_size_and_duration(
    client: ImageRouterClient,
) -> None:
    route = respx.post(VIDEOS).mock(return_value=httpx.Response(200, json=OK))

    url = await client.generate_video_url(
        model="kling/v2", prompt="an eagle", size="1280x720", seconds=5
    )

    assert url == "https://storage.imagerouter.io/out.mp4"
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer ir-token"
    assert json.loads(request.content) == {
        "model": "kling/v2",
        "prompt": "an eagle",
        "response_format": "url",
        "size": "1280x720",
        "duration": 5,
    }


@respx.mock
async def test_text_to_video_leaves_out_what_the_caller_left_out(
    client: ImageRouterClient,
) -> None:
    route = respx.post(VIDEOS).mock(return_value=httpx.Response(200, json=OK))

    await client.generate_video_url(model="kling/v2", prompt="an eagle")

    body = json.loads(route.calls.last.request.content)
    assert "size" not in body
    assert "duration" not in body


@respx.mock
async def test_image_to_video_goes_out_as_multipart_with_the_reference(
    client: ImageRouterClient,
) -> None:
    route = respx.post(VIDEOS).mock(return_value=httpx.Response(200, json=OK))

    await client.generate_video_url(
        model="kling/v2",
        prompt="make it move",
        size="720x720",
        seconds=3,
        input_reference=b"JPEGBYTES",
        input_reference_content_type="image/jpeg",
    )

    request = route.calls.last.request
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert b'name="image[]"; filename="image.jpg"' in request.content
    assert b"Content-Type: image/jpeg\r\n\r\nJPEGBYTES" in request.content
    assert _form_fields(request) == {
        "model": "kling/v2",
        "prompt": "make it move",
        "response_format": "url",
        "size": "720x720",
        "duration": "3",
    }


@respx.mock
async def test_image_to_video_reference_defaults_to_png(client: ImageRouterClient) -> None:
    route = respx.post(VIDEOS).mock(return_value=httpx.Response(200, json=OK))

    await client.generate_video_url(model="m", prompt="p", input_reference=b"PNGBYTES")

    assert b"Content-Type: image/png\r\n\r\nPNGBYTES" in route.calls.last.request.content


@respx.mock
async def test_edit_forwards_size_and_maps_a_rejection_to_a_client_error(
    client: ImageRouterClient,
) -> None:
    route = respx.post(EDITS).mock(
        return_value=httpx.Response(422, json={"error": "size not supported by this model"})
    )

    with pytest.raises(InvalidRequest, match="size not supported"):
        await client.edit_image_url(
            model="m",
            prompt="p",
            size="2048x2048",
            images=[InputImage(data=b"PNGBYTES", content_type="image/png")],
        )

    assert _form_fields(route.calls.last.request)["size"] == "2048x2048"


@pytest.mark.parametrize(
    ("body", "complaint"),
    [
        (["not", "an", "object"], "non-dict body"),
        ({"data": []}, "empty data array"),
        ({"data": "https://x"}, "empty data array"),
        ({"data": ["https://x"]}, "is not an object"),
        ({"data": [{"b64_json": "AAAA"}]}, "no usable url"),
        ({"data": [{"url": ""}]}, "no usable url"),
    ],
)
@respx.mock
async def test_a_malformed_envelope_is_an_upstream_error(
    client: ImageRouterClient, body: Any, complaint: str
) -> None:
    """A 200 that carries no usable URL has already been billed; it has to fail
    loudly as a 502, not turn into a KeyError the app reports as a 500."""
    respx.post(VIDEOS).mock(return_value=httpx.Response(200, json=body))

    with pytest.raises(UpstreamError, match=complaint) as exc:
        await client.generate_video_url(model="m", prompt="p")

    assert exc.value.status_code == 502
