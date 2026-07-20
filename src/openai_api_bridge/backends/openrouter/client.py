"""Image-via-chat translation helpers for OpenRouter.

OpenRouter's chat/embedding surface is straight OpenAI-compatible and we
reuse ``OpenAIClient`` directly for it. Where OpenRouter diverges is
image generation: the response carries images in a non-standard
``choices[0].message.images`` array (each entry: ``{image_url: {url: ...}}``
with the URL almost always being an inline base64 data URL). This module
holds the small bit of glue that pulls bytes out of that envelope.
"""

from __future__ import annotations

import base64
import binascii
import logging
from typing import Any

import httpx

from ...errors import UpstreamError

log = logging.getLogger(__name__)


# Modest ceiling; OpenRouter's data URLs are typically <2MB even for
# high-quality outputs. If an OpenRouter image model ever exceeds this, the
# bound trims a runaway payload without affecting normal use.
_MAX_IMAGE_BYTES = 50 * 1024 * 1024


def extract_image_data_urls(response: dict[str, Any]) -> list[str]:
    """Pull the data: URLs (or hosted URLs) out of an OpenRouter chat
    completion response.

    Raises UpstreamError when the response shape doesn't carry any usable
    image — covers both "model returned text only" and "envelope is
    malformed."
    """
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise UpstreamError(
            "OpenRouter response has no choices array; model may have returned an error"
        )
    first = choices[0]
    if not isinstance(first, dict):
        raise UpstreamError("OpenRouter response choices[0] is not an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise UpstreamError("OpenRouter response choices[0].message is missing or malformed")
    images = message.get("images") or []
    if not isinstance(images, list) or not images:
        # Model produced text-only — not a transport error, but no image to
        # return. Surface as upstream so the bridge's caller sees a clean
        # failure rather than an empty success.
        raise UpstreamError(
            "OpenRouter model did not return any images. Check that the model "
            "supports image output and that the prompt asks for image content."
        )

    urls: list[str] = []
    for img in images:
        if not isinstance(img, dict):
            continue
        # OpenRouter's documented shape: {image_url: {url: "..."}}.
        # Tolerate a flat ``url`` for future-compat.
        image_url = img.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else img.get("url")
        if isinstance(url, str) and url:
            urls.append(url)
    if not urls:
        raise UpstreamError("OpenRouter response carried an images array but no usable URLs")
    return urls


async def fetch_image_bytes(url: str, http_client: httpx.AsyncClient) -> tuple[bytes, str]:
    """Resolve a URL from extract_image_data_urls into raw bytes.

    Data URLs are decoded inline; HTTP(S) URLs are fetched. Returns
    ``(bytes, content_type)`` where content_type defaults to
    ``application/octet-stream`` if the upstream doesn't hint otherwise.
    """
    if url.startswith("data:"):
        # Format: ``data:image/png;base64,<base64>``
        # Tolerate missing media-type ("data:;base64,...") with a sensible default.
        header, _, payload = url.partition(",")
        if not header.endswith("base64"):
            raise UpstreamError("OpenRouter returned a non-base64 data URL — unsupported encoding")
        # Strip the leading "data:" and the trailing ";base64"
        media_type = header[len("data:") : -len(";base64")]
        if not media_type:
            media_type = "image/png"
        try:
            data = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as e:
            raise UpstreamError(f"OpenRouter data URL contained undecodable base64: {e}") from e
        if len(data) > _MAX_IMAGE_BYTES:
            raise UpstreamError(
                f"OpenRouter image exceeded size cap ({len(data)} > {_MAX_IMAGE_BYTES} bytes)"
            )
        return data, media_type

    # HTTP fetch path — OpenRouter occasionally hosts large outputs on a CDN
    # rather than inlining them.
    try:
        resp = await http_client.get(url, timeout=120.0, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise UpstreamError(
            f"OpenRouter image fetch returned {e.response.status_code} for {url}"
        ) from e
    except httpx.HTTPError as e:
        raise UpstreamError(f"OpenRouter image fetch failed for {url}: {e}") from e
    content_type = resp.headers.get("content-type", "application/octet-stream")
    content_type = content_type.split(";", 1)[0].strip()
    data = resp.content
    if len(data) > _MAX_IMAGE_BYTES:
        raise UpstreamError(
            f"OpenRouter image exceeded size cap ({len(data)} > {_MAX_IMAGE_BYTES} bytes)"
        )
    return data, content_type


def classify_kind(model: dict[str, Any]) -> str | None:
    """Map an OpenRouter ``/v1/models`` entry to a bridge ModelEntry kind.

    OpenRouter publishes ``architecture.output_modalities`` and
    ``architecture.input_modalities`` on each model — strict, source-of-
    truth metadata that's more reliable than substring matching on names.

    Returns one of: ``"image"``, ``"video"``, ``"chat"``, ``"embedding"``,
    or ``None`` if the model's output is something we don't surface (e.g.
    audio).
    """
    arch = model.get("architecture")
    if not isinstance(arch, dict):
        return None
    outputs = arch.get("output_modalities") or []
    if not isinstance(outputs, list):
        return None
    if "image" in outputs:
        return "image"
    if "video" in outputs:
        return "video"
    if "embedding" in outputs:
        return "embedding"
    if "text" in outputs:
        return "chat"
    return None
