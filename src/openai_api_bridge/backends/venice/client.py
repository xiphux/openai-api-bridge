"""Async HTTP client for the Venice.ai image API.

Wraps four endpoints:
  * ``GET  /api/v1/models?type=image`` — text-to-image models
  * ``GET  /api/v1/models?type=inpaint`` — image-to-image ("-edit") models
  * ``POST /api/v1/image/generate``    — synchronous text-to-image
  * ``POST /api/v1/image/edit``        — synchronous image-to-image (img2img)

Venice's image API is *not* OpenAI-shaped. ``/image/generate`` uses
``width``/``height`` ints (not ``size`` strings), exposes ``steps``/``cfg_scale``
explicitly, and returns base64-encoded image data under ``data["images"][0]``.
``/image/edit`` is different again: it's multipart (``image`` + ``prompt`` +
``model``) and returns the edited image as *raw binary*, not base64 JSON. The
bridge translates both shapes for OpenAI clients. (Note: img2img lives on the
dedicated ``/image/edit`` endpoint — the older ``inpaint`` flag on
``/image/generate`` was deprecated and disabled in May 2025.)
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from ...errors import UpstreamError
from ...util.http import parse_json

log = logging.getLogger(__name__)


class VeniceClient:
    def __init__(self, *, base_url: str, api_token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = api_token
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_image_models(self, model_type: str = "image") -> list[dict[str, Any]]:
        """List models of a given Venice type.

        Venice files text-to-image under ``image`` and image-to-image under
        ``inpaint`` — the edit models are a separate listing, not a flag on the
        generate ones.
        """
        try:
            response = await self._client.get(
                f"{self.base_url}/api/v1/models", params={"type": model_type}
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise UpstreamError(
                f"Venice /models returned {e.response.status_code}: {e.response.text[:300]}"
            ) from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"Venice /models failed: {e}") from e
        body = parse_json(response, "Venice /models")
        return list(body.get("data", []))

    async def generate_image(
        self,
        *,
        model: str,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
    ) -> bytes:
        """Generate one image and return the decoded bytes (PNG)."""
        payload = {
            "model": model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "hide_watermark": True,
            "return_binary": False,
            "safe_mode": False,
        }
        try:
            response = await self._client.post(
                f"{self.base_url}/api/v1/image/generate", json=payload
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise UpstreamError(
                f"Venice /image/generate returned {e.response.status_code}: {e.response.text[:300]}"
            ) from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"Venice /image/generate failed: {e}") from e

        body = parse_json(response, "Venice /image/generate")
        images = body.get("images") or []
        if not images:
            raise UpstreamError(f"Venice response contained no images: {str(body)[:200]}")
        try:
            return base64.b64decode(images[0])
        except (ValueError, TypeError) as e:
            raise UpstreamError(f"Venice returned undecodable base64: {e}") from e

    async def edit_image(
        self,
        *,
        model: str,
        prompt: str,
        image: bytes,
        image_content_type: str,
    ) -> tuple[bytes, str]:
        """Edit one image. Returns ``(bytes, content_type)``.

        Unlike ``/image/generate`` (base64 JSON), ``/image/edit`` is multipart
        and streams the edited image back as raw binary, so we read the bytes
        and the response's own content-type directly.
        """
        files = {"image": (_filename_for(image_content_type), image, image_content_type)}
        # safe_mode mirrors the generate path (operator opted out of Venice's
        # content filter there); the form encodes booleans as strings.
        data = {"model": model, "prompt": prompt, "safe_mode": "false"}
        try:
            response = await self._client.post(
                f"{self.base_url}/api/v1/image/edit", data=data, files=files
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise UpstreamError(
                f"Venice /image/edit returned {e.response.status_code}: {e.response.text[:300]}"
            ) from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"Venice /image/edit failed: {e}") from e

        content = response.content
        if not content:
            raise UpstreamError("Venice /image/edit returned an empty body")
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        # Venice should echo the output format (png/jpeg/webp); fall back to
        # PNG if the header is missing or non-image.
        if not content_type.startswith("image/"):
            content_type = "image/png"
        return content, content_type


def _filename_for(content_type: str) -> str:
    """Filename component for the multipart upload. Venice infers the input
    format from the extension when the content-type is generic."""
    ct = content_type.lower()
    if ct in ("image/jpeg", "image/jpg"):
        return "image.jpg"
    if ct == "image/webp":
        return "image.webp"
    if ct == "image/gif":
        return "image.gif"
    return "image.png"
