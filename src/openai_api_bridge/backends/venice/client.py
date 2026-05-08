"""Async HTTP client for the Venice.ai image API.

Wraps two endpoints:
  * ``GET  /api/v1/models?type=image`` — list available image models
  * ``POST /api/v1/image/generate``    — synchronous image generation

Venice's image-generation endpoint is *not* OpenAI-shaped: it uses
``width``/``height`` ints (not ``size`` strings), exposes ``steps``/``cfg_scale``
explicitly, and returns base64-encoded image data under ``data["images"][0]``.
The bridge translates those shapes for OpenAI clients.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from ...errors import UpstreamError

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

    async def list_image_models(self) -> list[dict[str, Any]]:
        try:
            response = await self._client.get(
                f"{self.base_url}/api/v1/models", params={"type": "image"}
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise UpstreamError(
                f"Venice /models returned {e.response.status_code}: "
                f"{e.response.text[:300]}"
            ) from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"Venice /models failed: {e}") from e
        body = response.json()
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
                f"Venice /image/generate returned {e.response.status_code}: "
                f"{e.response.text[:300]}"
            ) from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"Venice /image/generate failed: {e}") from e

        body = response.json()
        images = body.get("images") or []
        if not images:
            raise UpstreamError(
                f"Venice response contained no images: {str(body)[:200]}"
            )
        try:
            return base64.b64decode(images[0])
        except (ValueError, TypeError) as e:
            raise UpstreamError(f"Venice returned undecodable base64: {e}") from e
