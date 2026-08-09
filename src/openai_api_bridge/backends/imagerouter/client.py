"""Async HTTP client for imagerouter.io.

ImageRouter is *almost* OpenAI-compatible but the surface is split across two
base paths:

* ``GET  /v2/models``                       — model catalog (array of model
                                              objects with ``output`` and
                                              ``inputs`` capability metadata)
* ``POST /v1/openai/images/generations``    — t2i, JSON body
* ``POST /v1/openai/images/edits``          — i2i, multipart
* ``POST /v1/openai/videos/generations``    — video, sync (NOT OpenAI's
                                              async /v1/videos shape)

A single ``base_url`` of ``https://api.imagerouter.io`` lets us reach both
the model listing and the inference endpoints with consistent prefixing
inside this client.

We use the v2 model catalog rather than v1: it's the only one documented
on the user-facing docs site (v1 lingers only in the auto-generated
openapi.yaml), the shape is cleaner (array vs id-keyed dict), and it
supports server-side filtering by ``outputType`` for efficiency.

All generation calls use ``response_format=url`` so the response is just a
JSON envelope ``{"data": [{"url": "..."}]}``; the caller fetches bytes
separately. URL-format avoids inflating the response body with ~5MB of
base64 per image; the trade-off is one extra round trip per asset, which
is fine since ImageRouter's CDN is fast.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ...errors import UpstreamError
from ...util.http import fetch_asset_with_retry, parse_json, raise_for_upstream_status
from ...util.media import image_filename
from ..base import InputImage

log = logging.getLogger(__name__)


# ImageRouter generation can take a while — diffusion models on the slower
# tier, or video on any tier, easily run past 60s. Default httpx timeouts
# (5s read) would cut these short. The connect timeout stays low because
# DNS / TLS handshake should always be fast; only the read side needs the
# generous budget.
_DEFAULT_GENERATION_READ_TIMEOUT_S = 600.0
_DEFAULT_VIDEO_READ_TIMEOUT_S = 1800.0


class ImageRouterClient:
    def __init__(
        self, *, base_url: str, api_token: str, max_asset_bytes: int | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_asset_bytes = max_asset_bytes
        self._auth_headers = {"Authorization": f"Bearer {api_token}"}
        self._client = httpx.AsyncClient(
            headers=self._auth_headers,
            timeout=httpx.Timeout(_DEFAULT_GENERATION_READ_TIMEOUT_S, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # --- model catalog ---------------------------------------------------

    async def list_models(self) -> list[Any]:
        """Returns ImageRouter's raw model catalog: an array of model objects.

        Each entry is *expected* to carry an ``id`` field and an ``output``
        array listing the modalities the model produces (e.g. ``["image"]``
        or ``["video"]``), but this is unvalidated upstream JSON, so the
        element type is Any rather than a dict we haven't checked. The
        adapter does that checking and translates the result into the
        bridge's flat ModelEntry list.
        """
        try:
            resp = await self._client.get(f"{self.base_url}/v2/models", timeout=30.0)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise_for_upstream_status(
                status=e.response.status_code,
                body=e.response.text[:300],
                provider="ImageRouter",
                endpoint="/v2/models",
            )
        except httpx.HTTPError as e:
            raise UpstreamError(f"ImageRouter /v2/models failed: {e}") from e
        body = parse_json(resp, "ImageRouter /v2/models")
        if not isinstance(body, list):
            raise UpstreamError(
                f"ImageRouter /v2/models returned non-array body: {str(body)[:200]}"
            )
        return body

    # --- image generation ------------------------------------------------

    async def generate_image_url(self, *, model: str, prompt: str, size: str | None = None) -> str:
        """Returns the URL of the generated image.

        Caller fetches the bytes via :meth:`fetch_asset`.
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "response_format": "url",
        }
        if size:
            payload["size"] = size
        try:
            resp = await self._client.post(
                f"{self.base_url}/v1/openai/images/generations", json=payload
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise_for_upstream_status(
                status=e.response.status_code,
                body=e.response.text[:300],
                provider="ImageRouter",
                endpoint="/images/generations",
            )
        except httpx.HTTPError as e:
            raise UpstreamError(f"ImageRouter /images/generations failed: {e}") from e
        return _extract_first_url(
            parse_json(resp, "ImageRouter /images/generations"), "/images/generations"
        )

    # --- image editing ---------------------------------------------------

    async def edit_image_url(
        self,
        *,
        model: str,
        prompt: str,
        images: list[InputImage],
        size: str | None = None,
    ) -> str:
        """Returns the URL of the edited image.

        ImageRouter's /images/edits accepts ``image[]`` for one-or-more
        input images. We forward every supplied reference image under that
        repeated field, in order. Models that only accept a single image
        surface an upstream error rather than the bridge silently dropping
        the extras.
        """
        data: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "response_format": "url",
        }
        if size:
            data["size"] = size
        # Repeat-field name mirrors the OWUI pipe's behavior; ImageRouter
        # treats ``image[]`` as the canonical multi-image multipart form.
        files = [
            ("image[]", (_filename_for(img.content_type), img.data, img.content_type))
            for img in images
        ]
        try:
            resp = await self._client.post(
                f"{self.base_url}/v1/openai/images/edits",
                data=data,
                files=files,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise_for_upstream_status(
                status=e.response.status_code,
                body=e.response.text[:300],
                provider="ImageRouter",
                endpoint="/images/edits",
            )
        except httpx.HTTPError as e:
            raise UpstreamError(f"ImageRouter /images/edits failed: {e}") from e
        return _extract_first_url(parse_json(resp, "ImageRouter /images/edits"), "/images/edits")

    # --- video generation ------------------------------------------------

    async def generate_video_url(
        self,
        *,
        model: str,
        prompt: str,
        size: str | None = None,
        seconds: float | None = None,
        input_reference: bytes | None = None,
        input_reference_content_type: str | None = None,
    ) -> str:
        """Returns the URL of the generated video.

        ImageRouter's /videos/generations is synchronous — the HTTP request
        blocks until generation completes (can be minutes). The bridge's
        ``video_jobs`` runner already runs in the background and is happy
        to await this call; the long timeout below is what keeps httpx
        from giving up too early.

        With ``input_reference`` set, the request goes out as multipart
        (image-to-video). Otherwise it's plain JSON.
        """
        params: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "response_format": "url",
        }
        if size:
            params["size"] = size
        if seconds is not None:
            params["duration"] = seconds

        try:
            if input_reference is not None:
                ct = input_reference_content_type or "image/png"
                filename = _filename_for(ct)
                resp = await self._client.post(
                    f"{self.base_url}/v1/openai/videos/generations",
                    data=params,
                    files=[("image[]", (filename, input_reference, ct))],
                    timeout=httpx.Timeout(_DEFAULT_VIDEO_READ_TIMEOUT_S, connect=10.0),
                )
            else:
                resp = await self._client.post(
                    f"{self.base_url}/v1/openai/videos/generations",
                    json=params,
                    timeout=httpx.Timeout(_DEFAULT_VIDEO_READ_TIMEOUT_S, connect=10.0),
                )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise_for_upstream_status(
                status=e.response.status_code,
                body=e.response.text[:300],
                provider="ImageRouter",
                endpoint="/videos/generations",
            )
        except httpx.HTTPError as e:
            raise UpstreamError(f"ImageRouter /videos/generations failed: {e}") from e
        return _extract_first_url(
            parse_json(resp, "ImageRouter /videos/generations"), "/videos/generations"
        )

    # --- asset fetch -----------------------------------------------------

    async def fetch_asset(self, url: str) -> tuple[bytes, str]:
        """Download a generated asset (image or video) by URL.

        Returns ``(bytes, content_type)`` — converting ImageRouter's
        ``response_format=url`` envelope into the byte payload the FileStore
        expects. ImageRouter's asset URLs (``storage.imagerouter.io/...``) are
        publicly accessible, so the shared helper fetches them unauthenticated
        with retry/backoff, streaming the body and abandoning it once it passes
        ``max_asset_bytes``. See
        :func:`~openai_api_bridge.util.http.fetch_asset_with_retry`.
        """
        return await fetch_asset_with_retry(
            url, provider_label="ImageRouter", max_bytes=self.max_asset_bytes
        )


def _extract_first_url(body: Any, label: str) -> str:
    """Pull the first ``data[0].url`` out of an ImageRouter response, raising
    an UpstreamError if the envelope is malformed."""
    if not isinstance(body, dict):
        raise UpstreamError(f"ImageRouter {label} returned non-dict body: {str(body)[:200]}")
    data = body.get("data")
    if not isinstance(data, list) or not data:
        raise UpstreamError(f"ImageRouter {label} returned empty data array: {str(body)[:200]}")
    first = data[0]
    if not isinstance(first, dict):
        raise UpstreamError(f"ImageRouter {label} data[0] is not an object: {str(first)[:200]}")
    url = first.get("url")
    if not isinstance(url, str) or not url:
        raise UpstreamError(f"ImageRouter {label} data[0] has no usable url: {str(first)[:200]}")
    return url


def _filename_for(content_type: str) -> str:
    """Filename component for multipart uploads. ImageRouter uses the
    extension to infer the format when the content-type header is generic;
    keeping it accurate avoids spurious "unsupported format" errors."""
    return image_filename(content_type, fallback="image")
