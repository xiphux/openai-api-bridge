"""Async HTTP client for fal.ai.

fal exposes every model both as a queue endpoint (``queue.fal.run``, with
submit/poll/result) and a **synchronous** endpoint (``https://fal.run/{model_id}``)
that blocks until the result is ready and returns it inline. Image generation
finishes in seconds, so — like the ImageRouter backend — we use the synchronous
endpoint and treat the whole call as one long await. (Video, which can run for
minutes and is better served by the queue lifecycle, is intentionally out of
scope for this backend.)

Auth is a fal API key sent as ``Authorization: Key {token}`` (note: ``Key``,
not ``Bearer``). Successful responses carry an ``images`` array of hosted asset
descriptors; the caller fetches the bytes from the (public) ``fal.media`` URLs
separately, mirroring the URL-then-fetch pattern used elsewhere in the bridge.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from ...errors import UpstreamError

log = logging.getLogger(__name__)


# fal.run holds the connection open until generation completes. Tier-1 image
# models finish well inside a minute, but a busy queue or a 4K request can run
# longer, so the read budget is generous. Connect stays low — DNS/TLS is fast.
_DEFAULT_GENERATION_READ_TIMEOUT_S = 600.0
_FETCH_ASSET_TIMEOUT_S = 120.0


class FalClient:
    def __init__(self, *, base_url: str, api_token: str, request_timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self._auth_headers = {"Authorization": f"Key {api_token}"}
        self._client = httpx.AsyncClient(
            headers=self._auth_headers,
            timeout=httpx.Timeout(
                request_timeout_seconds or _DEFAULT_GENERATION_READ_TIMEOUT_S,
                connect=10.0,
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # --- inference -------------------------------------------------------

    async def run_image(self, model_id: str, body: dict[str, Any]) -> list[str]:
        """Run a fal image model synchronously and return the output image URLs.

        ``model_id`` is the fal model path (e.g.
        ``fal-ai/bytedance/seedream/v4/text-to-image``); it becomes the URL path
        against ``fal.run``. ``body`` is the model's native input schema, already
        assembled by the adapter (prompt, size, safety knobs, …). The caller
        fetches bytes for each returned URL via :meth:`fetch_asset`.
        """
        url = f"{self.base_url}/{model_id}"
        try:
            resp = await self._client.post(url, json=body)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise UpstreamError(
                f"fal {model_id} returned {e.response.status_code}: {e.response.text[:300]}"
            ) from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"fal {model_id} failed: {e}") from e
        return _extract_image_urls(resp.json(), model_id)

    # --- asset fetch -----------------------------------------------------

    async def fetch_asset(self, url: str) -> tuple[bytes, str]:
        """Download a generated asset by URL, returning ``(bytes, content_type)``.

        fal's output URLs (``*.fal.media``) are publicly accessible, so — as with
        the ImageRouter backend — we fetch them with a separate unauthenticated
        client to avoid attaching our fal key to a CDN request or tripping
        httpx's cross-origin header-stripping on redirects. Retries with
        exponential backoff cover the brief window where a just-minted URL isn't
        yet readable from storage.
        """
        max_attempts = 3
        base_delay = 1.0
        last_error: httpx.HTTPError | None = None
        resp: httpx.Response | None = None

        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        url,
                        timeout=_FETCH_ASSET_TIMEOUT_S,
                        follow_redirects=True,
                    )
                if (
                    resp is not None
                    and (resp.status_code in (401, 404) or resp.status_code >= 500)
                    and attempt < max_attempts - 1
                ):
                    delay = base_delay * (2**attempt)
                    log.warning(
                        f"fal asset fetch got {resp.status_code} for {url}, "
                        f"retrying in {delay}s (attempt {attempt + 1}/{max_attempts})"
                    )
                    await asyncio.sleep(delay)
                    continue
                if resp is not None:
                    resp.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                last_error = e
                if attempt < max_attempts - 1:
                    delay = base_delay * (2**attempt)
                    log.warning(
                        f"fal asset fetch failed for {url}, "
                        f"retrying in {delay}s (attempt {attempt + 1}/{max_attempts})"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise UpstreamError(
                    f"fal asset fetch returned {e.response.status_code} for {url} "
                    f"after {max_attempts} attempts"
                ) from e
            except httpx.HTTPError as e:
                last_error = e
                if attempt < max_attempts - 1:
                    delay = base_delay * (2**attempt)
                    log.warning(
                        f"fal asset fetch failed for {url}, "
                        f"retrying in {delay}s (attempt {attempt + 1}/{max_attempts})"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise UpstreamError(
                    f"fal asset fetch failed for {url} after {max_attempts} attempts: {e}"
                ) from e

        if resp is None:
            raise UpstreamError(
                f"fal asset fetch failed for {url} after {max_attempts} attempts"
            ) from last_error

        content_type = resp.headers.get("content-type", "application/octet-stream")
        content_type = content_type.split(";", 1)[0].strip()
        return resp.content, content_type


def _extract_image_urls(body: Any, model_id: str) -> list[str]:
    """Pull every ``images[].url`` out of a fal response.

    Raises UpstreamError if the envelope is malformed or carries no image —
    including fal's ``{"detail": ...}`` validation-error shape, so a rejected
    request surfaces as a clean upstream error rather than an empty result.
    """
    if not isinstance(body, dict):
        raise UpstreamError(f"fal {model_id} returned non-dict body: {str(body)[:200]}")
    images = body.get("images")
    if not isinstance(images, list) or not images:
        raise UpstreamError(f"fal {model_id} returned no images: {str(body)[:300]}")
    urls: list[str] = []
    for img in images:
        if not isinstance(img, dict):
            continue
        url = img.get("url")
        if isinstance(url, str) and url:
            urls.append(url)
    if not urls:
        raise UpstreamError(f"fal {model_id} images carried no usable url: {str(images)[:300]}")
    return urls
