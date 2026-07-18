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

import logging
from typing import Any

import httpx

from ...errors import UpstreamError
from ...util.http import fetch_asset_with_retry

log = logging.getLogger(__name__)


# fal.run holds the connection open until generation completes. Tier-1 image
# models finish well inside a minute, but a busy queue or a 4K request can run
# longer, so the read budget is generous. Connect stays low — DNS/TLS is fast.
_DEFAULT_GENERATION_READ_TIMEOUT_S = 600.0


class FalClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_token: str,
        request_timeout_seconds: float,
        models_api_url: str = "https://api.fal.ai/v1/models",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.models_api_url = models_api_url
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

    # --- model catalog ---------------------------------------------------

    async def fetch_catalog(self, categories: list[str]) -> list[dict[str, Any]]:
        """List active models in the given fal categories.

        Paginates fal's model API per category and returns the raw entries
        (``{"endpoint_id": ..., "metadata": {...}}``). ``status=active`` keeps
        deprecated models out of the bridge's ``/v1/models`` listing — fal does
        still serve them from the catalog otherwise.

        No ``expand`` here, so the 10-item truncation that afflicts schema
        responses doesn't apply and full pages come back.
        """
        out: list[dict[str, Any]] = []
        for category in categories:
            cursor: str | None = None
            # Guard against a malformed cursor loop; 50 pages is ~5000 models,
            # far beyond any real category.
            for _ in range(50):
                params: list[tuple[str, str | int | float | bool | None]] = [
                    ("category", category),
                    ("status", "active"),
                    ("limit", 100),
                ]
                if cursor:
                    params.append(("cursor", cursor))
                try:
                    resp = await self._client.get(self.models_api_url, params=params, timeout=60.0)
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    raise UpstreamError(
                        f"fal model API returned {e.response.status_code} listing "
                        f"{category}: {e.response.text[:300]}"
                    ) from e
                except httpx.HTTPError as e:
                    raise UpstreamError(f"fal model API listing {category} failed: {e}") from e
                body = resp.json()
                if not isinstance(body, dict):
                    raise UpstreamError(f"fal model API returned non-dict body: {str(body)[:200]}")
                for entry in body.get("models") or []:
                    if isinstance(entry, dict) and isinstance(entry.get("endpoint_id"), str):
                        out.append(entry)
                cursor = body.get("next_cursor")
                if not body.get("has_more") or not cursor:
                    break
        return out

    # --- model schemas ---------------------------------------------------

    async def fetch_model_schemas(self, model_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch OpenAPI documents for the given fal models, keyed by model id.

        Uses fal's model API in "Find Mode" (``?endpoint_id=a&endpoint_id=b``)
        with ``expand=openapi-3.0`` to inline each model's schema. Auth is
        optional there but we send our key anyway — it raises the rate limit,
        and unauthenticated ``expand`` calls get 429'd readily.

        Note the batch size: while Find Mode accepts up to 50 ids, a response
        carrying expanded schemas is **silently truncated to 10** — ask for 14
        and you get 10 back with no error and no pagination hint. Chunking at
        10 avoids losing schemas; any id still missing afterwards is retried
        on its own before we give up on it.

        Models the API never returns are simply absent from the result; the
        caller decides what to do about them.
        """
        out: dict[str, dict[str, Any]] = {}
        chunk_size = 10
        for start in range(0, len(model_ids), chunk_size):
            out.update(await self._fetch_schema_batch(model_ids[start : start + chunk_size]))

        # Guard against the truncation cap moving: re-ask for stragglers one at
        # a time, where a single-model response can't be trimmed.
        missing = [mid for mid in model_ids if mid not in out]
        for mid in missing:
            out.update(await self._fetch_schema_batch([mid]))
        return out

    async def _fetch_schema_batch(self, chunk: list[str]) -> dict[str, dict[str, Any]]:
        params: list[tuple[str, str | int | float | bool | None]] = [
            ("endpoint_id", mid) for mid in chunk
        ]
        params.append(("expand", "openapi-3.0"))
        try:
            resp = await self._client.get(self.models_api_url, params=params, timeout=60.0)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise UpstreamError(
                f"fal model API returned {e.response.status_code}: {e.response.text[:300]}"
            ) from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"fal model API request failed: {e}") from e
        body = resp.json()
        if not isinstance(body, dict):
            raise UpstreamError(f"fal model API returned non-dict body: {str(body)[:200]}")
        out: dict[str, dict[str, Any]] = {}
        for entry in body.get("models") or []:
            if not isinstance(entry, dict):
                continue
            mid = entry.get("endpoint_id")
            spec = entry.get("openapi")
            if isinstance(mid, str) and isinstance(spec, dict):
                out[mid] = spec
        return out

    # --- asset fetch -----------------------------------------------------

    async def fetch_asset(self, url: str) -> tuple[bytes, str]:
        """Download a generated asset by URL, returning ``(bytes, content_type)``.

        fal's output URLs (``*.fal.media``) are publicly accessible; the shared
        helper fetches them unauthenticated with retry/backoff. See
        :func:`~openai_api_bridge.util.http.fetch_asset_with_retry`.
        """
        return await fetch_asset_with_retry(url, provider_label="fal")


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
