"""Async HTTP client for fal.ai.

fal exposes every model both as a queue endpoint (``queue.fal.run``, with
submit/poll/result) and a **synchronous** endpoint (``https://fal.run/{model_id}``)
that blocks until the result is ready and returns it inline. This client uses
both, picked by modality:

* **Images** run against the synchronous endpoint and are treated as one long
  await, like the ImageRouter backend — they finish in seconds.
* **Video** goes through the queue. A clip takes minutes, well past what
  fal.run will hold a connection open for, and the queue's request id gives the
  bridge something to record on the job row.

Auth is a fal API key sent as ``Authorization: Key {token}`` (note: ``Key``,
not ``Bearer``). Successful responses carry an ``images`` array of hosted asset
descriptors; the caller fetches the bytes from the (public) ``fal.media`` URLs
separately, mirroring the URL-then-fetch pattern used elsewhere in the bridge.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from ...errors import RateLimited, UnsupportedOperation, UpstreamAuthError, UpstreamError
from ...util.http import fetch_asset_with_retry, parse_json, raise_for_upstream_status

log = logging.getLogger(__name__)


# fal.run holds the connection open until generation completes. Tier-1 image
# models finish well inside a minute, but a busy queue or a 4K request can run
# longer, so the read budget is generous. Connect stays low — DNS/TLS is fast.
_DEFAULT_GENERATION_READ_TIMEOUT_S = 600.0


def _status_error(e: httpx.HTTPStatusError, what: str) -> UpstreamError:
    """Map a status from the *queue lifecycle* calls onto an error.

    The inference entry points (``run_image``, ``submit_queued``) don't use
    this — they go through ``util.http.raise_for_upstream_status`` like every
    other adapter, so a request fal rejects as malformed reaches the client
    as a 400 rather than a 502.

    Status polling, result fetch and cancel keep this deliberately blunter
    mapping, which returns ``UpstreamError`` for every 4xx that isn't auth or
    a rate limit. Those callers retry a bounded run of ``UpstreamError`` as
    "not ready yet", and a just-submitted job's status URL can genuinely 404
    for a moment before fal's queue has it — the same propagation race the
    shared asset fetcher retries. Sharpening these to InvalidRequest would
    turn that race into an immediately failed render.

    401/403 become UpstreamAuthError so callers can tell "bad key" (back off
    hard) from "fal is having a moment" (retry on a cooldown). 429 becomes
    RateLimited so the client is told to retry rather than that it sent a bad
    request; since RateLimited subclasses UpstreamError, the queue poller
    still treats it as the transient condition it is.
    """
    status = e.response.status_code
    body = e.response.text[:300]
    if status == 429:
        return RateLimited(f"fal {what} rate-limited the bridge: {body}")
    if status in (401, 403):
        # Deliberately without the body: this is fal's complaint about the
        # credential *we* sent, and the message goes straight to a client that
        # has no business seeing it. Providers have been known to quote the
        # offending token back. The body is still available at DEBUG.
        log.debug("fal auth failure on %s: %s", what, body)
        return UpstreamAuthError(f"fal rejected our credentials ({status}) on {what}")
    return UpstreamError(f"fal {what} returned {status}: {body}")


@dataclass(frozen=True, slots=True)
class QueuedRequest:
    """Handles for an in-flight fal queue job (submit -> status -> result)."""

    request_id: str
    status_url: str
    response_url: str
    cancel_url: str


class FalClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_token: str,
        request_timeout_seconds: float,
        models_api_url: str = "https://api.fal.ai/v1/models",
        queue_base_url: str = "https://queue.fal.run",
        store_payloads: bool = True,
        output_expiration_seconds: int | None = None,
        max_asset_bytes: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.models_api_url = models_api_url
        self.queue_base_url = queue_base_url.rstrip("/")
        self.max_asset_bytes = max_asset_bytes
        self._auth_headers = {"Authorization": f"Key {api_token}"}
        # Retention controls ride on the *inference* calls only — they say what
        # fal should do with this job's payload and output, so they're
        # meaningless on catalogue/schema reads.
        self._inference_headers: dict[str, str] = {}
        if not store_payloads:
            self._inference_headers["X-Fal-Store-IO"] = "0"
        if output_expiration_seconds is not None:
            self._inference_headers["X-Fal-Object-Lifecycle-Preference"] = json.dumps(
                {"expiration_duration_seconds": output_expiration_seconds}
            )
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
            resp = await self._client.post(url, json=body, headers=self._inference_headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise_for_upstream_status(
                status=e.response.status_code,
                body=e.response.text[:300],
                provider="fal",
                endpoint=model_id,
            )
        except httpx.HTTPError as e:
            raise UpstreamError(f"fal {model_id} failed: {e}") from e
        return _extract_image_urls(parse_json(resp, f"fal {model_id}"), model_id)

    # --- queued inference (video) ----------------------------------------

    async def submit_queued(self, model_id: str, body: dict[str, Any]) -> QueuedRequest:
        """Submit a job to fal's queue and return its handles.

        Video runs for minutes, well past what the synchronous ``fal.run``
        endpoint will hold open, so it goes through ``queue.fal.run``:
        submit → poll status → fetch result. The submit response carries the
        status/response URLs, so we use those verbatim rather than rebuilding
        them and guessing at fal's path layout.
        """
        url = f"{self.queue_base_url}/{model_id}"
        try:
            resp = await self._client.post(url, json=body, headers=self._inference_headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise_for_upstream_status(
                status=e.response.status_code,
                body=e.response.text[:300],
                provider="fal",
                endpoint=model_id,
            )
        except httpx.HTTPError as e:
            raise UpstreamError(f"fal {model_id} queue submit failed: {e}") from e
        body_json = parse_json(resp, f"fal {model_id} queue submit")
        if not isinstance(body_json, dict):
            raise UpstreamError(f"fal {model_id} queue submit returned non-dict body")
        request_id = body_json.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise UpstreamError(
                f"fal {model_id} queue submit returned no request_id: {str(body_json)[:200]}"
            )

        def link(key: str, fallback: str) -> str:
            value = body_json.get(key)
            return value if isinstance(value, str) and value else fallback

        base = f"{url}/requests/{request_id}"
        return QueuedRequest(
            request_id=request_id,
            status_url=link("status_url", f"{base}/status"),
            response_url=link("response_url", base),
            cancel_url=link("cancel_url", f"{base}/cancel"),
        )

    async def cancel_queued(self, job: QueuedRequest, *, model_id: str) -> None:
        """Ask fal to stop rendering a job we're no longer going to collect.

        Best-effort by nature: fal answers 202 CANCELLATION_REQUESTED, and a
        job already past the point of no return simply won't stop. Raises so
        the caller can log; the caller is always on an unwind path.
        """
        try:
            resp = await self._client.put(job.cancel_url, timeout=30.0)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise _status_error(e, f"{model_id} cancel") from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"fal {model_id} cancel failed: {e}") from e

    async def poll_queued(self, job: QueuedRequest, *, model_id: str) -> str:
        """Current queue status: ``IN_QUEUE``, ``IN_PROGRESS`` or ``COMPLETED``."""
        try:
            resp = await self._client.get(job.status_url, timeout=60.0)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise _status_error(e, f"{model_id} status") from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"fal {model_id} status poll failed: {e}") from e
        body = parse_json(resp, f"fal {model_id} status")
        status = body.get("status") if isinstance(body, dict) else None
        if not isinstance(status, str):
            raise UpstreamError(f"fal {model_id} status had no status field: {str(body)[:200]}")
        return status

    async def fetch_queued_result(self, job: QueuedRequest, *, model_id: str) -> dict[str, Any]:
        try:
            resp = await self._client.get(job.response_url, timeout=120.0)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise _status_error(e, f"{model_id} result") from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"fal {model_id} result fetch failed: {e}") from e
        body = parse_json(resp, f"fal {model_id} result")
        if not isinstance(body, dict):
            raise UpstreamError(f"fal {model_id} result was not an object: {str(body)[:200]}")
        return body

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
        # Categories are independent, so walk them concurrently. Serially this
        # is ~10-13 round trips on the first /v1/models after boot — all of it
        # holding the catalogue lock, with every other provider queued behind
        # it. Pages *within* a category stay sequential; they're cursor-chained.
        # return_exceptions so a failing category doesn't leave its siblings
        # running detached with unretrieved exceptions — gather propagates the
        # first error immediately but does not cancel the rest.
        results = await asyncio.gather(
            *(self._fetch_category(category) for category in categories),
            return_exceptions=True,
        )
        out: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, BaseException):
                raise result
            out.extend(result)
        return out

    async def _fetch_category(self, category: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
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
                raise _status_error(e, f"listing {category}") from e
            except httpx.HTTPError as e:
                raise UpstreamError(f"fal model API listing {category} failed: {e}") from e
            body = parse_json(resp, f"fal listing {category}")
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
        # a time, where a single-model response can't be trimmed. Only worth it
        # for a real batch — re-asking for a lone id would repeat a
        # byte-identical request that already came back without it.
        if len(model_ids) > 1:
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
            raise _status_error(e, "the model API") from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"fal model API request failed: {e}") from e
        body = parse_json(resp, "fal the model API")
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
        helper fetches them unauthenticated with retry/backoff, streaming the
        body and abandoning it once it passes ``max_asset_bytes``. See
        :func:`~openai_api_bridge.util.http.fetch_asset_with_retry`.
        """
        return await fetch_asset_with_retry(
            url, provider_label="fal", max_bytes=self.max_asset_bytes
        )


# Output keys fal uses for the modalities this backend doesn't implement. A
# response carrying one of these is a well-formed success for a video/audio/3d
# model, not an upstream fault — the caller pointed an image endpoint at a
# non-image model.
_NON_IMAGE_OUTPUT_KEYS = (
    "video",
    "audio",
    "model_mesh",
    "mesh",
    "video_url",
    "audio_url",
)


def extract_video_url(body: Any, model_id: str) -> str:
    """Pull the output video URL out of a fal queue result.

    fal video models return ``{"video": {"url": ...}}``; a few use a ``videos``
    array. Anything else is reported as an upstream fault rather than guessed at.
    """
    if not isinstance(body, dict):
        raise UpstreamError(f"fal {model_id} returned non-dict result: {str(body)[:200]}")
    candidates: list[Any] = []
    video = body.get("video")
    if video is not None:
        candidates.append(video)
    videos = body.get("videos")
    if isinstance(videos, list):
        candidates.extend(videos)
    for candidate in candidates:
        if isinstance(candidate, dict):
            url = candidate.get("url")
            if isinstance(url, str) and url:
                return url
        elif isinstance(candidate, str) and candidate:
            return candidate
    raise UpstreamError(f"fal {model_id} returned no video: {str(body)[:300]}")


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
        for key in _NON_IMAGE_OUTPUT_KEYS:
            if key in body:
                hint = " — use /v1/videos for video models" if key in ("video", "video_url") else ""
                raise UnsupportedOperation(
                    f"fal model {model_id!r} produced {key!r} output, not an image{hint}",
                    param="model",
                )
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
