"""Async HTTP client for one ComfyUI instance.

Wraps the four endpoints we use:
  * ``POST /upload/image`` — multipart upload, returns assigned filename
  * ``POST /prompt``       — submit a workflow, returns prompt_id
  * ``GET  /history/{id}`` — poll for completion
  * ``GET  /view``         — download generated media
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import time
import uuid
from typing import Any

import httpx

from ...errors import GenerationTimeout, UpstreamError, WorkflowInvalid
from ...util.http import parse_json

log = logging.getLogger(__name__)

# Re-verify the prompt is still queued or running every N seconds. Long enough
# to not pummel ComfyUI's queue endpoint on busy generations; short enough that
# a dropped prompt is detected within ~couple minutes instead of the full
# poll_timeout_video_seconds (default 900s).
QUEUE_RECHECK_INTERVAL = 30.0

# A single negative queue check ("not in /queue, not in /history") isn't
# enough to declare the prompt dropped. ComfyUI's /queue endpoint can briefly
# misreport during heavy execution — kjnodes preview-node transitions,
# queue-lock contention while a worker is mid-step, restart-then-recover
# transients, etc. Require this many consecutive misses (each separated by
# QUEUE_RECHECK_INTERVAL) before we give up on the prompt. With the default
# 30s interval that's a ~90s tolerance window — plenty for transient state,
# still much better than waiting out the full generation timeout.
QUEUE_MISS_THRESHOLD = 3


class ComfyUIClient:
    def __init__(
        self,
        *,
        base_url: str,
        poll_interval_seconds: float = 1.0,
        request_timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval_seconds
        self.request_timeout = request_timeout_seconds
        self._client = httpx.AsyncClient()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def upload_image(self, image_data: bytes, content_type: str) -> str:
        """Upload an image; return ComfyUI's assigned filename."""
        ext = mimetypes.guess_extension(content_type) or ".png"
        filename = f"bridge_{uuid.uuid4().hex[:12]}{ext}"
        try:
            response = await self._client.post(
                f"{self.base_url}/upload/image",
                files={"image": (filename, image_data, content_type)},
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            result = parse_json(response, "ComfyUI image upload")
            return str(result.get("name", filename))
        except httpx.HTTPStatusError as e:
            raise UpstreamError(f"ComfyUI image upload returned {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"ComfyUI image upload failed: {e}") from e

    async def submit_prompt(self, workflow: dict[str, Any]) -> str:
        """Submit a workflow; return the prompt_id.

        We always send a fresh ``client_id``. ComfyUI's
        ``server.last_node_id`` is only populated when ``client_id`` is
        non-None (see ComfyUI's execution.py:483). Some preview-producing
        custom nodes (notably kjnodes' ``LTX2SamplingPreviewOverride``) crash
        with ``'NoneType' object has no attribute 'encode'`` if last_node_id
        is None, so this is a correctness requirement, not just hygiene.
        """
        try:
            response = await self._client.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow, "client_id": uuid.uuid4().hex},
                timeout=self.request_timeout,
            )
        except httpx.HTTPError as e:
            raise UpstreamError(f"ComfyUI /prompt failed: {e}") from e

        if response.status_code == 200:
            try:
                result = response.json()
            except ValueError as e:
                raise UpstreamError(
                    f"ComfyUI returned non-JSON 200: {response.text[:200]!r}"
                ) from e
            if "error" in result:
                raise WorkflowInvalid(f"ComfyUI rejected workflow: {result['error']}")
            prompt_id = result.get("prompt_id")
            if not prompt_id:
                raise UpstreamError(f"ComfyUI /prompt response missing prompt_id: {result!r}")
            return str(prompt_id)
        if 400 <= response.status_code < 500:
            raise WorkflowInvalid(f"ComfyUI returned {response.status_code}: {response.text[:500]}")
        raise UpstreamError(f"ComfyUI returned {response.status_code}: {response.text[:500]}")

    async def poll_completion(self, prompt_id: str, *, timeout_seconds: float) -> dict[str, Any]:
        """Poll ``/history/{prompt_id}`` until the entry appears or we time out.

        Per-request timeout is generous (30s) because ComfyUI's web thread can
        be CPU-starved during heavy generation. Transient network failures are
        treated as "not ready yet" — the overall ``timeout_seconds`` is what
        actually bounds us.

        Every ``QUEUE_RECHECK_INTERVAL`` seconds we also verify the prompt is
        still in ComfyUI's queue (running or pending). If history doesn't have
        it AND the queue doesn't either for ``QUEUE_MISS_THRESHOLD`` consecutive
        checks, ComfyUI has dropped the prompt — most commonly because the
        upstream restarted or a worker crashed mid-execution. Fail in that case
        instead of waiting out ``timeout_seconds`` (which used to mean we'd
        hold the bridge's video semaphore for up to 15 minutes per dropped job,
        eventually starving all subsequent video requests).

        The multi-miss threshold tolerates transient queue-state weirdness
        during heavy generation (custom-node transitions, queue-lock
        contention) without false-positive-failing healthy long-running jobs.
        """
        start = time.time()
        last_queue_check = 0.0
        queue_miss_streak = 0
        per_request_timeout = 30.0
        history_url = f"{self.base_url}/history/{prompt_id}"
        queue_url = f"{self.base_url}/queue"
        while True:
            elapsed = time.time() - start
            if elapsed > timeout_seconds:
                raise GenerationTimeout(
                    f"ComfyUI generation timed out after {timeout_seconds:.0f}s"
                )
            try:
                response = await self._client.get(history_url, timeout=per_request_timeout)
                if response.status_code == 200:
                    # A non-JSON 200 here is a transient upstream glitch, not a
                    # reason to fail the job — the loop retries.
                    history = parse_json(response, "ComfyUI /history")
                    if prompt_id in history:
                        return dict(history[prompt_id])
            except (
                httpx.ReadTimeout,
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                UpstreamError,  # includes a non-JSON 200 from a proxy hiccup
            ) as e:
                log.debug("Transient history poll error (will retry): %s", type(e).__name__)

            if elapsed - last_queue_check > QUEUE_RECHECK_INTERVAL:
                last_queue_check = elapsed
                try:
                    tracked = await self._is_prompt_tracked(
                        queue_url, prompt_id, per_request_timeout
                    )
                except (
                    httpx.ReadTimeout,
                    httpx.ConnectError,
                    httpx.RemoteProtocolError,
                ) as e:
                    # Don't fail just because the queue check itself was flaky;
                    # the main loop will retry on the next interval. Don't
                    # update the miss streak either — a network blip isn't a
                    # signal about whether the prompt is alive.
                    log.debug("Queue recheck failed (will retry): %s", type(e).__name__)
                else:
                    if tracked:
                        queue_miss_streak = 0
                    else:
                        queue_miss_streak += 1
                        log.info(
                            "ComfyUI prompt %s missing from /queue (%d/%d misses)",
                            prompt_id,
                            queue_miss_streak,
                            QUEUE_MISS_THRESHOLD,
                        )
                        if queue_miss_streak >= QUEUE_MISS_THRESHOLD:
                            raise UpstreamError(
                                f"ComfyUI dropped prompt {prompt_id}: missing from "
                                f"history and queue for {QUEUE_MISS_THRESHOLD} "
                                f"consecutive checks (~{QUEUE_MISS_THRESHOLD * QUEUE_RECHECK_INTERVAL:.0f}s). "
                                "The upstream likely restarted or a worker crashed "
                                "mid-execution."
                            )

            await asyncio.sleep(self.poll_interval)

    async def _is_prompt_tracked(self, queue_url: str, prompt_id: str, timeout: float) -> bool:
        """Return True if ``prompt_id`` is in ComfyUI's running or pending queue.

        ComfyUI's /queue response shape is:
          ``{"queue_running": [[prio, prompt_id, prompt, extra], ...],
              "queue_pending": [[...], ...]}``
        On any non-200 / non-JSON / unexpected shape we err on the side of
        "still tracked" — we'd rather wait out the overall timeout than abort
        a healthy job because of one weird queue snapshot.
        """
        response = await self._client.get(queue_url, timeout=timeout)
        if response.status_code != 200:
            return True
        try:
            queue = response.json()
        except ValueError:
            return True
        for key in ("queue_running", "queue_pending"):
            for item in queue.get(key) or []:
                if isinstance(item, list) and len(item) >= 2 and item[1] == prompt_id:
                    return True
        return False

    async def retrieve_media(
        self, history_entry: dict[str, Any], *, output_type: str
    ) -> tuple[bytes, str]:
        """Locate the generated media in a history entry and download it.

        Output keys vary by node type:
          * ``SaveImage`` / ``SaveVideo`` → ``images``
          * ``VHS_VideoCombine``          → ``gifs`` (legacy from animated GIF
            support; covers mp4/webm too)

        Prefer ``type="output"`` over ``type="temp"`` (preview frames).
        """
        outputs = history_entry.get("outputs", {})
        default_content_type = "video/mp4" if output_type == "video" else "image/png"

        candidates: list[dict[str, Any]] = []
        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            for key in ("images", "gifs"):
                items = node_output.get(key) or []
                if items and isinstance(items[0], dict) and items[0].get("filename"):
                    candidates.append(items[0])

        if not candidates:
            raise UpstreamError("ComfyUI history entry contained no output media")

        candidates.sort(key=lambda x: 0 if x.get("type") == "output" else 1)
        media = candidates[0]
        params = {
            "filename": media["filename"],
            "subfolder": media.get("subfolder", ""),
            "type": media.get("type", "output"),
        }
        try:
            response = await self._client.get(
                f"{self.base_url}/view",
                params=params,
                timeout=60.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise UpstreamError(f"ComfyUI /view fetch failed: {e}") from e

        content_type = (
            response.headers.get("content-type", default_content_type).split(";")[0].strip()
            or default_content_type
        )
        return response.content, content_type
