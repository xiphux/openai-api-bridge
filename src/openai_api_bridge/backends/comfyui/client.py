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
import time
import uuid
from typing import Any

import httpx

from ...errors import GenerationTimeout, UpstreamAuthError, UpstreamError, WorkflowInvalid
from ...util.http import parse_json
from ...util.media import image_extension

log = logging.getLogger(__name__)

# Re-verify the prompt is still queued or running every N seconds. Long enough
# to not pummel ComfyUI's queue endpoint on busy generations; short enough that
# a dropped prompt is detected within ~couple minutes instead of the full
# poll_timeout_video_seconds (default 900s).
QUEUE_RECHECK_INTERVAL = 30.0

# Growth applied to the poll interval after each unproductive check. A flat
# interval is wrong at both ends of ComfyUI's range: fast enough for a
# 3-second SDXL render is far too fast for a 15-minute video, where it means
# ~900 requests at a web thread this module already notes is CPU-starved
# during generation — polling that hard slows the very generation it waits on.
#
# Ramping fixes the volume; the *ceiling* is what decides the cost, and it is
# passed per call because the two output types disagree about it. See
# poll_completion. The default below is the video-shaped one, used only when a
# caller doesn't pass its own; ceilings must stay well inside
# QUEUE_RECHECK_INTERVAL so the dropped-prompt detection keeps its cadence.
POLL_BACKOFF_FACTOR = 1.5
MAX_POLL_INTERVAL = 5.0

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
        poll_interval_seconds: float = 0.25,
        max_poll_interval_seconds: float = MAX_POLL_INTERVAL,
        request_timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval_seconds
        # Never below the starting interval: an operator who deliberately slows
        # polling down shouldn't have the ceiling speed it back up.
        self.max_poll_interval = max(max_poll_interval_seconds, poll_interval_seconds)
        self.request_timeout = request_timeout_seconds
        # Every other adapter configures its client explicitly; this one
        # inherited httpx's 5s default for anything not passing a per-call
        # timeout. That's a tight *connect* budget for a self-hosted box that
        # can be slow to accept while loading a model, and it left any future
        # call site that forgets an explicit timeout on a different budget
        # from the rest of the bridge.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout_seconds, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def upload_image(self, image_data: bytes, content_type: str) -> str:
        """Upload an image; return ComfyUI's assigned filename."""
        # image_extension, not a general media map: the content type traces
        # back to the caller's own multipart part, and this names a file
        # written into ComfyUI's input directory. See util.media.
        filename = f"bridge_{uuid.uuid4().hex[:12]}{image_extension(content_type)}"
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
        if response.status_code in (401, 403):
            # ComfyUI is unauthenticated itself, but it's commonly fronted by a
            # reverse proxy that isn't. Same rule as every other provider: the
            # rejection concerns our credential, so the body doesn't go to the
            # client, and it's permanent rather than a retriable blip.
            log.debug("ComfyUI auth failure on /prompt: %s", response.text[:300])
            raise UpstreamAuthError(
                f"ComfyUI rejected our credentials ({response.status_code}) on /prompt"
            )
        if 400 <= response.status_code < 500:
            raise WorkflowInvalid(f"ComfyUI returned {response.status_code}: {response.text[:500]}")
        raise UpstreamError(f"ComfyUI returned {response.status_code}: {response.text[:500]}")

    async def delete_queued(self, prompt_ids: list[str]) -> None:
        """Drop *pending* prompts from ComfyUI's queue. Raises on failure.

        Only helps prompts that haven't started executing. ComfyUI's only
        control for one already running is ``POST /interrupt``, which is
        global — it would kill whatever the box is currently rendering,
        including another client's job — so this deliberately doesn't touch
        it. A running prompt finishes and is discarded by the caller.
        """
        response = await self._client.post(
            f"{self.base_url}/queue",
            json={"delete": list(prompt_ids)},
            timeout=self.request_timeout,
        )
        response.raise_for_status()

    async def poll_completion(
        self,
        prompt_id: str,
        *,
        timeout_seconds: float,
        max_interval: float | None = None,
    ) -> dict[str, Any]:
        """Poll ``/history/{prompt_id}`` until the entry appears or we time out.

        The interval starts at ``poll_interval_seconds`` and eases out towards
        ``max_interval`` (see ``POLL_BACKOFF_FACTOR``), so a fast workflow is
        collected almost as soon as it finishes while a long one stops
        hammering the upstream for its whole duration.

        ``max_interval`` is per call rather than per client because one ceiling
        cannot serve both output types: an image is collected inside the
        caller's synchronous request, where detection lag is latency the user
        feels, while a video is collected by a background job where the same
        lag costs nobody anything and the request volume is what matters. The
        adapter picks it from the workflow; ``None`` falls back to the
        client-level default.

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
        # Never below the starting interval: an operator who deliberately
        # slows polling down shouldn't have a ceiling speed it back up.
        ceiling = (
            self.max_poll_interval
            if max_interval is None
            else max(max_interval, self.poll_interval)
        )
        interval = self.poll_interval
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

            await asyncio.sleep(interval)
            interval = min(interval * POLL_BACKOFF_FACTOR, ceiling)

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
