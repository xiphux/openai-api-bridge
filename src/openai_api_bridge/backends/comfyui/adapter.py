"""ComfyUI Backend implementation."""

from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path

from ...config import ComfyUIProviderConfig
from ...errors import ImageRequired, ModelNotFound, UnsupportedOperation
from ...util.sizes import parse_size
from ..base import (
    Backend,
    GeneratedAsset,
    InputImage,
    ModelEntry,
    UpstreamIdCallback,
    make_capabilities,
)
from .client import ComfyUIClient
from .workflows import (
    WorkflowRecord,
    prepare_workflow,
    scan_workflows,
    seconds_to_frames,
)

log = logging.getLogger(__name__)

# Bounded because it runs while unwinding, often from a cancellation the
# caller is waiting on: DELETE /v1/videos/{id} doesn't return until this
# does. Kept short because the recall is a courtesy — losing it wastes GPU
# time on a self-hosted box, while a slow one stalls a user-visible cancel.
# The prompt ids are logged before the attempt, so a lapsed recall is still
# diagnosable.
_QUEUE_DISCARD_TIMEOUT_S = 2.0


def _dir_stamp(workflows_dir: Path) -> tuple[tuple[str, int, int], ...]:
    """A stat-only fingerprint of the workflow directory.

    Cheap enough to check per request, unlike a full scan: stat each JSON
    rather than reading and parsing it.
    """
    if not workflows_dir.is_dir():
        return ()
    entries: list[tuple[str, int, int]] = []
    for path in sorted(workflows_dir.glob("*.json")):
        try:
            st = path.stat()
        except OSError:  # vanished mid-scan; the next pass will settle it
            continue
        entries.append((path.name, st.st_mtime_ns, st.st_size))
    return tuple(entries)


def _scan_with_stamp(
    workflows_dir: Path,
) -> tuple[dict[str, WorkflowRecord], tuple[tuple[str, int, int], ...]]:
    """Scan the directory and fingerprint it, in one trip off the event loop.

    Stamped before the scan so a write landing mid-scan invalidates the
    result rather than being missed until the *next* change.
    """
    stamp = _dir_stamp(workflows_dir)
    return scan_workflows(workflows_dir), stamp


class ComfyUIBackend(Backend):
    def __init__(self, cfg: ComfyUIProviderConfig) -> None:
        self.cfg = cfg
        self.client = ComfyUIClient(
            base_url=cfg.url,
            poll_interval_seconds=cfg.poll_interval_seconds,
            max_poll_interval_seconds=cfg.max_poll_interval_seconds,
        )
        self._workflows: dict[str, WorkflowRecord] | None = None
        self._stamp: tuple[tuple[str, int, int], ...] | None = None

    async def aclose(self) -> None:
        await self.client.aclose()

    # --- discovery ---------------------------------------------------------

    async def _ensure_workflows(self) -> dict[str, WorkflowRecord]:
        """The workflow map, rescanned when the directory has actually changed.

        With ``cache_workflows = false`` this rescanned on every request —
        every /v1/models and every generation — and a scan reads and parses
        each ``.meta.json`` (plus the graph itself for any workflow needing
        output-type autodetection). With fifty workflows that's ~50-100 file
        reads and JSON parses per request, synchronously, on the event loop
        the whole bridge shares.

        A stat-only fingerprint tells us whether anything moved, which is the
        cheap version of the same guarantee, and the scan itself runs off the
        loop. Note ``prepare_workflow`` already re-reads the graph from disk
        per generation, so an edited *workflow* takes effect regardless of
        this cache; what the rescan buys is picking up edited or added
        *meta* files.
        """
        if self._workflows is not None:
            if self.cfg.cache_workflows:
                return self._workflows
            stamp = await asyncio.to_thread(_dir_stamp, self.cfg.workflows_dir)
            if stamp == self._stamp:
                return self._workflows

        records, stamp = await asyncio.to_thread(_scan_with_stamp, self.cfg.workflows_dir)
        self._workflows = records
        self._stamp = stamp
        return records

    async def _record_for(self, model_slug: str) -> WorkflowRecord:
        records = await self._ensure_workflows()
        record = records.get(model_slug)
        if record is None:
            raise ModelNotFound(
                f"Workflow {model_slug!r} not found in provider {self.cfg.id!r}",
                param="model",
            )
        return record

    async def list_models(self) -> list[ModelEntry]:
        records = await self._ensure_workflows()
        return [
            ModelEntry(
                id=r.slug,
                kind=r.output_type,
                display_name=r.display_name,
                prompt_style=r.meta.get("prompt_style"),
                prompt_hint=r.meta.get("prompt_hint"),
                # ``image_inputs`` is the same declaration edit_image and
                # generate_video gate on, so the listing can't disagree with
                # what a request will actually accept. Needs no new meta field.
                capabilities=make_capabilities(
                    ["text", "image"] if r.meta.get("image_inputs") else ["text"],
                    r.output_type,
                ),
            )
            for r in sorted(records.values(), key=lambda r: r.slug)
        ]

    # --- generation --------------------------------------------------------

    async def _submit_one(
        self,
        record: WorkflowRecord,
        *,
        prompt: str,
        size: str | None,
        image_filenames: list[str] | None = None,
        length: int | None = None,
        on_upstream_id: UpstreamIdCallback | None = None,
        rng: random.Random | None = None,
    ) -> str:
        """Queue one run and return ComfyUI's prompt_id."""
        width, height = parse_size(size)
        workflow = prepare_workflow(
            record,
            prompt_text=prompt,
            image_filenames=image_filenames,
            width=width or None,
            height=height or None,
            length=length,
            rng=rng,
        )
        prompt_id = await self.client.submit_prompt(workflow)
        if on_upstream_id is not None:
            await on_upstream_id(prompt_id)
        return prompt_id

    async def _discard_queued(self, prompt_ids: list[str]) -> None:
        """Give back prompts we queued but will never collect. Never raises.

        Logs *before* attempting the call: this runs while unwinding, often
        from a cancellation, so the request to ComfyUI may itself be
        interrupted. The operator should still learn which prompt ids were
        abandoned even when the recall doesn't get to happen.
        """
        log.warning(
            "Abandoning %d queued ComfyUI prompt(s) (%s); asking ComfyUI to drop any still pending",
            len(prompt_ids),
            ", ".join(prompt_ids),
        )
        try:
            await asyncio.wait_for(
                self.client.delete_queued(prompt_ids),
                timeout=_QUEUE_DISCARD_TIMEOUT_S,
            )
        except Exception as e:
            # Best-effort: they render uncollected, which costs GPU time on a
            # self-hosted box rather than money.
            log.warning("ComfyUI would not drop the abandoned prompts: %s", e)

    def _poll_timeout_for(self, record: WorkflowRecord) -> float:
        return (
            self.cfg.poll_timeout_video_seconds
            if record.output_type == "video"
            else self.cfg.poll_timeout_image_seconds
        )

    async def _collect_one(
        self,
        record: WorkflowRecord,
        prompt_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> GeneratedAsset:
        """Wait for a queued run to finish and download its output."""
        timeout = self._poll_timeout_for(record) if timeout_seconds is None else timeout_seconds
        history = await self.client.poll_completion(prompt_id, timeout_seconds=timeout)
        data, content_type = await self.client.retrieve_media(
            history, output_type=record.output_type
        )
        return GeneratedAsset(data=data, content_type=content_type, kind=record.output_type)

    async def _run_one(
        self,
        record: WorkflowRecord,
        *,
        prompt: str,
        size: str | None,
        image_filenames: list[str] | None = None,
        length: int | None = None,
        on_upstream_id: UpstreamIdCallback | None = None,
        rng: random.Random | None = None,
    ) -> GeneratedAsset:
        prompt_id = await self._submit_one(
            record,
            prompt=prompt,
            size=size,
            image_filenames=image_filenames,
            length=length,
            on_upstream_id=on_upstream_id,
            rng=rng,
        )
        try:
            return await self._collect_one(record, prompt_id)
        except BaseException:
            # Queued but never collected — most often because the client
            # disconnected or a video job was cancelled, which cancels us
            # here. Without this the prompt keeps rendering for nobody.
            await self._discard_queued([prompt_id])
            raise

    async def _run_batch(
        self,
        record: WorkflowRecord,
        *,
        n: int,
        prompt: str,
        size: str | None,
        image_filenames: list[str] | None = None,
        rng: random.Random | None = None,
    ) -> list[GeneratedAsset]:
        """Queue ``n`` runs up front, then collect them.

        Running these end-to-end one at a time meant the caller waited for
        the sum of n full generations inside a single synchronous
        POST /v1/images/generations — at the permitted n=4 with a 60s
        workflow that's four minutes, well past most clients' timeouts, and
        the abandoned work keeps running upstream unread.

        Submitting first lets ComfyUI queue and pipeline the runs, which is
        what it's built to do. Each submit re-randomises seeds via
        prepare_workflow, so the outputs still differ.
        """
        prompt_ids: list[str] = []
        try:
            for _ in range(n):
                prompt_ids.append(
                    await self._submit_one(
                        record,
                        prompt=prompt,
                        size=size,
                        image_filenames=image_filenames,
                        rng=rng,
                    )
                )
        except BaseException:
            # BaseException, not Exception: a client disconnecting cancels this
            # coroutine, and that is probably the most common way a batch gets
            # abandoned part-way through submitting. Catching only Exception
            # let exactly that case orphan prompts with no trace at all.
            if prompt_ids:
                await self._discard_queued(prompt_ids)
            raise

        # Every collector starts its clock now, but the runs execute one after
        # another — the last one waits out all the others before it even
        # starts. Budget for the whole queue rather than a single run, or a
        # healthy n=4 batch of slow workflows times out on position alone.
        budget = self._poll_timeout_for(record) * n

        tasks = [
            asyncio.create_task(self._collect_one(record, pid, timeout_seconds=budget))
            for pid in prompt_ids
        ]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            # gather surfaces the first error immediately but leaves the other
            # tasks running, so without this they keep polling ComfyUI and
            # buffering media into memory long after the request that owns
            # them has failed. Cancel explicitly, then wait for the
            # cancellations to land before propagating.
            #
            # return_exceptions=True is *not* the fix here: it cancels nothing
            # and merely makes gather wait for every sibling, turning a prompt
            # failure into one the client waits out for the whole batch budget.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # Cancelling promptly is right for the client, but it leaves the
            # cancelled collectors' prompts queued upstream — the same orphan
            # the submit path recalls, moved to what is now the likelier
            # route. Hand back everything that never produced a result.
            abandoned = [
                prompt_id
                for prompt_id, task in zip(prompt_ids, tasks, strict=True)
                if task.cancelled() or task.exception() is not None
            ]
            if abandoned:
                await self._discard_queued(abandoned)
            raise

    async def generate_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        record = await self._record_for(model_slug)
        if record.output_type != "image":
            raise UnsupportedOperation(
                f"Model {model_slug!r} produces {record.output_type}, not image"
            )
        if record.meta.get("image_required"):
            raise ImageRequired(
                f"Model {model_slug!r} requires an input image; use /v1/images/edits",
                param="image",
            )
        return await self._run_batch(record, n=n, prompt=prompt, size=size)

    async def edit_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        images: list[InputImage],
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        record = await self._record_for(model_slug)
        if record.output_type != "image":
            raise UnsupportedOperation(
                f"Model {model_slug!r} produces {record.output_type}, not image"
            )
        if not record.meta.get("image_inputs"):
            raise UnsupportedOperation(f"Workflow {model_slug!r} does not accept image input")
        # Upload once and reuse the filenames across the n runs. Uploads run
        # concurrently; gather preserves order so the filenames still line up
        # with the workflow's declared image_inputs. prepare_workflow then
        # distributes the list (a spec marked ``multiple`` consumes the rest).
        comfy_filenames = await asyncio.gather(
            *(self.client.upload_image(img.data, img.content_type) for img in images)
        )
        return await self._run_batch(
            record,
            n=n,
            prompt=prompt,
            size=size,
            image_filenames=list(comfy_filenames),
        )

    async def generate_video(
        self,
        *,
        model_slug: str,
        prompt: str,
        size: str | None = None,
        seconds: float | None = None,
        input_reference: bytes | None = None,
        input_reference_content_type: str | None = None,
        on_upstream_id: UpstreamIdCallback | None = None,
    ) -> GeneratedAsset:
        record = await self._record_for(model_slug)
        if record.output_type != "video":
            raise UnsupportedOperation(
                f"Model {model_slug!r} produces {record.output_type}, not video"
            )

        image_filenames: list[str] | None = None
        if record.meta.get("image_inputs"):
            if input_reference is None:
                if record.meta.get("image_required"):
                    raise ImageRequired(
                        f"Model {model_slug!r} requires an input_reference image",
                        param="input_reference",
                    )
            else:
                ct = input_reference_content_type or "image/png"
                comfy_filename = await self.client.upload_image(input_reference, ct)
                image_filenames = [comfy_filename]
        elif input_reference is not None:
            log.debug(
                "Workflow %r does not declare image_inputs; ignoring input_reference",
                model_slug,
            )

        # Translate seconds → frames only if the workflow declares an FPS hint.
        # Otherwise we leave the workflow's baked-in length untouched, which
        # matches how the existing pipe handled things.
        length = seconds_to_frames(seconds, record.meta)

        return await self._run_one(
            record,
            prompt=prompt,
            size=size,
            image_filenames=image_filenames,
            length=length,
            on_upstream_id=on_upstream_id,
        )
