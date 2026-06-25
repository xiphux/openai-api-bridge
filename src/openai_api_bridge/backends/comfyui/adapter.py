"""ComfyUI Backend implementation."""

from __future__ import annotations

import asyncio
import logging
import random

from ...config import ComfyUIProviderConfig
from ...errors import ImageRequired, ModelNotFound, UnsupportedOperation
from ...util.sizes import parse_size
from ..base import Backend, GeneratedAsset, InputImage, ModelEntry, UpstreamIdCallback
from .client import ComfyUIClient
from .workflows import (
    WorkflowRecord,
    prepare_workflow,
    scan_workflows,
    seconds_to_frames,
)

log = logging.getLogger(__name__)


class ComfyUIBackend(Backend):
    def __init__(self, cfg: ComfyUIProviderConfig) -> None:
        self.cfg = cfg
        self.client = ComfyUIClient(
            base_url=cfg.url,
            poll_interval_seconds=cfg.poll_interval_seconds,
        )
        self._workflows: dict[str, WorkflowRecord] | None = None

    async def aclose(self) -> None:
        await self.client.aclose()

    # --- discovery ---------------------------------------------------------

    def _ensure_workflows(self) -> dict[str, WorkflowRecord]:
        if self._workflows is None or not self.cfg.cache_workflows:
            self._workflows = scan_workflows(self.cfg.workflows_dir)
        return self._workflows

    def _record_for(self, model_slug: str) -> WorkflowRecord:
        records = self._ensure_workflows()
        record = records.get(model_slug)
        if record is None:
            raise ModelNotFound(
                f"Workflow {model_slug!r} not found in provider {self.cfg.id!r}",
                param="model",
            )
        return record

    async def list_models(self) -> list[ModelEntry]:
        records = self._ensure_workflows()
        return [
            ModelEntry(id=r.slug, kind=r.output_type, display_name=r.display_name)
            for r in sorted(records.values(), key=lambda r: r.slug)
        ]

    # --- generation --------------------------------------------------------

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

        timeout = (
            self.cfg.poll_timeout_video_seconds
            if record.output_type == "video"
            else self.cfg.poll_timeout_image_seconds
        )
        history = await self.client.poll_completion(prompt_id, timeout_seconds=timeout)
        data, content_type = await self.client.retrieve_media(
            history, output_type=record.output_type
        )
        return GeneratedAsset(data=data, content_type=content_type, kind=record.output_type)

    async def generate_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        record = self._record_for(model_slug)
        if record.output_type != "image":
            raise UnsupportedOperation(
                f"Model {model_slug!r} produces {record.output_type}, not image"
            )
        if record.meta.get("image_required"):
            raise ImageRequired(
                f"Model {model_slug!r} requires an input image; use /v1/images/edits",
                param="image",
            )
        return [await self._run_one(record, prompt=prompt, size=size) for _ in range(n)]

    async def edit_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        images: list[InputImage],
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        record = self._record_for(model_slug)
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
        return [
            await self._run_one(
                record,
                prompt=prompt,
                size=size,
                image_filenames=comfy_filenames,
            )
            for _ in range(n)
        ]

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
        record = self._record_for(model_slug)
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
