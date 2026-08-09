"""ImageRouter Backend implementation.

ImageRouter is OpenAI-compatible in *content* but not in *paths*: the
inference endpoints live under ``/v1/openai`` while the model catalog is
at ``/v1/models``, and the video endpoint is sync (single POST) rather
than OpenAI's async /v1/videos lifecycle. The bridge absorbs all three
divergences so downstream clients see a uniform OpenAI-shaped API.

Model catalog filtering: ImageRouter exposes hundreds of models across
modalities (text, image, video, audio). The bridge surfaces only image
and video models — chat / embedding models from ImageRouter aren't
useful here since they belong on a chat-only OpenAI-passthrough provider.
"""

from __future__ import annotations

import logging

from ...config import ImageRouterProviderConfig
from ...util.cache import AsyncTTLCache
from ...util.concurrency import run_all
from ..base import (
    Backend,
    GeneratedAsset,
    InputImage,
    ModelEntry,
    UpstreamIdCallback,
    make_capabilities,
)
from .client import ImageRouterClient

log = logging.getLogger(__name__)


class ImageRouterBackend(Backend):
    def __init__(self, cfg: ImageRouterProviderConfig) -> None:
        self.cfg = cfg
        self.client = ImageRouterClient(
            base_url=cfg.base_url,
            api_token=cfg.resolve_api_token(),
        )
        self._catalog: AsyncTTLCache[list[ModelEntry]] = AsyncTTLCache(
            cfg.catalog_ttl_seconds, cfg.catalog_retry_seconds
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def list_models(self) -> list[ModelEntry]:
        return await self._catalog.get(self._fetch_models)

    async def _fetch_models(self) -> list[ModelEntry]:
        raw = await self.client.list_models()
        entries: list[ModelEntry] = []
        for info in raw:
            if not isinstance(info, dict):
                continue
            model_id = info.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue
            outputs = info.get("output") or info.get("outputs") or []
            if not isinstance(outputs, list):
                continue
            # Surface only generation-side modalities. A model that lists both
            # image and video output gets a single 'image' entry — bridge
            # clients pick by request shape (POST /v1/images vs /v1/videos),
            # not by the catalog kind, so the field is just a hint.
            if "image" in outputs:
                kind = "image"
            elif "video" in outputs:
                kind = "video"
            else:
                continue
            # ImageRouter states accepted inputs outright
            # (``"inputs": {"text": true, "image": false, ...}``), so whether a
            # model can take a reference image is read, not inferred.
            raw_inputs = info.get("inputs")
            accepted = (
                [name for name, ok in raw_inputs.items() if ok is True]
                if isinstance(raw_inputs, dict)
                else []
            )
            entries.append(
                ModelEntry(
                    id=model_id,
                    kind=kind,
                    display_name=model_id,
                    capabilities=make_capabilities(accepted, kind),
                )
            )
        return entries

    async def generate_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        # ImageRouter doesn't support n > 1 natively (its response is single-
        # image), so honouring n means one request per image. They're
        # independent, so they go out concurrently: run serially, the caller
        # waited for the sum of n generations inside a single synchronous
        # POST /v1/images/generations. n=1 is by far the common case and is
        # unaffected either way.
        async def one() -> GeneratedAsset:
            url = await self.client.generate_image_url(model=model_slug, prompt=prompt, size=size)
            data, content_type = await self.client.fetch_asset(url)
            return GeneratedAsset(data=data, content_type=content_type, kind="image")

        return await run_all([one] * n)

    async def edit_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        images: list[InputImage],
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        async def one() -> GeneratedAsset:
            url = await self.client.edit_image_url(
                model=model_slug,
                prompt=prompt,
                images=images,
                size=size,
            )
            data, content_type = await self.client.fetch_asset(url)
            return GeneratedAsset(data=data, content_type=content_type, kind="image")

        return await run_all([one] * n)

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
        # ImageRouter's video endpoint is synchronous — there's no upstream
        # job id to surface mid-flight, so on_upstream_id stays unused.
        # (The bridge's video runner already handles the async-ish lifecycle
        # by treating this whole call as one long await.)
        del on_upstream_id
        url = await self.client.generate_video_url(
            model=model_slug,
            prompt=prompt,
            size=size,
            seconds=seconds,
            input_reference=input_reference,
            input_reference_content_type=input_reference_content_type,
        )
        data, content_type = await self.client.fetch_asset(url)
        return GeneratedAsset(data=data, content_type=content_type, kind="video")
