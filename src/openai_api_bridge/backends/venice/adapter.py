"""Venice Backend implementation.

Venice supports text-to-image (``/image/generate``) and image-to-image
(``/image/edit``), but no video. ``edit_image`` routes to Venice's dedicated
edit endpoint; ``generate_video`` raises ``UnsupportedOperation`` from the base
class (overridden here for a clearer message). Venice edits are single-image,
so more than one reference is rejected rather than silently dropped.
"""

from __future__ import annotations

import logging

from ...config import VeniceProviderConfig
from ...errors import InvalidRequest, UnsupportedOperation
from ...util.sizes import parse_size
from ..base import Backend, GeneratedAsset, InputImage, ModelEntry
from .client import VeniceClient

log = logging.getLogger(__name__)

# Venice's image-generation endpoint always returns PNG; the API doesn't
# include a content-type per image so we hard-code it (matches existing pipe).
_VENICE_CONTENT_TYPE = "image/png"


class VeniceBackend(Backend):
    def __init__(self, cfg: VeniceProviderConfig) -> None:
        self.cfg = cfg
        self.client = VeniceClient(
            base_url=cfg.base_url,
            api_token=cfg.resolve_api_token(),
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def list_models(self) -> list[ModelEntry]:
        raw = await self.client.list_image_models()
        return [
            ModelEntry(id=m["id"], kind="image", display_name=m.get("id")) for m in raw if "id" in m
        ]

    async def generate_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        w, h = parse_size(size)
        width = w or self.cfg.default_width
        height = h or self.cfg.default_height

        out: list[GeneratedAsset] = []
        for _ in range(n):
            data = await self.client.generate_image(
                model=model_slug,
                prompt=prompt,
                width=width,
                height=height,
                steps=self.cfg.steps,
                cfg_scale=self.cfg.cfg_scale,
            )
            out.append(GeneratedAsset(data=data, content_type=_VENICE_CONTENT_TYPE, kind="image"))
        return out

    async def edit_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        images: list[InputImage],
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        # Venice's /image/edit takes exactly one reference image. Reject extras
        # rather than silently dropping them (the edit_image contract).
        if len(images) > 1:
            raise InvalidRequest(
                f"Venice image edits accept exactly one reference image (got {len(images)})",
                param="image",
            )
        # /image/edit uses aspect_ratio/resolution, not OpenAI's size string;
        # we let Venice infer from the source image rather than guess a mapping.
        del size
        image = images[0]
        out: list[GeneratedAsset] = []
        for _ in range(n):
            data, content_type = await self.client.edit_image(
                model=model_slug,
                prompt=prompt,
                image=image.data,
                image_content_type=image.content_type,
            )
            out.append(GeneratedAsset(data=data, content_type=content_type, kind="image"))
        return out

    async def generate_video(
        self,
        *,
        model_slug: str,
        prompt: str,
        size: str | None = None,
        seconds: float | None = None,
        input_reference: bytes | None = None,
        input_reference_content_type: str | None = None,
        on_upstream_id=None,
    ) -> GeneratedAsset:
        raise UnsupportedOperation(
            "Venice does not support video generation. "
            "Use a ComfyUI provider with a video workflow."
        )
