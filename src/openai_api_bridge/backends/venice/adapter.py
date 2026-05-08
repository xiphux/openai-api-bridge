"""Venice Backend implementation.

Venice supports text-to-image only — no I2I, no video. The default
``edit_image``/``generate_video`` raise ``UnsupportedOperation`` from the base
class; we override here to surface clearer messages.
"""

from __future__ import annotations

import logging

from ...config import VeniceProviderConfig
from ...errors import UnsupportedOperation
from ...util.sizes import parse_size
from ..base import Backend, GeneratedAsset, ModelEntry
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
            ModelEntry(id=m["id"], kind="image", display_name=m.get("id"))
            for m in raw
            if "id" in m
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
            out.append(
                GeneratedAsset(
                    data=data, content_type=_VENICE_CONTENT_TYPE, kind="image"
                )
            )
        return out

    async def edit_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        image: bytes,
        image_content_type: str,
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        raise UnsupportedOperation(
            "Venice does not support image edits (img2img). "
            "Use a ComfyUI provider with a workflow that declares image_inputs."
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
        on_upstream_id=None,
    ) -> GeneratedAsset:
        raise UnsupportedOperation(
            "Venice does not support video generation. "
            "Use a ComfyUI provider with a video workflow."
        )
