"""fal.ai Backend implementation.

fal is a model-hosting broker whose defining advantage for this bridge is that
it exposes each model's *native* input schema — including the content-moderation
knob that flat OpenAI-shaped brokers (ImageRouter) hide. The catch is that the
knob's name and shape differ per model family, so this adapter carries a small
per-family map and injects the loosest setting when a model is configured with
``disable_safety = true`` (the default). Families the map doesn't cover — notably
fal's OpenAI GPT-Image wrapper, which exposes *no* moderation field at all — get
nothing injected; there's no universal off-switch to reach for.

fal has no model-catalog endpoint, so ``list_models`` reflects the models
declared in the provider's TOML block rather than a live upstream listing.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from ...config import FalModelConfig, FalProviderConfig
from ...errors import ModelNotFound
from ...util.sizes import parse_size
from ..base import Backend, GeneratedAsset, InputImage, ModelEntry
from .client import FalClient

log = logging.getLogger(__name__)


# Loosest-moderation body params per model family, matched as a substring of
# the fal model path (lower-cased). First match wins. Kept deliberately narrow:
# fal rejects unknown input fields with a 422, so injecting the *wrong* knob
# breaks generation — better to cover only families whose schema we've verified
# and leave the rest to the per-model ``params`` escape hatch.
_LOOSEST_SAFETY_RULES: tuple[tuple[str, dict[str, Any]], ...] = (
    # ByteDance Seedream (text-to-image + edit): boolean checker, default true.
    ("seedream", {"enable_safety_checker": False}),
    # Google Nano Banana / Gemini image (text-to-image + edit): a "1".."6"
    # string enum where 1 is strictest and 6 loosest; default "4".
    ("nano-banana", {"safety_tolerance": "6"}),
    ("gemini", {"safety_tolerance": "6"}),
)


def _loosest_safety_params(model_slug: str) -> dict[str, Any]:
    """Body fields that minimise moderation for this model's family, or ``{}``
    when the bridge doesn't recognise the family (nothing is guessed — an
    unknown field would 422)."""
    lowered = model_slug.lower()
    for needle, params in _LOOSEST_SAFETY_RULES:
        if needle in lowered:
            return dict(params)
    log.debug("fal: no known moderation knob for %r; leaving upstream defaults", model_slug)
    return {}


def _apply_size(body: dict[str, Any], model_slug: str, size: str | None) -> None:
    """Translate OpenAI's ``WxH`` size into fal's field for this model family.

    Most image models take ``image_size`` as ``{width, height}``. Nano Banana /
    Gemini image models instead use ``aspect_ratio`` + ``resolution`` and reject
    ``image_size`` outright, so an explicit ``WxH`` can't be honoured faithfully
    — we leave the model's default (a caller who cares sets ``params`` in config).
    """
    w, h = parse_size(size)
    if w <= 0 or h <= 0:
        return
    lowered = model_slug.lower()
    if "nano-banana" in lowered or "gemini" in lowered:
        log.debug(
            "fal: %r takes aspect_ratio/resolution, not image_size; ignoring size", model_slug
        )
        return
    body["image_size"] = {"width": w, "height": h}


def _data_uri(img: InputImage) -> str:
    b64 = base64.b64encode(img.data).decode("ascii")
    return f"data:{img.content_type};base64,{b64}"


class FalBackend(Backend):
    def __init__(self, cfg: FalProviderConfig) -> None:
        self.cfg = cfg
        self._models: dict[str, FalModelConfig] = {m.id: m for m in cfg.models}
        self.client = FalClient(
            base_url=cfg.base_url,
            api_token=cfg.resolve_api_token(),
            request_timeout_seconds=cfg.request_timeout_seconds,
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def list_models(self) -> list[ModelEntry]:
        # fal has no catalog endpoint — the models this provider serves are the
        # ones declared in config.
        return [
            ModelEntry(
                id=m.id,
                kind=m.kind,
                display_name=m.display_name or m.id,
                prompt_style=m.prompt_style,
                prompt_hint=m.prompt_hint,
            )
            for m in self.cfg.models
        ]

    def _require_model(self, model_slug: str) -> FalModelConfig:
        mcfg = self._models.get(model_slug)
        if mcfg is None:
            raise ModelNotFound(
                f"Model {model_slug!r} is not configured for fal provider {self.cfg.id!r}",
                param="model",
            )
        return mcfg

    def _build_body(
        self, mcfg: FalModelConfig, *, prompt: str, size: str | None, n: int
    ) -> dict[str, Any]:
        """Assemble the fal request body. Precedence (last wins): base fields →
        size → built-in loosest-safety defaults → per-model ``params`` override."""
        body: dict[str, Any] = {"prompt": prompt, "num_images": n}
        _apply_size(body, mcfg.id, size)
        if mcfg.disable_safety:
            body.update(_loosest_safety_params(mcfg.id))
        if mcfg.params:
            body.update(mcfg.params)
        return body

    async def _fetch_all(self, urls: list[str]) -> list[GeneratedAsset]:
        out: list[GeneratedAsset] = []
        for url in urls:
            data, content_type = await self.client.fetch_asset(url)
            out.append(GeneratedAsset(data=data, content_type=content_type, kind="image"))
        return out

    async def generate_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        mcfg = self._require_model(model_slug)
        # fal honours ``num_images`` server-side, so a single call yields n
        # images — no per-image request loop (unlike ImageRouter).
        body = self._build_body(mcfg, prompt=prompt, size=size, n=n)
        urls = await self.client.run_image(model_slug, body)
        return await self._fetch_all(urls)

    async def edit_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        images: list[InputImage],
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        mcfg = self._require_model(model_slug)
        # fal's edit endpoints take reference images as ``image_urls``; they
        # accept data URIs inline, so no separate upload step is needed. All
        # supplied references are forwarded in order — a model that only honours
        # one lets the upstream decide, matching the ABC's no-silent-drop rule.
        body = self._build_body(mcfg, prompt=prompt, size=size, n=n)
        body["image_urls"] = [_data_uri(img) for img in images]
        urls = await self.client.run_image(model_slug, body)
        return await self._fetch_all(urls)
