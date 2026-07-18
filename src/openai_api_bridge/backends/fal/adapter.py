"""fal.ai Backend implementation.

fal is a model-hosting broker whose defining advantage for this bridge is that
it exposes each model's *native* input schema — including the content-moderation
knob that flat OpenAI-shaped brokers (ImageRouter) hide.

Rather than mapping model name -> knob (a code change per new model version),
the loosest setting is read from the model's own OpenAPI schema and cached; see
``safety.py`` for the derivation and why it beats hardcoding. Models exposing no
knob we recognise — notably fal's OpenAI GPT-Image wrapper, which has no
moderation field at all — get nothing injected; there's no universal off-switch.
Introspection is applied when a model is configured with ``disable_safety =
true`` (the default), and degrades to a small static map if fal's model API is
unreachable.

fal's *inference* host has no catalog endpoint, so ``list_models`` reflects the
models declared in the provider's TOML block rather than a live listing; the
separate model API is used only to read schemas.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

from ...config import FalModelConfig, FalProviderConfig
from ...errors import ModelNotFound
from ...util.sizes import parse_size
from ..base import Backend, GeneratedAsset, InputImage, ModelEntry
from .client import FalClient
from .safety import fallback_safety_params, safety_params_from_schema

log = logging.getLogger(__name__)


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
            models_api_url=cfg.models_api_url,
        )
        # Schema-derived safety params, resolved lazily on first image request
        # (one batched call covering every configured model) and cached for the
        # process. Lazy rather than at startup so the lifespan graph keeps no
        # network dependency and a fal outage can't block boot.
        self._safety_cache: dict[str, dict[str, Any]] | None = None
        self._safety_lock = asyncio.Lock()
        self._introspect_failed = False

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

    async def _ensure_safety_cache(self) -> dict[str, dict[str, Any]] | None:
        """Populate the schema-derived safety map, once, for all configured models.

        Returns ``None`` if introspection is off or the lookup failed — callers
        then fall back to the static map. A failure is sticky for the process so
        an unreachable fal model API doesn't cost a round trip per request.
        """
        if not self.cfg.introspect_safety:
            return None
        # Always taken under the lock: it collapses a burst of concurrent first
        # requests into exactly one batched lookup, and once resolved the body
        # is a single cache check, so the cost is an uncontended acquire.
        async with self._safety_lock:
            if self._safety_cache is not None:
                return self._safety_cache
            if self._introspect_failed:
                return None
            model_ids = [m.id for m in self.cfg.models]
            if not model_ids:
                self._safety_cache = {}
                return self._safety_cache
            try:
                specs = await self.client.fetch_model_schemas(model_ids)
            except Exception as e:  # never fail a generation over an introspection blip
                self._introspect_failed = True
                log.warning(
                    "fal: could not introspect model schemas (%s); "
                    "falling back to built-in moderation defaults",
                    e,
                )
                return None
            resolved = {mid: safety_params_from_schema(spec) for mid, spec in specs.items()}
            for mid in model_ids:
                if mid not in resolved:
                    log.warning(
                        "fal: model %r absent from the model API; using fallback "
                        "moderation defaults",
                        mid,
                    )
            self._safety_cache = resolved
            return self._safety_cache

    async def _safety_params(self, mcfg: FalModelConfig) -> dict[str, Any]:
        """Loosest moderation settings for a model: schema-derived when we can
        read the schema, else the built-in fallback map."""
        if not mcfg.disable_safety:
            return {}
        cache = await self._ensure_safety_cache()
        if cache is not None and mcfg.id in cache:
            params = dict(cache[mcfg.id])
            if not params:
                log.debug("fal: %r exposes no moderation knob; leaving defaults", mcfg.id)
            return params
        return fallback_safety_params(mcfg.id)

    async def _build_body(
        self, mcfg: FalModelConfig, *, prompt: str, size: str | None, n: int
    ) -> dict[str, Any]:
        """Assemble the fal request body. Precedence (last wins): base fields →
        size → loosest-safety settings → per-model ``params`` override."""
        body: dict[str, Any] = {"prompt": prompt, "num_images": n}
        _apply_size(body, mcfg.id, size)
        body.update(await self._safety_params(mcfg))
        if mcfg.params:
            body.update(mcfg.params)
        return body

    async def _fetch_all(self, urls: list[str]) -> list[GeneratedAsset]:
        # fal returns every image from a single call, so the URLs are all in
        # hand at once — fetch them concurrently rather than serially. Order is
        # preserved; if any fetch exhausts its retries, gather propagates the
        # error and the whole request fails (all-or-nothing, as before).
        assets = await asyncio.gather(*(self.client.fetch_asset(url) for url in urls))
        return [
            GeneratedAsset(data=data, content_type=content_type, kind="image")
            for data, content_type in assets
        ]

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
        body = await self._build_body(mcfg, prompt=prompt, size=size, n=n)
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
        body = await self._build_body(mcfg, prompt=prompt, size=size, n=n)
        body["image_urls"] = [_data_uri(img) for img in images]
        urls = await self.client.run_image(model_slug, body)
        return await self._fetch_all(urls)
