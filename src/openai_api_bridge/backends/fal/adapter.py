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
true`` (the default). It degrades to a small static map per model — when
introspection is off, the lookup failed, or that model was absent from an
otherwise successful response.

``list_models`` reflects the models declared in the provider's TOML block rather
than a live listing. That's a deliberate choice, not a missing capability: fal
*does* publish a platform-wide model search API (which is what we read schemas
from), but it catalogues every model on the platform rather than a per-account
selection, so the set this provider serves is declared explicitly.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
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
        # Monotonic timestamp of the last failed lookup, or None if we haven't
        # failed. Drives the retry cooldown so an outage degrades temporarily
        # rather than for the life of the process.
        self._introspect_failed_at: float | None = None

    async def aclose(self) -> None:
        await self.client.aclose()

    async def list_models(self) -> list[ModelEntry]:
        # The models this provider serves are the ones declared in config —
        # fal's model API catalogues the whole platform, not a per-account
        # selection, so it isn't a listing we can surface directly.
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

    def _in_introspect_cooldown(self) -> bool:
        """True while a recent introspection failure should suppress retries."""
        if self._introspect_failed_at is None:
            return False
        elapsed = time.monotonic() - self._introspect_failed_at
        return elapsed < self.cfg.introspect_retry_seconds

    def _arm_introspect_retry(self, reason: str) -> None:
        """Record a lookup as unsuccessful so the cooldown — and therefore a
        later retry — engages."""
        self._introspect_failed_at = time.monotonic()
        log.warning(
            "fal: %s; using built-in moderation defaults for the affected models, "
            "retrying in %.0fs",
            reason,
            self.cfg.introspect_retry_seconds,
        )

    async def _ensure_safety_cache(self) -> dict[str, dict[str, Any]] | None:
        """Resolve schema-derived safety params for the configured models.

        Returns the map — which may be **partial** — or ``None`` when
        introspection is off or nothing has resolved yet. Callers fall back to
        the static map for any model absent from it.

        Anything short of a complete result arms the retry cooldown: a raised
        error, a response carrying no usable schemas, or a response that simply
        omits some models. That matters because a *successful* but empty or
        partial response would otherwise latch a cache that never retries —
        the same "degraded until restart" failure the cooldown exists to
        prevent, just reached without an exception. Only the still-missing
        models are re-fetched on a retry.
        """
        if not self.cfg.introspect_safety:
            return None
        model_ids = [m.id for m in self.cfg.models]
        if not model_ids:
            return {}
        # Always taken under the lock: it collapses a burst of concurrent first
        # requests into exactly one batched lookup, and once fully resolved the
        # body is a single completeness check — an uncontended acquire.
        async with self._safety_lock:
            cached = self._safety_cache
            missing = [mid for mid in model_ids if cached is None or mid not in cached]
            if not missing:
                return cached
            if self._in_introspect_cooldown():
                return cached
            try:
                specs = await self.client.fetch_model_schemas(missing)
            except Exception as e:  # never fail a generation over an introspection blip
                self._arm_introspect_retry(f"could not introspect model schemas ({e})")
                return cached
            resolved = {mid: safety_params_from_schema(spec) for mid, spec in specs.items()}
            if not resolved:
                # HTTP succeeded but nothing usable came back (empty catalog
                # response, changed payload shape, auth downgrade). Treat it as
                # a failure rather than caching the void.
                self._arm_introspect_retry(f"model API returned no schemas for {missing}")
                return cached
            merged = dict(cached or {})
            merged.update(resolved)
            self._safety_cache = merged
            still_missing = [mid for mid in model_ids if mid not in merged]
            if still_missing:
                self._arm_introspect_retry(f"no schema returned for {still_missing}")
            else:
                self._introspect_failed_at = None
            return merged

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
