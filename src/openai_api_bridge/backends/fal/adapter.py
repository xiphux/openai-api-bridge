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

``list_models`` is discovered from fal's model API, filtered to
``SUPPORTED_CATEGORIES`` — the modalities this backend actually implements —
and to active models. ``discover_models = false`` opts out and serves only the
configured models. Either way ``[[providers.models]]`` entries are per-model
*overrides* rather than a whitelist, and a configured model the catalogue
didn't return is still listed, since generation works for it regardless.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections import OrderedDict
from typing import Any

from ...config import FalModelConfig, FalProviderConfig
from ...errors import ModelNotFound, UpstreamAuthError
from ...util.sizes import parse_size
from ..base import Backend, GeneratedAsset, InputImage, ModelEntry
from .client import FalClient
from .safety import fallback_safety_params, safety_params_from_schema

log = logging.getLogger(__name__)

# fal categories this backend can actually serve. It implements the image
# surface (generate + edit) only, so the video/audio/3d categories fal also
# publishes are deliberately excluded — listing them would advertise models
# every request would fail on with UnsupportedOperation. Adding video here
# means implementing generate_video for fal first (its queue lifecycle), not
# just widening the filter.
SUPPORTED_CATEGORIES: tuple[str, ...] = ("text-to-image", "image-to-image")

# With discovery on, any slug a client sends is accepted (fal rejects unknown
# ids itself), so per-model bookkeeping is keyed by unvalidated input. Bound it
# so a client generating slugs — a uuid appended per request, say — can't grow
# these maps without limit. Well past the ~574 real models, and eviction is
# harmless: a dropped lock only risks a duplicate fetch, which the cache
# already tolerates, and a dropped cooldown only allows one earlier retry.
_MAX_TRACKED_MODELS = 1024


def _remember[T](mapping: OrderedDict[str, T], key: str, value: T) -> None:
    """Insert into an LRU-bounded map, evicting the oldest entries past the cap."""
    mapping[key] = value
    mapping.move_to_end(key)
    while len(mapping) > _MAX_TRACKED_MODELS:
        mapping.popitem(last=False)


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
        # Schema-derived safety params, resolved per model on first use and
        # cached for the process. Per-model rather than one batch over the whole
        # catalog: discovery surfaces hundreds of models and we only pay for the
        # ones actually generated with. Lazy rather than at startup so the
        # lifespan graph keeps no network dependency and a fal outage can't
        # block boot.
        self._schema_cache: dict[str, dict[str, Any]] = {}
        # One lock per model, so a fan-out across different models resolves
        # concurrently while duplicate requests for the *same* model still
        # collapse into a single lookup.
        self._schema_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._locks_guard = asyncio.Lock()
        # Monotonic timestamp of each model's last failed lookup. Drives the
        # retry cooldown so an outage degrades temporarily rather than for the
        # life of the process.
        self._introspect_failed_at: OrderedDict[str, float] = OrderedDict()
        # Discovered catalog, with its own cooldown on failure.
        self._catalog_cache: list[ModelEntry] | None = None
        self._catalog_lock = asyncio.Lock()
        self._catalog_failed_at: float | None = None
        # Set once fal rejects our key. Unlike an outage this can't heal at
        # runtime — the token is read from the environment at startup — so it
        # permanently disables discovery and introspection instead of retrying.
        self._auth_failed = False

    async def aclose(self) -> None:
        await self.client.aclose()

    # --- model catalog ---------------------------------------------------

    @property
    def _categories(self) -> list[str]:
        return list(self.cfg.categories) if self.cfg.categories else list(SUPPORTED_CATEGORIES)

    def _configured_entries(self) -> list[ModelEntry]:
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

    def _note_auth_failure(self, e: Exception) -> None:
        """Report a rejected credential once, loudly, and stop retrying."""
        if self._auth_failed:
            return
        self._auth_failed = True
        log.error(
            "fal: provider %r credentials were rejected (%s). Check that env var %s "
            "holds a valid fal API key — model discovery and moderation "
            "introspection are disabled until the bridge restarts with a working "
            "key, and generation requests will fail.",
            self.cfg.id,
            e,
            self.cfg.api_token_env,
        )

    async def list_models(self) -> list[ModelEntry]:
        """Models this provider serves.

        With ``discover_models`` on (the default) this is fal's catalogue,
        filtered to the categories the backend can actually serve; any
        ``[[providers.models]]`` entry enriches its match with per-model
        overrides. With it off, only the configured models are served.

        If the catalogue can't be fetched we fall back to whatever is
        explicitly configured rather than dropping the provider's models from
        ``/v1/models`` entirely.
        """
        if not self.cfg.discover_models or self._auth_failed:
            return self._configured_entries()
        async with self._catalog_lock:
            if self._catalog_cache is not None:
                return self._catalog_cache
            if self._catalog_failed_at is not None:
                elapsed = time.monotonic() - self._catalog_failed_at
                if elapsed < self.cfg.introspect_retry_seconds:
                    return self._configured_entries()
            try:
                raw = await self.client.fetch_catalog(self._categories)
            except UpstreamAuthError as e:
                self._note_auth_failure(e)
                return self._configured_entries()
            except Exception as e:  # never 500 /v1/models over a catalogue blip
                self._catalog_failed_at = time.monotonic()
                log.warning(
                    "fal: could not list models (%s); serving only explicitly "
                    "configured models, retrying in %.0fs",
                    e,
                    self.cfg.introspect_retry_seconds,
                )
                return self._configured_entries()
            entries = self._entries_from_catalog(raw)
            if not entries:
                self._catalog_failed_at = time.monotonic()
                log.warning(
                    "fal: model API returned no models for categories %s; serving "
                    "only explicitly configured models, retrying in %.0fs",
                    self._categories,
                    self.cfg.introspect_retry_seconds,
                )
                return self._configured_entries()
            # Union, not projection: an explicitly configured model the
            # catalogue didn't return must still be listed. fal's ids don't
            # always match what operators have configured (the gpt-image-2
            # endpoint is ``openai/gpt-image-2``, not ``fal-ai/gpt-image-2``),
            # and a model outside the filtered categories or marked deprecated
            # is absent too — yet `_model_config` still resolves it and
            # generation still works, so dropping it here would leave the
            # listing and generation surfaces disagreeing.
            listed = {e.id for e in entries}
            entries += [e for e in self._configured_entries() if e.id not in listed]
            self._catalog_failed_at = None
            self._catalog_cache = entries
            return entries

    def _entries_from_catalog(self, raw: list[dict[str, Any]]) -> list[ModelEntry]:
        entries: list[ModelEntry] = []
        seen: set[str] = set()
        for item in raw:
            model_id = item.get("endpoint_id")
            if not isinstance(model_id, str) or not model_id or model_id in seen:
                continue
            seen.add(model_id)
            meta = item.get("metadata")
            meta = meta if isinstance(meta, dict) else {}
            override = self._models.get(model_id)
            display = None
            if override is not None and override.display_name:
                display = override.display_name
            elif isinstance(meta.get("display_name"), str):
                display = meta["display_name"]
            entries.append(
                ModelEntry(
                    id=model_id,
                    # Both surfaced categories are image; the bridge picks the
                    # code path from the request shape, so this is just a hint.
                    kind="image",
                    display_name=display or model_id,
                    prompt_style=override.prompt_style if override else None,
                    prompt_hint=override.prompt_hint if override else None,
                )
            )
        return entries

    # --- moderation settings ---------------------------------------------

    def _model_config(self, model_slug: str) -> FalModelConfig:
        """Per-model config for a request, defaulting for discovered models.

        With discovery off an unlisted model is a 404. With it on we don't
        validate against the catalogue — that would put a listing fetch on the
        generation path — and let fal reject an unknown id itself.
        """
        mcfg = self._models.get(model_slug)
        if mcfg is not None:
            return mcfg
        if not self.cfg.discover_models:
            raise ModelNotFound(
                f"Model {model_slug!r} is not configured for fal provider {self.cfg.id!r}",
                param="model",
            )
        return FalModelConfig(id=model_slug)

    def _in_introspect_cooldown(self, model_id: str) -> bool:
        """True while a recent failure for this model should suppress retries."""
        failed_at = self._introspect_failed_at.get(model_id)
        if failed_at is None:
            return False
        return time.monotonic() - failed_at < self.cfg.introspect_retry_seconds

    def _arm_introspect_retry(self, model_id: str, reason: str) -> None:
        """Record a lookup as unsuccessful so the cooldown — and therefore a
        later retry — engages for this model."""
        _remember(self._introspect_failed_at, model_id, time.monotonic())
        log.warning(
            "fal: %s for %r; using built-in moderation defaults, retrying in %.0fs",
            reason,
            model_id,
            self.cfg.introspect_retry_seconds,
        )

    async def _lock_for(self, model_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._schema_locks.get(model_id)
            if lock is None:
                lock = asyncio.Lock()
            _remember(self._schema_locks, model_id, lock)
            return lock

    async def _derived_safety_params(self, model_id: str) -> dict[str, Any] | None:
        """Schema-derived moderation settings for one model.

        ``None`` means unavailable — introspection off, the lookup failed, the
        model wasn't in the response, or we're inside its retry cooldown — and
        the caller falls back to the static map. Nothing unsuccessful is ever
        cached, so a transient failure can't latch for the life of the process.
        """
        if not self.cfg.introspect_safety or self._auth_failed:
            return None
        cached = self._schema_cache.get(model_id)
        if cached is not None:
            return cached
        lock = await self._lock_for(model_id)
        async with lock:
            cached = self._schema_cache.get(model_id)
            if cached is not None:
                return cached
            if self._in_introspect_cooldown(model_id):
                return None
            try:
                specs = await self.client.fetch_model_schemas([model_id])
            except UpstreamAuthError as e:
                self._note_auth_failure(e)
                return None
            except Exception as e:  # never fail a generation over an introspection blip
                self._arm_introspect_retry(model_id, f"schema lookup failed ({e})")
                return None
            spec = specs.get(model_id)
            if spec is None:
                self._arm_introspect_retry(model_id, "no schema returned")
                return None
            params = safety_params_from_schema(spec)
            self._schema_cache[model_id] = params
            self._introspect_failed_at.pop(model_id, None)
            return params

    async def _safety_params(self, mcfg: FalModelConfig) -> dict[str, Any]:
        """Loosest moderation settings for a model: schema-derived when we can
        read the schema, else the built-in fallback map."""
        if not mcfg.disable_safety:
            return {}
        derived = await self._derived_safety_params(mcfg.id)
        if derived is not None:
            if not derived:
                log.debug("fal: %r exposes no moderation knob; leaving defaults", mcfg.id)
            return dict(derived)
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

    async def _run_image(self, model_slug: str, body: dict[str, Any]) -> list[str]:
        """Run a generation, reporting a rejected key on the way past.

        Generation is the one path that can reach fal without introspection
        having run first (``introspect_safety = false``, ``discover_models =
        false``, or a model with ``disable_safety = false``), so without this
        the "reported once at ERROR" guarantee wouldn't hold for those configs.
        ``_note_auth_failure`` is idempotent, so "once" survives.
        """
        try:
            return await self.client.run_image(model_slug, body)
        except UpstreamAuthError as e:
            self._note_auth_failure(e)
            raise

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
        mcfg = self._model_config(model_slug)
        # fal honours ``num_images`` server-side, so a single call yields n
        # images — no per-image request loop (unlike ImageRouter).
        body = await self._build_body(mcfg, prompt=prompt, size=size, n=n)
        urls = await self._run_image(model_slug, body)
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
        mcfg = self._model_config(model_slug)
        # fal's edit endpoints take reference images as ``image_urls``; they
        # accept data URIs inline, so no separate upload step is needed. All
        # supplied references are forwarded in order — a model that only honours
        # one lets the upstream decide, matching the ABC's no-silent-drop rule.
        body = await self._build_body(mcfg, prompt=prompt, size=size, n=n)
        body["image_urls"] = [_data_uri(img) for img in images]
        urls = await self._run_image(model_slug, body)
        return await self._fetch_all(urls)
