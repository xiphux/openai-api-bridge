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
from dataclasses import dataclass
from typing import Any

from ...config import FalModelConfig, FalProviderConfig
from ...errors import (
    GenerationTimeout,
    ModelNotFound,
    RateLimited,
    UpstreamAuthError,
    UpstreamError,
)
from ...infra.tasks import SingleFlight
from ...util.cache import AsyncTTLCache
from ...util.sizes import parse_size
from ..base import (
    Backend,
    GeneratedAsset,
    InputImage,
    ModelEntry,
    UpstreamIdCallback,
    disambiguate_display_names,
)
from .client import FalClient, QueuedRequest, extract_video_url
from .safety import fallback_safety_params, safety_params_from_schema
from .schema import duration_params, duration_property

log = logging.getLogger(__name__)

# fal categories this backend serves, mapped to the bridge's `kind` hint. The
# audio and 3d categories fal also publishes stay out: there's no code path for
# them, so listing them would advertise models every request would fail on.
CATEGORY_KINDS: dict[str, str] = {
    "text-to-image": "image",
    "image-to-image": "image",
    "text-to-video": "video",
    "image-to-video": "video",
}
SUPPORTED_CATEGORIES: tuple[str, ...] = tuple(CATEGORY_KINDS)

# With discovery on, any slug a client sends is accepted (fal rejects unknown
# ids itself), so per-model bookkeeping is keyed by unvalidated input. Bound it
# so a client generating slugs — a uuid appended per request, say — can't grow
# these maps without limit. Well past the ~574 real models, and eviction is
# harmless: a dropped lock only risks a duplicate fetch, which the cache
# already tolerates, and a dropped cooldown only allows one earlier retry.
_MAX_TRACKED_MODELS = 1024

# SingleFlight key for the background catalogue refresh. There is only one
# catalogue per provider, so the key is a constant — the map exists for its
# at-most-one-live-task-per-key semantics, not to distinguish keys.
_CATALOG_KEY = "catalog"

# A long video job issues hundreds of status polls; any one of them can blip
# without the render being in trouble. Tolerate a short run of consecutive
# failures rather than discarding a clip fal is still working on (and billing
# for), and retry the one-shot result fetch, which happens after the render is
# already paid for.
_MAX_CONSECUTIVE_POLL_ERRORS = 5
_RESULT_FETCH_ATTEMPTS = 3

# Cancelling an abandoned job runs on the unwind path, so it can't be allowed
# to stall shutdown or hold up a client's DELETE.
_UPSTREAM_CANCEL_TIMEOUT_S = 10.0

# fal publishes a model's text-driven and reference-image-driven halves as
# separate endpoints, which is an easy trap: a client sends an edit to
# `fal-ai/nano-banana-2` without knowing `/edit` exists, or a still to
# `fal-ai/veo3.1` without knowing `/image-to-video` does. Where we can pair the
# two *confidently*, the bridge lists one model and routes by request shape.
#
# "Confidently" means both ids exist in the catalogue and sit in the expected
# categories — never a guess from a name. Against the live catalogue this pairs
# ~81 of 194 text-to-image models and ~74 of 125 text-to-video ones; anything
# unpaired, and every reference-only model (inpainting, upscalers, background
# removal), is left exactly as it is.
#
# text-driven category -> (its reference-image counterpart, endpoint suffixes)
_VARIANT_PAIRS: dict[str, tuple[str, tuple[str, ...]]] = {
    "text-to-image": ("image-to-image", ("/edit", "/image-to-image", "/edit-image")),
    "text-to-video": ("image-to-video", ("/image-to-video", "/edit")),
}


def variant_candidates(text_model_id: str, category: str) -> list[str]:
    """Ids that would be the reference-image half of ``text_model_id``.

    Two shapes occur: a bare base gaining a suffix
    (``fal-ai/nano-banana-2`` -> ``fal-ai/nano-banana-2/edit``), and sibling
    suffixes under a shared stem
    (``bytedance/seedream/v5/pro/text-to-image`` -> ``.../pro/edit``).
    """
    pair = _VARIANT_PAIRS.get(category)
    if pair is None:
        return []
    _, suffixes = pair
    text_suffix = f"/{category}"
    stem = (
        text_model_id[: -len(text_suffix)] if text_model_id.endswith(text_suffix) else text_model_id
    )
    candidates = [stem + suffix for suffix in suffixes]
    if stem != text_model_id:
        # A stem-suffixed model can also pair with `<full id>/edit`.
        candidates += [text_model_id + suffix for suffix in suffixes]
    return candidates


def _remember[T](mapping: OrderedDict[str, T], key: str, value: T) -> None:
    """Insert into an LRU-bounded map, evicting the oldest entries past the cap."""
    mapping[key] = value
    mapping.move_to_end(key)
    while len(mapping) > _MAX_TRACKED_MODELS:
        mapping.popitem(last=False)


@dataclass(frozen=True, slots=True)
class _DerivedParams:
    """What we keep from a model's schema, per model.

    Deliberately not the whole OpenAPI document: discovery surfaces hundreds of
    models and the documents are large, while all we need is the moderation
    settings and the shape of the ``duration`` field.
    """

    safety: dict[str, Any]
    duration_prop: dict[str, Any] | None


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
            queue_base_url=cfg.queue_base_url,
            store_payloads=cfg.store_payloads,
            output_expiration_seconds=cfg.output_expiration_seconds,
            # 0 is the documented "no bound"; anything else is MB.
            max_asset_bytes=(cfg.max_asset_mb * 1024**2) if cfg.max_asset_mb > 0 else None,
        )
        # Schema-derived request settings (moderation, plus the shape of the
        # model's duration field), resolved per model on first use and cached
        # for the process. Per-model rather than one batch over the whole
        # catalog: discovery surfaces hundreds of models and we only pay for the
        # ones actually generated with. Lazy rather than at startup so the
        # lifespan graph keeps no network dependency and a fal outage can't
        # block boot.
        self._schema_cache: dict[str, _DerivedParams] = {}
        # One lock per model, so a fan-out across different models resolves
        # concurrently while duplicate requests for the *same* model still
        # collapse into a single lookup.
        self._schema_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._locks_guard = asyncio.Lock()
        # Monotonic timestamp of each model's last failed lookup. Drives the
        # retry cooldown so an outage degrades temporarily rather than for the
        # life of the process.
        self._introspect_failed_at: OrderedDict[str, float] = OrderedDict()
        # Discovered catalog. Driven through AsyncTTLCache's lock/fresh/store
        # primitives rather than its `get`, because this backend degrades to
        # the configured models on a failed fetch where `get` re-raises — but
        # the TTL, the failure cooldown and the single-flight lock are the same
        # concerns every other backend has, and were hand-rolled here.
        self._catalog: AsyncTTLCache[list[ModelEntry]] = AsyncTTLCache(
            cfg.catalog_ttl_seconds, cfg.catalog_retry_seconds
        )
        # Holds the background refresh started when the catalogue goes stale,
        # so it can't be garbage-collected mid-flight and is drained at
        # shutdown. At most one runs at a time; see `list_models`.
        self._catalog_refresh: SingleFlight[list[ModelEntry]] = SingleFlight()
        # text-driven id -> the sibling that accepts a reference image, filled
        # in alongside the catalogue. Empty when collapsing or discovery is off.
        self._variant_routes: dict[str, str] = {}
        # Set once fal rejects our key. Unlike an outage this can't heal at
        # runtime — the token is read from the environment at startup — so it
        # permanently disables discovery and introspection instead of retrying.
        self._auth_failed = False

    async def aclose(self) -> None:
        # Before the client: a refresh in flight is holding it, and closing
        # underneath one would fail the fetch on the way out for no reason.
        await self._catalog_refresh.cancel_all()
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

        The listing is cached for ``catalog_ttl_seconds``, as on every other
        backend. It used to be cached for the life of the process, so a model
        fal added — or a variant pairing that changed with it — only appeared
        after a bridge restart.

        **Stale-while-revalidate**, which the other backends don't need. Once
        the TTL lapses this serves the previous catalogue and refreshes in the
        background, rather than making the caller wait for the refetch. fal's
        listing is 10-13 paginated round trips, comfortably past the
        ``MODELS_TIMEOUT_SECONDS`` budget ``/v1/models`` gives each provider —
        so blocking here meant fal's ~886 models dropped out of one listing per
        TTL window, and dropped out *entirely*, since a caller that times out
        never reaches the degrade-to-configured path below. It also meant an
        image edit could land on the refetch, where nothing bounds it at all
        (see :meth:`_reference_variant`).

        The other backends fetch a single page, so their refresh fits inside
        the budget and there is nothing to hide.
        """
        if not self.cfg.discover_models or self._auth_failed:
            return self._configured_entries()

        cached = self._catalog.fresh()
        if cached is not None:
            return cached

        previous = self._catalog.stale()
        if previous is not None:
            # Aged out but we still have the last good listing. Hand that back
            # now and let the refetch happen off the request.
            #
            # Not while a failure cooldown is open: the refresh would return
            # immediately having done nothing, so starting one per request
            # would be pure churn. Serving `previous` through the cooldown is
            # also strictly better than what this used to do, which was to fall
            # back to the configured models and discard a catalogue that was
            # merely a few minutes old.
            if self._catalog.pending_failure() is None:
                self._catalog_refresh.join_or_start(
                    _CATALOG_KEY, self._fetch_catalog_entries, name="fal-catalog-refresh"
                )
            return previous

        # Nothing cached at all — the first call of the process, or caching is
        # disabled outright. Someone has to pay for the fetch.
        return await self._fetch_catalog_entries()

    async def _fetch_catalog_entries(self) -> list[ModelEntry]:
        """Fetch, translate and cache the catalogue.

        Runs either inline for the first caller or as the background refresh
        started by :meth:`list_models`. Returns what that caller should serve,
        which on any failure is the explicitly configured models — the return
        value is ignored when this runs in the background, where the point is
        the ``store`` on the way through.

        Raises nothing an upstream fault can produce; ``CancelledError``
        deliberately still propagates (see the handler below), which as a
        background task means shutdown can drain it.
        """
        async with self._catalog.lock:
            cached = self._catalog.fresh()
            if cached is not None:
                return cached
            if self._catalog.pending_failure() is not None:
                # Inside the retry cooldown. Serve what's configured rather
                # than re-raising, which is what `AsyncTTLCache.get` would do.
                return self._configured_entries()
            try:
                raw = await self.client.fetch_catalog(self._categories)
            except UpstreamAuthError as e:
                self._note_auth_failure(e)
                return self._configured_entries()
            # NB: asyncio.CancelledError is BaseException, so it deliberately
            # escapes this handler. A client disconnecting mid-fetch is no
            # evidence fal is unhealthy — arming the cooldown there would
            # degrade the listing for every other caller for the full retry
            # window because one client hung up. Cancellation leaves no partial
            # state and releases the lock, so the next request simply retries;
            # the only cost is discarded work, and only until the first
            # successful fetch caches the catalogue for the window.
            except Exception as e:  # never 500 /v1/models over a catalogue blip
                self._catalog.note_failure(e)
                log.warning(
                    "fal: could not list models (%s); serving only explicitly "
                    "configured models, retrying in %.0fs",
                    e,
                    self.cfg.catalog_retry_seconds,
                )
                return self._configured_entries()
            entries, variant_routes = self._entries_from_catalog(raw)
            if not entries:
                self._catalog.note_failure(
                    UpstreamError(
                        f"fal model API returned no models for categories {self._categories}"
                    )
                )
                log.warning(
                    "fal: model API returned no models for categories %s; serving "
                    "only explicitly configured models, retrying in %.0fs",
                    self._categories,
                    self.cfg.catalog_retry_seconds,
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
            # Two reasons a configured model can be missing from `entries`,
            # and only one of them wants re-adding: absent from the catalogue
            # (union it back in), versus deliberately collapsed into its
            # text-driven half (leave it out — re-listing is exactly what
            # collapsing exists to prevent). A re-added entry would also come
            # from config, which carries no capabilities and defaults kind to
            # "image", so a collapsed *video* half would reappear advertising
            # the wrong modality.
            listed = {e.id for e in entries} | set(variant_routes.values())
            entries += [e for e in self._configured_entries() if e.id not in listed]
            # Publish the routing map and the listing together, and only here:
            # every path above returns without touching `_variant_routes`, so a
            # failed or empty refresh leaves the last working one in place
            # rather than routing edits to the text-only endpoint.
            self._variant_routes = variant_routes
            self._catalog.store(entries)
            return entries

    def _entries_from_catalog(
        self, raw: list[dict[str, Any]]
    ) -> tuple[list[ModelEntry], dict[str, str]]:
        """Translate a raw catalogue into entries and the variant routing map.

        Returns the routes rather than assigning them, so the caller can
        publish both together only once the result is known good. Assigning
        here meant a fetch that succeeded at the HTTP level but yielded nothing
        usable — fal answering ``{"models": []}`` during an incident, a
        category rename — replaced a working routing map with an empty one on
        its way to reporting failure. Harmless while the catalogue was fetched
        once per process; not once it refreshes on a TTL.
        """
        by_id = {
            item["endpoint_id"]: item
            for item in raw
            if isinstance(item.get("endpoint_id"), str) and item["endpoint_id"]
        }
        variant_routes, collapsed = self._pair_variants(by_id)

        entries: list[ModelEntry] = []
        seen: set[str] = set()
        pinned: set[str] = set()
        for item in raw:
            model_id = item.get("endpoint_id")
            if not isinstance(model_id, str) or not model_id or model_id in seen:
                continue
            if model_id in collapsed:
                # Folded into its text-driven half, which routes here when a
                # reference image is supplied. Listing both is what invites
                # picking the wrong one.
                continue
            seen.add(model_id)
            meta = item.get("metadata")
            meta = meta if isinstance(meta, dict) else {}
            override = self._models.get(model_id)
            display = None
            if override is not None and override.display_name:
                display = override.display_name
                pinned.add(model_id)
            elif isinstance(meta.get("display_name"), str):
                display = meta["display_name"]
            category = meta.get("category")
            kind = CATEGORY_KINDS.get(category) if isinstance(category, str) else None
            # What this entry actually accepts. A collapsed model covers both
            # halves; without this the merged id would be indistinguishable
            # from a text-only one, and a client would find out by failing.
            # Only categories the backend knows are published: they double as
            # the capability strings, so an operator pointing `categories` at
            # something outside CATEGORY_KINDS would otherwise emit a value
            # that isn't in the documented {input}-to-{output} vocabulary.
            capabilities: tuple[str, ...] | None = None
            if isinstance(category, str) and category in CATEGORY_KINDS:
                routed = variant_routes.get(model_id)
                partner = self._category_of(by_id, routed) if routed else None
                capabilities = (
                    (category, partner) if partner in CATEGORY_KINDS and partner else (category,)
                )
            entries.append(
                ModelEntry(
                    id=model_id,
                    # The bridge picks the code path from the request shape
                    # (POST /v1/images vs /v1/videos), so this is just a hint.
                    kind=kind or "image",
                    display_name=display or model_id,
                    prompt_style=override.prompt_style if override else None,
                    prompt_hint=override.prompt_hint if override else None,
                    capabilities=capabilities,
                )
            )
        # fal titles an endpoint after its model family, so multi-endpoint
        # families arrive as a run of identically-named rows.
        return disambiguate_display_names(entries, pinned=frozenset(pinned)), variant_routes

    def _category_of(self, by_id: dict[str, dict[str, Any]], model_id: str) -> str | None:
        meta = by_id.get(model_id, {}).get("metadata")
        category = meta.get("category") if isinstance(meta, dict) else None
        return category if isinstance(category, str) else None

    def _pair_variants(self, by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, str], set[str]]:
        """Match text-driven models to their reference-image half.

        Returns ``(routes, collapsed)`` — where to send a request carrying a
        reference image, and which ids to drop from the listing because they're
        now reachable through their sibling.

        A pair is only made when both ids are in this catalogue *and* their
        categories are the expected halves, so nothing is inferred from a name
        alone. Models without a partner — including every reference-only
        endpoint — are untouched.
        """
        if not self.cfg.collapse_variants:
            return {}, set()
        routes: dict[str, str] = {}
        collapsed: set[str] = set()
        for model_id in by_id:
            category = self._category_of(by_id, model_id)
            pair = _VARIANT_PAIRS.get(category) if category else None
            if category is None or pair is None:
                continue
            partner_category, _ = pair
            for candidate in variant_candidates(model_id, category):
                if candidate in by_id and self._category_of(by_id, candidate) == partner_category:
                    routes[model_id] = candidate
                    collapsed.add(candidate)
                    break
        if routes:
            log.debug("fal: collapsed %d variant pairs into their base models", len(routes))
        return routes, collapsed

    async def _reference_variant(self, model_slug: str) -> str:
        """The endpoint a request carrying a reference image should hit.

        Pairing comes from the catalogue, so this makes sure it's loaded rather
        than letting routing depend on whether some earlier request happened to
        populate it — that would make the same call behave differently based on
        unrelated traffic. ``list_models`` is a lock acquisition and a freshness
        check when the catalogue is cached, which is the common case.
        """
        if not self.cfg.collapse_variants or not self.cfg.discover_models:
            return model_slug
        await self.list_models()
        return self._variant_routes.get(model_slug, model_slug)

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

    def _config_for(self, model_slug: str, target: str) -> FalModelConfig:
        """Per-model config for a request that may have been routed elsewhere.

        A block for the endpoint actually being called takes precedence, so
        ``[[providers.models]] id = "fal-ai/nano-banana-2/edit"`` governs an
        edit that arrived addressed to the base model. Falling back to the
        requested id (rather than looking the target up directly) matters with
        ``discover_models = false``, where an unconfigured target would 404.

        The sibling **layers over** the base rather than replacing it, and only
        for fields it actually sets — ``exclude_unset`` is what distinguishes
        "omitted" from "set to the default". Swapping wholesale would mean a
        sibling block written to set, say, a display name silently reverted an
        explicit ``disable_safety = false`` on the base back to the permissive
        default, and dropped any pinned ``params``.
        """
        base = self._model_config(model_slug)
        routed = self._models.get(target)
        if routed is None:
            return base
        overrides = routed.model_dump(exclude_unset=True)
        if "params" in overrides:
            # Merge rather than replace, so the sibling can add or override a
            # key without discarding the base's pins.
            overrides["params"] = {**base.params, **routed.params}
        return base.model_copy(update=overrides)

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

    async def _derived_params(self, model_id: str) -> _DerivedParams | None:
        """Schema-derived request settings for one model.

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
            params = _DerivedParams(
                safety=safety_params_from_schema(spec),
                duration_prop=duration_property(spec),
            )
            self._schema_cache[model_id] = params
            self._introspect_failed_at.pop(model_id, None)
            return params

    async def _safety_params(self, mcfg: FalModelConfig, model_id: str) -> dict[str, Any]:
        """Loosest moderation settings for a model: schema-derived when we can
        read the schema, else the built-in fallback map.

        ``model_id`` is the endpoint actually being called, which differs from
        ``mcfg.id`` for a routed edit — the sibling has its own schema, and its
        knob is the one that will be honoured.
        """
        if not mcfg.disable_safety:
            return {}
        derived = await self._derived_params(model_id)
        if derived is not None:
            if not derived.safety:
                log.debug("fal: %r exposes no moderation knob; leaving defaults", model_id)
            return dict(derived.safety)
        return fallback_safety_params(model_id)

    async def _build_body(
        self, mcfg: FalModelConfig, *, model_id: str, prompt: str, size: str | None, n: int
    ) -> dict[str, Any]:
        """Assemble the fal request body. Precedence (last wins): base fields →
        size → loosest-safety settings → per-model ``params`` override."""
        body: dict[str, Any] = {"prompt": prompt, "num_images": n}
        _apply_size(body, model_id, size)
        body.update(await self._safety_params(mcfg, model_id))
        if mcfg.params:
            body.update(mcfg.params)
        return body

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
        """Generate a video through fal's queue lifecycle.

        Unlike images, video goes through ``queue.fal.run`` rather than the
        synchronous endpoint: a clip takes minutes, well past what fal.run will
        hold a connection open for. Submit returns a request id — surfaced via
        ``on_upstream_id`` so the job row can be cross-referenced — then we poll
        to completion and fetch the result.

        ``size`` is not forwarded: video models take ``aspect_ratio`` and
        ``resolution`` enums rather than pixel dimensions, and those don't
        follow from a ``WxH`` string. Pin them per model via ``params``.
        """
        del size
        # A still means this is really an image-to-video request, which fal may
        # serve from a sibling endpoint — /v1/videos is one endpoint for both,
        # so the caller has no way to express the difference itself.
        target = await self._reference_variant(model_slug) if input_reference else model_slug
        if target != model_slug:
            log.debug("fal: routing image-to-video for %r to %r", model_slug, target)
        mcfg = self._config_for(model_slug, target)
        body: dict[str, Any] = {"prompt": prompt}
        if input_reference is not None:
            # fal's image-to-video models take the still as `image_url`, and
            # accept a data URI inline — no separate upload step.
            content_type = input_reference_content_type or "image/png"
            encoded = base64.b64encode(input_reference).decode("ascii")
            body["image_url"] = f"data:{content_type};base64,{encoded}"
        derived = await self._derived_params(target)
        if seconds is not None and derived is not None:
            # Duration spellings differ per model ("8s" vs "10"), so the value
            # comes from this model's own enum. See schema.duration_params.
            body.update(duration_params(derived.duration_prop, seconds))
        if mcfg.disable_safety:
            body.update(await self._safety_params(mcfg, target))
        if mcfg.params:
            body.update(mcfg.params)

        # One handler for the whole submit -> poll -> fetch sequence, so a key
        # revoked mid-job is reported like one rejected at submit.
        try:
            return await self._run_video_job(target, body, on_upstream_id)
        except UpstreamAuthError as e:
            self._note_auth_failure(e)
            raise

    async def _run_video_job(
        self,
        model_slug: str,
        body: dict[str, Any],
        on_upstream_id: UpstreamIdCallback | None,
    ) -> GeneratedAsset:
        job = await self.client.submit_queued(model_slug, body)
        if on_upstream_id is not None:
            await on_upstream_id(job.request_id)
        try:
            return await self._await_queued_video(job, model_slug)
        except BaseException:
            # Every way out of here that isn't a finished clip means we've
            # stopped caring about a job fal is still rendering — a client
            # DELETE, our own timeout, a poll that gave up. fal bills the
            # render regardless of whether anyone collects it, so say so on
            # the way out rather than leaving it to run to completion unread.
            await self._cancel_upstream(job, model_slug)
            raise

    async def _await_queued_video(self, job: QueuedRequest, model_slug: str) -> GeneratedAsset:
        deadline = time.monotonic() + self.cfg.video_poll_timeout_seconds
        consecutive_errors = 0
        while True:
            try:
                status = await self.client.poll_queued(job, model_id=model_slug)
                consecutive_errors = 0
            except UpstreamAuthError:
                raise  # a rejected key is permanent; no point polling on
            except UpstreamError as e:
                # A single blip must not discard a clip fal is still rendering
                # (and billing for). Treat a bounded run of transient failures
                # as "not ready yet", as the ComfyUI poller does, and let the
                # deadline bound the loop.
                consecutive_errors += 1
                if consecutive_errors > _MAX_CONSECUTIVE_POLL_ERRORS:
                    raise
                if time.monotonic() >= deadline:
                    raise
                log.warning(
                    "fal: status poll for job %s (%s) failed, %d in a row: %s",
                    job.request_id,
                    model_slug,
                    consecutive_errors,
                    e,
                )
                status = "IN_PROGRESS"
            if status == "COMPLETED":
                break
            if time.monotonic() >= deadline:
                raise GenerationTimeout(
                    f"fal video job {job.request_id} for {model_slug!r} did not finish "
                    f"within {self.cfg.video_poll_timeout_seconds:.0f}s (last status "
                    f"{status!r})"
                )
            await asyncio.sleep(self.cfg.video_poll_interval_seconds)

        result = await self._fetch_video_result(job, model_slug)
        url = extract_video_url(result, model_slug)
        data, content_type = await self._fetch_asset(url)
        return GeneratedAsset(data=data, content_type=content_type, kind="video")

    async def _cancel_upstream(self, job: QueuedRequest, model_slug: str) -> None:
        """Tell fal to stop a job we're abandoning. Never raises.

        Bounded because this runs while unwinding — often mid-cancellation, and
        the scheduler is waiting on us during shutdown. A second cancellation
        arriving here is honoured rather than swallowed: it propagates and
        replaces the error we were already unwinding with.
        """
        try:
            await asyncio.wait_for(
                self.client.cancel_queued(job, model_id=model_slug),
                timeout=_UPSTREAM_CANCEL_TIMEOUT_S,
            )
            log.info("fal: asked to cancel abandoned job %s (%s)", job.request_id, model_slug)
        except Exception as e:
            log.warning(
                "fal: could not cancel abandoned job %s (%s); it may keep rendering "
                "and billing: %s",
                job.request_id,
                model_slug,
                e,
            )

    async def _fetch_video_result(self, job: QueuedRequest, model_slug: str) -> dict[str, Any]:
        """Collect a finished job's payload, retrying transient failures.

        This runs exactly once per job, *after* fal reports COMPLETED — the
        render is done and paid for, so letting one hiccup discard it would be
        the most expensive possible failure in the whole path.
        """
        delay = 1.0
        for attempt in range(1, _RESULT_FETCH_ATTEMPTS + 1):
            try:
                return await self.client.fetch_queued_result(job, model_id=model_slug)
            except UpstreamAuthError:
                raise
            except UpstreamError as e:
                if attempt == _RESULT_FETCH_ATTEMPTS:
                    raise
                log.warning(
                    "fal: fetching result for completed job %s (%s) failed "
                    "(attempt %d/%d), retrying in %.0fs: %s",
                    job.request_id,
                    model_slug,
                    attempt,
                    _RESULT_FETCH_ATTEMPTS,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")  # pragma: no cover

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

    async def _fetch_asset(self, url: str) -> tuple[bytes, str]:
        """Download a generated asset, naming expiry as a suspect if we set one.

        A too-short ``output_expiration_seconds`` surfaces as an ordinary fetch
        failure — the object is simply gone — which is a confusing way to learn
        the setting is wrong. Point at it.
        """
        try:
            return await self.client.fetch_asset(url)
        except (UpstreamAuthError, RateLimited):
            # Neither is an expiry problem, and re-wrapping would flatten them
            # into a generic 502, losing the retry signal a rate limit carries.
            raise
        except UpstreamError as e:
            if self.cfg.output_expiration_seconds is None:
                raise
            raise UpstreamError(
                f"{e.message} — this provider sets output_expiration_seconds="
                f"{self.cfg.output_expiration_seconds}; if the asset expired before the "
                "bridge could download it, raise that value"
            ) from e

    async def _fetch_all(self, urls: list[str]) -> list[GeneratedAsset]:
        # fal returns every image from a single call, so the URLs are all in
        # hand at once — fetch them concurrently rather than serially. Order is
        # preserved; if any fetch exhausts its retries, gather propagates the
        # error and the whole request fails (all-or-nothing, as before).
        assets = await asyncio.gather(*(self._fetch_asset(url) for url in urls))
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
        body = await self._build_body(mcfg, model_id=model_slug, prompt=prompt, size=size, n=n)
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
        # A model listed as text-to-image may have its edits served by a
        # sibling endpoint; send this there so callers don't have to know the
        # `/edit` half exists.
        target = await self._reference_variant(model_slug)
        if target != model_slug:
            log.debug("fal: routing edit for %r to %r", model_slug, target)
        mcfg = self._config_for(model_slug, target)
        # fal's edit endpoints take reference images as ``image_urls``; they
        # accept data URIs inline, so no separate upload step is needed. All
        # supplied references are forwarded in order — a model that only honours
        # one lets the upstream decide, matching the ABC's no-silent-drop rule.
        body = await self._build_body(mcfg, model_id=target, prompt=prompt, size=size, n=n)
        body["image_urls"] = [_data_uri(img) for img in images]
        urls = await self._run_image(target, body)
        return await self._fetch_all(urls)
