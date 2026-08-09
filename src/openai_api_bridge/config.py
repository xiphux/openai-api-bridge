"""Bridge configuration: env-var infrastructure settings + TOML provider definitions.

Two layers:

* `BridgeSettings` — read from env (`BRIDGE_*`, `FILES_DIR`, etc.) via pydantic-settings.
* `ProvidersFile` — read from a TOML file pointed to by `BRIDGE_CONFIG_PATH`. Each
  provider is a discriminated union member keyed on `backend`.

Secrets in the TOML are stored by the *name of an env var* (any field whose schema
name ends in ``_env``). The actual secret is read from `os.environ` at the point of
use by the backend adapter — never copied into pydantic state.
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import BridgeError, InvalidRequest


class ConfigError(BridgeError):
    """Raised when the bridge's own config is unusable. Surfaces as 500 if
    it ever escapes startup, but normally aborts process startup."""

    status_code = 500
    error_type = "api_error"
    code = "configuration_error"


class BridgeSettings(BaseSettings):
    """Infrastructure & secrets. Read once at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: str = Field(..., alias="BRIDGE_API_KEY")
    host: str = Field(default="0.0.0.0", alias="BRIDGE_HOST")
    port: int = Field(default=8080, alias="BRIDGE_PORT")
    public_base_url: str = Field(default="", alias="BRIDGE_PUBLIC_BASE_URL")
    config_path: Path = Field(
        default=Path("/etc/openai-api-bridge/config.toml"),
        alias="BRIDGE_CONFIG_PATH",
    )
    files_dir: Path = Field(
        default=Path("/var/lib/openai-api-bridge/files"),
        alias="FILES_DIR",
    )
    sqlite_path: Path = Field(
        default=Path("/var/lib/openai-api-bridge/state.db"),
        alias="SQLITE_PATH",
    )
    retention_days: int = Field(default=30, alias="RETENTION_DAYS")
    max_cache_gb: int = Field(default=50, alias="MAX_CACHE_GB")
    eviction_interval_seconds: int = Field(default=600, alias="EVICTION_INTERVAL_SECONDS")
    max_concurrent_video_jobs: int = Field(default=2, alias="MAX_CONCURRENT_VIDEO_JOBS")
    # How long `GET /v1/models` waits for any one provider's catalogue before
    # leaving it out of *this* listing. The endpoint awaits every provider at
    # once, so without a bound its latency is the slowest upstream's read
    # timeout — two minutes for an openai-passthrough provider whose upstream
    # is wedged, on every single model-picker refresh, taking every healthy
    # provider's models with it.
    #
    # The provider isn't dropped permanently: the fetch is left running so its
    # own catalogue cache still fills, and the next request serves it from
    # there. A cold fal catalogue (10-13 paginated round trips) legitimately
    # exceeds this on the first request after boot and appears on the second.
    models_timeout_seconds: float = Field(default=5.0, alias="MODELS_TIMEOUT_SECONDS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def max_cache_bytes(self) -> int:
        return self.max_cache_gb * 1024**3


# --- Provider config (TOML-backed) ------------------------------------------


class ComfyUIProviderConfig(BaseModel):
    backend: Literal["comfyui"]
    id: str
    url: str = "http://127.0.0.1:8188"
    workflows_dir: Path
    poll_interval_seconds: float = 1.0
    poll_timeout_image_seconds: float = 300.0
    poll_timeout_video_seconds: float = 900.0
    cache_workflows: bool = True


class VeniceProviderConfig(BaseModel):
    backend: Literal["venice"]
    id: str
    base_url: str = "https://api.venice.ai"
    api_token_env: str
    # Venice diffusion knobs. Defaults match the legacy pipe; tunable per
    # provider in TOML if a user wants higher/lower fidelity by default.
    steps: int = 16
    cfg_scale: float = 4.0
    default_width: int = 1024
    default_height: int = 1024
    # Cooldown before re-reading the model catalogue after a failed attempt to
    # resolve edit routing. During the window edits go out unrouted (Venice's
    # edit endpoint only accepts the "-edit" ids, so they fail) — the cost is a
    # bounded tail after Venice recovers, traded against re-fetching the
    # catalogue on every single edit while it's down.
    route_retry_seconds: float = 60.0
    # How long a successfully-read model catalogue is reused before being
    # re-fetched. /v1/models costs two upstream calls on Venice (the
    # text-to-image and image-to-image listings), and edit routing reads the
    # same catalogue, so without this every listing request and every
    # first-of-process edit pays for both. A TTL rather than a permanent cache
    # so models Venice adds appear without restarting the bridge. 0 disables
    # caching entirely.
    catalog_ttl_seconds: float = 300.0
    # After a failed catalogue fetch, how long before another is attempted.
    # The fetch runs under a lock, so without this a burst arriving during an
    # upstream hang would each start their own fetch and queue behind one
    # another; instead the first pays the timeout and the rest fail fast.
    # 0 retries immediately. This also bounds how long an *incomplete* listing
    # is served — for that, whichever of this and catalog_ttl_seconds is
    # shorter applies — so when that bound exceeds route_retry_seconds it, not
    # that knob, is what governs when edit routing recovers.
    catalog_retry_seconds: float = 30.0

    def resolve_api_token(self) -> str:
        token = os.environ.get(self.api_token_env)
        if not token:
            raise ConfigError(f"Provider '{self.id}': env var '{self.api_token_env}' is not set")
        return token


class ImageRouterProviderConfig(BaseModel):
    """ImageRouter (https://imagerouter.io) — partial-OpenAI gateway with
    image + video generation across many providers.

    Path-divergent: model catalog at ``/v1/models``, inference under
    ``/v1/openai/{images,videos}/...``, video endpoint is sync. The
    bridge's imagerouter backend smooths these over so clients see
    OpenAI-shaped traffic.
    """

    backend: Literal["imagerouter"]
    id: str
    base_url: str = "https://api.imagerouter.io"
    api_token_env: str
    # How long the model catalogue is reused before being re-fetched.
    # /v1/models fans out to every provider on every request, so without this
    # each client refresh costs an upstream round trip. A TTL rather than a
    # permanent cache so newly added models appear without a restart.
    # 0 disables caching.
    catalog_ttl_seconds: float = 300.0
    # After a failed catalogue fetch, how long before another is attempted.
    # The fetch runs under a lock, so without this a burst arriving during an
    # upstream hang would each start their own fetch and queue behind one
    # another; instead the first pays the timeout and the rest fail fast.
    # 0 retries immediately.
    catalog_retry_seconds: float = 30.0

    def resolve_api_token(self) -> str:
        token = os.environ.get(self.api_token_env)
        if not token:
            raise ConfigError(f"Provider '{self.id}': env var '{self.api_token_env}' is not set")
        return token


class OpenRouterProviderConfig(BaseModel):
    """OpenRouter (https://openrouter.ai) — multi-vendor aggregator that's
    OpenAI-compatible for chat and embeddings but routes image generation
    through chat completions with a non-standard ``message.images``
    response field. The bridge's openrouter backend composes the OpenAI
    passthrough client for spec-compliant surfaces and layers image-via-
    chat translation on top, so clients see a single OpenAI-shaped API.
    """

    backend: Literal["openrouter"]
    id: str
    base_url: str = "https://openrouter.ai/api"
    api_token_env: str
    request_timeout_seconds: float = 120.0
    # How long the model catalogue is reused before being re-fetched.
    # /v1/models fans out to every provider on every request, so without this
    # each client refresh costs an upstream round trip. A TTL rather than a
    # permanent cache so newly added models appear without a restart.
    # 0 disables caching.
    catalog_ttl_seconds: float = 300.0
    # After a failed catalogue fetch, how long before another is attempted.
    # The fetch runs under a lock, so without this a burst arriving during an
    # upstream hang would each start their own fetch and queue behind one
    # another; instead the first pays the timeout and the rest fail fast.
    # 0 retries immediately.
    catalog_retry_seconds: float = 30.0

    def resolve_api_token(self) -> str:
        token = os.environ.get(self.api_token_env)
        if not token:
            raise ConfigError(f"Provider '{self.id}': env var '{self.api_token_env}' is not set")
        return token


# Below this, a generated asset would plausibly expire before the bridge could
# retrieve it — the fetch retries with backoff, and video detects completion by
# polling first. Not a guarantee, just a floor that rejects values that cannot
# work; see FalProviderConfig.output_expiration_seconds.
_MIN_OUTPUT_EXPIRATION_SECONDS = 60


class FalModelConfig(BaseModel):
    """One fal.ai model exposed by a ``fal`` provider.

    Per-model settings for a model served by a ``fal`` provider. These are
    *overrides*: with ``discover_models`` on (the default) the served set comes
    from fal's model API, and an entry here only adjusts its match — it doesn't
    restrict what else is listed. With discovery off, these entries are the
    authoritative list. ``id`` is the fal model path, used both as the
    ``/v1/models`` slug and as the URL path against ``fal.run`` (e.g.
    ``fal-ai/bytedance/seedream/v4/text-to-image``).
    """

    id: str
    # Surfaced on /v1/models as a hint. With discovery on this is taken from
    # the catalogue category; set it here only for a model you're declaring
    # explicitly. The bridge picks the actual code path from the request shape
    # (POST /v1/images vs /v1/videos), not from this field.
    kind: Literal["image", "video"] = "image"
    display_name: str | None = None
    prompt_style: str | None = None
    prompt_hint: str | None = None
    # Loosen content moderation as far as this model allows. The setting is read
    # from the model's own OpenAPI schema (see ``backends/fal/safety.py``); set
    # false to leave the upstream's own defaults in place. Models exposing no
    # knob the bridge recognises — e.g. fal's gpt-image wrapper, which has no
    # moderation field at all — get nothing injected regardless.
    disable_safety: bool = True
    # Arbitrary extra fields merged into the fal request body, applied last so
    # they override the derived safety setting. Escape hatch to pin
    # aspect_ratio/resolution/quality, or to set a knob the bridge doesn't
    # recognise.
    params: dict[str, Any] = Field(default_factory=dict)


class FalProviderConfig(BaseModel):
    """fal.ai (https://fal.ai) — model-hosting broker for tier-1 image models
    (Seedream, Nano Banana / Gemini image, GPT Image, FLUX, …).

    Unlike ImageRouter/OpenRouter, fal exposes each model's *native* input
    schema, which includes per-model content-moderation knobs. The bridge reads
    that schema and sets the knob to its loosest value (see
    ``backends/fal/safety.py``), so tier-1 models can be driven past the
    over-eager default guardrails other brokers don't let you touch — and new
    model versions are picked up without a code change.

    Calls hit fal's synchronous endpoint (``POST https://fal.run/{model_id}``)
    with ``Authorization: Key {token}``; the response carries hosted asset URLs
    the bridge fetches and stores like any other image provider.
    """

    backend: Literal["fal"]
    id: str
    base_url: str = "https://fal.run"
    api_token_env: str
    request_timeout_seconds: float = 600.0
    # Per-model overrides (moderation, extra body params, prompt metadata).
    # These do NOT restrict what's served while ``discover_models`` is on — set
    # that false to serve only what's listed here.
    models: list[FalModelConfig] = Field(default_factory=list)
    # When true (default), ``/v1/models`` is populated from fal's model API,
    # filtered to ``categories``. False serves only the ``models`` above, and
    # a model id outside that list is a 404.
    discover_models: bool = True
    # fal categories to surface. None uses the backend's own set — the ones it
    # can actually serve: text-to-image, image-to-image, text-to-video and
    # image-to-video. fal also publishes audio/3d categories; listing those
    # here would advertise models the fal backend has no code path for.
    categories: list[str] | None = None
    # fal splits a model's text-driven and reference-image halves across two
    # endpoints — `fal-ai/nano-banana-2` and `.../edit`, `fal-ai/veo3.1` and
    # `.../image-to-video` — which is easy to pick wrong. When true (default)
    # the bridge lists only the text-driven id for pairs it can identify
    # confidently, and routes requests carrying a reference image to the
    # sibling. Models without an identifiable partner — including every
    # reference-only endpoint — are listed unchanged. The merged entry
    # advertises both halves in its `capabilities`, so a client can still tell
    # whether an image may be attached.
    collapse_variants: bool = True
    # fal's model API, used both to list models and to introspect each model's
    # OpenAPI input schema so the moderation knob can be derived, not hardcoded.
    models_api_url: str = "https://api.fal.ai/v1/models"
    # fal's queue host, used for video. Images run against the synchronous
    # endpoint; a video clip takes minutes, past what that will hold open.
    queue_base_url: str = "https://queue.fal.run"
    video_poll_interval_seconds: float = 3.0
    video_poll_timeout_seconds: float = 1800.0
    # --- upstream data retention -------------------------------------------
    # fal keeps request payloads (your prompts, and any inline input images)
    # for 30 days by default, and serves generated media from a public CDN.
    # These two knobs hand that back: the bridge has already downloaded the
    # asset into its own FileStore by the time a request completes, so the
    # upstream copies are redundant for a self-hosted setup.
    #
    # False sends `X-Fal-Store-IO: 0`, so fal never stores the payload at all.
    store_payloads: bool = True
    # Seconds until generated media expires from fal's CDN, via
    # `X-Fal-Object-Lifecycle-Preference`. None leaves fal's default (no
    # expiry). Mind the retrieval budget: the clock starts when fal creates the
    # object, and the bridge still has to notice, fetch, and possibly retry.
    # An asset fetch alone can span three attempts with backoff, and video adds
    # poll detection plus a retried result fetch before that — so short values
    # risk the asset expiring out from under a generation you already paid for.
    # A few hundred seconds is comfortable; the floor below just rejects values
    # that could never work.
    output_expiration_seconds: int | None = None

    @field_validator("output_expiration_seconds")
    @classmethod
    def _expiration_must_allow_retrieval(cls, v: int | None) -> int | None:
        if v is not None and v < _MIN_OUTPUT_EXPIRATION_SECONDS:
            raise ValueError(
                f"output_expiration_seconds={v} is below the {_MIN_OUTPUT_EXPIRATION_SECONDS}s "
                "floor — the bridge would likely lose the generated asset before it "
                "could download it (fetches retry with backoff, and video polls for "
                "completion first). Use a few hundred seconds for comfort."
            )
        return v

    # When true (default), request settings are read from each model's own
    # schema — the loosest moderation value, and the accepted spelling of
    # `duration` for video. That keeps working across new model versions and
    # picks up per-model enums (moderation is "1".."6" on most models but tops
    # out at "5" on flux-2-flex; duration is "8s" on veo3 and "10" on Kling).
    # False skips the lookup: moderation falls back to a small built-in map and
    # video duration is left to the model's default. The bridge also falls back
    # automatically if the fetch fails.
    introspect_safety: bool = True
    # Cooldown before a *failed* introspection is retried. Until it elapses,
    # requests use the built-in fallback map, so a fal outage costs one round
    # trip per window rather than one per request — but it heals on its own
    # instead of staying degraded until the process restarts. 0 retries on the
    # very next request.
    introspect_retry_seconds: float = 300.0

    def resolve_api_token(self) -> str:
        token = os.environ.get(self.api_token_env)
        if not token:
            raise ConfigError(f"Provider '{self.id}': env var '{self.api_token_env}' is not set")
        return token


class OpenAIPassthroughProviderConfig(BaseModel):
    """Generic OpenAI-API-compatible upstream (llama-server, vLLM, OpenAI itself,
    Venice's chat endpoint, any vendor that speaks the OpenAI wire protocol).

    Requests go through unchanged — we don't translate field names or response
    shapes. The bridge only routes, prefixes model ids, and aggregates the
    upstream's /v1/models listing alongside everything else.
    """

    backend: Literal["openai"]
    id: str
    base_url: str
    # Optional: many local OpenAI-compat servers (llama-server, vLLM with no
    # auth, lmstudio) require no Authorization header. Leave api_token_env
    # unset and the bridge sends nothing.
    api_token_env: str | None = None
    # Per-request timeout for non-streaming calls (seconds). Streaming ignores
    # this — the bridge holds the connection open for as long as the upstream
    # keeps writing.
    request_timeout_seconds: float = 120.0
    # How long the model catalogue is reused before being re-fetched. Same
    # knobs, and the same reasoning, as every other backend: /v1/models fans
    # out to every provider on every request, so an uncached listing costs an
    # upstream round trip per client refresh. This backend went without one
    # for longer than the others, which made it the sole reason a wedged
    # upstream could stall the whole endpoint on every request rather than
    # once per window. 0 disables caching.
    #
    # Note for llama.cpp: a model's `meta.n_ctx` only appears while it is
    # loaded, so a cached listing can report a context window the upstream has
    # since unloaded. Router mode also publishes `status.args`, which is read
    # as a cold fallback, so the field survives the model being swapped out.
    catalog_ttl_seconds: float = 300.0
    # After a failed catalogue fetch, how long before another is attempted.
    # The fetch runs under a lock, so without this a burst arriving during an
    # upstream hang would each start their own fetch and queue behind one
    # another; instead the first pays the timeout and the rest fail fast.
    # 0 retries immediately.
    catalog_retry_seconds: float = 30.0

    def resolve_api_token(self) -> str | None:
        if not self.api_token_env:
            return None
        token = os.environ.get(self.api_token_env)
        if not token:
            raise ConfigError(f"Provider '{self.id}': env var '{self.api_token_env}' is not set")
        return token


ProviderConfig = Annotated[
    ComfyUIProviderConfig
    | VeniceProviderConfig
    | ImageRouterProviderConfig
    | OpenRouterProviderConfig
    | FalProviderConfig
    | OpenAIPassthroughProviderConfig,
    Field(discriminator="backend"),
]


class Defaults(BaseModel):
    cache_workflows: bool = True


class ProvidersFile(BaseModel):
    defaults: Defaults = Field(default_factory=Defaults)
    providers: list[ProviderConfig] = Field(default_factory=list)

    def by_id(self, provider_id: str) -> ProviderConfig | None:
        for p in self.providers:
            if p.id == provider_id:
                return p
        return None


def load_providers(path: Path) -> ProvidersFile:
    """Read and validate the providers TOML file.

    Raises ConfigError on missing file, parse error, or schema violation.
    """
    if not path.exists():
        raise ConfigError(f"Providers config file not found: {path}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Failed to parse {path}: {e}") from e
    file = ProvidersFile.model_validate(data)
    _enforce_unique_ids(file)
    return file


def _enforce_unique_ids(file: ProvidersFile) -> None:
    seen: set[str] = set()
    for p in file.providers:
        if p.id in seen:
            raise ConfigError(f"Duplicate provider id: {p.id!r}")
        seen.add(p.id)


# --- Model id parsing -------------------------------------------------------


def parse_model_id(model_id: str) -> tuple[str, str]:
    """Split an OpenAI-shaped `model` field into (provider_id, model_slug).

    Pattern: "{provider_id}/{model_slug}". Both parts must be non-empty.
    """
    if "/" not in model_id:
        raise InvalidRequest(
            f"Model id must be 'provider/model' (got {model_id!r})",
            param="model",
        )
    provider_id, _, slug = model_id.partition("/")
    if not provider_id or not slug:
        raise InvalidRequest(
            f"Model id must be 'provider/model' (got {model_id!r})",
            param="model",
        )
    return provider_id, slug


# --- Singleton accessors (FastAPI dependency-friendly) ----------------------


@lru_cache(maxsize=1)
def get_settings() -> BridgeSettings:
    # api_key is required but supplied by the environment, not the caller.
    return BridgeSettings()


_providers_cache: ProvidersFile | None = None


def get_providers() -> ProvidersFile:
    """Return the loaded ProvidersFile. Caller must have called `init_providers` first."""
    if _providers_cache is None:
        raise RuntimeError("Providers not loaded — call init_providers() at startup")
    return _providers_cache


def init_providers(path: Path | None = None) -> ProvidersFile:
    """Load (or reload) the providers TOML and update the module-level cache."""
    global _providers_cache
    target = path or get_settings().config_path
    _providers_cache = load_providers(target)
    return _providers_cache


def reset_caches_for_tests() -> None:
    """Test hook to clear cached settings/providers."""
    global _providers_cache
    get_settings.cache_clear()
    _providers_cache = None
