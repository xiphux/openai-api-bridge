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

from pydantic import BaseModel, Field
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

    def resolve_api_token(self) -> str:
        token = os.environ.get(self.api_token_env)
        if not token:
            raise ConfigError(f"Provider '{self.id}': env var '{self.api_token_env}' is not set")
        return token


class FalModelConfig(BaseModel):
    """One fal.ai model exposed by a ``fal`` provider.

    fal has no "list my models" catalog endpoint the way ImageRouter does —
    each model is its own inference endpoint — so the models a ``fal`` provider
    serves are declared explicitly here. ``id`` is the fal model path used both
    as the ``/v1/models`` slug and as the URL path against ``fal.run`` (e.g.
    ``fal-ai/bytedance/seedream/v4/text-to-image``).
    """

    id: str
    # fal serves image and video; the bridge's fal backend implements the
    # image surface (generate + edit). Only "image" is accepted for now.
    kind: Literal["image"] = "image"
    display_name: str | None = None
    prompt_style: str | None = None
    prompt_hint: str | None = None
    # Loosen content moderation to the minimum this model's family allows.
    # The bridge knows the per-family knob (Seedream -> enable_safety_checker=
    # false, Nano Banana / Gemini image -> safety_tolerance="6"); set false to
    # leave the upstream's own defaults in place. Families the bridge doesn't
    # recognise (e.g. fal's gpt-image wrapper, which exposes no moderation
    # field at all) get nothing injected regardless.
    disable_safety: bool = True
    # Arbitrary extra fields merged into the fal request body, applied last so
    # they override the built-in safety defaults. Escape hatch to pin
    # aspect_ratio/resolution/quality, or to set a safety knob for a family the
    # bridge doesn't special-case.
    params: dict[str, Any] = Field(default_factory=dict)


class FalProviderConfig(BaseModel):
    """fal.ai (https://fal.ai) — model-hosting broker for tier-1 image models
    (Seedream, Nano Banana / Gemini image, GPT Image, FLUX, …).

    Unlike ImageRouter/OpenRouter, fal exposes each model's *native* input
    schema, which includes per-model content-moderation knobs. The bridge's fal
    backend hardcodes the loosest setting per model family (see
    ``FalModelConfig.disable_safety``) so tier-1 models can be driven past the
    over-eager default guardrails other brokers don't let you touch.

    Calls hit fal's synchronous endpoint (``POST https://fal.run/{model_id}``)
    with ``Authorization: Key {token}``; the response carries hosted asset URLs
    the bridge fetches and stores like any other image provider.
    """

    backend: Literal["fal"]
    id: str
    base_url: str = "https://fal.run"
    api_token_env: str
    request_timeout_seconds: float = 600.0
    models: list[FalModelConfig] = Field(default_factory=list)
    # fal's model-catalog API, used to introspect each model's OpenAPI input
    # schema so the moderation knob can be derived rather than hardcoded.
    models_api_url: str = "https://api.fal.ai/v1/models"
    # When true (default), the loosest moderation setting is read from each
    # model's own schema — which keeps working across new model versions and
    # picks up per-model enum ceilings (most accept "1".."6"; flux-2-flex tops
    # out at "5"). Set false to skip the lookup and use the built-in fallback
    # map instead; the bridge also falls back automatically if the fetch fails.
    introspect_safety: bool = True

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
    return BridgeSettings()  # type: ignore[call-arg]  # required field comes from env


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
