"""Provider-id → Backend lookup, plus lifecycle management.

One BackendDispatcher per process; built at startup from the loaded
ProvidersFile and torn down at shutdown.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from .backends.base import Backend
from .backends.comfyui.adapter import ComfyUIBackend
from .backends.fal.adapter import FalBackend
from .backends.imagerouter.adapter import ImageRouterBackend
from .backends.openai.adapter import OpenAIPassthroughBackend
from .backends.openrouter.adapter import OpenRouterBackend
from .backends.venice.adapter import VeniceBackend
from .config import (
    ComfyUIProviderConfig,
    ConfigError,
    FalProviderConfig,
    ImageRouterProviderConfig,
    OpenAIPassthroughProviderConfig,
    OpenRouterProviderConfig,
    ProvidersFile,
    VeniceProviderConfig,
)
from .errors import ProviderNotFound

log = logging.getLogger(__name__)


def _build_backend(cfg) -> Backend:
    if isinstance(cfg, ComfyUIProviderConfig):
        return ComfyUIBackend(cfg)
    if isinstance(cfg, VeniceProviderConfig):
        return VeniceBackend(cfg)
    if isinstance(cfg, ImageRouterProviderConfig):
        return ImageRouterBackend(cfg)
    if isinstance(cfg, OpenRouterProviderConfig):
        return OpenRouterBackend(cfg)
    if isinstance(cfg, FalProviderConfig):
        return FalBackend(cfg)
    if isinstance(cfg, OpenAIPassthroughProviderConfig):
        return OpenAIPassthroughBackend(cfg)
    raise ConfigError(f"Unhandled provider backend type: {type(cfg).__name__}")


class BackendDispatcher:
    def __init__(self, providers: ProvidersFile) -> None:
        self._backends: dict[str, Backend] = {}
        for cfg in providers.providers:
            self._backends[cfg.id] = _build_backend(cfg)
            log.info("Registered provider %r (backend=%s)", cfg.id, cfg.backend)

    def for_provider(self, provider_id: str) -> Backend:
        backend = self._backends.get(provider_id)
        if backend is None:
            raise ProviderNotFound(
                f"No provider configured with id {provider_id!r}",
                param="model",
            )
        return backend

    def all_providers(self) -> Iterable[tuple[str, Backend]]:
        return list(self._backends.items())

    async def aclose(self) -> None:
        for provider_id, backend in self._backends.items():
            try:
                await backend.aclose()
            except Exception as e:
                log.warning("Error closing backend %r: %s", provider_id, e)
        self._backends.clear()
