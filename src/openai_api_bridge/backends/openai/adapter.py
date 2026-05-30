"""OpenAI passthrough Backend implementation.

Multiplexer adapter for any OpenAI-compatible upstream — local llama-server
or vLLM, hosted OpenRouter (chat only), Venice's chat surface, or OpenAI
itself. Image generation is intentionally not supported here even though
some upstreams could in principle do it via /v1/images/generations: the
existing image flow assumes the bridge owns the file-store side, and the
plumbing for "passthrough image" is meaningfully different from "translate
ComfyUI/Venice." Defer until someone actually needs it.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from ...config import OpenAIPassthroughProviderConfig
from ..base import Backend, ModelEntry
from .client import OpenAIClient

log = logging.getLogger(__name__)


class OpenAIPassthroughBackend(Backend):
    def __init__(self, cfg: OpenAIPassthroughProviderConfig) -> None:
        self.cfg = cfg
        self.client = OpenAIClient(
            base_url=cfg.base_url,
            api_token=cfg.resolve_api_token(),
            request_timeout_seconds=cfg.request_timeout_seconds,
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def list_models(self) -> list[ModelEntry]:
        raw = await self.client.list_models()
        # `kind` is intentionally None — OpenAI's /v1/models doesn't expose
        # modality, and trying to guess from name patterns ("text-embedding-*"
        # → embedding) is fragile. Routing happens by *endpoint*, not kind:
        # /v1/chat/completions handles chat models, /v1/embeddings handles
        # embedding models, and the upstream sorts out which is which.
        #
        # `supports_tools` is also intentionally None: this backend
        # multiplexes for *any* OpenAI-compatible upstream — actual OpenAI
        # (where all chat models support tools), local llama-server (where
        # small models often don't), vLLM, OpenRouter's chat surface, etc.
        # The bridge can't tell from the upstream's catalog. Clients fall
        # back to their per-endpoint config flag.
        return [ModelEntry(id=m["id"], display_name=m.get("id")) for m in raw if "id" in m]

    async def chat_completion(
        self,
        body: dict[str, Any],
        *,
        stream: bool,
    ) -> dict[str, Any] | AsyncIterator[bytes]:
        if stream:
            return await self.client.chat_completion_stream(body)
        return await self.client.chat_completion(body)

    async def create_embedding(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self.client.create_embedding(body)
