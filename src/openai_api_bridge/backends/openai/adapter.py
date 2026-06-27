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


def _extract_context_window(m: dict[str, Any]) -> int | None:
    """Best-effort context-window size from a raw upstream /v1/models row.

    The OpenAI spec has no context-size field, so we try the vendor extensions
    in order of trustworthiness, then fall back to parsing a llama.cpp router
    child's launch argv:

      1. ``context_window`` — an already-normalized field, if the upstream is
         itself a gateway that set one.
      2. ``meta.n_ctx`` — llama.cpp's *configured* context (= ``--ctx-size``),
         present only while the model is loaded. ``meta.n_ctx_train`` is the
         model's trained ceiling (often far larger than the real window) and is
         deliberately NOT used.
      3. ``max_model_len`` — vLLM.
      4. ``status.args`` ``--ctx-size``/``-c`` — llama.cpp router mode lists the
         child's argv even when the model is unloaded; the only cold source.
    """

    def positive(v: object) -> int | None:
        if isinstance(v, bool):  # bool is an int subclass — reject it
            return None
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
        return None

    meta = m.get("meta")
    meta_n_ctx = meta.get("n_ctx") if isinstance(meta, dict) else None
    for candidate in (m.get("context_window"), meta_n_ctx, m.get("max_model_len")):
        n = positive(candidate)
        if n is not None:
            return n

    status = m.get("status")
    args = status.get("args") if isinstance(status, dict) else None
    if isinstance(args, list):
        for i, a in enumerate(args):
            if not isinstance(a, str):
                continue
            if a in ("--ctx-size", "-c") and i + 1 < len(args):
                try:
                    n = positive(int(args[i + 1]))
                except (TypeError, ValueError):
                    n = None
                if n is not None:
                    return n
            if a.startswith(("--ctx-size=", "-c=")):
                try:
                    n = positive(int(a.split("=", 1)[1]))
                except (TypeError, ValueError):
                    n = None
                if n is not None:
                    return n
    return None


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
        #
        # `context_window` IS extracted when the upstream exposes it (mainly
        # llama.cpp / vLLM): the bridge otherwise strips the `meta` / `status`
        # blocks that carry it, so a frontend behind the bridge would lose the
        # only signal it has for a "N / max tokens" budget.
        return [
            ModelEntry(
                id=m["id"],
                display_name=m.get("id"),
                context_window=_extract_context_window(m),
            )
            for m in raw
            if "id" in m
        ]

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
