"""Backend protocol.

Each provider in the TOML config instantiates one Backend implementation. The
dispatcher routes incoming requests to the right backend by parsing the
``model`` field as ``{provider_id}/{model_slug}``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..errors import UnsupportedOperation

# Async callback invoked once when the upstream backend assigns a job/prompt id.
# The bridge persists this so video jobs can be cross-referenced for debugging.
UpstreamIdCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """One row in the response of `GET /v1/models` for this backend.

    The dispatcher prefixes ``id`` with the provider's id before returning to
    the client. ``kind`` is "image", "video", "chat", "embedding", or None
    when the backend can't tell (e.g. an OpenAI-compat upstream that lists
    every model uniformly without modality hints).

    ``supports_tools`` is a non-standard extension surfaced for gateway-aware
    frontends — the bridge knows per-backend (and sometimes per-model) which
    models accept the OpenAI ``tools`` array. ``None`` means the backend
    didn't say; the client can fall back to its own per-endpoint default.
    """

    id: str
    kind: str | None = None
    display_name: str | None = None
    supports_tools: bool | None = None


@dataclass(frozen=True, slots=True)
class GeneratedAsset:
    """The result of any generation: bytes + content type + kind."""

    data: bytes
    content_type: str
    kind: str  # "image" | "video"


@dataclass(frozen=True, slots=True)
class InputImage:
    """One reference image supplied to an image-edit request."""

    data: bytes
    content_type: str


class Backend(ABC):
    """Async backend protocol. Implementations are *not* expected to be thread-safe;
    they live for the duration of the bridge process and serve all requests for
    a single configured provider."""

    @abstractmethod
    async def list_models(self) -> list[ModelEntry]: ...

    async def generate_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        """Default: not supported. Override in backends that do text-to-image.

        Originally abstract — relaxed when the OpenAI-passthrough backend
        landed (chat/embedding upstreams have no image surface, but the
        Backend ABC is a single union of all backend capabilities).
        """
        raise UnsupportedOperation("Image generation is not supported by this provider")

    async def edit_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        images: list[InputImage],
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        """Default: not supported. Override in backends that do img2img.

        ``images`` carries one or more reference images in client order.
        Backends that can forward multiples (ImageRouter's ``image[]``,
        OpenRouter's per-image content parts, ComfyUI multi-input workflows)
        pass the whole list through. The invariant is that a backend which
        can't use every supplied image must raise rather than silently drop
        one — either by letting the upstream reject it (ImageRouter) or by
        erroring at the bridge (ComfyUI, when a workflow has fewer image
        slots than images supplied).
        """
        raise UnsupportedOperation("Image edits are not supported by this provider")

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
        """Default: not supported. Override in backends that produce video.

        ``on_upstream_id`` is awaited once with the upstream's job id (e.g.
        ComfyUI's prompt_id) as soon as it's known, so the runner can persist
        it to the video_jobs row for resume/debug.
        """
        raise UnsupportedOperation("Video generation is not supported by this provider")

    # --- chat / embedding (OpenAI-passthrough territory) ----------------

    async def chat_completion(
        self,
        body: dict[str, Any],
        *,
        stream: bool,
    ) -> dict[str, Any] | AsyncIterator[bytes]:
        """Forward an OpenAI-shaped chat completion request to the backend.

        Default: not supported. Override in OpenAI-passthrough backends.

        When ``stream=False``, returns the upstream's parsed JSON response.
        When ``stream=True``, returns an async iterator of raw SSE byte chunks
        the bridge will pipe straight to the client without re-parsing — so a
        client's typewriter UI sees tokens land as the upstream emits them.
        The opaque-bytes shape on the streaming path is deliberate: chat
        completions chunks include vendor extensions (function calls, vision,
        tool outputs, JSON mode) we don't need to understand to forward.
        """
        raise UnsupportedOperation("Chat completions are not supported by this provider")

    async def create_embedding(
        self,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Forward an OpenAI-shaped embeddings request. Default: not supported."""
        raise UnsupportedOperation("Embeddings are not supported by this provider")

    # --- lifecycle -------------------------------------------------------

    async def aclose(self) -> None:
        """Optional cleanup hook (e.g. close a shared httpx client)."""
        return
