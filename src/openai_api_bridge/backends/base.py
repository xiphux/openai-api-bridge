"""Backend protocol.

Each provider in the TOML config instantiates one Backend implementation. The
dispatcher routes incoming requests to the right backend by parsing the
``model`` field as ``{provider_id}/{model_slug}``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..errors import UnsupportedOperation

# Async callback invoked once when the upstream backend assigns a job/prompt id.
# The bridge persists this so video jobs can be cross-referenced for debugging.
UpstreamIdCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """One row in the response of `GET /v1/models` for this backend.

    The dispatcher prefixes ``id`` with the provider's id before returning to
    the client. ``kind`` is "image" or "video".
    """

    id: str
    kind: str
    display_name: str | None = None


@dataclass(frozen=True, slots=True)
class GeneratedAsset:
    """The result of any generation: bytes + content type + kind."""

    data: bytes
    content_type: str
    kind: str  # "image" | "video"


class Backend(ABC):
    """Async backend protocol. Implementations are *not* expected to be thread-safe;
    they live for the duration of the bridge process and serve all requests for
    a single configured provider."""

    @abstractmethod
    async def list_models(self) -> list[ModelEntry]:
        ...

    @abstractmethod
    async def generate_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        ...

    async def edit_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        image: bytes,
        image_content_type: str,
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        """Default: not supported. Override in backends that do img2img."""
        raise UnsupportedOperation(
            "Image edits are not supported by this provider"
        )

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
        raise UnsupportedOperation(
            "Video generation is not supported by this provider"
        )

    async def aclose(self) -> None:
        """Optional cleanup hook (e.g. close a shared httpx client)."""
        return
