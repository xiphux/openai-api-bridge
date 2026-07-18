"""Venice Backend implementation.

Venice supports text-to-image (``/image/generate``) and image-to-image
(``/image/edit``), but no video. ``edit_image`` routes to Venice's dedicated
edit endpoint; ``generate_video`` raises ``UnsupportedOperation`` from the base
class (overridden here for a clearer message). Venice edits are single-image,
so more than one reference is rejected rather than silently dropped.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

from ...config import VeniceProviderConfig
from ...errors import InvalidRequest, UnsupportedOperation
from ...util.cache import AsyncTTLCache
from ...util.sizes import parse_size
from ..base import Backend, GeneratedAsset, InputImage, ModelEntry, make_capabilities
from .client import VeniceClient

log = logging.getLogger(__name__)

# Venice names a model's image-to-image half by suffixing the base id, and
# files it under a separate model type: `gpt-image-2` (type=image) pairs with
# `gpt-image-2-edit` (type=inpaint). Same idea as fal's `/edit` endpoints, so
# it gets the same treatment — list one model, route by request shape — which
# also means the edit half is reachable at all. It previously wasn't: only
# type=image was listed, and an edit sent to `gpt-image-2` was rejected by
# Venice, whose edit endpoint only accepts the `-edit` ids.
_EDIT_SUFFIX = "-edit"

# Without a cooldown every edit during an outage would re-run the catalogue
# fetch — and because waiters acquire the lock in turn rather than sharing one
# result, N concurrent edits serialise into N sequential attempts. This is the
# same shape as fal's *catalogue* cooldown (not its schema introspection):
# while the window is open the routes stay empty and edits go out unrouted,
# so the residual cost is a bounded tail after the upstream recovers. The
# window is configurable via `route_retry_seconds`.

# Venice's image-generation endpoint always returns PNG; the API doesn't
# include a content-type per image so we hard-code it (matches existing pipe).
_VENICE_CONTENT_TYPE = "image/png"


class VeniceBackend(Backend):
    def __init__(self, cfg: VeniceProviderConfig) -> None:
        self.cfg = cfg
        self.client = VeniceClient(
            base_url=cfg.base_url,
            api_token=cfg.resolve_api_token(),
        )
        # base id -> its "-edit" counterpart, learned from the catalogue.
        self._edit_routes: dict[str, str] = {}
        self._routes_lock = asyncio.Lock()
        self._routes_loaded = False
        self._routes_failed_at: float | None = None
        # A degraded listing (inpaint half missing) is cached too, but only
        # for the short failure window (clamped to the TTL) — long enough that a burst can't queue
        # up behind the lock, short enough that the missing half is retried
        # soon rather than pinned for the full TTL.
        # Caches ``(entries, complete)``: whether both halves arrived decides
        # how long the result is worth keeping, so it travels with the value
        # rather than through a side-channel flag.
        self._catalog: AsyncTTLCache[tuple[list[ModelEntry], bool]] = AsyncTTLCache(
            cfg.catalog_ttl_seconds, cfg.catalog_retry_seconds
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def list_models(self) -> list[ModelEntry]:
        # A listing whose inpaint half failed is still worth serving — dropping
        # the provider over its narrower query is what an earlier review
        # rejected — but it is *incomplete*, so it's held only for the failure
        # window (clamped to the TTL, which an override may shorten but never
        # extend) rather than for the full TTL. Without caching it at all,
        # a burst during an inpaint hang would queue up behind the lock and
        # each waiter would start its own fetch: the exact storm the failure
        # cooldown exists to damp, reached without ever raising.
        entries, _complete = await self._catalog.get(
            self._fetch_catalog,
            ttl_for=lambda result: (
                self.cfg.catalog_ttl_seconds if result[1] else self.cfg.catalog_retry_seconds
            ),
        )
        return entries

    async def _fetch_catalog(self) -> tuple[list[ModelEntry], bool]:
        # Two independent listings, fetched together — neither feeds the other,
        # and serialising them would double this endpoint's tail latency while
        # every other provider waits behind it.
        # Sequence[Any] because return_exceptions widens each slot to
        # "result or exception", which defeats inference on a tuple unpack.
        results: Sequence[Any] = await asyncio.gather(
            self.client.list_image_models("image"),
            self.client.list_image_models("inpaint"),
            return_exceptions=True,
        )
        generate_result, edit_result = results[0], results[1]
        # The generate listing is the provider's reason to exist; without it
        # there is nothing to serve.
        if isinstance(generate_result, BaseException):
            raise generate_result
        # The inpaint listing is not. `type=inpaint` is a narrower query than
        # `type=image`, so a Venice-compatible proxy or an older API version
        # could reject it outright — and letting that take the whole provider
        # down with it would drop a perfectly healthy text-to-image catalogue
        # from /v1/models. Degrade instead, as the fal backend does.
        edit: list[dict[str, Any]] = []
        edit_available = not isinstance(edit_result, BaseException)
        if edit_available:
            edit = edit_result
        else:
            log.warning(
                "Venice: inpaint catalogue unavailable (%s); listing text-to-image "
                "models only, and edits will not be routed until it recovers",
                edit_result,
            )

        gen_ids = [m["id"] for m in generate_result if isinstance(m, dict) and "id" in m]
        edit_ids = [m["id"] for m in edit if isinstance(m, dict) and "id" in m]

        routes = {
            e[: -len(_EDIT_SUFFIX)]: e
            for e in edit_ids
            if e.endswith(_EDIT_SUFFIX) and e[: -len(_EDIT_SUFFIX)] in set(gen_ids)
        }
        if edit_available:
            # Only treat routing as resolved when the half it depends on was
            # actually read, so a degraded listing retries later instead of
            # latching an empty map.
            self._edit_routes = routes
            self._routes_loaded = True
            self._routes_failed_at = None
        else:
            self._routes_failed_at = time.monotonic()
        collapsed = set(routes.values())

        entries = [
            ModelEntry(
                id=model_id,
                kind="image",
                display_name=model_id,
                capabilities=make_capabilities(
                    ["text", "image"] if model_id in routes else ["text"], "image"
                ),
            )
            for model_id in gen_ids
        ]
        # Edit models with no generate half — Venice's uncensored//inpaint-only
        # entries — stay listed in their own right; they just can't do t2i.
        entries += [
            ModelEntry(
                id=model_id,
                kind="image",
                display_name=model_id,
                capabilities=make_capabilities(["image"], "image"),
            )
            for model_id in edit_ids
            if model_id not in collapsed
        ]
        return entries, edit_available

    def _in_route_cooldown(self) -> bool:
        if self._routes_failed_at is None:
            return False
        return time.monotonic() - self._routes_failed_at < self.cfg.route_retry_seconds

    async def _edit_target(self, model_slug: str) -> str:
        """The model id Venice's edit endpoint will actually accept.

        Loads the catalogue if it hasn't been read yet, so routing doesn't
        depend on whether some earlier request happened to populate it. A
        failed read is remembered for a cooldown, so an outage costs one
        attempt per window rather than one per request.
        """
        if not self._routes_loaded and not self._in_route_cooldown():
            async with self._routes_lock:
                if not self._routes_loaded and not self._in_route_cooldown():
                    try:
                        await self.list_models()
                    except Exception as e:  # fall through to the id as given
                        self._routes_failed_at = time.monotonic()
                        log.warning("Venice: could not load model list for edit routing: %s", e)
        return self._edit_routes.get(model_slug, model_slug)

    async def generate_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        w, h = parse_size(size)
        width = w or self.cfg.default_width
        height = h or self.cfg.default_height

        out: list[GeneratedAsset] = []
        for _ in range(n):
            data = await self.client.generate_image(
                model=model_slug,
                prompt=prompt,
                width=width,
                height=height,
                steps=self.cfg.steps,
                cfg_scale=self.cfg.cfg_scale,
            )
            out.append(GeneratedAsset(data=data, content_type=_VENICE_CONTENT_TYPE, kind="image"))
        return out

    async def edit_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        images: list[InputImage],
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        # Venice's /image/edit takes exactly one reference image. Reject extras
        # rather than silently dropping them (the edit_image contract).
        if len(images) > 1:
            raise InvalidRequest(
                f"Venice image edits accept exactly one reference image (got {len(images)})",
                param="image",
            )
        # /image/edit uses aspect_ratio/resolution, not OpenAI's size string;
        # we let Venice infer from the source image rather than guess a mapping.
        del size
        # Venice's edit endpoint only accepts the "-edit" ids, so a request
        # naming the base model has to be routed to its counterpart.
        target = await self._edit_target(model_slug)
        if target != model_slug:
            log.debug("Venice: routing edit for %r to %r", model_slug, target)
        image = images[0]
        out: list[GeneratedAsset] = []
        for _ in range(n):
            data, content_type = await self.client.edit_image(
                model=target,
                prompt=prompt,
                image=image.data,
                image_content_type=image.content_type,
            )
            out.append(GeneratedAsset(data=data, content_type=content_type, kind="image"))
        return out

    async def generate_video(
        self,
        *,
        model_slug: str,
        prompt: str,
        size: str | None = None,
        seconds: float | None = None,
        input_reference: bytes | None = None,
        input_reference_content_type: str | None = None,
        on_upstream_id=None,
    ) -> GeneratedAsset:
        raise UnsupportedOperation(
            "Venice does not support video generation. "
            "Use a ComfyUI provider with a video workflow."
        )
