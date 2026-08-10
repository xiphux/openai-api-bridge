"""``GET /v1/models`` — flat enumeration across all configured providers."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, Request

from ..auth import require_api_key
from ..backends.base import Backend, ModelEntry
from ..errors import BridgeError
from ..infra.tasks import SingleFlight
from ..resources import resources

log = logging.getLogger(__name__)

router = APIRouter()

# Catalogue fetches that outlive the request that started them, one per
# provider. See SingleFlight for why the entry is created with the task rather
# than when the starter gives up, and why removal is identity-checked.
_lingering: SingleFlight[list[ModelEntry]] = SingleFlight()


async def drain_lingering() -> None:
    """Stop orphaned catalogue fetches. Called from the app's shutdown path.

    Ordered *before* the dispatcher closes its backends: these tasks are
    awaiting on an httpx2 client ``BackendDispatcher.aclose`` is about to close
    underneath them, which fails silently since ``_entries_for`` swallows
    everything — and the loop would otherwise be torn down with them pending.
    """
    await _lingering.cancel_all()


async def _entries_for(provider_id: str, backend: Backend) -> list[ModelEntry]:
    """One provider's catalogue, or an empty list if it failed."""
    try:
        return await backend.list_models()
    except BridgeError as e:
        # One flaky provider shouldn't break the whole listing.
        log.warning("Provider %r list_models failed: %s", provider_id, e.message)
        return []
    except Exception:
        # Same intent, but the guarantee can't rest on every adapter
        # remembering to wrap its upstream errors: a bare httpx2 error or an
        # unexpected catalogue shape would otherwise 500 the whole endpoint
        # and take every healthy provider's models with it.
        log.exception("Provider %r list_models raised an unexpected error", provider_id)
        return []


async def _entries_within(provider_id: str, backend: Backend, budget: float) -> list[ModelEntry]:
    """``_entries_for``, but bounded — a slow provider sits out this listing.

    Containment used to cover failure but not *slowness*: the endpoint awaits
    every provider at once, so its latency was the maximum over providers of
    their worst-case upstream read timeout. One wedged upstream stalled the
    whole listing, healthy providers included, on every request.

    The fetch is shielded rather than cancelled, which is the part that makes
    this safe to do. Every backend caches its catalogue, and a cancelled fetch
    populates nothing — so cancelling here would drop a merely-slow provider
    from *every* listing forever, each request killing the fetch that would
    have made the next one fast. Left running, it fills the backend's own
    cache and the provider reappears on a subsequent request.

    A request arriving while a previous one's fetch is still running **joins**
    that fetch rather than starting a second one. Joining rather than declining
    outright is the useful choice: the wait is bounded by this request's own
    budget either way, so joining costs nothing extra and returns real data
    whenever the in-flight fetch happens to land inside it — where declining
    guarantees the provider stays missing for as long as the fetch runs.

    A non-positive budget disables the bound, for an operator who would rather
    wait than see a provider omitted.
    """
    if budget <= 0:
        return await _entries_for(provider_id, backend)
    task = _lingering.join_or_start(
        provider_id,
        lambda: _entries_for(provider_id, backend),
        name=f"list-models-{provider_id}",
    )
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=budget)
    except TimeoutError:
        # No bookkeeping on the way out: the task has been tracked since it was
        # created, so it is already held and already drainable — including when
        # this request is cancelled rather than timing out.
        log.warning(
            "Provider %r did not return its catalogue within %.1fs; omitting it from "
            "this listing while the fetch continues in the background",
            provider_id,
            budget,
        )
        return []


@router.get("/v1/models", dependencies=[Depends(require_api_key)])
async def list_models(request: Request) -> dict[str, Any]:
    res = resources(request)
    now = int(time.time())
    out: list[dict[str, Any]] = []
    # Fan out concurrently: awaiting each provider in turn made this endpoint
    # cost the *sum* of every upstream catalogue fetch, and it's on the path a
    # client's model-picker refresh hits. Providers parallelise internally
    # already (Venice's two listings, fal's asset fetches) — this is the one
    # level that didn't. Order is preserved, so the listing stays stable.
    providers = list(res.dispatcher.all_providers())
    budget = res.settings.models_timeout_seconds
    per_provider = await asyncio.gather(
        *(_entries_within(provider_id, backend, budget) for provider_id, backend in providers)
    )
    for (provider_id, _backend), entries in zip(providers, per_provider, strict=True):
        for entry in entries:
            # `display_name`, `kind`, and `supports_tools` are non-standard
            # extensions — strict OpenAI SDKs ignore unknown fields, while
            # gateway-aware frontends (LiteLLM, our own Open WebUI pipe,
            # GlyphStream) use `display_name` for a human-readable label,
            # `kind` to pick the right endpoint (/v1/images/* vs /v1/videos),
            # and `supports_tools` to know whether to send the OpenAI
            # `tools` array on chat completions.
            row: dict[str, Any] = {
                "id": f"{provider_id}/{entry.id}",
                "object": "model",
                "created": now,
                "owned_by": provider_id,
                "display_name": entry.display_name or entry.id,
                "kind": entry.kind,
            }
            # Only surface the field when the backend actually set it —
            # `None` (unknown) is conveyed by omission so clients can
            # apply their own fallback policy.
            if entry.supports_tools is not None:
                row["supports_tools"] = entry.supports_tools
            # Likewise additive: a model's max context window in tokens, when
            # the upstream exposed it. Omitted (not null) when unknown so the
            # client falls back to its own per-endpoint config.
            if entry.context_window is not None:
                row["context_window"] = entry.context_window
            # Which operations the model accepts ("text-to-image",
            # "image-to-image", ...). Backends that merge a model's text-driven
            # and reference-image halves into one id set this so a client can
            # still tell whether an image may be attached; omitted when unknown.
            if entry.capabilities is not None:
                row["capabilities"] = list(entry.capabilities)
            # Additive image-model hints for a frontend's prompt-enhancement
            # pass — the preferred prompt format and an optional per-model nudge.
            # Omitted when unset.
            if entry.prompt_style is not None:
                row["prompt_style"] = entry.prompt_style
            if entry.prompt_hint is not None:
                row["prompt_hint"] = entry.prompt_hint
            out.append(row)
    return {"object": "list", "data": out}
