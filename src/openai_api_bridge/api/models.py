"""``GET /v1/models`` — flat enumeration across all configured providers."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, Request

from ..auth import require_api_key
from ..dispatcher import BackendDispatcher
from ..errors import BridgeError

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/v1/models", dependencies=[Depends(require_api_key)])
async def list_models(request: Request) -> dict:
    dispatcher: BackendDispatcher = request.app.state.dispatcher
    now = int(time.time())
    out: list[dict] = []
    for provider_id, backend in dispatcher.all_providers():
        try:
            entries = await backend.list_models()
        except BridgeError as e:
            # One flaky provider shouldn't break the whole listing.
            log.warning("Provider %r list_models failed: %s", provider_id, e.message)
            continue
        for entry in entries:
            # `display_name`, `kind`, and `supports_tools` are non-standard
            # extensions — strict OpenAI SDKs ignore unknown fields, while
            # gateway-aware frontends (LiteLLM, our own Open WebUI pipe,
            # GlyphStream) use `display_name` for a human-readable label,
            # `kind` to pick the right endpoint (/v1/images/* vs /v1/videos),
            # and `supports_tools` to know whether to send the OpenAI
            # `tools` array on chat completions.
            row: dict = {
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
            out.append(row)
    return {"object": "list", "data": out}
