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
            log.warning(
                "Provider %r list_models failed: %s", provider_id, e.message
            )
            continue
        for entry in entries:
            # `display_name` is a non-standard extension — strict OpenAI SDKs
            # ignore unknown fields, while frontends that look for it (LiteLLM,
            # several custom gateways) can render the human-readable label
            # from meta.json alongside the path-prefixed id.
            out.append(
                {
                    "id": f"{provider_id}/{entry.id}",
                    "object": "model",
                    "created": now,
                    "owned_by": provider_id,
                    "display_name": entry.display_name or entry.id,
                }
            )
    return {"object": "list", "data": out}
