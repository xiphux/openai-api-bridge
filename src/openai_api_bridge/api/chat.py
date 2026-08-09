"""``POST /v1/chat/completions`` — sync + streaming passthrough."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from ..auth import require_api_key
from ..backends.base import Backend
from ..config import parse_model_id
from ..dispatcher import BackendDispatcher
from ..errors import InvalidRequest, UnsupportedOperation, UpstreamError

log = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(require_api_key)],
    # Disable FastAPI's auto-derived response model: this endpoint returns
    # either a plain Response (sync) or a StreamingResponse (SSE), and the
    # union confuses Pydantic's field-builder.
    response_model=None,
)
async def chat_completions(request: Request) -> Response | StreamingResponse:
    """Forward a chat-completions request to the right backend.

    Body parsing is intentionally minimal — we extract `model` to dispatch and
    pass the rest of the body through unchanged. That way function calls,
    tools, vision, JSON mode, response_format, and any future OpenAI request
    fields all work without bridge code changes.
    """
    try:
        body: dict[str, Any] = await request.json()
    except ValueError as e:
        raise InvalidRequest(f"Request body must be JSON: {e}") from e
    if not isinstance(body, dict):
        raise InvalidRequest("Request body must be a JSON object")

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise InvalidRequest("Missing or invalid 'model' field", param="model")

    provider_id, model_slug = parse_model_id(model)
    dispatcher: BackendDispatcher = request.app.state.dispatcher
    backend = dispatcher.for_provider(provider_id)

    # Rewrite the model id in the forwarded body so the upstream sees its
    # native slug, not our prefixed form.
    forwarded_body = {**body, "model": model_slug}
    stream_requested = bool(body.get("stream"))

    if not _backend_supports_chat(backend):
        raise UnsupportedOperation(f"Provider {provider_id!r} does not support chat completions")

    if stream_requested:
        sse_iterator = await backend.chat_completion(forwarded_body, stream=True)
        if isinstance(sse_iterator, bytes):
            # A backend that answered stream=True with a whole body. Worth
            # naming rather than forwarding: StreamingResponse would iterate
            # the bytes and emit one SSE frame per *byte*. Previously this
            # couldn't be caught — the non-streaming return type was a dict,
            # which satisfies Iterable[str], so the same mistake type-checked
            # and only showed up as a mangled stream at the client.
            raise UpstreamError(
                f"Provider {provider_id!r} returned a non-streaming body for a streaming request"
            )
        # The upstream's SSE stream is forwarded byte-for-byte. Setting
        # X-Accel-Buffering disables nginx-style proxy buffering so tokens
        # don't get coalesced if there's a reverse proxy in front of us.
        return StreamingResponse(
            sse_iterator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # The upstream's bytes, forwarded as-is. Decoding the body into Python
    # objects and re-encoding them would burn CPU on the single event loop —
    # blocking every other client — to reproduce exactly what we received.
    result = await backend.chat_completion(forwarded_body, stream=False)
    return Response(content=result, media_type="application/json")


def _backend_supports_chat(backend: Backend) -> bool:
    """Quick check that this backend actually overrides chat_completion.

    The base `Backend.chat_completion` raises `UnsupportedOperation`, but we
    can avoid the exception path entirely by detecting that a backend hasn't
    overridden it. Cleaner error message, no stack trace in logs for the
    common "user picked a comfyui model for chat" mistake.
    """
    own = backend.__class__.chat_completion
    base = Backend.chat_completion
    # If the bound method's underlying function is still the base ABC's, the
    # backend hasn't overridden it.
    return own is not base
