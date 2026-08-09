"""``POST /v1/chat/completions`` — sync + streaming passthrough."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from ..auth import require_api_key
from ..backends.base import Backend
from ..errors import UpstreamError
from ._passthrough import prepare

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
    fields all work without bridge code changes. See ``_passthrough.prepare``,
    shared with the embeddings endpoint.
    """
    resolved = await prepare(
        request,
        base_method=Backend.chat_completion,
        operation="chat completions",
    )
    stream_requested = bool(resolved.body.get("stream"))

    if stream_requested:
        sse_iterator = await resolved.backend.chat_completion(resolved.body, stream=True)
        if isinstance(sse_iterator, bytes):
            # A backend that answered stream=True with a whole body. Worth
            # naming rather than forwarding: iterating ``bytes`` yields ints,
            # and StreamingResponse encodes anything that isn't bytes or
            # memoryview — so this raises AttributeError mid-response, after
            # the headers have already gone out, and the client sees an
            # aborted connection rather than an error.
            #
            # Previously this couldn't be caught at all. The non-streaming
            # return type was a dict, which satisfies Iterable[str] because
            # iterating a dict yields its keys, so the same mistake
            # type-checked and degraded silently instead: SSE frames
            # containing the JSON key names, and no failure anywhere.
            raise UpstreamError(
                f"Provider {resolved.provider_id!r} returned a non-streaming body "
                "for a streaming request"
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
    result = await resolved.backend.chat_completion(resolved.body, stream=False)
    return Response(content=result, media_type="application/json")
