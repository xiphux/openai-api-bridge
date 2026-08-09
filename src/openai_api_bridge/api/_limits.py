"""A ceiling on request body size, enforced before anything buffers the body.

The bridge reads request bodies whole. ``/v1/images/edits`` calls
``UploadFile.read()`` on up to 16 uploads, ``/v1/videos`` does the same for its
``input_reference``, and the chat and embedding passthroughs call
``request.json()``. Starlette spools a multipart part past 1MB to a temp file,
so an oversized upload costs disk on the way to costing memory. With a single
uvicorn worker — the bridge's deployment model, since SQLite and in-memory job
state don't survive forking — one such request stalls or OOMs every other
client on the box.

This is a guard against a body nobody meant to send (a client handed the wrong
file, a runaway retry loop) as much as against a hostile one: the endpoints all
sit behind the bearer token, so the realistic failure is an accident with a
process-wide blast radius.

Enforced in two places, because either alone has a hole:

* **The declared ``Content-Length``**, checked before the first
  ``http.request`` message is pulled. This is the one that matters — an honest
  client is refused without a byte being buffered or spooled.
* **The bytes actually received**, counted as they arrive. A chunked request
  carries no ``Content-Length`` at all, and a dishonest one can understate it.

Rendered here rather than raised into ``bridge_error_handler``: exception
handlers live *inside* the middleware stack, so an error raised out here would
never reach them and would surface as a bare 500.
"""

from __future__ import annotations

import logging

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..errors import RequestTooLarge, error_payload

log = logging.getLogger(__name__)


class _BodyTooLarge(Exception):
    """Signals an over-cap body from inside the wrapped ``receive``.

    Raised rather than answered on the spot because ``receive`` has no way to
    send a response. It is the signal to *stop reading*, and it reliably does
    that — but it is explicitly **not** relied on to reach ``__call__``: see
    :meth:`BodySizeLimitMiddleware.__call__` for why the response is driven by
    a flag instead.
    """

    def __init__(self, received: int) -> None:
        super().__init__(received)
        self.received = received


def _too_large_response(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=RequestTooLarge.status_code,
        content=error_payload(
            message=message,
            type_=RequestTooLarge.error_type,
            code=RequestTooLarge.code,
        ),
        # Nothing about this request is going to get smaller on a retry, and
        # the connection may still be mid-upload of a body we are refusing to
        # read. Closing is both the honest signal and what stops the rest of
        # the bytes arriving.
        headers={"connection": "close"},
    )


class BodySizeLimitMiddleware:
    """Reject request bodies over ``max_bytes``. A non-positive cap disables it.

    Pure ASGI rather than ``BaseHTTPMiddleware``: the latter buffers the body
    to hand a ``Request`` to the handler, which is the exact cost this exists
    to avoid paying.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.max_bytes <= 0:
            await self.app(scope, receive, send)
            return

        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            try:
                length = int(declared)
            except ValueError:
                # Malformed; let the server layer have its own opinion rather
                # than inventing a verdict here. The received-bytes counter
                # below still bounds it.
                length = -1
            if length > self.max_bytes:
                log.warning(
                    "Refusing %s %s: declared Content-Length %d exceeds the %d byte cap",
                    scope.get("method", "?"),
                    scope.get("path", "?"),
                    length,
                    self.max_bytes,
                )
                response = _too_large_response(
                    f"Request body of {length} bytes exceeds the "
                    f"{self.max_bytes} byte limit (BRIDGE_MAX_REQUEST_MB)."
                )
                await response(scope, receive, send)
                return

        received = 0
        # Bytes received when the cap was crossed; 0 means it never was. A
        # flag rather than the exception alone, because the exception does not
        # reliably come back to us — see below.
        exceeded = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = received
                    raise _BodyTooLarge(received)
            return message

        async def guarded_send(message: Message) -> None:
            """Forward the app's response — unless the cap was crossed.

            FastAPI reads the body for any route that binds it as a parameter
            (``Form``/``File``/``UploadFile``, or a pydantic body model) inside
            its own request handler, which wraps that read in a catch-all
            ``except Exception`` and turns anything it catches into
            ``HTTPException(400, "There was an error parsing the body")``.
            ``_BodyTooLarge`` is a plain exception, so on exactly those routes
            — ``/v1/images/edits``, ``/v1/videos``, ``/v1/images/generations``,
            the ones this middleware exists for — it was swallowed there and
            the client got a 400 in FastAPI's ``{"detail": ...}`` shape rather
            than the bridge's 413 envelope, with no ``Connection: close`` and
            nothing in the log.

            Subclassing ``HTTPException`` does not fix it: Starlette's
            ``ExceptionMiddleware`` sits *inside* this one and would render its
            own ``{"detail": ...}`` before we ever saw it. So the verdict is
            carried by ``exceeded``, which no downstream handler can swallow,
            and the app's response is replaced on its way out.
            """
            nonlocal response_started
            if exceeded and not response_started:
                response_started = True
                await _log_and_send(exceeded)
                return
            if exceeded:
                # Our 413 is already on the wire; drop whatever the app is
                # still trying to say about the request.
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        async def _log_and_send(size: int) -> None:
            log.warning(
                "Refusing %s %s: body reached %d bytes, past the %d byte cap",
                scope.get("method", "?"),
                scope.get("path", "?"),
                size,
                self.max_bytes,
            )
            response = _too_large_response(
                f"Request body exceeded the {self.max_bytes} byte limit "
                "(BRIDGE_MAX_REQUEST_MB) while being received."
            )
            await response(scope, receive, send)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _BodyTooLarge as e:
            # The routes that read the body themselves (the chat and embedding
            # passthroughs call ``request.json()`` in the handler) do let it
            # propagate. Nothing has been sent in that case, so answer here.
            if response_started:
                # Headers are already on the wire; there is no status left to
                # set. Let it surface as a failed response rather than
                # pretending we can answer twice.
                raise
            response_started = True
            await _log_and_send(e.received)
