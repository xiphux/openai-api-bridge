"""Conditional-request handling for the two endpoints that serve stored bytes.

``/v1/files/{id}/content`` and ``/v1/videos/{id}/content`` both hand back a
file the bridge generated once and will never change. Without validators that
say so, a client that already holds the image re-downloads it in full on every
render — the single most visible cost in the bridge for any frontend showing
a gallery, and bandwidth paid for twice.

Starlette's ``FileResponse`` sets an ``ETag`` but never acts on
``If-None-Match`` (that logic lives in ``StaticFiles``), so the conditional
half has to happen here.
"""

from __future__ import annotations

from fastapi import Response
from fastapi.responses import FileResponse

from ..infra.filestore import OpenedFile

# A generated asset is addressed by a 32-hex random id and its bytes never
# change, so the entity really is immutable and a year is not an overstatement.
# Outliving the eviction window (RETENTION_DAYS, LRU past MAX_CACHE_GB) is
# intended: a client that kept the bytes is still holding a correct copy after
# the bridge has retired its own.
#
# `private`, not `public`: these endpoints sit behind the bridge's bearer
# token, and a shared proxy caching the response would hand it to callers that
# never presented one.
_CACHE_CONTROL = "private, max-age=31536000, immutable"


def _etag_for(file_id: str) -> str:
    """A validator derived from the id rather than from mtime and size.

    Starlette derives its own from the stat, which changes whenever the bytes
    are re-copied — restoring a backup or recreating a volume would invalidate
    every client's cache despite the content being identical.
    """
    return f'"{file_id}"'


def _validator_headers(etag: str) -> dict[str, str]:
    """Caching headers shared by the 200 and the 304.

    ``Vary: Authorization`` because these endpoints sit behind a bearer token
    and the response is cacheable. ``private`` already tells a well-behaved
    shared cache to keep out; ``Vary`` is what stops one that stores the
    response anyway from serving it to a request bearing a different
    credential. Both, because the cost is a header and the failure is handing
    someone else's asset to the wrong caller.
    """
    return {
        "etag": etag,
        "cache-control": _CACHE_CONTROL,
        "vary": "Authorization",
    }


def _matches(if_none_match: str | None, etag: str) -> bool:
    """Whether the client's ``If-None-Match`` covers this entity.

    Comparison is weak per RFC 9110 §13.1.2, so a ``W/`` prefix on either side
    is stripped before comparing — for an immutable entity the weak and strong
    forms mean the same thing anyway.
    """
    if not if_none_match:
        return False
    for raw in if_none_match.split(","):
        candidate = raw.strip()
        if candidate == "*":
            return True
        if candidate.removeprefix("W/") == etag.removeprefix("W/"):
            return True
    return False


def asset_response(
    opened: OpenedFile,
    *,
    if_none_match: str | None,
    filename: str | None = None,
) -> Response:
    """A cacheable response for a stored asset, or 304 if the client has it."""
    etag = _etag_for(opened.meta.id)
    if _matches(if_none_match, etag):
        # Cache-Control repeated on the 304: RFC 9110 §15.4.5 asks for the
        # headers that would have been sent on a 200, and a client that drops
        # the directive on revalidation would revalidate again next time.
        return Response(
            status_code=304,
            headers=_validator_headers(etag),
        )
    return FileResponse(
        opened.path,
        media_type=opened.meta.content_type,
        # setdefault semantics inside FileResponse mean these win over the
        # stat-derived ETag it would otherwise generate.
        headers=_validator_headers(etag),
        filename=filename,
        # Deliberately NOT passing stat_result, even though FileStore already
        # took one and Starlette is about to take another. Supplying it makes
        # FileResponse skip its own stat — and that stat is also its existence
        # check: without it, a file that vanishes between the store's stat and
        # the send (the eviction race FileStore.open_for_read documents) is
        # answered as 200 with a Content-Length from the stale stat and a
        # truncated body, instead of failing loudly. A silent short read is a
        # worse answer than an error, and one stat is not worth it.
    )
