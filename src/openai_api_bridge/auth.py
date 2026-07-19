"""Bearer-token auth dependency.

Single shared API key model: clients send `Authorization: Bearer <BRIDGE_API_KEY>`.
Constant-time comparison to prevent timing attacks on key recovery.
"""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Depends, Header

from .config import BridgeSettings, get_settings
from .errors import Unauthorized

_BEARER = "Bearer "


async def require_api_key(
    settings: Annotated[BridgeSettings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    if not authorization or not authorization.startswith(_BEARER):
        raise Unauthorized("Missing or malformed Authorization header")
    presented = authorization[len(_BEARER) :]
    # Compare as bytes: hmac.compare_digest raises TypeError on a str holding
    # non-ASCII, which would surface a bad key as a 500 (plus a traceback in
    # the log) instead of a 401 — and that path is reachable pre-auth by
    # anyone who can reach the port.
    if not hmac.compare_digest(presented.encode("utf-8"), settings.api_key.encode("utf-8")):
        raise Unauthorized("Invalid API key")
