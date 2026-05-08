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
    presented = authorization[len(_BEARER):]
    if not hmac.compare_digest(presented, settings.api_key):
        raise Unauthorized("Invalid API key")
