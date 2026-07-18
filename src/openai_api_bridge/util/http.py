"""Shared HTTP helpers for backend adapters."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from ..errors import UpstreamError

log = logging.getLogger(__name__)

_DEFAULT_FETCH_TIMEOUT_S = 120.0


def parse_json(resp: httpx.Response, what: str) -> Any:
    """Decode a JSON response body, reporting a non-JSON one as an upstream fault.

    ``httpx``'s ``.json()`` raises ``ValueError``, which is not a
    ``BridgeError`` — so an upstream answering 200 with HTML (a captive portal,
    a CDN error page, a WAF interstitial) would escape the per-provider
    handlers as an unhandled exception, surfacing as a 500 and a full
    ``log.exception`` traceback rather than a 502 naming the provider.
    """
    try:
        return resp.json()
    except ValueError as e:
        raise UpstreamError(f"{what} returned a non-JSON body ({e}): {resp.text[:200]}") from e


async def fetch_asset_with_retry(
    url: str,
    *,
    provider_label: str,
    timeout: float = _DEFAULT_FETCH_TIMEOUT_S,
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> tuple[bytes, str]:
    """Download a generated asset by URL, returning ``(bytes, content_type)``.

    URL-format brokers (ImageRouter, fal) return generation results as links to
    a public CDN and expect the caller to fetch the bytes separately. Two shared
    concerns this centralises:

    * **No auth leakage.** A fresh unauthenticated client is used per attempt so
      the bridge's upstream ``Authorization`` header is never attached to a CDN
      request, and so httpx's cross-origin header-stripping on redirects can't
      surprise us.
    * **Transient races.** A just-minted asset URL can briefly 401/404 before
      storage catches up, and CDNs occasionally 5xx or drop the connection.
      Retries with exponential backoff smooth these over.

    ``provider_label`` appears only in log/error text (e.g. "fal", "ImageRouter").
    """
    last_error: httpx.HTTPError | None = None
    resp: httpx.Response | None = None

    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=timeout, follow_redirects=True)
            # Retry on 401/404 (storage race) or 5xx (transient CDN error).
            if (
                resp is not None
                and (resp.status_code in (401, 404) or resp.status_code >= 500)
                and attempt < max_attempts - 1
            ):
                delay = base_delay * (2**attempt)
                log.warning(
                    f"{provider_label} asset fetch got {resp.status_code} for {url}, "
                    f"retrying in {delay}s (attempt {attempt + 1}/{max_attempts})"
                )
                await asyncio.sleep(delay)
                continue
            if resp is not None:
                resp.raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            last_error = e
            if attempt < max_attempts - 1:
                delay = base_delay * (2**attempt)
                log.warning(
                    f"{provider_label} asset fetch failed for {url}, "
                    f"retrying in {delay}s (attempt {attempt + 1}/{max_attempts})"
                )
                await asyncio.sleep(delay)
                continue
            raise UpstreamError(
                f"{provider_label} asset fetch returned {e.response.status_code} for {url} "
                f"after {max_attempts} attempts"
            ) from e
        except httpx.HTTPError as e:
            last_error = e
            if attempt < max_attempts - 1:
                delay = base_delay * (2**attempt)
                log.warning(
                    f"{provider_label} asset fetch failed for {url}, "
                    f"retrying in {delay}s (attempt {attempt + 1}/{max_attempts})"
                )
                await asyncio.sleep(delay)
                continue
            raise UpstreamError(
                f"{provider_label} asset fetch failed for {url} after {max_attempts} attempts: {e}"
            ) from e

    if resp is None:
        raise UpstreamError(
            f"{provider_label} asset fetch failed for {url} after {max_attempts} attempts"
        ) from last_error

    content_type = resp.headers.get("content-type", "application/octet-stream")
    # Strip charset / boundary suffixes for clean storage.
    content_type = content_type.split(";", 1)[0].strip()
    return resp.content, content_type
