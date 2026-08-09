"""Shared HTTP helpers for backend adapters."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, NoReturn

import httpx

from ..errors import (
    InvalidRequest,
    RateLimited,
    UnsupportedOperation,
    UpstreamAuthError,
    UpstreamError,
)

log = logging.getLogger(__name__)

_DEFAULT_FETCH_TIMEOUT_S = 120.0

# One unauthenticated client for every asset fetch in the process. Built per
# running event loop: a pool's connections belong to the loop that opened them,
# and the bridge has exactly one — but the test suite runs each case in a fresh
# loop, and a client carried across would hand out connections attached to a
# closed one.
_asset_client: httpx.AsyncClient | None = None
_asset_client_loop: asyncio.AbstractEventLoop | None = None

# httpx defaults to a 5s keepalive expiry, which is shorter than the gap
# between almost every pair of asset fetches this bridge makes: image
# generations are seconds to a minute apart, video minutes. At the default,
# measurement showed a second connection opened after a 6s idle gap — i.e. the
# pool existed but never actually got reused, so the handshake was still paid
# every time. Five minutes covers the realistic spacing.
#
# Two honest caveats. This does nothing for a `run_all` batch: n concurrent
# fetches open n connections whatever the expiry. And it was never the only
# win from sharing a client — building one costs ~2.5ms of blocking event-loop
# time to construct an SSLContext and load the certifi bundle, which the
# previous client-per-attempt paid on every fetch and every retry.
_ASSET_LIMITS = httpx.Limits(
    max_connections=100,
    max_keepalive_connections=20,
    keepalive_expiry=300.0,
)


def _asset_fetch_client() -> httpx.AsyncClient:
    """The shared client used to download generated assets.

    Deliberately built with **no default headers**. Asset URLs point at public
    CDNs (fal.media, storage.imagerouter.io, OpenRouter's host), and the
    bridge's upstream ``Authorization`` header has no business being attached
    to them — some providers would log it. Keeping the client unauthenticated
    is also what makes httpx's cross-origin header stripping on redirects a
    non-question rather than something to reason about per provider.
    """
    global _asset_client, _asset_client_loop
    loop = asyncio.get_running_loop()
    if _asset_client is None or _asset_client_loop is not loop or _asset_client.is_closed:
        _asset_client = httpx.AsyncClient(follow_redirects=True, limits=_ASSET_LIMITS)
        _asset_client_loop = loop
    return _asset_client


async def aclose_asset_client() -> None:
    """Close the shared asset client. Called from the app's shutdown path."""
    global _asset_client, _asset_client_loop
    client, _asset_client, _asset_client_loop = _asset_client, None, None
    if client is not None and not client.is_closed:
        await client.aclose()


def raise_for_upstream_status(*, status: int, body: str, provider: str, endpoint: str) -> NoReturn:
    """Map an upstream error status onto the bridge's typed errors.

    Every adapter has to answer "what does this upstream status mean to our
    client", and each one answering separately is how the bridge ended up
    telling clients that the *same* rejection was a 400 on one provider and a
    502 on another: a malformed prompt refused by an OpenAI-compatible
    upstream surfaced as ``invalid_request_error``, while Venice and
    ImageRouter routed every status — 4xx included — to ``UpstreamError``.
    Clients with retry-on-5xx logic then retried requests that could never
    succeed, against providers that may bill for them.

    One mapping, used by every client, so the answer can't drift again.
    """
    if status in (401, 403):
        # UpstreamAuthError, not a generic UpstreamError: provider tokens are
        # read from the environment at startup, so a rejected credential is
        # unlikely to fix itself, and callers back off harder on this type
        # than on a transient blip. Until this lived here, only the fal
        # adapter raised it, so a bad key on any other provider was retried
        # as if it were a hiccup. Note 403 lands here too and isn't always
        # about the credential, which is why the backoff is a long window
        # rather than a permanent latch — see AsyncTTLCache._cooldown_for.
        #
        # Deliberately without the body. This is the upstream's complaint
        # about the credential *we* sent, and it goes straight to a client
        # that has no business seeing it — some providers quote the token back.
        raise UpstreamAuthError(f"{provider} rejected our credentials ({status}) on {endpoint}")
    if status == 404:
        # Most likely the model slug doesn't exist on this upstream (typo, or
        # withdrawn from its catalogue).
        raise InvalidRequest(f"{provider} {endpoint} returned 404 — model not found upstream")
    if status == 405:
        raise UnsupportedOperation(f"{provider} does not implement {endpoint}")
    if status == 429:
        # The one retriable 4xx. Sweeping it into the catch-all below would
        # report a rate limit as a malformed request, telling OpenAI-shaped
        # clients not to retry precisely when they should.
        raise RateLimited(f"{provider} {endpoint} rate-limited the bridge: {body}")
    if 400 <= status < 500:
        raise InvalidRequest(f"{provider} {endpoint} returned {status}: {body}")
    raise UpstreamError(f"{provider} {endpoint} returned {status}: {body}")


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


def _is_retriable_fetch_status(status: int) -> bool:
    """Whether re-requesting a generated asset could plausibly succeed.

    Named rather than inlined because getting this set wrong is silent and
    expensive in both directions: too broad and a hopeless 400 costs three
    upstream calls and two sleeps, too narrow and a finished, already-billed
    render is discarded over a momentary throttle.

    * 401/404 — a just-minted asset URL before storage catches up.
    * 408/425 — the request timed out, or arrived "too early" behind a CDN.
    * 429 — throttled, which is by definition worth waiting out. Notably the
      one that a "don't retry 4xx" rule sweeps up by mistake, and the costliest
      to lose: fal fetches a video *after* the render completes and is billed.
    * 5xx — a transient CDN or origin fault, except 501, which says the origin
      does not implement the request at all and will keep saying so.
    """
    return status in (401, 404, 408, 425, 429) or (status >= 500 and status != 501)


def redact_url(url: str) -> str:
    """An asset URL with its query string dropped, for a client-facing message.

    Provider asset URLs are routinely pre-signed — fal and ImageRouter both
    hand back CDN links whose query carries the signature and expiry that make
    them fetchable. Those messages are rendered into the OpenAI error envelope
    and returned over the wire, so the credential travels with them and lands
    wherever the client logs its errors.

    The path still names the asset, which is what makes an error readable, so
    nothing diagnostic is lost. Log lines deliberately keep the full URL: they
    stay on the bridge's own host, and an expired or malformed signature is
    exactly the thing an operator needs to see to debug a failing fetch.

    Query only. A signature carried in the *path* would survive this, which no
    provider here does today — the point is to stop the routine leak, not to
    claim a URL can never identify itself.
    """
    return url.split("?", 1)[0]


def _asset_too_large(provider_label: str, url: str, size: int, max_bytes: int) -> UpstreamError:
    return UpstreamError(
        f"{provider_label} asset exceeded the {max_bytes} byte cap "
        f"({size} bytes) for {redact_url(url)}"
    )


async def _read_capped(
    response: httpx.Response,
    *,
    provider_label: str,
    url: str,
    max_bytes: int | None,
) -> bytes:
    """Buffer a streaming response body, stopping the moment it passes the cap.

    The cap used to be applied to ``response.content`` — i.e. *after* the whole
    body was already in memory, which is the one moment it could no longer
    help. The bridge runs a single uvicorn worker, so an oversized asset is
    felt by every other client on the box, and a hostile or malfunctioning CDN
    was bounded only by the disk it was streamed onto.

    Raising mid-iteration leaves the ``async with`` to close the response,
    which drops the connection and stops the remaining bytes arriving.
    """
    if max_bytes is None:
        return await response.aread()

    # A truthful Content-Length lets us refuse before reading anything at all.
    # Absent or understated, the running count below is what actually binds.
    declared = response.headers.get("content-length")
    if declared is not None:
        try:
            length = int(declared)
        except ValueError:
            pass
        else:
            if length > max_bytes:
                raise _asset_too_large(provider_label, url, length, max_bytes)

    buffer = bytearray()
    async for chunk in response.aiter_bytes():
        buffer += chunk
        if len(buffer) > max_bytes:
            raise _asset_too_large(provider_label, url, len(buffer), max_bytes)
    return bytes(buffer)


async def fetch_asset_with_retry(
    url: str,
    *,
    provider_label: str,
    timeout: float = _DEFAULT_FETCH_TIMEOUT_S,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_bytes: int | None = None,
) -> tuple[bytes, str]:
    """Download a generated asset by URL, returning ``(bytes, content_type)``.

    URL-format brokers (ImageRouter, fal) return generation results as links to
    a public CDN and expect the caller to fetch the bytes separately. Three
    shared concerns this centralises:

    * **No auth leakage.** The shared client is unauthenticated, so the
      bridge's upstream ``Authorization`` header is never attached to a CDN
      request, and httpx's cross-origin header-stripping on redirects can't
      surprise us. See :func:`_asset_fetch_client`.
    * **Transient conditions.** A just-minted asset URL can briefly 401/404
      before storage catches up, CDNs occasionally 5xx or drop the connection,
      and a throttled fetch answers 429. Retries with exponential backoff
      smooth these over; see :func:`_is_retriable_fetch_status` for the set.
      Statuses that can never succeed are raised on the first response.
    * **A bounded payload.** The body is streamed and abandoned as soon as it
      passes ``max_bytes``; see :func:`_read_capped`.

    ``provider_label`` appears only in log/error text (e.g. "fal", "ImageRouter").

    ``max_bytes`` of ``None`` means unbounded, which is now only what a caller
    gets by explicitly asking for it — the video-bearing providers pass their
    configured ceiling.
    """
    last_error: httpx.HTTPError | None = None

    for attempt in range(max_attempts):
        try:
            # Reused, not built per attempt. See _ASSET_LIMITS for what that
            # actually buys and what it doesn't. Streamed rather than fetched
            # whole so the cap can be enforced against bytes as they land.
            async with _asset_fetch_client().stream("GET", url, timeout=timeout) as response:
                status = response.status_code
                if status >= 400:
                    if _is_retriable_fetch_status(status) and attempt < max_attempts - 1:
                        delay = base_delay * (2**attempt)
                        log.warning(
                            f"{provider_label} asset fetch got {status} for {url}, "
                            f"retrying in {delay}s (attempt {attempt + 1}/{max_attempts})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    # No retry past here. Everything worth re-attempting is
                    # handled by the branch above (see
                    # _is_retriable_fetch_status), so reaching this point means
                    # either a status that can never succeed (400, 403, 410, …)
                    # or the last attempt for one that could. Retrying
                    # regardless, as this once did, spent three upstream calls
                    # and two backoff sleeps to confirm a guaranteed failure.
                    message = (
                        f"{provider_label} asset fetch returned {status} for "
                        f"{redact_url(url)} after {attempt + 1} attempt(s)"
                    )
                    # Keep the type: a throttled fetch that outlasts our
                    # attempts is still a rate limit, and callers branch on
                    # that — fal's _fetch_asset skips its "did the asset
                    # expire?" hint for it, which would otherwise be appended
                    # to a message plainly saying 429. Untyped, that guard
                    # could never fire.
                    if status == 429:
                        raise RateLimited(message)
                    raise UpstreamError(message)

                content_type = response.headers.get("content-type", "application/octet-stream")
                # Strip charset / boundary suffixes for clean storage.
                content_type = content_type.split(";", 1)[0].strip()
                data = await _read_capped(
                    response,
                    provider_label=provider_label,
                    url=url,
                    max_bytes=max_bytes,
                )
                return data, content_type
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
                f"{provider_label} asset fetch failed for {redact_url(url)} "
                f"after {max_attempts} attempts: {e}"
            ) from e

    raise UpstreamError(
        f"{provider_label} asset fetch failed for {redact_url(url)} after {max_attempts} attempts"
    ) from last_error
