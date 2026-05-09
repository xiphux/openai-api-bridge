"""HTTP client for an upstream OpenAI-compatible server.

Pure passthrough — request bodies and response bodies are forwarded with
zero translation. The only thing this client does beyond a shared httpx is
manage the optional Authorization header and surface upstream HTTP errors as
typed bridge exceptions.

Streaming chat completions yield raw SSE byte chunks from the upstream so
the bridge can forward them straight to the client. We never parse the SSE
stream — function calls, tool use, vision, and JSON mode all flow through
unchanged because their payloads are inside opaque ``data:`` lines we don't
crack open.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ...errors import InvalidRequest, UnsupportedOperation, UpstreamError

log = logging.getLogger(__name__)


class OpenAIClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_token: str | None,
        request_timeout_seconds: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # Auth header is set per-client so we don't accidentally mix tokens
        # between providers if multiple openai-passthrough providers coexist.
        headers: dict[str, str] = {}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        # Use a generous default timeout for non-streaming calls; streaming
        # opens its own client with no read timeout.
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(request_timeout_seconds, connect=10.0),
        )
        self._streaming_client = httpx.AsyncClient(
            headers=headers,
            # Streaming chat completions can sit idle between tokens — disable
            # the read timeout so a slow token stream isn't mistaken for a
            # dead connection. Connect timeout still bounds initial setup.
            timeout=httpx.Timeout(None, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._streaming_client.aclose()

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            response = await self._client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise UpstreamError(
                f"Upstream /v1/models returned {e.response.status_code}: "
                f"{e.response.text[:300]}"
            ) from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"Upstream /v1/models failed: {e}") from e
        body = response.json()
        return list(body.get("data", []))

    async def chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        """Non-streaming chat completion. Returns the parsed JSON response."""
        try:
            response = await self._client.post(
                f"{self.base_url}/v1/chat/completions", json=body
            )
        except httpx.HTTPError as e:
            raise UpstreamError(
                f"Upstream /v1/chat/completions failed: {e}"
            ) from e
        return self._parse_response(response, "/v1/chat/completions")

    async def chat_completion_stream(
        self, body: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Streaming chat completion. Yields raw SSE byte chunks until the
        upstream signals ``data: [DONE]``.

        We deliberately don't parse the SSE frames — the bytes carry vendor
        extensions (function calls, vision, JSON mode chunks) the bridge has
        no business interpreting. Forwarding bytes verbatim means new
        upstream features work without code changes here.
        """
        # Force stream=True; some upstreams will refuse the stream context
        # if the body says otherwise.
        body = {**body, "stream": True}
        try:
            req = self._streaming_client.build_request(
                "POST", f"{self.base_url}/v1/chat/completions", json=body
            )
            response = await self._streaming_client.send(req, stream=True)
        except httpx.HTTPError as e:
            raise UpstreamError(
                f"Upstream /v1/chat/completions stream failed to open: {e}"
            ) from e

        if response.status_code != 200:
            # Read the (small) error body so we can surface a useful message.
            await response.aread()
            text = response.text[:500]
            await response.aclose()
            self._raise_for_status(response.status_code, text, "/v1/chat/completions")

        async def iterator() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()

        return iterator()

    async def create_embedding(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(
                f"{self.base_url}/v1/embeddings", json=body
            )
        except httpx.HTTPError as e:
            raise UpstreamError(f"Upstream /v1/embeddings failed: {e}") from e
        return self._parse_response(response, "/v1/embeddings")

    # --- helpers ----------------------------------------------------------

    def _parse_response(
        self, response: httpx.Response, endpoint: str
    ) -> dict[str, Any]:
        if response.status_code == 200:
            try:
                return dict(response.json())
            except ValueError as e:
                raise UpstreamError(
                    f"Upstream {endpoint} returned non-JSON 200: "
                    f"{response.text[:200]!r}"
                ) from e
        self._raise_for_status(response.status_code, response.text[:500], endpoint)
        # Unreachable, but mypy needs it
        raise UpstreamError(f"unreachable: {endpoint} status={response.status_code}")

    @staticmethod
    def _raise_for_status(status: int, text: str, endpoint: str) -> None:
        if status == 401:
            # Re-frame as a bridge config error rather than passing the
            # upstream's auth complaint to the chat client.
            raise UpstreamError(
                f"Upstream {endpoint} rejected our credentials (401)"
            )
        if status == 404:
            # Most likely the model id slug doesn't exist on this upstream
            # (e.g. user typed it wrong, or it was removed from the upstream's
            # catalog). Surface as a 404-equivalent for the client.
            raise InvalidRequest(
                f"Upstream {endpoint} returned 404 — model not found upstream"
            )
        if status == 405:
            raise UnsupportedOperation(
                f"Upstream does not implement {endpoint}"
            )
        if 400 <= status < 500:
            raise InvalidRequest(f"Upstream {endpoint} returned {status}: {text}")
        raise UpstreamError(f"Upstream {endpoint} returned {status}: {text}")
