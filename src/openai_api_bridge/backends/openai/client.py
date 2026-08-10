"""HTTP client for an upstream OpenAI-compatible server.

Pure passthrough — request bodies and response bodies are forwarded with
zero translation. The only thing this client does beyond a shared httpx2 is
manage the optional Authorization header and surface upstream HTTP errors as
typed bridge exceptions.

Streaming chat completions yield raw SSE byte chunks from the upstream so
the bridge can forward them straight to the client. We never parse the SSE
stream — function calls, tool use, vision, and JSON mode all flow through
unchanged because their payloads are inside opaque ``data:`` lines we don't
crack open.

The non-streaming responses are bytes for the same reason. Passthrough means
the bridge has no opinion about the *contents* of the body, so decoding it
into Python objects only to re-encode them is CPU spent on the single event
loop to reproduce what the upstream already sent. It still checks that a 200
looks like a JSON object before forwarding it — see :meth:`OpenAIClient._raw_body`
for why an unexamined 200 is the one thing this shortcut must not become.
"""

from __future__ import annotations

import codecs
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx2

from ...errors import UpstreamError
from ...util.http import parse_json, raise_for_upstream_status

log = logging.getLogger(__name__)

# Encoding preambles that mean "this is text in a declared encoding", not
# "this is HTML". ``json.loads`` on bytes runs ``detect_encoding`` and handles
# every one of them, so a body opening with a BOM is JSON as far as any client
# that parses it is concerned — including the OpenAI SDK, which decodes via
# httpx v1's ``.json()`` (that SDK depends on httpx, not the bridge's httpx2).
# Listed rather than stripped because the big-endian forms put a NUL before
# the ``{``, so stripping and re-testing for ``{`` would reject exactly the
# payloads this exists to admit.
_UTF_BOMS = (
    codecs.BOM_UTF8,
    codecs.BOM_UTF32_LE,
    codecs.BOM_UTF32_BE,
    codecs.BOM_UTF16_LE,
    codecs.BOM_UTF16_BE,
)

# Bodies at or under this are validated by an actual parse; larger ones get a
# first-byte check instead. See OpenAIClient._raw_body for where the number
# comes from — it sits above every chat completion and every proxy error page,
# and below the embedding batches the byte path exists to keep cheap.
_MAX_VALIDATED_BODY = 256 * 1024


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
        self._client = httpx2.AsyncClient(
            headers=headers,
            timeout=httpx2.Timeout(request_timeout_seconds, connect=10.0),
        )
        self._streaming_client = httpx2.AsyncClient(
            headers=headers,
            # Streaming chat completions can sit idle between tokens — disable
            # the read timeout so a slow token stream isn't mistaken for a
            # dead connection. Connect timeout still bounds initial setup.
            timeout=httpx2.Timeout(None, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._streaming_client.aclose()

    async def list_models(self) -> list[Any]:
        """The upstream's raw ``/v1/models`` data array.

        Unvalidated upstream JSON: entries are Any, not a dict shape we
        have actually checked. Callers validate before indexing.
        """
        try:
            response = await self._client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
        except httpx2.HTTPStatusError as e:
            raise_for_upstream_status(
                status=e.response.status_code,
                body=e.response.text[:300],
                provider="Upstream",
                endpoint="/v1/models",
            )
        except httpx2.HTTPError as e:
            raise UpstreamError(f"Upstream /v1/models failed: {e}") from e
        body = parse_json(response, "Upstream /v1/models")
        return list(body.get("data", []))

    async def chat_completion(self, body: dict[str, Any]) -> bytes:
        """Non-streaming chat completion. Returns the raw JSON response body.

        Bytes rather than a parsed dict because nothing on the passthrough
        path reads the response — decoding it only to re-encode it byte-for-
        byte is CPU spent on the single event loop for no result. A caller
        that genuinely needs the object (OpenRouter's image-via-chat
        translation) parses it itself.
        """
        try:
            response = await self._client.post(f"{self.base_url}/v1/chat/completions", json=body)
        except httpx2.HTTPError as e:
            raise UpstreamError(f"Upstream /v1/chat/completions failed: {e}") from e
        return self._raw_body(response, "/v1/chat/completions")

    async def chat_completion_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
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
        except httpx2.HTTPError as e:
            raise UpstreamError(f"Upstream /v1/chat/completions stream failed to open: {e}") from e

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

    async def create_embedding(self, body: dict[str, Any]) -> bytes:
        """Embeddings passthrough. Returns the raw JSON response body.

        The same argument as ``chat_completion``, but this is where it bites:
        a RAG ingestion batch is megabytes of float arrays, and parsing then
        re-serialising one measured 53ms of event-loop block — time every
        other client of the bridge spends waiting, for a byte-identical
        result.
        """
        try:
            response = await self._client.post(f"{self.base_url}/v1/embeddings", json=body)
        except httpx2.HTTPError as e:
            raise UpstreamError(f"Upstream /v1/embeddings failed: {e}") from e
        return self._raw_body(response, "/v1/embeddings")

    # --- helpers ----------------------------------------------------------

    def _raw_body(self, response: httpx2.Response, endpoint: str) -> bytes:
        """The upstream's 200 body, forwarded whole; a typed error otherwise.

        Forwarding without decoding is the point — see the module docstring —
        but "don't decode it" must not become "don't look at it". A 200 whose
        body isn't real JSON is the failure this bridge has been bitten by
        before: a captive portal, a CDN error page or a WAF interstitial
        answering 200 with HTML, or a proxy that committed the status line and
        then truncated. Passed through, that reaches the client as a 200
        labelled ``application/json``, so an OpenAI SDK raises an opaque decode
        error naming nothing, retry-on-5xx never fires, and an empty body looks
        like a successful zero-length result — which a RAG ingestion run will
        happily store.

        The body is therefore **validated by parsing** and then forwarded as
        the original bytes. The parse result is discarded; only its success
        matters. That restores exactly the guarantee ``dict(response.json())``
        used to give, without the re-serialisation that motivated the byte
        path in the first place — the expensive half was always the *encode*,
        not the decode.

        Above ``_MAX_VALIDATED_BODY`` the parse is skipped for a first-byte
        check instead. Measured on this codebase: a typical chat completion is
        ~2KB and parses in under 0.01ms, a verbose n=4 one 32KB in 0.03ms, but
        a 100 x 1536 embedding batch is 3MB and costs 18ms of event-loop block
        — time every other client of the bridge spends waiting. The gate lands
        between them by design. Note which cases that leaves unvalidated: an
        HTML interstitial or a truncated small response is always well under
        the threshold and is still caught; what slips through is only a
        multi-megabyte body that begins with ``{`` and is malformed later on.

        ``{`` specifically, not any JSON value — both endpoints return an
        object, and the parse this replaced rejected arrays and scalars too.
        A leading BOM counts as a JSON start above the threshold, since
        ``json.loads`` sniffs the encoding and would have accepted it.

        A non-200 is still decoded: the error path needs the text to build a
        useful message, and an error body is small.
        """
        if response.status_code == 200:
            body = response.content
            if len(body) <= _MAX_VALIDATED_BODY:
                try:
                    parsed = json.loads(body)
                except ValueError as e:
                    raise UpstreamError(
                        f"Upstream {endpoint} returned non-JSON 200: {body[:200]!r}"
                    ) from e
                # An object, matching the check above the threshold. The parse
                # this replaced expressed the same requirement as
                # ``dict(response.json())``, but a JSON array raised an
                # *uncaught* TypeError there and surfaced as a 500; this is
                # the same verdict delivered as a 502 that names the provider.
                if not isinstance(parsed, dict):
                    raise UpstreamError(
                        f"Upstream {endpoint} returned non-JSON 200: {body[:200]!r}"
                    )
            else:
                probe = body.lstrip()
                if not (probe.startswith(b"{") or probe.startswith(_UTF_BOMS)):
                    raise UpstreamError(
                        f"Upstream {endpoint} returned non-JSON 200: {body[:200]!r}"
                    )
            return body
        self._raise_for_status(response.status_code, response.text[:500], endpoint)
        # Unreachable, but mypy needs it
        raise UpstreamError(f"unreachable: {endpoint} status={response.status_code}")

    @staticmethod
    def _raise_for_status(status: int, text: str, endpoint: str) -> None:
        # The mapping this class used to own now lives in util.http so every
        # adapter answers an upstream status the same way.
        raise_for_upstream_status(status=status, body=text, provider="Upstream", endpoint=endpoint)
