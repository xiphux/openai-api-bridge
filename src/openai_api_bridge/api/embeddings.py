"""``POST /v1/embeddings`` — sync passthrough.

Used by RAG pipelines (e.g. Open WebUI's vector store ingestion) that hit
the bridge as their OpenAI-compatible embedding endpoint.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, Response

from ..auth import require_api_key
from ..backends.base import Backend
from ._passthrough import prepare

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/embeddings", dependencies=[Depends(require_api_key)])
async def embeddings(request: Request) -> Response:
    resolved = await prepare(
        request,
        base_method=Backend.create_embedding,
        operation="embeddings",
    )
    # Forwarded as bytes. An ingestion batch is megabytes of float arrays, and
    # parsing one into Python objects only to re-serialise it measured 53ms of
    # event-loop block per 3MB — time every other client of the bridge spends
    # waiting, for a byte-identical result.
    result = await resolved.backend.create_embedding(resolved.body)
    return Response(content=result, media_type="application/json")
