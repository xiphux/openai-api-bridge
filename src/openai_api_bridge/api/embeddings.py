"""``POST /v1/embeddings`` — sync passthrough.

Used by RAG pipelines (e.g. Open WebUI's vector store ingestion) that hit
the bridge as their OpenAI-compatible embedding endpoint.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request, Response

from ..auth import require_api_key
from ..backends.base import Backend
from ..config import parse_model_id
from ..dispatcher import BackendDispatcher
from ..errors import InvalidRequest, UnsupportedOperation

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/embeddings", dependencies=[Depends(require_api_key)])
async def embeddings(request: Request) -> Response:
    try:
        body: dict[str, Any] = await request.json()
    except ValueError as e:
        raise InvalidRequest(f"Request body must be JSON: {e}") from e
    if not isinstance(body, dict):
        raise InvalidRequest("Request body must be a JSON object")

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise InvalidRequest("Missing or invalid 'model' field", param="model")

    provider_id, model_slug = parse_model_id(model)
    dispatcher: BackendDispatcher = request.app.state.dispatcher
    backend = dispatcher.for_provider(provider_id)

    if not _backend_supports_embeddings(backend):
        raise UnsupportedOperation(f"Provider {provider_id!r} does not support embeddings")

    forwarded_body = {**body, "model": model_slug}
    # Forwarded as bytes. An ingestion batch is megabytes of float arrays, and
    # parsing one into Python objects only to re-serialise it measured 53ms of
    # event-loop block per 3MB — time every other client of the bridge spends
    # waiting, for a byte-identical result.
    result = await backend.create_embedding(forwarded_body)
    return Response(content=result, media_type="application/json")


def _backend_supports_embeddings(backend: Backend) -> bool:
    own = backend.__class__.create_embedding
    base = Backend.create_embedding
    return own is not base
