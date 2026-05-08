"""``POST /v1/images/generations`` and ``POST /v1/images/edits``."""

from __future__ import annotations

import base64
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from ..auth import require_api_key
from ..backends.base import GeneratedAsset
from ..config import BridgeSettings, get_settings, parse_model_id
from ..dispatcher import BackendDispatcher
from ..errors import InvalidRequest
from ..infra.filestore import FileStore
from ..schemas.openai import ImagesGenerationRequest

router = APIRouter()

_MAX_N = 4


def _build_url(settings: BridgeSettings, file_id: str) -> str:
    """Construct the URL we return to clients for a stored asset."""
    if settings.public_base_url:
        return f"{settings.public_base_url.rstrip('/')}/v1/files/{file_id}/content"
    return f"/v1/files/{file_id}/content"


async def _render_assets(
    assets: list[GeneratedAsset],
    *,
    response_format: str,
    filestore: FileStore,
    settings: BridgeSettings,
    provider_id: str,
    model_slug: str,
    prompt: str,
) -> list[dict]:
    out: list[dict] = []
    for asset in assets:
        if response_format == "b64_json":
            out.append({"b64_json": base64.b64encode(asset.data).decode("ascii")})
            continue
        file_id = await filestore.put(
            asset.data,
            content_type=asset.content_type,
            kind=asset.kind,
            source_backend=provider_id,
            source_model=model_slug,
            prompt_excerpt=prompt,
        )
        out.append({"url": _build_url(settings, file_id)})
    return out


@router.post("/v1/images/generations", dependencies=[Depends(require_api_key)])
async def images_generations(req: ImagesGenerationRequest, request: Request) -> dict:
    provider_id, model_slug = parse_model_id(req.model)
    dispatcher: BackendDispatcher = request.app.state.dispatcher
    backend = dispatcher.for_provider(provider_id)

    assets = await backend.generate_image(
        model_slug=model_slug,
        prompt=req.prompt,
        size=req.size,
        n=req.n,
    )

    settings: BridgeSettings = request.app.state.settings
    filestore: FileStore = request.app.state.filestore
    data = await _render_assets(
        assets,
        response_format=req.response_format,
        filestore=filestore,
        settings=settings,
        provider_id=provider_id,
        model_slug=model_slug,
        prompt=req.prompt,
    )
    return {"created": int(time.time()), "data": data}


@router.post("/v1/images/edits", dependencies=[Depends(require_api_key)])
async def images_edits(
    request: Request,
    image: Annotated[UploadFile, File(description="Source image (multipart upload)")],
    prompt: Annotated[str, Form()],
    model: Annotated[str, Form()],
    n: Annotated[int, Form()] = 1,
    size: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "url",
) -> dict:
    if not 1 <= n <= _MAX_N:
        raise InvalidRequest(f"n must be between 1 and {_MAX_N} (got {n})", param="n")
    if response_format not in ("url", "b64_json"):
        raise InvalidRequest(
            f"response_format must be 'url' or 'b64_json' (got {response_format!r})",
            param="response_format",
        )

    # Validate the upload first so an empty/missing file surfaces as a 400
    # rather than getting masked by a downstream provider/model 404.
    image_bytes = await image.read()
    if not image_bytes:
        raise InvalidRequest("Uploaded image is empty", param="image")
    image_content_type = image.content_type or "image/png"

    provider_id, model_slug = parse_model_id(model)
    dispatcher: BackendDispatcher = request.app.state.dispatcher
    backend = dispatcher.for_provider(provider_id)

    assets = await backend.edit_image(
        model_slug=model_slug,
        prompt=prompt,
        image=image_bytes,
        image_content_type=image_content_type,
        size=size,
        n=n,
    )

    settings: BridgeSettings = get_settings()
    filestore: FileStore = request.app.state.filestore
    data = await _render_assets(
        assets,
        response_format=response_format,
        filestore=filestore,
        settings=settings,
        provider_id=provider_id,
        model_slug=model_slug,
        prompt=prompt,
    )
    return {"created": int(time.time()), "data": data}
