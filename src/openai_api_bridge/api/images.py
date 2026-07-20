"""``POST /v1/images/generations`` and ``POST /v1/images/edits``."""

from __future__ import annotations

import base64
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from ..auth import require_api_key
from ..backends.base import GeneratedAsset, InputImage
from ..config import BridgeSettings, get_settings, parse_model_id
from ..dispatcher import BackendDispatcher
from ..errors import InvalidRequest
from ..infra.filestore import FileStore
from ..schemas.openai import ImagesGenerationRequest

router = APIRouter()

_MAX_N = 4
# Cap on reference images per edit. Matches OpenAI's /images/edits limit and
# bounds memory on backends that buffer every image (OpenRouter base64-expands
# them); edits are authenticated, so this is a sanity guard, not a DoS control.
_MAX_EDIT_IMAGES = 16


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
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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
async def images_generations(req: ImagesGenerationRequest, request: Request) -> dict[str, Any]:
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
    prompt: Annotated[str, Form()],
    model: Annotated[str, Form()],
    # Accept both the OpenAI single-field ``image`` (which some clients
    # repeat to send multiples) and the ``image[]`` array convention. A
    # single declared ``UploadFile`` silently keeps only the last of a
    # repeated field, which is what dropped all-but-one reference image.
    image: Annotated[
        list[UploadFile] | None, File(description="Source image(s) (multipart upload)")
    ] = None,
    image_array: Annotated[list[UploadFile] | None, File(alias="image[]")] = None,
    n: Annotated[int, Form()] = 1,
    size: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "url",
) -> dict[str, Any]:
    if not 1 <= n <= _MAX_N:
        raise InvalidRequest(f"n must be between 1 and {_MAX_N} (got {n})", param="n")
    if response_format not in ("url", "b64_json"):
        raise InvalidRequest(
            f"response_format must be 'url' or 'b64_json' (got {response_format!r})",
            param="response_format",
        )

    # Preserve client order: ``image`` entries first, then any ``image[]``.
    uploads = [*(image or []), *(image_array or [])]
    if not uploads:
        raise InvalidRequest("At least one input image is required", param="image")
    if len(uploads) > _MAX_EDIT_IMAGES:
        raise InvalidRequest(
            f"At most {_MAX_EDIT_IMAGES} input images are allowed (got {len(uploads)})",
            param="image",
        )

    # Validate each upload first so an empty file surfaces as a 400 rather
    # than getting masked by a downstream provider/model 404.
    images: list[InputImage] = []
    for upload in uploads:
        raw = await upload.read()
        if not raw:
            raise InvalidRequest("Uploaded image is empty", param="image")
        images.append(InputImage(data=raw, content_type=upload.content_type or "image/png"))

    provider_id, model_slug = parse_model_id(model)
    dispatcher: BackendDispatcher = request.app.state.dispatcher
    backend = dispatcher.for_provider(provider_id)

    assets = await backend.edit_image(
        model_slug=model_slug,
        prompt=prompt,
        images=images,
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
