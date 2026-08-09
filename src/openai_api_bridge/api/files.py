"""Bridge-internal asset URL endpoint.

This is the URL we return in the ``url`` field of every image/video API response.
``FileResponse`` handles HTTP range requests natively, which video clients need
for seeking, and ``_assets.asset_response`` adds the caching validators that let
a client skip re-downloading an asset it already holds.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request, Response

from ..auth import require_api_key
from ..errors import JobNotFound
from ..infra.filestore import FileStore
from ._assets import asset_response

router = APIRouter()


@router.get("/v1/files/{file_id}/content", dependencies=[Depends(require_api_key)])
async def get_file_content(
    file_id: str,
    request: Request,
    if_none_match: Annotated[str | None, Header()] = None,
) -> Response:
    filestore: FileStore = request.app.state.filestore
    opened = await filestore.open_for_read(file_id)
    if opened is None:
        raise JobNotFound(f"File {file_id!r} not found")
    return asset_response(
        opened,
        if_none_match=if_none_match,
        # Set a stable filename so curl -O / browser downloads name the file sensibly.
        filename=f"{file_id}{opened.path.suffix}",
    )
