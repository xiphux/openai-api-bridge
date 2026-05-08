"""Bridge-internal asset URL endpoint.

This is the URL we return in the ``url`` field of every image/video API response.
``FileResponse`` handles HTTP range requests natively, which video clients need
for seeking.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse

from ..auth import require_api_key
from ..errors import JobNotFound
from ..infra.filestore import FileStore

router = APIRouter()


@router.get("/v1/files/{file_id}/content", dependencies=[Depends(require_api_key)])
async def get_file_content(file_id: str, request: Request) -> FileResponse:
    filestore: FileStore = request.app.state.filestore
    result = await filestore.open_for_read(file_id)
    if result is None:
        raise JobNotFound(f"File {file_id!r} not found")
    abs_path, meta = result
    return FileResponse(
        abs_path,
        media_type=meta.content_type,
        # Set a stable filename so curl -O / browser downloads name the file sensibly.
        filename=f"{file_id}{abs_path.suffix}",
    )
