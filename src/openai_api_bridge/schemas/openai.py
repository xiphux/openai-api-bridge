"""OpenAI-shaped request and response models.

We intentionally accept-and-ignore extra fields the OpenAI API supports but the
bridge has no use for (``user``, ``quality``, ``style``, etc.) so well-behaved
clients don't error out. Validation happens at the field level — invalid
``size`` formats fail fast with a 400.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Permissive(BaseModel):
    """Accept and silently drop unknown fields."""

    model_config = ConfigDict(extra="ignore")


# --- /v1/images/generations -------------------------------------------------

# Ceiling on images per request. Lives here rather than beside the endpoint
# because this model is what enforces it on the JSON path — but the multipart
# edits path can't use a pydantic model (it binds Form fields), so it checks
# the same value by hand. Two enforcement points, one number: they disagreed
# silently when it was written out twice.
MAX_IMAGES_PER_REQUEST = 4


class ImagesGenerationRequest(_Permissive):
    model: str
    prompt: str
    n: int = Field(default=1, ge=1, le=MAX_IMAGES_PER_REQUEST)
    size: str | None = None
    response_format: Literal["url", "b64_json"] = "url"


class ImageData(BaseModel):
    url: str | None = None
    b64_json: str | None = None


class ImagesResponse(BaseModel):
    created: int
    data: list[ImageData]


# --- /v1/models -------------------------------------------------------------


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


# --- /v1/videos -------------------------------------------------------------


VideoStatus = Literal["queued", "in_progress", "completed", "failed"]


class VideoErrorBlock(BaseModel):
    message: str
    type: str | None = None
    code: str | None = None


class VideoObject(BaseModel):
    id: str
    object: Literal["video"] = "video"
    model: str
    status: VideoStatus
    progress: int | None = None
    seconds: float | None = None
    size: str | None = None
    created_at: int
    completed_at: int | None = None
    error: VideoErrorBlock | None = None
