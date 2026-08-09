"""Media-type policy in one place: what an asset is called, and what we serve.

Four separate tables used to answer "what do I call this image", each with its
own coverage: the file store's extension map, ComfyUI's upload-extension
allowlist, and a ``_filename_for`` apiece in the Venice and ImageRouter
clients. They disagreed — ImageRouter knew about AVIF and Venice didn't,
ComfyUI knew about BMP and TIFF and the file store didn't — so the same upload
was named three different ways depending on which provider it reached, and a
newly supported format had to be added in up to four places to take effect.

Two axes, kept separate on purpose:

* **Extensions** (:func:`asset_extension`, :func:`image_extension`) — what the
  bridge writes on disk or sends as a multipart filename. Labels, not
  decisions: every consumer here sniffs the real format on load.
* **Serveability** (:func:`sanitize_content_type`) — what the bridge is willing
  to repeat back to a client as a response media type. Wider than the extension
  table, because a type can be perfectly safe to serve without there being an
  extension worth writing.
"""

from __future__ import annotations

# Image types the bridge will name. This set is also what bounds ComfyUI
# uploads: the content type traces straight back to the client's own multipart
# part (see ``api/images.py``, which takes ``upload.content_type`` at face
# value), so an unbounded mapping — ``mimetypes.guess_extension``, as this once
# was — let the caller choose the extension of a file written into ComfyUI's
# input directory: ``application/x-sh`` to ``.sh``, ``text/html`` to ``.html``.
# The stem is random and nothing executes it, but it handed an authenticated
# caller an arbitrary-bytes-with-chosen-extension write into an upstream that
# is very often unauthenticated itself, for no benefit.
#
# Video types are therefore deliberately kept out of this table and added only
# where they're wanted (see ``asset_extension``) — the upload path must not be
# able to reach them.
_IMAGE_EXTENSIONS: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/avif": ".avif",
}

_VIDEO_EXTENSIONS: dict[str, str] = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
}

_ASSET_EXTENSIONS: dict[str, str] = _IMAGE_EXTENSIONS | _VIDEO_EXTENSIONS

# Media types the bridge will repeat back to a client verbatim. The stored
# content type is whatever the upstream's `content-type` header claimed when
# the asset was fetched, and it is served back as the response's media type —
# so an upstream, CDN error page or WAF interstitial answering with markup
# would otherwise decide what a browser does with bytes served from the
# bridge's own origin.
#
# Wider than the extension tables above, which only need an entry where there
# is a filename worth writing: apng, heic and heif are real outputs from these
# providers and belong here even though the bridge has no extension for them.
#
# image/svg+xml is deliberately absent. It is a document format that carries
# script, none of these providers emit it, and admitting it would undo the
# point of the list.
_SERVEABLE_TYPES: frozenset[str] = frozenset(
    set(_ASSET_EXTENSIONS)
    | {
        "image/apng",
        "image/heic",
        "image/heif",
        "video/x-matroska",
        "video/mpeg",
        "video/ogg",
    }
)

# What an unrecognised type is recorded as. Not a rejection: the bytes are a
# generation the caller has already paid for, and the store's job is to keep
# them. It only declines to vouch for what they are.
_OPAQUE_TYPE = "application/octet-stream"


def normalize_content_type(content_type: str) -> str:
    """Strip charset/boundary parameters and case, e.g. ``image/PNG; q=1``."""
    return content_type.split(";", 1)[0].strip().lower()


def sanitize_content_type(content_type: str) -> str:
    """Normalise an upstream-declared media type to one safe to serve back."""
    normalized = normalize_content_type(content_type)
    if normalized in _SERVEABLE_TYPES:
        return normalized
    return _OPAQUE_TYPE


def asset_extension(content_type: str) -> str:
    """File extension for a stored asset, or ``""`` when there isn't a known one.

    Empty rather than a guess: the id is what addresses the file, so an
    extensionless path is merely less descriptive, not broken.
    """
    return _ASSET_EXTENSIONS.get(normalize_content_type(content_type), "")


def image_extension(content_type: str, *, default: str = ".png") -> str:
    """Extension for an image the bridge is *uploading* to an upstream.

    Video types are unreachable here by construction; see ``_IMAGE_EXTENSIONS``.
    """
    return _IMAGE_EXTENSIONS.get(normalize_content_type(content_type), default)


def image_filename(content_type: str, *, fallback: str, stem: str = "image") -> str:
    """Multipart filename for a reference image.

    Several providers infer the input format from the extension when the
    content-type header is generic, so keeping it accurate avoids spurious
    "unsupported format" rejections. ``fallback`` is returned whole for a type
    with no known extension, because providers differ on what they'd rather
    receive — ImageRouter takes a bare ``image``, Venice a definite
    ``image.png``.
    """
    ext = _IMAGE_EXTENSIONS.get(normalize_content_type(content_type))
    return f"{stem}{ext}" if ext else fallback
