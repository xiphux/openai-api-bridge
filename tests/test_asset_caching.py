"""Conditional-request handling for stored assets.

A generated asset is addressed by a random id and its bytes never change, but
the responses carried no ``Cache-Control`` and Starlette's ``FileResponse``
never acts on ``If-None-Match`` — so a client that already held an image
re-downloaded it in full on every render.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.responses import FileResponse

from openai_api_bridge.api._assets import _etag_for, _matches, asset_response
from openai_api_bridge.infra.filestore import FileMetadata, OpenedFile

FILE_ID = "0123456789abcdef0123456789abcdef"


def _meta(file_id: str = FILE_ID) -> FileMetadata:
    return FileMetadata(
        id=file_id,
        storage_path=f"01/23/{file_id}.png",
        content_type="image/png",
        byte_size=8,
        kind="image",
        source_backend="p",
        source_model="m",
        prompt_excerpt=None,
        created_at=0,
        last_accessed_at=0,
        pinned=False,
    )


@pytest.fixture
def stored(tmp_path: Path) -> OpenedFile:
    path = tmp_path / f"{FILE_ID}.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return OpenedFile(path=path, meta=_meta())


def test_first_fetch_is_cacheable_and_carries_a_validator(stored: OpenedFile) -> None:
    response = asset_response(stored, if_none_match=None)

    assert isinstance(response, FileResponse)
    assert response.headers["etag"] == f'"{FILE_ID}"'
    assert response.headers["cache-control"] == "private, max-age=31536000, immutable"


def test_validator_comes_from_the_id_not_the_stat(stored: OpenedFile, tmp_path: Path) -> None:
    """Starlette derives its ETag from mtime+size, which a restore invalidates.

    Re-copying the bytes — restoring a backup, recreating a volume — must not
    tell every client its cached copy is stale when the content is identical.
    """
    response = asset_response(stored, if_none_match=None)
    etag_before = response.headers["etag"]

    moved = tmp_path / "restored.png"
    moved.write_bytes(stored.path.read_bytes())
    restored = OpenedFile(path=moved, meta=_meta())
    response_after = asset_response(restored, if_none_match=None)

    assert response_after.headers["etag"] == etag_before


def test_matching_validator_returns_304_without_a_body(stored: OpenedFile) -> None:
    response = asset_response(stored, if_none_match=f'"{FILE_ID}"')

    assert response.status_code == 304
    assert response.body == b""
    # RFC 9110 asks a 304 to repeat the headers a 200 would have carried; a
    # client that lost the directive here would revalidate again next time.
    assert response.headers["cache-control"] == "private, max-age=31536000, immutable"


def test_stale_validator_still_sends_the_body(stored: OpenedFile) -> None:
    response = asset_response(stored, if_none_match='"some-other-asset"')

    assert isinstance(response, FileResponse)
    assert response.status_code == 200


@pytest.mark.parametrize(
    "header",
    [
        f'"{FILE_ID}"',
        f'W/"{FILE_ID}"',
        "*",
        f'"other", "{FILE_ID}"',
        f'  W/"{FILE_ID}" ,  "other"  ',
    ],
    ids=["strong", "weak", "wildcard", "list", "list-with-whitespace"],
)
def test_if_none_match_forms_that_should_match(header: str) -> None:
    """Weak comparison per RFC 9110 §13.1.2, and the header may carry a list."""
    assert _matches(header, _etag_for(FILE_ID))


@pytest.mark.parametrize(
    "header",
    [None, "", '"other"', '"0123456789abcdef0123456789abcdee"', FILE_ID],
    ids=["absent", "empty", "different", "off-by-one", "unquoted"],
)
def test_if_none_match_forms_that_should_not_match(header: str | None) -> None:
    assert not _matches(header, _etag_for(FILE_ID))


def test_responses_vary_on_authorization(stored: OpenedFile) -> None:
    """These endpoints are cacheable AND behind a bearer token.

    `private` tells a well-behaved shared cache to keep out; Vary is what
    stops one that stored the response anyway from serving it to a request
    bearing a different credential.
    """
    served = asset_response(stored, if_none_match=None)
    not_modified = asset_response(stored, if_none_match=f'"{FILE_ID}"')

    assert served.headers["vary"] == "Authorization"
    assert not_modified.headers["vary"] == "Authorization"


def test_stat_is_not_handed_to_starlette(stored: OpenedFile) -> None:
    """FileResponse's own stat is also its existence check.

    Supplying stat_result makes it skip that stat, so a file evicted between
    the store's check and the send is answered as 200 with a Content-Length
    from the stale stat and a truncated body — a silent short read where
    there should be a loud failure.
    """
    response = asset_response(stored, if_none_match=None)

    assert isinstance(response, FileResponse)
    assert response.stat_result is None


# --- content handling -------------------------------------------------------


def test_response_forbids_content_type_sniffing(stored: OpenedFile) -> None:
    """The media type is whatever the upstream claimed when the bytes were
    fetched, and it is echoed as the response's own. Sniffing on top of that
    would let a browser make its own guess about bytes served from the
    bridge's origin."""
    response = asset_response(stored, if_none_match=None)
    assert response.headers["x-content-type-options"] == "nosniff"


def test_304_also_forbids_sniffing(stored: OpenedFile) -> None:
    response = asset_response(stored, if_none_match=f'"{FILE_ID}"')
    assert response.status_code == 304
    assert response.headers["x-content-type-options"] == "nosniff"


def test_disposition_is_set_even_without_an_explicit_filename(stored: OpenedFile) -> None:
    """The video endpoint passed no filename, so it was the one response here a
    browser would render inline rather than download."""
    response = asset_response(stored, if_none_match=None)
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment")
    assert FILE_ID in disposition


def test_explicit_filename_still_wins(stored: OpenedFile) -> None:
    response = asset_response(stored, if_none_match=None, filename="chosen.png")
    assert "chosen.png" in response.headers["content-disposition"]
