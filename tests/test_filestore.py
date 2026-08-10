"""FileStore: put/get/delete + atomic-write semantics + sharding."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from openai_api_bridge.infra.filestore import FileStore


async def test_put_and_get_roundtrip(filestore: FileStore) -> None:
    data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    file_id = await filestore.put(
        data,
        content_type="image/png",
        kind="image",
        source_backend="comfyui",
        source_model="ltxv-t2i",
        prompt_excerpt="a red panda",
    )
    assert len(file_id) == 32 and all(c in "0123456789abcdef" for c in file_id)

    meta = await filestore.get_metadata(file_id)
    assert meta is not None
    assert meta.content_type == "image/png"
    assert meta.byte_size == len(data)
    assert meta.kind == "image"
    assert meta.source_backend == "comfyui"
    assert meta.source_model == "ltxv-t2i"
    assert meta.prompt_excerpt == "a red panda"
    assert meta.pinned is False


async def test_open_for_read_returns_the_real_path(
    filestore: FileStore,
    files_dir: Path,
) -> None:
    file_id = await filestore.put(
        b"abc",
        content_type="image/png",
        kind="image",
        source_backend="comfyui",
        source_model="m",
    )
    result = await filestore.open_for_read(file_id)
    assert result is not None
    abs_path = result.path
    assert abs_path.exists()
    assert abs_path.read_bytes() == b"abc"
    assert files_dir in abs_path.parents


async def test_open_for_read_missing_returns_none(filestore: FileStore) -> None:
    assert await filestore.open_for_read("does-not-exist") is None


async def test_disk_layout_uses_two_level_shard(filestore: FileStore, files_dir: Path) -> None:
    file_id = await filestore.put(
        b"x",
        content_type="image/png",
        kind="image",
        source_backend="comfyui",
        source_model="m",
    )
    expected = files_dir / file_id[0:2] / file_id[2:4] / f"{file_id}.png"
    assert expected.exists()


async def test_extension_inference(filestore: FileStore, files_dir: Path) -> None:
    cases = [
        ("image/png", ".png"),
        ("image/jpeg", ".jpg"),
        ("video/mp4", ".mp4"),
        ("video/webm", ".webm"),
        ("application/octet-stream", ""),
    ]
    for ct, expected_ext in cases:
        kind = "video" if ct.startswith("video/") else "image"
        fid = await filestore.put(
            b"x", content_type=ct, kind=kind, source_backend="x", source_model="m"
        )
        meta = await filestore.get_metadata(fid)
        assert meta is not None
        assert meta.storage_path.endswith(f"{fid}{expected_ext}"), (
            f"content_type={ct} expected_ext={expected_ext!r} got={meta.storage_path}"
        )


async def test_set_pinned_and_delete(filestore: FileStore, files_dir: Path) -> None:
    fid = await filestore.put(
        b"x",
        content_type="image/png",
        kind="image",
        source_backend="x",
        source_model="m",
    )
    abs_path = files_dir / (await filestore.get_metadata(fid)).storage_path

    await filestore.set_pinned(fid, True)
    meta = await filestore.get_metadata(fid)
    assert meta is not None and meta.pinned is True

    assert abs_path.exists()
    await filestore.delete(fid)
    assert await filestore.get_metadata(fid) is None
    assert not abs_path.exists()


async def test_total_byte_size(filestore: FileStore) -> None:
    assert await filestore.total_byte_size() == 0
    await filestore.put(
        b"a" * 100, content_type="image/png", kind="image", source_backend="x", source_model="m"
    )
    await filestore.put(
        b"b" * 50, content_type="image/png", kind="image", source_backend="x", source_model="m"
    )
    assert await filestore.total_byte_size() == 150


async def test_put_truncates_long_prompt(filestore: FileStore) -> None:
    huge = "x" * 5000
    fid = await filestore.put(
        b"d",
        content_type="image/png",
        kind="image",
        source_backend="x",
        source_model="m",
        prompt_excerpt=huge,
    )
    meta = await filestore.get_metadata(fid)
    assert meta is not None
    assert meta.prompt_excerpt is not None
    assert len(meta.prompt_excerpt) == 500


async def test_invalid_kind_rejected(filestore: FileStore) -> None:
    with pytest.raises(ValueError, match="kind must be"):
        await filestore.put(
            b"x",
            content_type="image/png",
            kind="audio",
            source_backend="x",
            source_model="m",
        )


async def test_open_for_read_returns_none_when_bytes_are_gone(filestore: FileStore) -> None:
    """A row whose file vanished must read as absent, not hand out a dead path.

    The caller opens the path after we return it (FileResponse stats it at
    send time), so returning a path to a missing file surfaced as a 500
    instead of a 404.
    """
    fid = await filestore.put(
        b"payload",
        content_type="image/png",
        kind="image",
        source_backend="p",
        source_model="m",
    )
    found = await filestore.open_for_read(fid)
    assert found is not None

    found.path.unlink()

    assert await filestore.open_for_read(fid) is None


async def test_open_for_read_reaps_the_orphan_row(filestore: FileStore) -> None:
    """The stale row is dropped, so its byte_size stops counting toward the cap."""
    fid = await filestore.put(
        b"payload",
        content_type="image/png",
        kind="image",
        source_backend="p",
        source_model="m",
    )
    found = await filestore.open_for_read(fid)
    assert found is not None
    found.path.unlink()

    await filestore.open_for_read(fid)

    assert await filestore.get_metadata(fid) is None
    assert await filestore.total_byte_size() == 0


async def test_delete_many_removes_rows_and_files(filestore: FileStore) -> None:
    ids = [
        await filestore.put(
            b"x" * 10,
            content_type="image/png",
            kind="image",
            source_backend="p",
            source_model="m",
        )
        for _ in range(5)
    ]
    paths = []
    for fid in ids:
        found = await filestore.open_for_read(fid)
        assert found is not None
        paths.append(found.path)

    removed = await filestore.delete_many(ids)

    assert removed == 5
    assert await filestore.total_byte_size() == 0
    assert not any(p.exists() for p in paths)


async def test_delete_many_chunks_past_the_sqlite_parameter_limit(
    filestore: FileStore,
) -> None:
    """SQLite caps bound parameters (999 by default); a big sweep must not blow it."""
    ids = [
        await filestore.put(
            b"x",
            content_type="image/png",
            kind="image",
            source_backend="p",
            source_model="m",
        )
        for _ in range(450)
    ]

    removed = await filestore.delete_many(ids)

    assert removed == 450
    assert await filestore.total_byte_size() == 0


async def test_delete_many_tolerates_unknown_and_empty_input(filestore: FileStore) -> None:
    assert await filestore.delete_many([]) == 0
    assert await filestore.delete_many(["deadbeef"]) == 0


async def test_a_fresh_read_does_not_rewrite_the_access_time(filestore: FileStore) -> None:
    """The common case — fetching the asset you just generated — costs no write.

    ``put`` stamps last_accessed_at at write time, so the read that follows
    has nothing to correct. Doing it anyway put an UPDATE and a commit on the
    one shared sqlite connection ahead of the first byte of every download,
    which a gallery pays N times at once.
    """
    fid = await filestore.put(
        b"x", content_type="image/png", kind="image", source_backend="p", source_model="m"
    )
    before = await filestore.get_metadata(fid)
    assert before is not None

    assert await filestore.open_for_read(fid) is not None

    after = await filestore.get_metadata(fid)
    assert after is not None
    assert after.last_accessed_at == before.last_accessed_at


async def test_a_stale_access_time_is_still_refreshed(filestore: FileStore) -> None:
    """Throttling the write must not stop LRU ordering from tracking real use."""
    fid = await filestore.put(
        b"x", content_type="image/png", kind="image", source_backend="p", source_model="m"
    )
    stale = int(time.time()) - 86400
    await filestore.db.execute(
        "UPDATE generated_files SET last_accessed_at = ? WHERE id = ?", (stale, fid)
    )

    assert await filestore.open_for_read(fid) is not None

    after = await filestore.get_metadata(fid)
    assert after is not None
    assert after.last_accessed_at > stale


async def test_a_directory_in_place_of_a_file_reads_as_absent(
    filestore: FileStore, files_dir: Path
) -> None:
    """Only a regular file counts. FileResponse would raise at send time on
    anything else, surfacing as a 500 rather than the 404 the caller expects."""
    fid = await filestore.put(
        b"x", content_type="image/png", kind="image", source_backend="p", source_model="m"
    )
    found = await filestore.open_for_read(fid)
    assert found is not None
    found.path.unlink()
    found.path.mkdir()

    assert await filestore.open_for_read(fid) is None


# --- content-type sanitisation ----------------------------------------------


async def test_put_neutralises_a_markup_content_type(filestore: FileStore) -> None:
    """The stored type is echoed as the response media type, so an upstream (or
    a CDN error page, or a WAF interstitial) answering with markup would
    otherwise decide what a browser does with bytes served from the bridge."""
    file_id = await filestore.put(
        b"<script>alert(1)</script>",
        content_type="text/html",
        kind="image",
        source_backend="p",
        source_model="m",
    )
    meta = await filestore.get_metadata(file_id)
    assert meta is not None
    assert meta.content_type == "application/octet-stream"


async def test_put_neutralises_svg(filestore: FileStore) -> None:
    """SVG carries script and no provider here emits it."""
    file_id = await filestore.put(
        b"<svg xmlns='http://www.w3.org/2000/svg'></svg>",
        content_type="image/svg+xml",
        kind="image",
        source_backend="p",
        source_model="m",
    )
    meta = await filestore.get_metadata(file_id)
    assert meta is not None
    assert meta.content_type == "application/octet-stream"


async def test_put_keeps_the_bytes_it_will_not_vouch_for(filestore: FileStore) -> None:
    """Sanitising is not rejecting — the generation was already paid for."""
    payload = b"<script>alert(1)</script>"
    file_id = await filestore.put(
        payload,
        content_type="text/html",
        kind="image",
        source_backend="p",
        source_model="m",
    )
    opened = await filestore.open_for_read(file_id)
    assert opened is not None
    assert opened.path.read_bytes() == payload


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("image/png", "image/png"),
        ("IMAGE/PNG", "image/png"),
        ("image/jpeg; charset=binary", "image/jpeg"),
        ("image/avif", "image/avif"),
        # Real provider outputs with no extension in the asset-extension table
        # (util.media._ASSET_EXTENSIONS) — the serveable set is deliberately
        # wider than the extension map, since a type can be safe to serve
        # without there being a filename worth writing for it.
        ("image/heic", "image/heic"),
        ("video/x-matroska", "video/x-matroska"),
    ],
)
async def test_put_preserves_legitimate_media_types(
    filestore: FileStore, declared: str, expected: str
) -> None:
    file_id = await filestore.put(
        b"data",
        content_type=declared,
        kind="image",
        source_backend="p",
        source_model="m",
    )
    meta = await filestore.get_metadata(file_id)
    assert meta is not None
    assert meta.content_type == expected


async def test_read_sanitises_a_row_written_before_the_allowlist(filestore: FileStore) -> None:
    """Rows predating the narrowing must not still dictate the served type.

    `put` covers everything written from now on, but a database carried across
    the upgrade holds whatever the upstream claimed, and that value is handed
    to FileResponse as the response media type.
    """
    file_id = await filestore.put(
        b"data",
        content_type="image/png",
        kind="image",
        source_backend="p",
        source_model="m",
    )
    # Simulate a row written by the previous version, bypassing put()'s sanitiser.
    await filestore.db.execute(
        "UPDATE generated_files SET content_type = ? WHERE id = ?",
        ("text/html", file_id),
    )

    meta = await filestore.get_metadata(file_id)
    assert meta is not None
    assert meta.content_type == "application/octet-stream"

    opened = await filestore.open_for_read(file_id)
    assert opened is not None
    assert opened.meta.content_type == "application/octet-stream"
