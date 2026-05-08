"""Eviction policy: TTL phase, LRU phase, and pinning exemption."""

from __future__ import annotations

import time

import pytest

from openai_api_bridge.infra.eviction import run_eviction_pass
from openai_api_bridge.infra.filestore import FileStore


async def _put_with_ages(
    filestore: FileStore,
    fixtures: list[tuple[bytes, int, int, bool]],
) -> list[str]:
    """Helper: insert files then back-date their created_at / last_accessed_at.

    ``fixtures`` is a list of (data, seconds_old_created, seconds_old_atime, pinned).
    """
    now = int(time.time())
    ids: list[str] = []
    for data, old_create, old_atime, pinned in fixtures:
        fid = await filestore.put(
            data,
            content_type="image/png",
            kind="image",
            source_backend="x",
            source_model="m",
            pinned=pinned,
        )
        await filestore.db.execute(
            "UPDATE generated_files SET created_at = ?, last_accessed_at = ? WHERE id = ?",
            (now - old_create, now - old_atime, fid),
        )
        ids.append(fid)
    return ids


async def test_ttl_deletes_expired_unpinned_files(filestore: FileStore) -> None:
    [old_id, fresh_id, pinned_old_id] = await _put_with_ages(
        filestore,
        [
            (b"old",          10 * 86400, 10 * 86400, False),  # 10d old
            (b"fresh",         1 * 86400,  1 * 86400, False),  # 1d old
            (b"pinned-old",  100 * 86400, 100 * 86400, True),  # ancient but pinned
        ],
    )
    ttl_count, lru_count = await run_eviction_pass(
        filestore, retention_seconds=7 * 86400, max_cache_bytes=10**12,
    )
    assert ttl_count == 1
    assert lru_count == 0
    assert await filestore.get_metadata(old_id) is None
    assert await filestore.get_metadata(fresh_id) is not None
    assert await filestore.get_metadata(pinned_old_id) is not None


async def test_lru_deletes_oldest_unpinned_until_under_cap(
    filestore: FileStore,
) -> None:
    [a_id, b_id, c_id, pinned_id] = await _put_with_ages(
        filestore,
        [
            # All recent so TTL doesn't fire. Increasing atime = increasing recency.
            (b"a" * 100,  60,  300, False),  # oldest atime
            (b"b" * 100,  60,  200, False),  # middle
            (b"c" * 100,  60,  100, False),  # newest
            (b"p" * 200,  60,  900, True),   # pinned, oldest atime — exempt
        ],
    )
    # Total = 100+100+100+200 = 500 bytes. Cap at 350 bytes:
    # eviction must delete a (-> 400) and b (-> 300), then stop. c survives.
    # If pinned weren't filtered, it'd be deleted first (oldest atime); that
    # this test passes proves the filter works.
    ttl_count, lru_count = await run_eviction_pass(
        filestore, retention_seconds=86400, max_cache_bytes=350,
    )
    assert ttl_count == 0
    assert lru_count == 2  # a then b (oldest atime first; pinned skipped)
    assert await filestore.get_metadata(a_id) is None
    assert await filestore.get_metadata(b_id) is None
    assert await filestore.get_metadata(c_id) is not None
    assert await filestore.get_metadata(pinned_id) is not None


async def test_pinned_pinning_exempts_from_both_phases(
    filestore: FileStore,
) -> None:
    [pinned_id, victim_id] = await _put_with_ages(
        filestore,
        [
            (b"p" * 1_000, 100 * 86400, 100 * 86400, True),
            (b"v" * 100,    50 * 86400,  50 * 86400, False),
        ],
    )
    await run_eviction_pass(
        filestore, retention_seconds=7 * 86400, max_cache_bytes=10,
    )
    # Pinned file is older, larger, AND below cap — but pinned, so it survives both phases.
    assert await filestore.get_metadata(pinned_id) is not None
    assert await filestore.get_metadata(victim_id) is None


async def test_no_eviction_when_under_cap(filestore: FileStore) -> None:
    fid = (await _put_with_ages(filestore, [(b"x" * 100, 60, 60, False)]))[0]
    ttl, lru = await run_eviction_pass(
        filestore, retention_seconds=86400, max_cache_bytes=10**12,
    )
    assert ttl == 0 and lru == 0
    assert await filestore.get_metadata(fid) is not None


@pytest.mark.parametrize("retention_days", [1, 7, 30, 365])
async def test_ttl_threshold_is_inclusive_of_retention_window(
    filestore: FileStore, retention_days: int,
) -> None:
    # File aged exactly to the cutoff should NOT be deleted (created_at < cutoff is the predicate).
    await _put_with_ages(
        filestore,
        [(b"borderline", retention_days * 86400 - 1, retention_days * 86400 - 1, False)],
    )
    ttl, lru = await run_eviction_pass(
        filestore,
        retention_seconds=retention_days * 86400,
        max_cache_bytes=10**12,
    )
    assert ttl == 0
    assert lru == 0
