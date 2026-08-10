"""``Database.transaction`` has to hold its guarantee on a *shared* connection.

One aiosqlite connection serves every task in the process, and SQLite's commit
and rollback are scoped to the connection rather than to a block of statements.
So the interesting cases here aren't about SQL — they're about what a second
task can do to a transaction while the first one is between two awaits.
"""

from __future__ import annotations

import asyncio

import pytest

from openai_api_bridge.infra.db import Database


async def _rows(db: Database) -> set[str]:
    return {r["id"] for r in await db.fetchall("SELECT id FROM t")}


@pytest.fixture
async def tdb(db: Database) -> Database:
    await db.execute("CREATE TABLE t (id TEXT PRIMARY KEY)")
    return db


async def test_execute_cannot_commit_an_open_transaction(tdb: Database) -> None:
    """A concurrent `execute` must not commit a transaction mid-flight.

    Without the write lock the `execute` below runs while the block is parked
    on its sleep, and its commit commits the block's first insert too — so the
    rollback that follows has nothing left to undo and 'doomed' survives.
    """
    started = asyncio.Event()

    async def failing_transaction() -> None:
        async with tdb.transaction() as conn:
            await conn.execute("INSERT INTO t (id) VALUES ('doomed')")
            started.set()
            await asyncio.sleep(0.05)
            raise RuntimeError("boom")

    task = asyncio.create_task(failing_transaction())
    await started.wait()
    await tdb.execute("INSERT INTO t (id) VALUES ('bystander')")

    with pytest.raises(RuntimeError, match="boom"):
        await task

    assert await _rows(tdb) == {"bystander"}, "the rolled-back insert must not survive"


async def test_a_failing_transaction_does_not_ride_on_a_neighbours_commit(
    tdb: Database,
) -> None:
    """Two overlapping blocks, one failing. Unserialized, the successful one's
    commit is connection-wide and carries the failing one's insert with it —
    so 'doomed' lands despite its own block raising."""
    started = asyncio.Event()

    async def failing() -> None:
        async with tdb.transaction() as conn:
            await conn.execute("INSERT INTO t (id) VALUES ('doomed')")
            started.set()
            await asyncio.sleep(0.05)
            raise RuntimeError("boom")

    async def succeeding() -> None:
        await started.wait()
        async with tdb.transaction() as conn:
            await conn.execute("INSERT INTO t (id) VALUES ('kept')")

    results = await asyncio.gather(failing(), succeeding(), return_exceptions=True)
    assert isinstance(results[0], RuntimeError)
    assert await _rows(tdb) == {"kept"}


async def test_concurrent_transactions_are_serialized(tdb: Database) -> None:
    """Two blocks that both succeed both land, and neither observes the
    other's uncommitted rows while it runs."""

    async def insert(name: str) -> None:
        async with tdb.transaction() as conn:
            await conn.execute("INSERT INTO t (id) VALUES (?)", (f"{name}-a",))
            await asyncio.sleep(0.01)
            await conn.execute("INSERT INTO t (id) VALUES (?)", (f"{name}-b",))

    await asyncio.gather(*(insert(n) for n in ("x", "y", "z")))
    assert await _rows(tdb) == {"x-a", "x-b", "y-a", "y-b", "z-a", "z-b"}


async def test_a_cancelled_transaction_is_rolled_back(tdb: Database) -> None:
    """CancelledError is a BaseException, so `except Exception` never saw it.

    A block cancelled between its statements and its commit would release the
    lock with an open transaction on the shared connection, and the next
    writer's commit would adopt the abandoned rows — the exact failure the
    lock exists to prevent, reached by the one path that skipped the rollback.
    """
    started = asyncio.Event()

    async def cancelled_transaction() -> None:
        async with tdb.transaction() as conn:
            await conn.execute("INSERT INTO t (id) VALUES ('abandoned')")
            started.set()
            await asyncio.sleep(3600)

    task = asyncio.create_task(cancelled_transaction())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The next writer must not inherit the cancelled block's row.
    await tdb.execute("INSERT INTO t (id) VALUES ('next-writer')")
    assert await _rows(tdb) == {"next-writer"}


async def test_a_cancelled_execute_does_not_leave_a_write_for_the_next_committer(
    tdb: Database,
) -> None:
    """`execute()` has the same two-await window `transaction()` does.

    aiosqlite queues the statement onto its worker thread, so cancelling the
    awaiting task doesn't cancel the SQL. Without a rollback the lock is
    released with that statement in an open transaction, and whichever writer
    commits next adopts a write its caller believes was cancelled.
    """
    started = asyncio.Event()
    real_commit = tdb.conn.commit

    async def slow_commit() -> None:
        # Widen the window between the statement and its commit so the
        # cancellation lands inside it deterministically.
        started.set()
        await asyncio.sleep(3600)
        await real_commit()

    async def cancelled_write() -> None:
        await tdb.execute("INSERT INTO t (id) VALUES ('cancelled')")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tdb.conn, "commit", slow_commit)
        task = asyncio.create_task(cancelled_write())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    await tdb.execute("INSERT INTO t (id) VALUES ('next-writer')")
    assert await _rows(tdb) == {"next-writer"}


async def test_execute_inside_transaction_raises_rather_than_deadlocking(
    tdb: Database,
) -> None:
    """The lock is not reentrant, so the obvious misuse would otherwise hang
    forever. Fail loudly and say what to use instead."""
    with pytest.raises(RuntimeError, match="inside transaction"):
        async with tdb.transaction():
            await tdb.execute("INSERT INTO t (id) VALUES ('nope')")

    assert await _rows(tdb) == set()
