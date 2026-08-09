"""Shared pytest fixtures.

The unit-style fixtures (``db``, ``filestore``, ``jobstore``) operate against
a fresh sqlite + tmp directory per test, so tests are fully isolated.

The ``client`` fixture spins the full FastAPI app via TestClient with an
empty providers config — useful for testing auth, validation, and 404 paths.
For endpoint tests against real backends, stub upstream HTTP via respx.
"""

from __future__ import annotations

import os
import textwrap
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openai_api_bridge.config import reset_caches_for_tests
from openai_api_bridge.infra.db import Database, run_migrations
from openai_api_bridge.infra.filestore import FileStore
from openai_api_bridge.infra.jobstore import JobStore


@pytest.fixture
def files_dir(tmp_path: Path) -> Path:
    d = tmp_path / "files"
    d.mkdir()
    return d


@pytest.fixture
def sqlite_path(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture
async def db(sqlite_path: Path) -> AsyncIterator[Database]:
    database = Database(sqlite_path)
    await database.connect()
    await run_migrations(database)
    try:
        yield database
    finally:
        await database.close()


@pytest.fixture
async def filestore(db: Database, files_dir: Path) -> FileStore:
    return FileStore(db, files_dir)


@pytest.fixture
async def jobstore(db: Database) -> JobStore:
    return JobStore(db)


# --- TestClient fixture for endpoint tests ----------------------------------


@pytest.fixture
def empty_config(tmp_path: Path) -> Path:
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent("""
        [defaults]
        cache_workflows = true
    """)
    )
    return config


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    empty_config: Path,
    files_dir: Path,
    sqlite_path: Path,
) -> Iterator[TestClient]:
    # Set required env *before* the app is constructed; clear cached settings.
    monkeypatch.setenv("BRIDGE_API_KEY", "test-bridge-api-key")
    monkeypatch.setenv("BRIDGE_CONFIG_PATH", str(empty_config))
    monkeypatch.setenv("FILES_DIR", str(files_dir))
    monkeypatch.setenv("SQLITE_PATH", str(sqlite_path))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    reset_caches_for_tests()

    # Late import so env is read fresh.
    from openai_api_bridge.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    # Clean any leftover env-driven caches between tests.
    reset_caches_for_tests()


@pytest.fixture
def docs_client(
    monkeypatch: pytest.MonkeyPatch,
    empty_config: Path,
    files_dir: Path,
    sqlite_path: Path,
) -> Iterator[TestClient]:
    """``client``, but with the interactive docs opted back in."""
    monkeypatch.setenv("BRIDGE_API_KEY", "test-bridge-api-key")
    monkeypatch.setenv("BRIDGE_CONFIG_PATH", str(empty_config))
    monkeypatch.setenv("FILES_DIR", str(files_dir))
    monkeypatch.setenv("SQLITE_PATH", str(sqlite_path))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("BRIDGE_ENABLE_DOCS", "true")
    reset_caches_for_tests()

    from openai_api_bridge.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    reset_caches_for_tests()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-bridge-api-key"}


# --- Misc -------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent ambient env vars from leaking into tests."""
    for var in (
        "BRIDGE_API_KEY",
        "BRIDGE_CONFIG_PATH",
        "BRIDGE_ENABLE_DOCS",
        "BRIDGE_MAX_REQUEST_MB",
        "BRIDGE_PUBLIC_BASE_URL",
        "FILES_DIR",
        "SQLITE_PATH",
        "VENICE_API_TOKEN",
    ):
        if var in os.environ and not os.environ[var].startswith("/var/folders/"):
            monkeypatch.delenv(var, raising=False)
