"""FastAPI app factory + uvicorn entry point.

The lifespan context manages the bridge's resource graph in well-defined order:

  startup:  settings -> providers -> db -> migrations -> file/job stores ->
            mark stale jobs failed -> dispatcher -> scheduler -> eviction loop
  shutdown: eviction loop -> scheduler drain -> dispatcher (closes httpx) -> db
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api import chat as chat_api
from .api import embeddings as embeddings_api
from .api import files as files_api
from .api import images as images_api
from .api import models as models_api
from .api import videos as videos_api
from .auth import require_api_key  # noqa: F401  (imported here to keep DI graph alive)
from .config import (
    BridgeSettings,
    get_settings,
    init_providers,
)
from .dispatcher import BackendDispatcher
from .errors import BridgeError, bridge_error_handler, error_payload
from .infra.db import Database, run_migrations
from .infra.eviction import EvictionLoop
from .infra.filestore import FileStore
from .infra.jobstore import JobStore
from .infra.tasks import TaskScheduler

log = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: BridgeSettings = get_settings()
    _configure_logging(settings.log_level)
    log.info("Starting openai-api-bridge")

    providers = init_providers(settings.config_path)
    log.info("Loaded %d provider(s) from %s", len(providers.providers), settings.config_path)

    db = Database(settings.sqlite_path)
    await db.connect()
    schema_version = await run_migrations(db)
    log.info("DB at %s, schema_version=%d", settings.sqlite_path, schema_version)

    filestore = FileStore(db, settings.files_dir)
    settings.files_dir.mkdir(parents=True, exist_ok=True)
    jobstore = JobStore(db)

    stale = await jobstore.mark_stale_failed("Bridge restarted before this job completed")
    if stale:
        log.warning("Marked %d stale video jobs as failed on startup", stale)

    dispatcher = BackendDispatcher(providers)
    scheduler = TaskScheduler(max_concurrent=settings.max_concurrent_video_jobs)
    eviction = EvictionLoop(
        filestore,
        retention_seconds=settings.retention_days * 86400,
        max_cache_bytes=settings.max_cache_bytes,
        interval_seconds=settings.eviction_interval_seconds,
    )
    eviction.start()

    app.state.settings = settings
    app.state.providers = providers
    app.state.db = db
    app.state.filestore = filestore
    app.state.jobstore = jobstore
    app.state.dispatcher = dispatcher
    app.state.scheduler = scheduler
    app.state.eviction = eviction

    log.info(
        "Ready on %s:%d  files=%s  cache=%dGB  retention=%dd",
        settings.host,
        settings.port,
        settings.files_dir,
        settings.max_cache_gb,
        settings.retention_days,
    )

    try:
        yield
    finally:
        log.info("Shutting down...")
        await eviction.stop()
        await scheduler.shutdown(timeout=30.0)
        await dispatcher.aclose()
        await db.close()
        log.info("Shutdown complete")


def _validation_handler(_request, exc: RequestValidationError) -> JSONResponse:
    """Map FastAPI's pydantic validation errors to the OpenAI error envelope."""
    # Surface the first error concisely; fuller details would leak schema internals.
    errs = exc.errors()
    if errs:
        first = errs[0]
        loc = ".".join(str(p) for p in first.get("loc", []))
        msg = first.get("msg", "Invalid request")
        message = f"{loc}: {msg}" if loc else msg
        param = first.get("loc", ["body"])[-1] if first.get("loc") else None
    else:
        message = "Invalid request"
        param = None
    return JSONResponse(
        status_code=400,
        content=error_payload(
            message=message,
            type_="invalid_request_error",
            code="invalid_request",
            param=str(param) if param else None,
        ),
    )


def _unhandled_handler(_request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content=error_payload(
            message="Internal server error",
            type_="api_error",
            code="internal_error",
        ),
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="OpenAI API Bridge",
        description="OpenAI-compatible HTTP bridge for ComfyUI and Venice.",
        lifespan=lifespan,
        # Disable the default OpenAPI/docs auth-free routes? For v1 we leave
        # them enabled since they don't hit any backend.
    )
    app.add_exception_handler(BridgeError, bridge_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(Exception, _unhandled_handler)

    app.include_router(models_api.router)
    app.include_router(images_api.router)
    app.include_router(videos_api.router)
    app.include_router(files_api.router)
    app.include_router(chat_api.router)
    app.include_router(embeddings_api.router)
    return app


app = create_app()


def run() -> None:
    """Console-script entry point (``openai-api-bridge``)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "openai_api_bridge.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        # Single worker for our single-process model. SQLite + in-memory state
        # would not survive worker forking.
        workers=1,
    )


if __name__ == "__main__":
    run()
