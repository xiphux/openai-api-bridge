# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An OpenAI-compatible HTTP gateway (FastAPI) that aggregates multiple generation
backends behind one endpoint. Clients speak standard OpenAI endpoints; the bridge
dispatches each request to the configured upstream, translating where the upstream
isn't OpenAI-compatible (ComfyUI, Venice, ImageRouter, OpenRouter image) and
passing through where it is (any OpenAI-compatible chat/embedding server).

The README is the authoritative reference for the API surface, provider matrix,
config layers, and deployment (Docker/systemd) — consult it before changing
public behavior.

## Commands

```bash
uv sync                      # install (creates .venv); add --frozen --no-dev for prod
uv run openai-api-bridge     # run the server (needs BRIDGE_API_KEY in env)

uv run pytest                # full suite
uv run pytest -m live        # opt-in live tests against real backends (marker only; not yet wired)
uv run pytest tests/test_video_lifecycle.py::test_name   # single test
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy src              # type-check (strict mode is on)
pre-commit run --all-files   # runs ruff + ruff-format + file hygiene hooks
```

Local dev run: `cp config.toml.example config.toml`, `cp .env.example .env` (set
`BRIDGE_API_KEY`), then `set -a && source .env && set +a && uv run openai-api-bridge`.

Python 3.12+. mypy is **strict** and ruff lint includes bugbear/simplify/pyupgrade —
new code must pass both clean.

## Architecture

Request flow: client → FastAPI router (`api/`) → `BackendDispatcher` → `Backend`
adapter (`backends/<name>/`) → upstream HTTP.

- **Model IDs are `{provider_id}/{model_slug}`.** `config.parse_model_id` splits
  them; the dispatcher maps `provider_id` → the `Backend` instance built for that
  `[[providers]]` block. A single backend *type* can be configured multiple times
  (e.g. two ComfyUI boxes) — each gets its own `id`.

- **`Backend` ABC (`backends/base.py`)** is one union of all capabilities
  (image/video generation+edit, chat, embeddings). Every method defaults to
  raising `UnsupportedOperation`; adapters override only what they support. This
  is deliberate — don't split it into per-capability protocols.

- **Adding a backend** = a new `*ProviderConfig` (discriminated-union member keyed
  on `backend` literal in `config.py`), a new `backends/<name>/` adapter+client,
  and a branch in `dispatcher._build_backend`. No router changes needed.

- **Two config layers (`config.py`):** infra/secrets via env (`BridgeSettings`,
  pydantic-settings, `BRIDGE_*` aliases) loaded once at startup; provider
  definitions via TOML (`ProvidersFile`). Secrets in TOML are stored as the
  *name* of an env var (any field ending in `_env`) and read from `os.environ`
  at point of use — never copied into pydantic state. Don't break this.

- **Lifespan resource graph (`main.py`)** builds and tears down in a fixed order
  (settings → providers → db → migrations → stores → dispatcher → scheduler →
  eviction). Single uvicorn worker only — SQLite + in-memory job state don't
  survive forking. The built graph is installed as one frozen `BridgeResources`
  (`resources.py`) and routes reach it via `resources(request)` — don't read
  `request.app.state` directly, that's an untyped cast the checker can't verify.

### Video jobs (the one async/stateful path)

Images/chat/embeddings are synchronous. Video is async to match OpenAI's Sora
lifecycle: `POST /v1/videos` returns `{id, status: "queued"}`; clients poll
`GET /v1/videos/{id}` and fetch `GET /v1/videos/{id}/content` once completed.

- `TaskScheduler` (`infra/tasks.py`) is a bounded asyncio pool — **not** FastAPI
  `BackgroundTasks` (those die on client disconnect; video must outlive the request).
  It keeps strong task refs (GC footgun) and indexes by name for `DELETE` cancel.
- `_videos_runner.run_video_job` drives queued→in_progress→completed/failed and
  persists every transition to the `video_jobs` table via `JobStore`.
- On startup, `JobStore.mark_stale_failed` fails any job left mid-flight by a
  restart (in-memory `input_reference` bytes are gone).

### File store & eviction

`FileStore` (`infra/filestore.py`) caches generated assets on disk
(`${FILES_DIR}/{id[:2]}/{id[2:4]}/{id}{ext}`) with SQLite metadata. Writes are
atomic (`.tmp` → `Path.replace`, row inserted after rename). Returned to clients
as bridge-internal `/v1/files/{id}/content` URLs. `EvictionLoop` (`infra/eviction.py`)
runs periodically, deleting by retention TTL and LRU past `MAX_CACHE_GB`.

### ComfyUI specifics

ComfyUI has no OpenAI surface, so a "model" is a workflow file pair on disk in
`workflows_dir`: `{slug}.json` (API-format graph) + `{slug}.meta.json` (declarative
metadata — which node receives the prompt/images, dimension/length nodes, output
type). `backends/comfyui/workflows.py` scans and prepares these; the meta schema
is documented in the README.

## Conventions

- All errors derive from `BridgeError` (`errors.py`) with `status_code` /
  `error_type` / `code`; handlers in `main.py` render the OpenAI error envelope.
  Raise a `BridgeError` subclass rather than returning ad-hoc JSON.
- Non-standard `/v1/models` fields (`owned_by`, `display_name`, `kind`,
  `supports_tools`) come from `ModelEntry`; standard clients ignore them. See the
  README "Model metadata extensions" table before changing them.
- Tests stub upstream HTTP with `respx`; `pytest-asyncio` is in auto mode. The
  `client` fixture (`conftest.py`) runs the full app with an empty provider config
  for auth/validation/404 paths; infra fixtures use a fresh sqlite + tmp dir per test.
- **The runtime speaks `httpx2`, but tests author respx mocks in `httpx` v1.**
  respx patches httpcore, so `conftest.py` points its default mocker at
  `httpcore2` to intercept httpx2 traffic — but `return_value=` still type
  checks against `httpx.Response` and rejects an `httpx2.Response`
  (lundberg/respx#324). Rule of thumb: objects handed to **respx** are `httpx`,
  objects handed to **bridge code** are `httpx2`. Only `httpx` is a dev dep,
  kept purely so respx has something to build mocks with; `httpx2` is a
  production dependency in `[project.dependencies]` that every backend imports.
