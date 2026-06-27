# openai-api-bridge

OpenAI-compatible HTTP gateway that aggregates multiple generation backends
behind a single endpoint. Clients (LibreChat, LobeChat, custom OpenAI SDK
code, `curl`) point at the bridge and use standard OpenAI endpoints; the
bridge dispatches each request to the right upstream — translating where
the upstream isn't OpenAI-compatible (ComfyUI, Venice image), pure-passing
where it is (any OpenAI-compatible chat / embedding server).

> **Need a chat UI to drive this?** [GlyphStream][glyphstream] is a
> companion project — a lightweight self-hostable SvelteKit chat
> frontend designed against the OpenAI spec, with first-class support
> for inline image and video rendering when the upstream serves them.

[glyphstream]: https://github.com/xiphux/glyphstream

```
client (LibreChat/LobeChat/curl/...)
        │  OpenAI API: /v1/{models,chat,embeddings,images,videos,files}
        ▼
   openai-api-bridge   ←  config.toml: [[providers]] = comfyui | venice | openai | ...
        │
        ├──► ComfyUI workflow (image or video, via user-defined workflow JSON)
        ├──► Venice.ai native /api/v1/image/{generate,edit} (image gen + img2img)
        ├──► ImageRouter / OpenRouter (cloud image · video · chat across many vendors)
        └──► Any OpenAI-compatible upstream (llama-server, vLLM, OpenAI, ...)
              passthrough for /v1/chat/completions and /v1/embeddings
```

## Supported providers

| Provider type | What it covers | Why it's a separate adapter |
|---|---|---|
| **ComfyUI** | Image and video generation via user-defined workflow files | ComfyUI has no OpenAI-compatible HTTP surface; bridge translates |
| **Venice.ai** | Image generation (`/api/v1/image/generate`) and img2img edits (`/api/v1/image/edit`, single reference image) | Venice's OpenAI-compat surface is **chat-only**; image is proprietary |
| **ImageRouter** | Image and video generation across many providers | OpenAI-compat *content* but path-divergent — model catalog is at `/v2/models`, inference at `/v1/openai/...`, and the video endpoint is sync (single POST) rather than OpenAI's async `/v1/videos` lifecycle |
| **OpenRouter** | Chat, embeddings, and image generation across many vendors | Chat/embeddings are spec-compliant; image generation diverges — OpenRouter exposes it via chat completions with a non-standard `message.images` array on the response. The bridge translates so clients see standard `/v1/images/generations` and `/v1/images/edits` |
| **OpenAI passthrough** | Chat completions (sync + streaming) and embeddings against any OpenAI-compatible upstream | No translation needed — bridge forwards bytes; the value is *aggregation* (one bridge endpoint, many upstreams in the model list) |

Configure as many of each as you want. The most common deployment fronts a
single ComfyUI box (image + video), Venice or ImageRouter (cloud image /
video), and one or more local llama-server / vLLM instances (chat +
embedding) — all behind one bridge URL.

Adding a future backend (Replicate, fal.ai, OpenRouter image, a second
ComfyUI instance, etc.) is a single new `[[providers]]` block in
`config.toml` plus an adapter module — see `src/openai_api_bridge/backends/`
for the existing implementations.

## What this bridge does *not* do

- **Audio (TTS / Whisper), Realtime API, Assistants API, fine-tuning,
  batch.** Not currently in scope; could be added per the same pattern if
  needed.

## API surface

| Method | Path | Notes |
|---|---|---|
| GET    | `/v1/models`                       | Aggregate listing across all providers |
| POST   | `/v1/chat/completions`             | Sync + SSE streaming passthrough to openai-compat upstreams |
| POST   | `/v1/embeddings`                   | Sync passthrough to openai-compat upstreams |
| POST   | `/v1/images/generations`           | Sync; JSON body |
| POST   | `/v1/images/edits`                 | Sync; multipart (`image` + `prompt` + `model`). Send multiple reference images by repeating `image` or using `image[]` (up to 16); backends that can't use all of them error rather than silently drop |
| POST   | `/v1/videos`                       | Async; multipart; returns `{id, status: "queued"}` |
| GET    | `/v1/videos/{id}`                  | Poll job status |
| GET    | `/v1/videos/{id}/content`          | Stream final mp4 once `status: "completed"` |
| DELETE | `/v1/videos/{id}`                  | Cancel a queued/in-progress job; releases the runner slot |
| GET    | `/v1/files/{id}/content`           | Bridge-internal asset URLs returned in responses |

Auth: `Authorization: Bearer ${BRIDGE_API_KEY}`.

Model IDs follow `{provider_id}/{model_slug}` — so a request like
`{"model": "comfyui/ltxv-t2i", "prompt": "..."}` routes to the ComfyUI provider's
`ltxv-t2i` workflow.

### Model metadata extensions (nonstandard)

The OpenAI `/v1/models` shape is too thin for a multi-modal catalog — it
doesn't say what a model *does*. The bridge adds a few fields to each model
entry so clients can build a useful picker without per-model hardcoding
([GlyphStream][glyphstream] consumes all of these):

| Field | Values | Source |
|---|---|---|
| `owned_by` | the `[[providers]]` block's `id` | always set — lets an aggregating client group models by real provider |
| `display_name` | human-readable name | ComfyUI: meta.json `display_name`; others: upstream catalog. Falls back to the model id |
| `kind` | `"chat"` \| `"image"` \| `"video"` \| `"embedding"`, omitted when unknown | ComfyUI: workflow output type; Venice/ImageRouter: inherent; OpenRouter: the model's `output_modalities`. OpenAI passthrough sets nothing — generic upstreams don't report modality reliably |
| `supports_tools` | `true` / `false`, omitted when unknown | OpenRouter only today, read from each model's advertised capabilities. Clients should treat *omitted* as "configure it yourself" |
| `context_window` | max context size in tokens, omitted when unknown | OpenAI-passthrough upstreams that expose it: llama.cpp's `meta.n_ctx` (a loaded model) or router `--ctx-size`, vLLM's `max_model_len`. The bridge strips the `meta`/`status` blocks otherwise, so this is the only way the size survives the proxy. Clients use it for a "N / max tokens" budget |

Standard OpenAI clients ignore the extra fields; nothing nonstandard is
*required* to use the bridge.

## Configuration

Two layers, by concern:

* **Infrastructure & secrets** → environment variables (see `.env.example`)
* **Provider definitions** → `config.toml` (see `config.toml.example`)

Adding a second ComfyUI instance is a single new `[[providers]]` block — no
code changes needed.

### Environment variables

`BRIDGE_API_KEY` is the only required variable. Everything else defaults
sensibly:

| Variable | Default | Purpose |
|---|---|---|
| `BRIDGE_API_KEY` | — (required) | Bearer token clients must send |
| `BRIDGE_HOST` / `BRIDGE_PORT` | `0.0.0.0` / `8080` | Bind address |
| `BRIDGE_PUBLIC_BASE_URL` | empty | Public origin used to build asset URLs in responses (e.g. `https://bridge.example.com`). Empty → relative URLs, fine when clients reach the bridge at the same host they requested from |
| `BRIDGE_CONFIG_PATH` | `/etc/openai-api-bridge/config.toml` | Provider config location |
| `FILES_DIR` | `/var/lib/openai-api-bridge/files` | Generated-asset cache directory |
| `SQLITE_PATH` | `/var/lib/openai-api-bridge/state.db` | Job + file state database |
| `RETENTION_DAYS` | `30` | TTL for cached generated files (see [Storage & retention](#storage--retention)) |
| `MAX_CACHE_GB` | `50` | Size cap for the file cache (LRU past this) |
| `EVICTION_INTERVAL_SECONDS` | `600` | How often the eviction loop runs |
| `MAX_CONCURRENT_VIDEO_JOBS` | `2` | Parallel video generations (see [Video jobs](#video-jobs)) |
| `LOG_LEVEL` | `INFO` | TRACE / DEBUG / INFO / WARNING / ERROR |

Provider API tokens (`VENICE_API_TOKEN`, `IMAGEROUTER_API_KEY`,
`OPENROUTER_API_KEY`, …) are referenced by *name* from `config.toml`'s
`*_env` fields — the bridge reads whatever variable names you declare
there, so secrets never live in the config file.

## Running locally (uv)

```bash
uv sync
cp config.toml.example config.toml
cp .env.example .env  # set BRIDGE_API_KEY at minimum

# Run with env loaded from .env
set -a && source .env && set +a
uv run openai-api-bridge
```

The bridge listens on `0.0.0.0:8080` by default.

## Docker

### Local build (development)

```bash
cp config.toml.example config.toml
echo "BRIDGE_API_KEY=$(openssl rand -hex 24)" > .env
# Optional: set VENICE_API_TOKEN, COMFYUI_WORKFLOWS_HOST_DIR, etc.

docker compose up -d --build
docker compose logs -f bridge
```

### Published image from GHCR (deployment)

The repo ships a GitHub Actions workflow (`.github/workflows/docker.yml`) that
builds and publishes a multi-arch (amd64 + arm64) image to GitHub Container
Registry on every push to `main` and on `v*.*.*` tags. To pull it:

```bash
# in your .env
BRIDGE_IMAGE=ghcr.io/<your-username>/openai-api-bridge:latest

# private package? authenticate the host first with a PAT (read:packages scope):
echo "$GHCR_PAT" | docker login ghcr.io -u <your-username> --password-stdin

docker compose pull
docker compose up -d
```

Tag conventions published by the workflow:

| Tag                | When | Use it for |
|---|---|---|
| `latest`           | only when a semver `v*.*.*` tag is pushed (matches `nginx` / `postgres` / `python` convention — *not* HEAD of `main`) | production |
| `v1.2.3` / `1.2` / `1` | semver tag pushes | pinning to a specific release |
| `main`             | every push to `main` | bleeding edge / dev integration |
| `pr-42`            | PR builds (built but not pushed) | n/a |
| `sha-abc1234`      | every build | rollback / audit |

State (SQLite + cached files) lives in the named volume `bridge-state`.

## systemd (Arch / CachyOS / any Linux with systemd ≥ 253)

```bash
# Layout — /opt/openai-api-bridge/ holds the source + .venv
sudo mkdir -p /opt/openai-api-bridge
sudo rsync -a --exclude=.venv --exclude=.git . /opt/openai-api-bridge/
sudo chown -R root:root /opt/openai-api-bridge
cd /opt/openai-api-bridge && sudo uv sync --frozen --no-dev

# Config
sudo install -d /etc/openai-api-bridge
sudo install -m 0644 config.toml.example /etc/openai-api-bridge/config.toml
sudo install -m 0600 .env.example /etc/openai-api-bridge.env
# … edit both files (BRIDGE_API_KEY, provider URLs, workflows_dir, etc.)

# Unit
sudo install -m 0644 systemd/openai-api-bridge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now openai-api-bridge
sudo journalctl -u openai-api-bridge -f
```

The unit runs as a `DynamicUser` with `StateDirectory=openai-api-bridge`, so
data lives at `/var/lib/openai-api-bridge` and is owned by an ephemeral UID.

## Smoke check

```bash
KEY=$(grep ^BRIDGE_API_KEY .env | cut -d= -f2)
H="Authorization: Bearer $KEY"

# Should 401:
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/v1/models

# Should list providers' models:
curl -sH "$H" http://localhost:8080/v1/models | jq .

# Generate an image:
curl -sH "$H" -H "Content-Type: application/json" \
  -d '{"model":"comfyui/<your-slug>","prompt":"a red panda","size":"1024x1024"}' \
  http://localhost:8080/v1/images/generations | jq .
```

## Storage & retention

Generated assets are cached on disk (`FILES_DIR`) and indexed in SQLite so
response URLs like `/v1/files/{id}/content` stay valid across requests. The
cache is **not permanent storage** — an eviction loop (every
`EVICTION_INTERVAL_SECONDS`) applies two policies:

- **TTL**: files older than `RETENTION_DAYS` (default 30) are deleted.
- **LRU**: when the cache exceeds `MAX_CACHE_GB` (default 50), the
  least-recently-accessed files are deleted until it fits.

Files referenced by an in-flight video job are pinned and never evicted
mid-job. Clients that want to keep generated media should download and
persist it on their side — [GlyphStream][glyphstream] does exactly this,
pulling each asset into its own media store on first generation.

Completed/failed rows in the video-jobs table are kept as history; they're
metadata-sized and have no eviction policy today.

## Video jobs

`/v1/videos` follows OpenAI's async lifecycle (submit → poll → fetch
content). Operationally:

- At most `MAX_CONCURRENT_VIDEO_JOBS` (default 2) run at once; excess jobs
  queue in `status: "queued"` order.
- Each job has a hard 30-minute wall-clock cap, independent of the
  per-provider poll timeouts.
- On bridge restart, jobs that were queued/in-progress are marked `failed`
  ("bridge restarted") rather than left dangling — clients never poll a
  zombie job forever.
- **ComfyUI dropped-prompt detection**: ComfyUI can silently lose a queued
  prompt (server restart, queue contention). The bridge re-verifies the
  prompt against ComfyUI's `/queue` + `/history` every 30 s and fails the
  job after 3 consecutive misses (~90 s tolerance window), instead of
  holding a runner slot for the full generation timeout.

## Workflow meta.json schema (ComfyUI)

Each workflow file `<name>.json` needs a companion `<name>.meta.json` declaring
where the bridge should inject things:

```json
{
  "positive_prompt_node": "10",
  "display_name": "LTX-V T2V (Fast)",
  "image_inputs": [{ "node": "7", "field": "image" }],
  "dimensions_node": "5",
  "length_node": "8",
  "fps": 24,
  "output_type": "video"
}
```

Only `positive_prompt_node` is required:

| Field | Default | Purpose |
|---|---|---|
| `positive_prompt_node` | — (required) | Node ID that receives the prompt text |
| `positive_prompt_field` | `"text"` | Field name on that node |
| `display_name` | the workflow filename | Human-readable name surfaced in `/v1/models` |
| `image_inputs` | `[]` | Where attached input images land — each entry is `{node, field}` plus optional `format` (`"filename"`, the default, or `"list"`) and `multiple: true` to route all remaining images into one input |
| `dimensions_node` / `width_field` / `height_field` | — / `"width"` / `"height"` | Node that receives the request's `size` |
| `length_node` / `length_field` | — / `"value"` | Node that receives the frame count (video) |
| `fps` | — | Enables OpenAI's `seconds` parameter: `seconds × fps` → frame count injected into `length_node` |
| `seed_nodes` | all nodes with a `seed` / `noise_seed` field | Node IDs to randomize per request; list them explicitly to leave other seeds untouched |
| `output_type` | auto-detected | `"image"` or `"video"`; auto-detection keys off the presence of `SaveVideo` / `VHS_VideoCombine` nodes — set explicitly to override |

## Tests

```bash
uv run pytest                # full suite
uv run pytest -m live        # opt-in live tests against real backends (not yet wired)
uv run ruff check .          # lint
```
