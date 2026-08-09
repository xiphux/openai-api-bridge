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
        ├──► fal.ai (image + video, with per-model moderation control)
        └──► Any OpenAI-compatible upstream (llama-server, vLLM, OpenAI, ...)
              passthrough for /v1/chat/completions and /v1/embeddings
```

## Supported providers

| Provider type | What it covers | Why it's a separate adapter |
|---|---|---|
| **ComfyUI** | Image and video generation via user-defined workflow files | ComfyUI has no OpenAI-compatible HTTP surface; bridge translates |
| **Venice.ai** | Image generation (`/api/v1/image/generate`) and img2img edits (`/api/v1/image/edit`, single reference image) | Venice's OpenAI-compat surface is **chat-only**; image is proprietary. Its edit models are a separate catalogue (`type=inpaint`) named by suffixing the base id — `gpt-image-2` / `gpt-image-2-edit` — and the edit endpoint accepts only the suffixed form, so the bridge pairs them and routes |
| **ImageRouter** | Image and video generation across many providers | OpenAI-compat *content* but path-divergent — model catalog is at `/v2/models`, inference at `/v1/openai/...`, and the video endpoint is sync (single POST) rather than OpenAI's async `/v1/videos` lifecycle |
| **OpenRouter** | Chat, embeddings, and image generation across many vendors | Chat/embeddings are spec-compliant; image generation diverges — OpenRouter exposes it via chat completions with a non-standard `message.images` array on the response. The bridge translates so clients see standard `/v1/images/generations` and `/v1/images/edits` |
| **fal.ai** | Image generation + edits *and* video across fal's catalogue (Seedream, Nano Banana / Gemini image, GPT Image, FLUX, Veo, Kling, …) | fal exposes each model's *native* input schema, so the bridge can reach the per-model **content-moderation** knob that flat brokers hide — and reads the loosest value out of that schema. Models are discovered from fal's model API and filtered to the categories this backend serves. Images run against fal's synchronous `fal.run/{model}`; video goes through the `queue.fal.run` lifecycle and is served as a standard async `/v1/videos` job |
| **OpenAI passthrough** | Chat completions (sync + streaming) and embeddings against any OpenAI-compatible upstream | No translation needed — bridge forwards bytes; the value is *aggregation* (one bridge endpoint, many upstreams in the model list) |

Configure as many of each as you want. The most common deployment fronts a
single ComfyUI box (image + video), Venice or ImageRouter (cloud image /
video), and one or more local llama-server / vLLM instances (chat +
embedding) — all behind one bridge URL.

Adding a future backend (Replicate, a second ComfyUI instance, etc.) is a
single new `[[providers]]` block in `config.toml` plus an adapter module —
see `src/openai_api_bridge/backends/` for the existing implementations.

### Model catalogue caching

`GET /v1/models` fans out to every configured provider, so an uncached backend
pays an upstream round trip each time a client refreshes its model picker.
ImageRouter, OpenRouter, Venice and OpenAI passthrough cache their listing for
`catalog_ttl_seconds` (default 300; `0` disables). It's a TTL rather than a
permanent cache so a model added upstream appears without restarting the
bridge. (ComfyUI is separate — it scans workflow files from disk, governed by
`cache_workflows`.)

The endpoint also bounds how long it waits on any one provider, at
`MODELS_TIMEOUT_SECONDS` (default 5; `0` disables the bound). Providers are
queried concurrently, so without it the endpoint's latency is the slowest
upstream's read timeout — a wedged upstream stalls the whole listing, healthy
providers included. A provider that misses the budget is left out of *that*
listing only: its fetch is deliberately not cancelled, so its own catalogue
cache still fills and it reappears on a later request, and a request arriving
while that fetch is still running joins it rather than starting a second. A
cold fal catalogue (10–13 paginated round trips) warms up this way on the
first request after boot.

Note this recurs rather than happening only at startup: the catalogue is
re-fetched whenever `catalog_ttl_seconds` lapses, so a provider whose cold
fetch is slower than the budget drops out of the listing once per TTL window
until the refresh lands. Raise `MODELS_TIMEOUT_SECONDS` past that provider's
cold-fetch time, or raise `catalog_ttl_seconds`, if a consumer treats the
listing as authoritative and doesn't merge across refreshes.

Concurrent requests collapse into the fetch already in flight rather than each
starting their own. A **failed** fetch is remembered for `catalog_retry_seconds`
(default 30; `0` retries immediately) and re-raised to callers arriving inside
that window, so an upstream hang costs one timeout rather than one per caller.
It's remembered, not latched — once the window closes the provider is retried
and recovers on its own.

A fetch rejected for **credentials** (upstream 401/403) is held for at least
300s instead, since provider tokens are read from the environment at startup
and a genuinely bad key won't fix itself before a restart. It's still a longer
window rather than a permanent one: 403 is often not about the credential at
all (a WAF interstitial, a geo block, an org quota), so the provider still
recovers on its own. Setting `catalog_retry_seconds = 0` disables failure
caching entirely, credentials included.

Venice can also return a **partial** listing, its `type=inpaint` half failing
while text-to-image succeeds. That's served rather than dropped, but held for
`catalog_retry_seconds` instead of the full TTL so the missing half is
re-attempted soon; edit routing stays unresolved until a complete listing
arrives.

### Edit models on Venice

Venice files text-to-image under `type=image` and image-to-image under
`type=inpaint`, naming the latter by suffixing the base id — `gpt-image-2` /
`gpt-image-2-edit` — and its edit endpoint accepts **only** the suffixed form.
The bridge fetches both listings, lists a matched pair once (advertising
`["text-to-image", "image-to-image"]`), and routes edits to the counterpart, so
a client addresses one model id whichever operation it wants. Edit-only models
are listed in their own right as `["image-to-image"]`.

Routing needs the catalogue, so the first edit loads it if `/v1/models` hasn't
already. A failed load is retried after **`route_retry_seconds`** (default 60);
during that window edits go out unrouted and Venice rejects them, which trades
a bounded tail after recovery against re-fetching the catalogue on every edit
while it's down. Setting it to `0` disables the cooldown and reinstates that
per-request retry. If the `type=inpaint` listing fails, the text-to-image
catalogue is still served — a narrower query failing shouldn't take a healthy
provider off `/v1/models`.

### fal.ai

fal exposes each model's **native** input schema — including its content
moderation knob — which is the reason to reach for it over ImageRouter or
OpenRouter, both of which flatten that away.

**Discovery.** `/v1/models` is populated from fal's model API (~886 models),
filtered to the four categories the bridge serves: `text-to-image`,
`image-to-image`, `text-to-video`, `image-to-video`. Audio and 3D are excluded —
there's no code path for them. `[[providers.models]]` entries are per-model
*overrides*, not a whitelist; set `discover_models = false` to serve a fixed list
instead.

**Paired variants are collapsed.** fal splits a model's text-driven and
reference-image halves across two endpoints (`fal-ai/nano-banana-2` and
`.../edit`). Where the bridge can pair them confidently it lists only the
text-driven id and routes edits to the sibling, so `POST /v1/images/edits`
against the base id just works. That matters most for video, where `/v1/videos`
is a single endpoint for both halves and the caller can't express which it wants.
Each entry's `capabilities` list says what it actually accepts. Set
`collapse_variants = false` to list both halves separately.

**Moderation is loosened by default.** The bridge reads each model's OpenAPI
schema and sets whichever moderation field it exposes to the loosest value that
model allows — deriving it per model rather than hardcoding, so new model
versions need no code change. Set `disable_safety = false` on a model to leave
the upstream's own default in place. Note that **Nano Banana can't be fully
unlocked by anyone**: Google runs a second, non-configurable safety layer no
broker can reach.

**Upstream retention.** Two opt-in knobs, both off unless set:
`store_payloads = false` stops fal storing your prompts and inline input images
(otherwise kept 30 days), and `output_expiration_seconds = N` expires generated
media from fal's CDN. Mind the retrieval budget on the latter — values under 60s
are rejected.

**Video** runs through fal's queue endpoint and surfaces as the standard async
`/v1/videos` lifecycle. `seconds` is mapped to whatever spelling the model
accepts; `size` is *not* forwarded, since these models take `aspect_ratio` and
`resolution` enums instead — pin those via `params`. Abandoned jobs are cancelled
upstream so fal stops billing, except for jobs already in flight when the bridge
restarts.

See the `fal` block in `config.toml.example` for a worked example, and
**[docs/fal.md](docs/fal.md)** for the internals: schema introspection, pairing
rules, display-name disambiguation, retention mechanics, and video job handling.

## What this bridge does *not* do

- **Audio (TTS / Whisper), Realtime API, Assistants API, fine-tuning,
  batch.** Not currently in scope; could be added per the same pattern if
  needed.

## Security model

The bridge assumes a **trusted network**. It is an aggregation layer for your
own generation backends, not a multi-tenant gateway, and the design reflects
that — read this before putting it anywhere reachable from the internet.

**One shared credential, all or nothing.** `BRIDGE_API_KEY` is a single bearer
token with no scopes, no per-client keys, and no per-caller isolation. Anyone
holding it can use every configured provider (spending whatever those
providers bill), read every cached asset, and cancel any video job. There is
no notion of "your" files versus someone else's — asset ids are unguessable
128-bit random values, and that is the only thing separating one caller's
generations from another's.

What the bridge does enforce:

- Every route requires the bearer token, compared in constant time. The
  interactive docs are off by default (`BRIDGE_ENABLE_DOCS`) because FastAPI
  mounts them outside that check, and the process refuses to start on an
  empty, trivial, or placeholder key.
- Request bodies are capped (`BRIDGE_MAX_REQUEST_MB`) before being buffered,
  and downloaded assets are capped (`max_asset_mb`) as they stream.
- Stored assets are served `nosniff` + `Content-Disposition: attachment`, with
  media types narrowed to a known-good set.
- Provider secrets live in environment variables referenced *by name* from
  `config.toml`, are read from `os.environ` at the point of use, and are never
  copied into the pydantic config objects or written back to the config file.
  They are not hidden from the process: each backend adapter holds its resolved
  token in memory for the process lifetime, in the HTTP client's default
  headers, as any long-lived client must. Asset downloads go out on a separate
  unauthenticated client, so an upstream token can never be attached to a CDN
  request or survive a redirect.
- An upstream's `401`/`403` response body is never echoed to clients — some
  providers quote the offending token back.

What it does **not** do, by design or by omission:

- **No rate limiting and no lockout** on failed authentication. A strong key
  plus constant-time comparison is what stands between the bridge and a brute
  force; there is nothing to slow an attacker who can reach the port.
- **No audit log.** Failed auth attempts are not recorded, so probing is
  invisible.
- **No TLS.** Terminate it at a reverse proxy in front of the bridge.
- **Prompts are persisted in the clear.** `video_jobs.prompt` is kept
  indefinitely (see [Storage & retention](#storage--retention)) and
  `generated_files.prompt_excerpt` holds the first 500 characters of every
  image prompt until that file is evicted. The SQLite database is not
  encrypted; treat it as sensitive and back it up accordingly.

Deployment guidance: bind to loopback and front the bridge with a reverse
proxy (`BRIDGE_HOST=127.0.0.1`) unless it genuinely needs to answer on other
interfaces. The default is `0.0.0.0` because the Docker image needs it to be
reachable outside the container; on a bare-metal or systemd install that
default puts the bridge on every interface the host has.

### Upgrading from 0.5.x

One of the changes above can stop an existing deployment from starting.
`BRIDGE_API_KEY` is now validated at boot: a key under 16 characters, or the
`.env.example` placeholder, aborts the process instead of serving. There is no
override — the empty-key case it closes made `Authorization: Bearer ` a valid
credential, and a warn-and-continue mode would leave that open. **Mint a new
key (`openssl rand -hex 24`) and roll it out to clients before upgrading**,
since every client authenticating against the old key has to change with it.

The other new defaults are reversible without re-keying. Compose users: these
are env vars, and `docker-compose.yml`'s `environment:` block is an allowlist —
setting one in `.env` alone will not reach the container unless the variable is
also listed there (the shipped file lists both).

- `/docs`, `/redoc` and `/openapi.json` now 404 — set `BRIDGE_ENABLE_DOCS=true`
  to restore them.
- Request bodies over 100MB are refused with `413` — set
  `BRIDGE_MAX_REQUEST_MB` higher, or `0` to disable the check.
- Assets downloaded from fal and ImageRouter are capped at 512MB, where they
  were previously unbounded. A larger generation now fails *after* the provider
  has billed for it, so raise `max_asset_mb` in the provider's `[[providers]]`
  block if you generate long video, or set `0` to restore the old behaviour.
- `GET /v1/videos/{id}/content` now sends `Content-Disposition: attachment`
  (`/v1/files/{id}/content` always did). `<video>` and `<img>` embedding is
  unaffected; only opening the URL directly in a browser changes, from playing
  inline to downloading.

## API surface

| Method | Path | Notes |
|---|---|---|
| GET    | `/v1/models`                       | Aggregate listing across all providers, best-effort per provider — one that fails or exceeds `MODELS_TIMEOUT_SECONDS` is omitted from that response rather than failing it (see [Model catalogue caching](#model-catalogue-caching)) |
| POST   | `/v1/chat/completions`             | Sync + SSE streaming passthrough to openai-compat upstreams |
| POST   | `/v1/embeddings`                   | Sync passthrough to openai-compat upstreams |
| POST   | `/v1/images/generations`           | Sync; JSON body |
| POST   | `/v1/images/edits`                 | Sync; multipart (`image` + `prompt` + `model`). Send multiple reference images by repeating `image` or using `image[]` (up to 16); backends that can't use all of them error rather than silently drop |
| POST   | `/v1/videos`                       | Async; multipart; returns `{id, status: "queued"}` |
| GET    | `/v1/videos/{id}`                  | Poll job status |
| GET    | `/v1/videos/{id}/content`          | Stream final mp4 once `status: "completed"`; supports conditional GET (see [Conditional requests](#conditional-requests)) |
| DELETE | `/v1/videos/{id}`                  | Cancel a queued/in-progress job; releases the runner slot |
| GET    | `/v1/files/{id}/content`           | Bridge-internal asset URLs returned in responses; supports conditional GET (see [Conditional requests](#conditional-requests)) |

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
| `kind` | `"chat"` \| `"image"` \| `"video"` \| `"embedding"`, omitted when unknown | ComfyUI: workflow output type; Venice/ImageRouter: inherent; OpenRouter: the model's `output_modalities`; fal: the per-model `kind` from config. OpenAI passthrough sets nothing — generic upstreams don't report modality reliably |
| `supports_tools` | `true` / `false`, omitted when unknown | OpenRouter only today, read from each model's advertised capabilities. Clients should treat *omitted* as "configure it yourself" |
| `context_window` | max context size in tokens, omitted when unknown | OpenAI-passthrough upstreams that expose it: llama.cpp's `meta.n_ctx` (a loaded model) or router `--ctx-size`, vLLM's `max_model_len`. The bridge strips the `meta`/`status` blocks otherwise, so this is the only way the size survives the proxy. Clients use it for a "N / max tokens" budget |
| `capabilities` | list of `{input}-to-{output}` operations, omitted when unknown | What a model actually accepts — `["text-to-image"]` vs `["text-to-image", "image-to-image"]`, or `["text-to-text", "image-to-text"]` for a vision chat model. A frontend uses it to enable or grey out image attachment per model. Read from each upstream's own metadata: fal's catalogue categories, ImageRouter's `inputs` map, OpenRouter's `architecture.input_modalities`, ComfyUI's `image_inputs` meta declaration, Venice's `type=image` / `type=inpaint` listings. **Always treat it as optional**: OpenAI-passthrough never has it, and every other backend omits it per-model whenever that model's upstream metadata is silent — an ImageRouter entry with no `inputs` map, say. Omission means "the upstream didn't say", never "accepts nothing" |
| `prompt_style` / `prompt_hint` | preferred prompt format + a freeform per-model nudge, omitted when unset | ComfyUI: the workflow's `meta.json` (see the meta schema below). Image and video models — a gateway-aware client uses them to rewrite a prompt into the model's preferred format before generation |

Standard OpenAI clients ignore the extra fields; nothing nonstandard is
*required* to use the bridge.

`capabilities` has a fuller, client-facing spec in
[docs/model-capabilities.md](docs/model-capabilities.md) — the value grammar,
ordering guarantees, and how a frontend should turn it into image-attachment UI.

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
| `BRIDGE_API_KEY` | — (required) | Bearer token clients must send. Must be at least 16 characters and not the `.env.example` placeholder — the bridge refuses to start otherwise, because an empty value makes `Authorization: Bearer ` a valid credential for every caller that can reach the port. Generate one with `openssl rand -hex 24` |
| `BRIDGE_HOST` / `BRIDGE_PORT` | `0.0.0.0` / `8080` | Bind address |
| `BRIDGE_ENABLE_DOCS` | `false` | Serve the interactive API docs (`/docs`, `/redoc`, `/openapi.json`). FastAPI mounts these outside the bearer-token dependency, so they answer *unauthenticated* — off by default; turn on only on a trusted network |
| `BRIDGE_MAX_REQUEST_MB` | `100` | Ceiling on an inbound request body; over it the bridge answers `413` before buffering. The bridge reads bodies whole on a single worker, so one oversized request is felt by every other client. `0` disables the check |
| `BRIDGE_PUBLIC_BASE_URL` | empty | Public origin used to build asset URLs in responses (e.g. `https://bridge.example.com`). Empty → relative URLs, fine when clients reach the bridge at the same host they requested from |
| `BRIDGE_CONFIG_PATH` | `/etc/openai-api-bridge/config.toml` | Provider config location |
| `FILES_DIR` | `/var/lib/openai-api-bridge/files` | Generated-asset cache directory |
| `SQLITE_PATH` | `/var/lib/openai-api-bridge/state.db` | Job + file state database |
| `RETENTION_DAYS` | `30` | TTL for cached generated files (see [Storage & retention](#storage--retention)) |
| `MAX_CACHE_GB` | `50` | Size cap for the file cache (LRU past this) |
| `EVICTION_INTERVAL_SECONDS` | `600` | How often the eviction loop runs |
| `MAX_CONCURRENT_VIDEO_JOBS` | `2` | Parallel video generations (see [Video jobs](#video-jobs)) |
| `MODELS_TIMEOUT_SECONDS` | `5` | Per-provider budget for `GET /v1/models` (see [Model catalogue caching](#model-catalogue-caching)); `0` disables |
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

**Nothing is exempt.** The schema carries a `pinned` flag and both sweeps skip
rows that set it, but nothing in the bridge sets it today — so a completed
video's file is subject to LRU like any other asset, and a client that polls
`GET /v1/videos/{id}` slowly enough can find its finished render already gone
(the content endpoint answers `404` with "evicted from cache" rather than
anything ambiguous). In practice this needs the cache to be at `MAX_CACHE_GB`
already; it is a narrow window, not a routine one.

Clients that want to keep generated media should download and persist it on
their side — [GlyphStream][glyphstream] does exactly this, pulling each asset
into its own media store on first generation.

### Conditional requests

A generated asset is addressed by a random id and its bytes never change, so
both content endpoints (`/v1/files/{id}/content` and
`/v1/videos/{id}/content`) serve it as an immutable resource:

- `ETag` — the asset's file id. Derived from the id rather than the file's
  mtime and size, so re-copying the bytes (restoring a backup, recreating a
  volume) doesn't invalidate a client's cached copy.
- `Cache-Control: private, max-age=31536000, immutable` — `private` because
  these endpoints sit behind the bridge's bearer token and a shared cache must
  not hand the response to a caller that never presented one. `Vary:
  Authorization` is sent for the same reason.
- `If-None-Match` → `304 Not Modified`, so a client that already holds the
  asset skips the transfer entirely.
- `X-Content-Type-Options: nosniff` and `Content-Disposition: attachment` on
  both endpoints. The served media type is whatever the upstream's
  `content-type` header claimed when the bytes were fetched, so these stop a
  browser rendering an unexpected payload at the bridge's own origin. Neither
  affects `<img>` or `<video>` embedding, which ignores both. Stored types are
  additionally narrowed to a known-good set — anything outside it is served as
  `application/octet-stream`, bytes kept, type not vouched for. Narrowing
  happens both when an asset is stored and when it is served, so assets already
  in the cache from an earlier version are covered too; for those, only the
  `Content-Type` header changes, never the bytes or the URL.

The cache window deliberately outlives the eviction window above: a client
that kept the bytes is still holding a correct copy after the bridge has
retired its own. Range requests (video seeking) are unaffected.

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
| `prompt_style` | — | Image or video models. The prompt FORMAT this model prefers, surfaced in `/v1/models` for a gateway-aware frontend's prompt-enhancement pass — image e.g. `"natural-language"`, `"booru-tags"`, `"keyword-soup"`, `"hybrid"`; video e.g. `"cinematic-prose"`, `"structured-cinematic"`. Passed through verbatim (not validated here). Omitted from the model row when unset |
| `prompt_hint` | — | Image or video models. A freeform per-model nudge surfaced alongside `prompt_style` (e.g. a quality-tag prefix, a length cap, an audio-cue reminder). Omitted when unset |

## Tests

```bash
uv run pytest                # full suite
uv run pytest -m live        # opt-in live tests against real backends (not yet wired)
uv run ruff check .          # lint
```
