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
ImageRouter, OpenRouter and Venice cache their listing for
`catalog_ttl_seconds` (default 300; `0` disables). It's a TTL rather than a
permanent cache so a model added upstream appears without restarting the
bridge. (ComfyUI is separate — it scans workflow files from disk, governed by
`cache_workflows`.)

A fetch runs under a lock, so a burst of concurrent requests waits on the one
already in flight rather than each starting its own — one round trip on
ImageRouter and OpenRouter, and one pair of them on Venice, whose catalogue is
split across two listings.

That lock is also why a **failure** is remembered for `catalog_retry_seconds`
(default 30; `0` retries immediately) and re-raised to callers arriving inside
the window. Without it, a burst during an upstream hang would queue up, each
waiter starting a fresh fetch after the previous one timed out — so the Nth
caller would wait N × the timeout. With it, the first caller pays the timeout
and the rest fail fast. The failure is remembered, not latched: once the window
closes the provider is retried and returns on its own.

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

### Model discovery on fal.ai

`/v1/models` is populated from fal's model API, filtered to the categories this
backend serves — **`text-to-image`, `image-to-image`, `text-to-video` and
`image-to-video`**, ~886 models. The listing is fetched once and cached, and
excludes deprecated models. Each model's `kind` comes from its catalogue
category, so a frontend can tell image models from video ones.

fal's audio and 3D categories are deliberately **not** listed — there's no code
path for them, so listing them would advertise models every request would fail
on. Nothing stops a client naming such an endpoint by hand, though; fal will
run the job and return a non-image envelope, which the bridge reports as
`unsupported_operation` (HTTP 400) rather than a retryable upstream error.

**Paired variants are collapsed.** fal publishes a model's text-driven and
reference-image halves as separate endpoints — `fal-ai/nano-banana-2` and
`.../edit`, `fal-ai/veo3.1` and `.../image-to-video` — which is easy to pick
wrong. Where the bridge can pair them *confidently* (both ids present in the
catalogue, in the expected categories) it lists only the text-driven id and
routes requests carrying a reference image to the sibling, so `POST
/v1/images/edits` against `fal-ai/nano-banana-2` just works. This matters most
for video, where `/v1/videos` is a **single endpoint** for both text-to-video
and image-to-video — the caller has no way to express which half it wants.

Against the live catalogue that pairs **81 of 194** text-to-image models and
**74 of 125** text-to-video ones — Nano Banana 2/Pro, both Seedream
generations, GPT Image 2, FLUX 2 Flex, Veo, Kling, Seedance. Nothing is
inferred from a name alone: a same-shaped id in the wrong category doesn't
pair. Models without a partner, and every reference-only endpoint (inpainting,
upscalers, background removal), are listed untouched. Set
`collapse_variants = false` to list both halves separately.

Because collapsing removes the `/edit` suffix that used to signal modality,
each entry carries a **`capabilities`** list saying what it actually accepts —
`["text-to-image"]`, `["text-to-image", "image-to-image"]`,
`["image-to-video"]`. Without it a merged id would be indistinguishable from a
text-only one, and a client would discover the difference from a failed
request; with it, a frontend can enable or grey out image attachment per
model.

When a request is routed to a collapsed sibling, a `[[providers.models]]` block
for the **sibling** takes precedence — it's the endpoint actually running — and
**layers over** the base model's block rather than replacing it. Only fields
the sibling actually sets win; anything it omits keeps the base's value, and
`params` are merged key-by-key. So a sibling block written just to set a
`display_name` won't quietly revert an explicit `disable_safety = false` on the
base, or drop its pinned `params`.

`[[providers.models]]` entries are per-model **overrides**, not a whitelist —
they set `disable_safety`, `params`, `display_name`, or prompt metadata on a
discovered model without restricting the rest. A configured model the catalogue
doesn't return (a deprecated one, or an id outside the filtered categories) is
still listed, since generation works for it either way. To serve a fixed set
instead, set `discover_models = false` and list them; a model outside that list
is then a 404.

If the catalogue can't be fetched, the provider degrades to whatever is
explicitly configured (rather than serving nothing) and retries after
`introspect_retry_seconds`.

**API key failures.** A *missing* key fails fast: `api_token_env` is required,
and the token is resolved while the dispatcher is built, so the bridge refuses
to start (`ConfigError: Provider 'fal': env var 'FAL_API_KEY' is not set`). A
*wrong* key can't be caught that way without putting a network call in the
startup path — which would let a fal outage block the bridge from booting — so
it's handled on first use — on discovery, introspection, or generation,
whichever touches fal first: fal's `401`/`403` is reported **once at ERROR**,
naming the env var to fix, and is never retried, since a token read from the
environment at startup cannot start working again. Discovery and introspection
switch off for the process, and generation fails with
`code: "upstream_auth_error"` (HTTP 502) rather than degrading quietly.

### Upstream data retention on fal.ai

By the time a request completes the bridge has already copied the asset into
its own FileStore, so fal's copies are redundant for a self-hosted setup. Two
opt-in knobs hand that back:

| Setting | Effect |
|---|---|
| `store_payloads = false` | Sends `X-Fal-Store-IO: 0` — fal never stores the request payload. Otherwise prompts (and inline input images) are kept for **30 days** |
| `output_expiration_seconds = N` | Sends `X-Fal-Object-Lifecycle-Preference` — generated media expires from fal's CDN after `N` seconds instead of persisting |

These are request headers rather than after-the-fact deletion, deliberately.
Deletion would need a request id, and **the synchronous image path doesn't have
one** — only queued video does — so headers are the only mechanism that covers
both uniformly, with no second round trip and no window where a failed delete
leaves data behind. Because reference images are sent as inline data URIs
rather than uploaded to fal's CDN, they ride inside the payload that
`store_payloads = false` suppresses.

**Mind the retrieval budget** when setting an expiry: the clock starts when fal
creates the object, and the bridge still has to notice it, fetch it, and
possibly retry — an asset fetch alone spans three attempts with backoff, and
video detects completion by polling before that. Values below 60s are rejected
at config load; a few hundred seconds is comfortable. If a fetch fails while an
expiry is configured, the error names the setting so the cause is obvious.

### Content moderation on fal.ai

The reason to reach for the `fal` backend over ImageRouter/OpenRouter is
control over the tier-1 models' safety guardrails. fal surfaces each model's
own moderation parameter, and the bridge sets it to the loosest value whenever
a model is configured with `disable_safety = true` (the default).

Rather than mapping model name → knob (which would need a code change for every
new model version), the bridge **reads the model's own OpenAPI schema** from
fal's model API and derives the setting. A sweep of fal's image catalogue found
the knobs collapse to two field names, both stable across versions (and they
apply to video models too):

| Input field | Models | What the bridge sets |
|---|---|---|
| `enable_safety_checker` (bool) | ~132 | `false` |
| `safety_tolerance` (enum) | ~23 | the **highest value in that model's own enum** |
| *(no recognized field)* | — | nothing injected — e.g. fal's **GPT Image** wrapper has no moderation field at all |

Reading the enum matters: most models accept `"1"`–`"6"`, but `fal-ai/flux-2-flex`
tops out at `"5"`, so a hardcoded `"6"` would be rejected. Defaults vary
independently of the ceiling (`"4"` for Gemini, `"2"` for FLUX), so the default
tells you nothing about the maximum.

Two look-alike fields are deliberately ignored: `safety_checker_version`
(`v1`/`v2`) selects *which* checker runs rather than how strict it is, and
`has_nsfw_concepts` is an *output* field, not an input knob.

Schemas are fetched lazily **per model** on first use and cached for the
process, so you only pay for models you actually generate with — discovery
surfaces hundreds. Each model has its own lock: a burst of concurrent
generations for the *same* model collapses into a single lookup that all of
them wait for and benefit from, while *different* models resolve concurrently
rather than queueing behind one another.

If fal's model API is unreachable the bridge logs a warning and falls back to a
small built-in map, so generation never fails over an introspection blip. The
failure is retried after `introspect_retry_seconds` (default 300), so an outage
degrades temporarily instead of until the next restart. Set
`introspect_safety = false` to skip the lookup entirely.

Two caveats worth knowing:

- **Nano Banana can't be fully unlocked by anyone.** Google runs a second,
  non-configurable `IMAGE_SAFETY` layer under the `safety_tolerance` knob —
  no broker (or Google's own API) can disable it. `safety_tolerance = "6"`
  softens Layer 1 only.
- **Per-model overrides** live in a `[providers.models.params]` table (merged
  last, so it beats the derived setting) plus a `disable_safety = false` toggle
  — use these to pin `aspect_ratio`/`resolution`/`quality`, or to set a knob the
  bridge doesn't recognize. `size` (`WxH`) maps to
  `image_size` for most families but is dropped for Nano Banana / Gemini, which
  take `aspect_ratio` + `resolution` instead.

### Video on fal.ai

Video uses fal's **queue** endpoint (`queue.fal.run`) rather than the
synchronous one images use: a clip runs for minutes, past what a held
connection tolerates. The bridge submits, records fal's `request_id` on the job
row, polls to completion, then fetches the asset — so clients see the standard
async `/v1/videos` lifecycle with no fal-specific handling.

`seconds` is mapped onto whatever the model actually accepts, read from its
schema, because the spellings don't agree: `fal-ai/veo3` takes `"4s"`/`"6s"`/`"8s"`,
Kling takes `"5"`/`"10"`, Hailuo `"6"`/`"10"`, and `wan` has no duration field
at all (it counts frames, so nothing is sent). The closest accepted value wins;
ties go to the longer clip, since silently returning less than asked for loses
content. A reference image for image-to-video is forwarded as `image_url`.

`size` is **not** forwarded for video — these models take `aspect_ratio` and
`resolution` enums rather than pixel dimensions, which don't follow from a
`WxH` string. Pin them per model via `params`. Polling is governed by
`video_poll_interval_seconds` (default 3) and `video_poll_timeout_seconds`
(default 1800).

Because a long job issues hundreds of status polls, no single one is fatal: a
bounded run of consecutive transient failures is treated as "not ready yet"
(as the ComfyUI poller does), with the deadline bounding the loop. The result
fetch — which happens *after* fal has rendered and billed for the clip — is
retried with backoff, so one hiccup can't discard a finished video. A rejected
key stays fatal throughout, and is reported wherever in the sequence it
surfaces.

Moderation works exactly as it does for images — the same schema introspection
finds `safety_tolerance` on `fal-ai/veo3` and `enable_safety_checker` on `wan`.

See the `fal` block in `config.toml.example` for a full example.

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
| `kind` | `"chat"` \| `"image"` \| `"video"` \| `"embedding"`, omitted when unknown | ComfyUI: workflow output type; Venice/ImageRouter: inherent; OpenRouter: the model's `output_modalities`; fal: the per-model `kind` from config. OpenAI passthrough sets nothing — generic upstreams don't report modality reliably |
| `supports_tools` | `true` / `false`, omitted when unknown | OpenRouter only today, read from each model's advertised capabilities. Clients should treat *omitted* as "configure it yourself" |
| `context_window` | max context size in tokens, omitted when unknown | OpenAI-passthrough upstreams that expose it: llama.cpp's `meta.n_ctx` (a loaded model) or router `--ctx-size`, vLLM's `max_model_len`. The bridge strips the `meta`/`status` blocks otherwise, so this is the only way the size survives the proxy. Clients use it for a "N / max tokens" budget |
| `capabilities` | list of `{input}-to-{output}` operations, omitted when unknown | What a model actually accepts — `["text-to-image"]` vs `["text-to-image", "image-to-image"]`, or `["text-to-text", "image-to-text"]` for a vision chat model. A frontend uses it to enable or grey out image attachment per model. Read from each upstream's own metadata: fal's catalogue categories, ImageRouter's `inputs` map, OpenRouter's `architecture.input_modalities`, ComfyUI's `image_inputs` meta declaration, Venice's `type=image` / `type=inpaint` listings. **Always treat it as optional**: OpenAI-passthrough never has it, and every other backend omits it per-model whenever that model's upstream metadata is silent — an ImageRouter entry with no `inputs` map, say. Omission means "the upstream didn't say", never "accepts nothing" |
| `prompt_style` / `prompt_hint` | preferred prompt format + a freeform per-model nudge, omitted when unset | ComfyUI: the workflow's `meta.json` (see the meta schema below). Image and video models — a gateway-aware client uses them to rewrite a prompt into the model's preferred format before generation |

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
| `prompt_style` | — | Image or video models. The prompt FORMAT this model prefers, surfaced in `/v1/models` for a gateway-aware frontend's prompt-enhancement pass — image e.g. `"natural-language"`, `"booru-tags"`, `"keyword-soup"`, `"hybrid"`; video e.g. `"cinematic-prose"`, `"structured-cinematic"`. Passed through verbatim (not validated here). Omitted from the model row when unset |
| `prompt_hint` | — | Image or video models. A freeform per-model nudge surfaced alongside `prompt_style` (e.g. a quality-tag prefix, a length cap, an audio-cue reminder). Omitted when unset |

## Tests

```bash
uv run pytest                # full suite
uv run pytest -m live        # opt-in live tests against real backends (not yet wired)
uv run ruff check .          # lint
```
