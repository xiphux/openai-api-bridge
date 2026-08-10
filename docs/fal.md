# fal.ai backend internals

Reference for the `fal` backend's behaviour beyond what the
[README](../README.md#falai) covers. Read this when you're debugging fal
specifically; the README section and the `fal` block in `config.toml.example`
are enough to configure it.

## Model discovery

`/v1/models` is populated from fal's model API, filtered to the categories this
backend serves — `text-to-image`, `image-to-image`, `text-to-video` and
`image-to-video`, ~886 models. The listing is cached for `catalog_ttl_seconds`
(default 300), and excludes deprecated models. Each model's `kind` comes from
its catalogue category.

fal's audio and 3D categories are deliberately not listed — there's no code path
for them, so listing them would advertise models every request would fail on.
Nothing stops a client naming such an endpoint by hand, though; fal will run the
job and return a non-image envelope, which the bridge reports as
`unsupported_operation` (HTTP 400) rather than a retryable upstream error.

`[[providers.models]]` entries are per-model **overrides**, not a whitelist —
they set `disable_safety`, `params`, `display_name`, or prompt metadata on a
discovered model without restricting the rest. A configured model the catalogue
doesn't return (a deprecated one, or an id outside the filtered categories) is
still listed, since generation works for it either way. To serve a fixed set
instead, set `discover_models = false` and list them; a model outside that list
is then a 404.

The refresh is **stale-while-revalidate**: once the TTL lapses, the next
caller gets the previous listing immediately and the refetch runs in the
background. This matters more here than on the other backends because a fal
listing is 10–13 paginated round trips — past the `MODELS_TIMEOUT_SECONDS`
budget `/v1/models` gives each provider, and unbounded on the image-edit path,
which consults the catalogue to resolve variant routing. Blocking on it meant
fal dropped out of one listing per TTL window, and could stall a generation.
Calls before the *first successful* fetch still pay for it inline — if fal is
unreachable at boot, one caller per retry window blocks until a fetch lands.

If the catalogue can't be fetched, the provider keeps serving the last good
listing if it has one, and otherwise degrades to whatever is explicitly
configured (rather than serving nothing); it retries after
`catalog_retry_seconds` (default 30). That's separate from
`introspect_retry_seconds`, which covers the per-model schema lookups below.
A refresh that returns no usable models is treated as a failure, and
specifically does not overwrite the variant-routing map — otherwise a
transient empty response would silently send edits to the text-only endpoint.

### Display names

fal titles every endpoint of a multi-endpoint family after the *family*, so
Florence-2 Large's eight task endpoints all arrive titled `Florence-2 Large`.
The ids stay distinct and each is separately callable, but a picker rendering
`display_name` would show eight identical rows — 232 of 731 catalogue entries
collide this way.

The bridge appends the part of the id that actually differs within the colliding
group, so `Wan` becomes `Wan (V2.2-A14B Text-to-Video Turbo)`. Where the
fragment would only restate the family (`Fooocus`, `Veo 3.1`) that endpoint is
the family's base and keeps its bare name. A `display_name` set in
`[[providers.models]]` is never rewritten.

### Paired variants

fal publishes a model's text-driven and reference-image halves as separate
endpoints — `fal-ai/nano-banana-2` and `.../edit`, `fal-ai/veo3.1` and
`.../image-to-video`. Where the bridge can pair them *confidently* (both ids
present in the catalogue, in the expected categories) it lists only the
text-driven id and routes requests carrying a reference image to the sibling.

Against the live catalogue that pairs **81 of 194** text-to-image models and
**74 of 125** text-to-video ones — Nano Banana 2/Pro, both Seedream generations,
GPT Image 2, FLUX 2 Flex, Veo, Kling, Seedance. Nothing is inferred from a name
alone: a same-shaped id in the wrong category doesn't pair. Models without a
partner, and every reference-only endpoint (inpainting, upscalers, background
removal), are listed untouched.

Because collapsing removes the `/edit` suffix that used to signal modality, each
entry carries a `capabilities` list saying what it actually accepts — see
[model-capabilities.md](model-capabilities.md).

When a request is routed to a collapsed sibling, a `[[providers.models]]` block
for the **sibling** takes precedence — it's the endpoint actually running — and
**layers over** the base model's block rather than replacing it. Only fields the
sibling actually sets win; anything it omits keeps the base's value, and `params`
are merged key-by-key. So a sibling block written just to set a `display_name`
won't quietly revert an explicit `disable_safety = false` on the base, or drop
its pinned `params`.

### API key failures

A *missing* key fails fast: `api_token_env` is required, and the token is
resolved while the dispatcher is built, so the bridge refuses to start
(`ConfigError: Provider 'fal': env var 'FAL_API_KEY' is not set`).

A *wrong* key can't be caught that way without putting a network call in the
startup path — which would let a fal outage block the bridge from booting — so
it's handled on first use, on discovery, introspection, or generation, whichever
touches fal first. fal's `401`/`403` is reported **once at ERROR**, naming the
env var to fix, and is never retried, since a token read from the environment at
startup cannot start working again. Discovery and introspection switch off for
the process, and generation fails with `code: "upstream_auth_error"` (HTTP 502)
rather than degrading quietly.

This takes precedence over the stale-while-revalidate behaviour above: the
latch is checked before the cached catalogue, so once a key is rejected the
listing falls back to the explicitly configured models even though a
previously-fetched catalogue is still held. Every *other* refresh failure
keeps serving that catalogue.

## Content moderation

Rather than mapping model name → knob (which would need a code change for every
new model version), the bridge reads the model's own OpenAPI schema from fal's
model API and derives the setting. A sweep of fal's image catalogue found the
knobs collapse to two field names, both stable across versions, and they apply to
video models too:

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

Schemas are fetched lazily **per model** on first use and cached for the process,
so you only pay for models you actually generate with — discovery surfaces
hundreds. Each model has its own lock: a burst of concurrent generations for the
*same* model collapses into a single lookup that all of them wait for and benefit
from, while *different* models resolve concurrently rather than queueing behind
one another.

If fal's model API is unreachable the bridge logs a warning and falls back to a
small built-in map, so generation never fails over an introspection blip. The
failure is retried after `introspect_retry_seconds` (default 300), so an outage
degrades temporarily instead of until the next restart. Set
`introspect_safety = false` to skip the lookup entirely.

**Nano Banana can't be fully unlocked by anyone.** Google runs a second,
non-configurable `IMAGE_SAFETY` layer under the `safety_tolerance` knob — no
broker, including Google's own API, can disable it. `safety_tolerance = "6"`
softens Layer 1 only.

Per-model overrides live in a `[providers.models.params]` table (merged last, so
it beats the derived setting) plus a `disable_safety = false` toggle — use these
to pin `aspect_ratio`/`resolution`/`quality`, or to set a knob the bridge doesn't
recognize. `size` (`WxH`) maps to `image_size` for most families but is dropped
for Nano Banana / Gemini, which take `aspect_ratio` + `resolution` instead.

## Upstream data retention

By the time a request completes the bridge has already copied the asset into its
own FileStore, so fal's copies are redundant for a self-hosted setup. Two opt-in
knobs hand that back:

| Setting | Effect |
|---|---|
| `store_payloads = false` | Sends `X-Fal-Store-IO: 0` — fal never stores the request payload. Otherwise prompts (and inline input images) are kept for **30 days** |
| `output_expiration_seconds = N` | Sends `X-Fal-Object-Lifecycle-Preference` — generated media expires from fal's CDN after `N` seconds instead of persisting |

These are request headers rather than after-the-fact deletion, deliberately.
Deletion would need a request id, and **the synchronous image path doesn't have
one** — only queued video does — so headers are the only mechanism that covers
both uniformly, with no second round trip and no window where a failed delete
leaves data behind. Because reference images are sent as inline data URIs rather
than uploaded to fal's CDN, they ride inside the payload that
`store_payloads = false` suppresses.

**Mind the retrieval budget** when setting an expiry: the clock starts when fal
creates the object, and the bridge still has to notice it, fetch it, and possibly
retry — an asset fetch alone spans three attempts with backoff, and video detects
completion by polling before that. Values below 60s are rejected at config load;
a few hundred seconds is comfortable. If a fetch fails while an expiry is
configured, the error names the setting so the cause is obvious.

## Video

Video uses fal's **queue** endpoint (`queue.fal.run`) rather than the synchronous
one images use: a clip runs for minutes, past what a held connection tolerates.
The bridge submits, records fal's `request_id` on the job row, polls to
completion, then fetches the asset — so clients see the standard async
`/v1/videos` lifecycle with no fal-specific handling.

`seconds` is mapped onto whatever the model actually accepts, read from its
schema, because the spellings don't agree: `fal-ai/veo3` takes
`"4s"`/`"6s"`/`"8s"`, Kling takes `"5"`/`"10"`, Hailuo `"6"`/`"10"`, and `wan`
has no duration field at all (it counts frames, so nothing is sent). The closest
accepted value wins; ties go to the longer clip, since silently returning less
than asked for loses content. A reference image for image-to-video is forwarded
as `image_url`.

`size` is **not** forwarded for video — these models take `aspect_ratio` and
`resolution` enums rather than pixel dimensions, which don't follow from a `WxH`
string. Pin them per model via `params`. Polling is governed by
`video_poll_interval_seconds` (default 3) and `video_poll_timeout_seconds`
(default 1800).

### Abandoned jobs

fal bills a render whether or not anyone collects it, so every exit from the poll
loop that isn't a finished clip — a client `DELETE /v1/videos/{id}`, the bridge's
own timeout, a poll that gave up — asks fal to stop via the queue's cancel
endpoint. It's best-effort and bounded (a job past the point of no return won't
stop, and the attempt can't stall shutdown), and a failed cancel is logged
without masking the error that caused the abandonment.

**Not covered:** a job in flight when the bridge restarts. Its row is failed by
`mark_stale_failed`, but the fal job keeps rendering and billing. The
`upstream_id` is persisted, so cancelling those is possible, just not wired up.

### Poll resilience

Because a long job issues hundreds of status polls, no single one is fatal: a
bounded run of consecutive transient failures is treated as "not ready yet" (as
the ComfyUI poller does), with the deadline bounding the loop. The result fetch —
which happens *after* fal has rendered and billed for the clip — is retried with
backoff, so one hiccup can't discard a finished video. A rejected key stays fatal
throughout, and is reported wherever in the sequence it surfaces.

Moderation works exactly as it does for images — the same schema introspection
finds `safety_tolerance` on `fal-ai/veo3` and `enable_safety_checker` on `wan`.
