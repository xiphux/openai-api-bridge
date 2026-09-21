# Changelog

What changed for people running the bridge and for the clients calling it,
newest first.

Internal work — refactors, tests, dependency bumps, CI — is deliberately absent,
and so is any fix to a problem that never reached a release. See the
[README](README.md) for how a feature works; entries here only say that it
arrived.

## Unreleased

## v0.7.0

### Added

- `/v1/images/generations` and the video endpoints accept an `aspect_ratio`, and
  echo back the ratio a job was actually rendered at.
- ComfyUI workflows can be sized by aspect ratio rather than by explicit pixel
  dimensions. The selector is disabled for a workflow that declares no ratio
  field, and a renumbered ratio node is reported rather than silently ignored.

## v0.6.0

### Security

- The bridge refuses to start on an API key that would authenticate everyone,
  rather than coming up wide open.
- The interactive API docs are no longer served without a credential.
- Inbound request bodies are capped before anything buffers them.
- A stored asset can no longer choose how a browser treats it, and a caller can
  no longer pick the extension an uploaded file is saved under.
- Signed asset URLs have their signature stripped before appearing in an error.

### Changed

- The runtime moved from httpx to httpx2.

### Added

- Stored assets are cacheable and answer conditional requests.

### Fixed

- The fal model catalogue has a TTL and its own retry setting, and the last good
  catalogue is served while a refresh is in flight rather than failing.
- Database transactions no longer interleave on a shared connection, and roll
  back on cancellation as well as on error.
- A non-positive width or height is rejected instead of being passed upstream.
- ComfyUI reports a proxy auth rejection the same way on both of its calls.
- Generated-asset downloads stream and are abandoned at the size cap instead of
  being buffered whole.
- `n > 1` image generations run concurrently on Venice, ImageRouter and
  OpenRouter; `/v1/models` bounds its fan-out and caches the passthrough
  catalogue; ComfyUI ramps its completion poll instead of holding a flat one
  second. Eviction and LRU bookkeeping no longer sit on the download path.

## v0.5.1

### Fixed

- ComfyUI batches are budgeted as a whole and contain their own failures; a
  cancelled run now actually cancels its sibling collectors, and prompts
  abandoned mid-collect are recalled instead of being left queued upstream.
- An upstream 429 is reported as a rate limit rather than as a malformed
  request, and throttled asset fetches are retried.
- Upstream auth complaints are no longer echoed back to the client.
- A non-ASCII bearer token returns 401 instead of 500.
- Cancelling a video job is conditional on the job still being active.
- One provider failing no longer takes down `/v1/models`, which now fans out to
  providers concurrently.
- A stored file whose bytes have gone is treated as absent rather than served.
- The catalogue cache backs off on repeated auth failures instead of latching
  into a failed state forever.
- OpenRouter downloads through the shared asset fetcher, so it honours the same
  size cap as everything else.
- Generated assets are written off the event loop and retired in batches.

## v0.5.0

### Added

- **fal.ai backend.** Models are discovered from fal's own API, with per-model
  moderation settings derived from each model's OpenAPI schema, video generation
  through fal's queue lifecycle, upstream retention controls, and collapsed edit
  variants. A rejected API key is reported loudly instead of retried.
- Per-model capabilities are exposed across the other backends too, and Venice's
  `-edit` models are paired with the models they edit.
- Model catalogues are cached for Venice, ImageRouter and OpenRouter.

### Fixed

- Venice's listing degrades quietly when partially available, and a non-JSON 200
  from it no longer escapes as a success.

## v0.4.4

### Added

- `/v1/models` surfaces `context_window`, and `prompt_style` / `prompt_hint` for
  image models.

## v0.4.3

### Added

- Venice image-to-image, via `/api/v1/image/edit`.

### Fixed

- `/v1/images/edits` forwards every reference image rather than only the first.
- ComfyUI reports an error for surplus reference images instead of silently
  dropping them.

## v0.4.2

### Fixed

- ImageRouter image retrieval sends the right credential. (The v0.3.2 fix was
  incomplete.)

## v0.4.1

### Changed

- The container declares its own health check, so orchestrators get one without
  extra configuration.

## v0.4.0

### Added

- `/v1/models` reports `supports_tools` per model.

## v0.3.2

### Fixed

- ImageRouter image retrieval passes authentication.

## v0.3.1

### Changed

- The runtime image is based on Alpine rather than Debian slim, which makes it
  considerably smaller.

## v0.3.0

### Added

- OpenRouter as a backend, for chat and image generation.
- ImageRouter as a backend, for image and video generation.

## v0.2.0

### Added

- Passthrough and aggregation of an upstream OpenAI-compatible endpoint, so its
  models appear alongside the bridge's own.
- An endpoint for cancelling a running video job.

### Fixed

- A video run no longer leaks its semaphore slot on the cancellation path.
- The dropped-prompt detector requires several consecutive misses before it
  reports, rather than reacting to a single one.

## v0.1.0

First release: an OpenAI-compatible HTTP bridge in front of ComfyUI and Venice,
shipped as a container image.

### Added

- `/v1/models` carries two non-standard fields, `kind` and `display_name`, so a
  client can tell an image model from a chat model and show a readable name.
