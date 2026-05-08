# openai-api-bridge

Thin OpenAI-compatible HTTP front-end for **ComfyUI** and **Venice.ai**.
Any chat client that speaks the OpenAI image/video API works against this
bridge without per-host customization.

```
client (LibreChat/LobeChat/curl/...)
        │  OpenAI API: /v1/{models,images,videos,files}
        ▼
   openai-api-bridge   ←  config.toml: [[providers]] = comfyui | venice | ...
        │
        ├──► ComfyUI workflow (image or video)
        └──► Venice.ai native API (image)
```

## API surface

| Method | Path | Notes |
|---|---|---|
| GET    | `/v1/models`                       | Aggregate listing across all providers |
| POST   | `/v1/images/generations`           | Sync; JSON body |
| POST   | `/v1/images/edits`                 | Sync; multipart (`image` + `prompt` + `model`) |
| POST   | `/v1/videos`                       | Async; multipart; returns `{id, status: "queued"}` |
| GET    | `/v1/videos/{id}`                  | Poll job status |
| GET    | `/v1/videos/{id}/content`          | Stream final mp4 once `status: "completed"` |
| GET    | `/v1/files/{id}/content`           | Bridge-internal asset URLs returned in responses |

Auth: `Authorization: Bearer ${BRIDGE_API_KEY}`.

Model IDs follow `{provider_id}/{model_slug}` — so a request like
`{"model": "comfyui/ltxv-t2i", "prompt": "..."}` routes to the ComfyUI provider's
`ltxv-t2i` workflow.

## Configuration

Two layers, by concern:

* **Infrastructure & secrets** → environment variables (see `.env.example`)
* **Provider definitions** → `config.toml` (see `config.toml.example`)

Adding a second ComfyUI instance is a single new `[[providers]]` block — no
code changes needed.

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

## Workflow meta.json schema (ComfyUI)

Each workflow file `<name>.json` needs a companion `<name>.meta.json` declaring
where the bridge should inject things:

```json
{
  "positive_prompt_node": "10",
  "positive_prompt_field": "text",
  "display_name": "LTX-V T2V (Fast)",
  "image_inputs": [{"node": "7", "field": "image"}],
  "image_required": true,
  "dimensions_node": "5",
  "width_field": "width",
  "height_field": "height",
  "length_node": "8",
  "length_field": "value",
  "fps": 24,
  "seed_nodes": ["11", "12"],
  "output_type": "video"
}
```

Only `positive_prompt_node` is required. `output_type` auto-detects between
`image` and `video` based on the presence of `SaveVideo` / `VHS_VideoCombine`
nodes; set it explicitly to override. `fps` enables OpenAI's `seconds`
parameter to translate into a frame count for the workflow's `length_node`.

## Tests

```bash
uv run pytest                # full suite (~80 unit + integration tests)
uv run pytest -m live        # opt-in live tests against real backends (not yet wired)
uv run ruff check .          # lint
```
