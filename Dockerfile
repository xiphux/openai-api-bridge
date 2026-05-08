# Multi-stage build using uv. The first stage installs deps + the package
# into a venv; the second stage copies the venv into a slim runtime image.
# The cache mount keeps `uv sync` fast across builds without bloating layers.

FROM python:3.12-slim AS builder

# Pin uv from the official image. Renovate-friendly tag.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# 1) Install dependencies first (without the project) for better layer caching.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# 2) Copy the rest of the source and install the project itself.
COPY src /app/src
COPY pyproject.toml uv.lock README.md /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

# Run as a non-root user. Image-time UID/GID; the actual data volume should
# be owned by this UID on the host, OR the user can override via --user.
RUN groupadd --system --gid 10001 bridge \
 && useradd  --system --uid 10001 --gid 10001 --no-create-home bridge

COPY --from=builder --chown=bridge:bridge /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    BRIDGE_HOST=0.0.0.0 \
    BRIDGE_PORT=8080 \
    BRIDGE_CONFIG_PATH=/etc/openai-api-bridge/config.toml \
    FILES_DIR=/var/lib/openai-api-bridge/files \
    SQLITE_PATH=/var/lib/openai-api-bridge/state.db

# Pre-create runtime data dir so a tmpfs-only deployment still has a writable
# location. Real deployments should mount a persistent volume here.
RUN mkdir -p /var/lib/openai-api-bridge \
 && chown -R bridge:bridge /var/lib/openai-api-bridge

USER bridge
WORKDIR /app
EXPOSE 8080

CMD ["openai-api-bridge"]
