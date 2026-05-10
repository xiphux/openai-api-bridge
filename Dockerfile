# syntax=docker/dockerfile:1.7
#
# Multi-stage build using uv. Builder installs deps + the package into a
# venv; runtime copies the venv into a fresh image. Both stages run on
# python:3.12-alpine for a ~100MB savings over python:3.12-slim — every
# native dep in this project (pydantic_core, uvloop, httptools,
# watchfiles) ships a musllinux wheel, so `uv sync` pulls pre-built
# wheels and never has to compile from source.

FROM python:3.12-alpine AS builder

# uv ships as a statically-linked musl binary in the scratch image, so
# the same artifact runs on alpine just as it would on glibc — no need
# to switch to an alpine-tagged uv image.
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
# LICENSE must be in the build context for `uv sync` to honor the
# `license-files = ["LICENSE"]` declaration in pyproject.toml.
COPY src /app/src
COPY pyproject.toml uv.lock README.md LICENSE /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-alpine AS runtime

# sqlite CLI for ad-hoc inspection of the state.db (file metadata, job
# tracking, eviction queue). ~1.5 MB and doesn't run unless invoked —
# avoids needing to docker cp the file out to look at it.
RUN apk add --no-cache sqlite

# Run as a non-root user. Image-time UID/GID; the actual data volume
# should be owned by this UID on the host, OR overridden via --user.
# Alpine's BusyBox provides addgroup/adduser (not the util-linux
# groupadd/useradd we'd use on debian); flag names differ.
RUN addgroup -S -g 10001 bridge \
 && adduser  -S -u 10001 -G bridge -H -D bridge

COPY --from=builder --chown=bridge:bridge /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    BRIDGE_HOST=0.0.0.0 \
    BRIDGE_PORT=8080 \
    BRIDGE_CONFIG_PATH=/etc/openai-api-bridge/config.toml \
    FILES_DIR=/var/lib/openai-api-bridge/files \
    SQLITE_PATH=/var/lib/openai-api-bridge/state.db

# Pre-create runtime data dir so a tmpfs-only deployment still has a
# writable location. Real deployments should mount a persistent volume.
RUN mkdir -p /var/lib/openai-api-bridge \
 && chown -R bridge:bridge /var/lib/openai-api-bridge

USER bridge
WORKDIR /app
EXPOSE 8080

CMD ["openai-api-bridge"]
