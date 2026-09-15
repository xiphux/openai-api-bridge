#!/usr/bin/env bash
# Boot a built runtime image and prove it works, the way CI's "Docker image
# smoke" job does. Usable locally too:
#
#   docker build -t openai-api-bridge:smoke .
#   tests/image-smoke/run.sh openai-api-bridge:smoke
#
# Checks, in order:
#   1. the server boots on a fresh data dir and answers /v1/models
#   2. an unauthenticated request is refused
#   3. smoke.py, run with the image's own interpreter and venv, imports every
#      bridge module, does real work with each native dependency, and drives
#      the OpenAI passthrough against stub_upstream.py over a real network:
#      a JSON completion, a stream forwarded as it arrives, and an unreachable
#      upstream answered with a 502 envelope
#   4. the server logged no traceback while doing all of the above
#
# With --imports-only, only step 3's import and native-extension checks run,
# in a throwaway container with no server. CI uses that for the arm64 image,
# which it runs under QEMU emulation.
set -euo pipefail

imports_only=no
if [ "${1:-}" = "--imports-only" ]; then
	imports_only=yes
	shift
fi
image="${1:?usage: run.sh [--imports-only] <image>}"
platform="${2:-}"
here="$(cd "$(dirname "$0")" && pwd)"
name="bridge-smoke-$$"
upstream="bridge-smoke-upstream-$$"
network="bridge-smoke-net-$$"
port="${SMOKE_PORT:-18080}"

if [ "$imports_only" = yes ]; then
	# Every check that needs no server. python -c keeps smoke.py's main() —
	# and its server check — out of it. main.py builds the app at import time,
	# which reads settings, so a key has to exist even with no server.
	exec docker run --rm ${platform:+--platform "$platform"} \
		-e BRIDGE_API_KEY=imports-only-not-a-real-key \
		-v "$here:/smoke:ro" -w /smoke --entrypoint python "$image" -c '
import asyncio, smoke
smoke.check_every_module_imports()
smoke.check_dev_dependencies_absent()
smoke.check_native_extensions()
asyncio.run(smoke.check_aiosqlite())'
fi

work="$(mktemp -d)"
cleanup() {
	echo "--- container log ---"
	docker logs "$name" 2>&1 || true
	echo "--- stub upstream log ---"
	docker logs "$upstream" 2>&1 || true
	docker rm -f "$name" "$upstream" >/dev/null 2>&1 || true
	docker network rm "$network" >/dev/null 2>&1 || true
	rm -rf "$work"
}
trap cleanup EXIT

# Three providers:
#   comfyui  catalogue is a local workflow pair, so it lists without an upstream
#            (its URL points nowhere)
#   stub     OpenAI passthrough to stub_upstream.py, across the Docker network
#   down     OpenAI passthrough to a port nothing listens on, for the 502 path
mkdir -p "$work/workflows"
cat >"$work/config.toml" <<EOF
[[providers]]
id = "comfyui"
backend = "comfyui"
url = "http://127.0.0.1:9"
workflows_dir = "/smoke-config/workflows"

[[providers]]
id = "stub"
backend = "openai"
base_url = "http://$upstream:9000"

[[providers]]
id = "down"
backend = "openai"
base_url = "http://$upstream:9001"
EOF
echo '{"1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}}}' >"$work/workflows/smoke.json"
echo '{"positive_prompt_node": "1"}' >"$work/workflows/smoke.meta.json"
# mktemp -d is 0700, and the image runs as uid 10001.
chmod -R a+rX "$work"

key="smoke-$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')"

docker network create "$network" >/dev/null

# The stub runs from the image under test too: it only needs a Python.
docker run -d --name "$upstream" --network "$network" \
	-v "$here:/smoke:ro" --entrypoint python "$image" /smoke/stub_upstream.py 9000 >/dev/null

# Before the bridge starts: its catalogue cache remembers a failed listing for a
# while, so a first /v1/models that raced the stub would hide its model.
echo "--- waiting for the stub upstream"
for _ in $(seq 1 30); do
	if docker exec "$upstream" python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:9000/v1/models', timeout=2)" 2>/dev/null; then
		break
	fi
	sleep 1
done
docker exec "$upstream" python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:9000/v1/models', timeout=2)"

docker run -d --name "$name" --network "$network" \
	-p "127.0.0.1:$port:8080" \
	-e BRIDGE_API_KEY="$key" \
	-e BRIDGE_CONFIG_PATH=/smoke-config/config.toml \
	-v "$work:/smoke-config:ro" \
	-v "$here:/smoke:ro" \
	"$image" >/dev/null

# Every curl is bounded: a server that accepts the connection but never
# answers would otherwise hang one call forever, and the 60-try budget below
# would never get to count.
curl_bounded() { curl --max-time 5 "$@"; }

echo "--- waiting for the server"
for _ in $(seq 1 60); do
	if curl_bounded -fsS -o /dev/null -H "Authorization: Bearer $key" "http://127.0.0.1:$port/v1/models" 2>/dev/null; then
		break
	fi
	if [ "$(docker inspect -f '{{.State.Running}}' "$name")" != true ]; then
		echo "::error::container exited before the server answered"
		exit 1
	fi
	sleep 1
done
curl_bounded -fsS -o /dev/null -H "Authorization: Bearer $key" "http://127.0.0.1:$port/v1/models"

echo "--- unauthenticated request is refused"
status=$(curl_bounded -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/v1/models")
if [ "$status" != 401 ]; then
	echo "::error::expected 401 without a key, got $status"
	exit 1
fi

echo "--- smoke.py inside the container"
docker exec -w /smoke "$name" python smoke.py

echo "--- server log"
if docker logs "$name" 2>&1 | grep -q 'Traceback'; then
	echo "::error::the server logged a traceback"
	exit 1
fi
echo "image smoke passed"
