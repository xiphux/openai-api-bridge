"""Exercise the runtime image's production dependencies from inside it.

Run by run.sh through ``docker exec``, with the image's own interpreter and
its ``--no-dev`` venv — never the developer's. Not a pytest module (nothing
here is collected); every check raises on failure, so a clean exit is the pass.

What this exists to catch is what the test suite structurally cannot: tests
run on a full dev install on glibc with every upstream mocked below httpx2,
while the image installs ``--no-dev`` on Alpine/musl and talks to real sockets.
So:

* every module under ``openai_api_bridge`` must import, which fails if source
  imports something that is only a dev dependency (``httpx`` is one, on
  purpose — see pyproject.toml);
* every native extension must load and do real work, which fails if a release
  dropped its musllinux wheel or shipped a broken one;
* the running server must parse a real multipart body and answer through the
  bridge's own error envelope, not a 500;
* the OpenAI passthrough must reach run.sh's stub upstream over the network:
  httpx2's real transport, a stream uvicorn forwards as it arrives rather than
  buffering, and an unreachable upstream answered with a 502.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import pkgutil
import sys

BASE_URL = "http://127.0.0.1:8080"


def check_every_module_imports() -> None:
    import openai_api_bridge

    names = [
        m.name
        for m in pkgutil.walk_packages(openai_api_bridge.__path__, prefix="openai_api_bridge.")
    ]
    for name in names:
        importlib.import_module(name)
    print(f"imported {len(names)} openai_api_bridge modules")


def check_dev_dependencies_absent() -> None:
    # If these were importable, the check above would prove nothing about
    # dev-only imports: the venv would have to be a dev install.
    for name in ("pytest", "respx", "mypy"):
        try:
            importlib.import_module(name)
        except ImportError:
            continue
        raise AssertionError(f"dev dependency {name!r} is installed in the runtime image")
    print("no dev dependencies in the venv")


def check_native_extensions() -> None:
    import httptools
    import pydantic
    import uvloop

    loop = uvloop.new_event_loop()
    try:
        assert loop.run_until_complete(asyncio.sleep(0, result=42)) == 42
    finally:
        loop.close()

    seen: dict[str, bytes] = {}

    class Callbacks:
        def on_url(self, url: bytes) -> None:
            seen["url"] = url

    parser = httptools.HttpRequestParser(Callbacks())
    parser.feed_data(b"GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n")
    assert seen["url"] == b"/v1/models", seen

    # pydantic_core is the compiled half of pydantic.
    class Probe(pydantic.BaseModel):
        n: int

    assert Probe.model_validate_json('{"n": "7"}').n == 7
    print("uvloop, httptools and pydantic_core loaded and ran")


async def check_aiosqlite() -> None:
    import aiosqlite

    async with aiosqlite.connect(":memory:") as db:
        await db.execute("CREATE TABLE t (v TEXT)")
        await db.execute("INSERT INTO t VALUES ('ok')")
        async with db.execute("SELECT v FROM t") as cur:
            row = await cur.fetchone()
    assert row == ("ok",), row
    print("aiosqlite round-tripped a row")


async def check_server_over_httpx2() -> None:
    import httpx2

    key = os.environ["BRIDGE_API_KEY"]
    auth = {"Authorization": f"Bearer {key}"}
    async with httpx2.AsyncClient(base_url=BASE_URL, timeout=10) as client:
        r = await client.get("/v1/models", headers=auth)
        assert r.status_code == 200, (r.status_code, r.text)
        ids = [m["id"] for m in r.json()["data"]]
        assert "comfyui/smoke" in ids, ids

        # A real multipart body through python-multipart. The smoke workflow
        # takes no input image, so the bridge must refuse the edit — with its
        # own error envelope and a 4xx, which proves the body was parsed and
        # routed rather than falling over.
        r = await client.post(
            "/v1/images/edits",
            headers=auth,
            data={"model": "comfyui/smoke", "prompt": "smoke"},
            files={"image": ("in.png", b"\x89PNG\r\n\x1a\n", "image/png")},
        )
        assert 400 <= r.status_code < 500, (r.status_code, r.text)
        assert "error" in r.json(), r.text
    print(f"server answered /v1/models and a multipart edit ({r.status_code})")


async def check_passthrough_over_the_network() -> None:
    """The bridge's outbound path, end to end, against run.sh's stub upstream.

    Nothing in pytest reaches this code: every backend test mocks httpx2's
    transport, so httpcore2, h11 and the socket layer only ever run here — and
    neither does uvicorn forwarding a response while the upstream is still
    sending it.
    """
    import time

    import httpx2

    key = os.environ["BRIDGE_API_KEY"]
    auth = {"Authorization": f"Bearer {key}"}
    chat = {"messages": [{"role": "user", "content": "ping"}]}
    async with httpx2.AsyncClient(base_url=BASE_URL, timeout=20) as client:
        r = await client.get("/v1/models", headers=auth)
        assert r.status_code == 200, (r.status_code, r.text)
        ids = [m["id"] for m in r.json()["data"]]
        assert "stub/stub-model" in ids, ids

        r = await client.post(
            "/v1/chat/completions", headers=auth, json={**chat, "model": "stub/stub-model"}
        )
        assert r.status_code == 200, (r.status_code, r.text)
        assert r.json()["choices"][0]["message"]["content"] == "pong", r.text

        # The stub pauses between events. A forwarded stream delivers the first
        # event well before the last; a buffered one delivers everything at
        # once, after all the pauses.
        started = time.monotonic()
        first_at: float | None = None
        received = b""
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=auth,
            json={**chat, "model": "stub/stub-model", "stream": True},
        ) as r:
            assert r.status_code == 200, r.status_code
            assert r.headers["content-type"].startswith("text/event-stream"), r.headers
            async for chunk in r.aiter_bytes():
                if first_at is None and b"Hello" in received + chunk:
                    first_at = time.monotonic() - started
                received += chunk
        total = time.monotonic() - started
        for text in (b"Hello", b" from", b" upstream", b"[DONE]"):
            assert text in received, received
        assert first_at is not None
        assert total - first_at >= 0.75, (
            f"stream was buffered: first event at {first_at:.2f}s, end at {total:.2f}s"
        )

        r = await client.post(
            "/v1/chat/completions", headers=auth, json={**chat, "model": "down/stub-model"}
        )
        assert r.status_code == 502, (r.status_code, r.text)
        assert r.json()["error"]["code"] == "upstream_error", r.text
    print(
        "passthrough: JSON, a stream forwarded as it arrived "
        f"(first event {first_at:.2f}s of {total:.2f}s), and a 502 for a dead upstream"
    )


def main() -> int:
    check_every_module_imports()
    check_dev_dependencies_absent()
    check_native_extensions()
    asyncio.run(check_aiosqlite())
    asyncio.run(check_server_over_httpx2())
    asyncio.run(check_passthrough_over_the_network())
    print(json.dumps({"python": sys.version.split()[0], "platform": sys.platform}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
