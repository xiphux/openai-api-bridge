"""A minimal OpenAI-compatible upstream for the image smoke. Standard library only.

run.sh starts it in its own container (from the image under test, which has a
Python) and points an ``openai`` provider at it, so the bridge's outbound
requests cross a real socket: httpx2 → httpcore2 → h11 → TCP. Every pytest test
mocks that transport away, so this is the only place it runs.

Serves:
  GET  /v1/models            one model, ``stub-model``
  POST /v1/chat/completions  JSON, or with ``"stream": true`` an SSE stream sent
                             with chunked transfer encoding, one event per chunk
                             and a pause between them, so the smoke can tell a
                             forwarded stream from a buffered one
"""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "stub-model"
STREAM_EVENTS = ["Hello", " from", " upstream"]
# Long enough that buffering the whole stream is unmistakable in the smoke's
# timings, short enough to keep the job fast.
STREAM_PAUSE_SECONDS = 0.5


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"stub-upstream: {format % args}\n")

    def _json(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        else:
            self._json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": {"message": f"no route {self.path}"}})
            return
        if request.get("model") != MODEL:
            self._json(400, {"error": {"message": f"unknown model {request.get('model')!r}"}})
            return

        if not request.get("stream"):
            self._json(
                200,
                {
                    "object": "chat.completion",
                    "model": MODEL,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}}],
                },
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for i, text in enumerate(STREAM_EVENTS):
            if i:
                time.sleep(STREAM_PAUSE_SECONDS)
            event = {"choices": [{"index": 0, "delta": {"content": text}}]}
            self._chunk(f"data: {json.dumps(event)}\n\n".encode())
        self._chunk(b"data: [DONE]\n\n")
        self._chunk(b"")  # terminating zero-length chunk


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
