"""A tiny JSON REST API built on the standard library's ``http.server``.

Endpoints:

* ``GET /health`` - liveness check plus corpus size.
* ``POST /ask`` - body ``{"question": "...", "top_k": 3}``; returns the answer,
  its citations and the retrieved chunks.

This is meant for demos and local integration, not for production traffic.
Put a real ASGI/WSGI server in front of the pipeline if you need that.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Tuple, Type

from .pipeline import RAGPipeline

MAX_BODY_BYTES = 64 * 1024


def handle_ask(pipeline: RAGPipeline, payload: Any) -> Tuple[int, Dict[str, Any]]:
    """Validate an ``/ask`` payload and run it through ``pipeline``.

    Kept separate from the HTTP handler so it can be tested directly.
    Returns ``(status_code, response_body)``.
    """
    if not isinstance(payload, dict):
        return HTTPStatus.BAD_REQUEST, {"error": "Request body must be a JSON object"}
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return HTTPStatus.BAD_REQUEST, {"error": "'question' must be a non-empty string"}
    top_k = payload.get("top_k", pipeline.top_k)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
        return HTTPStatus.BAD_REQUEST, {"error": "'top_k' must be an integer between 1 and 20"}
    answer = pipeline.ask(question, k=top_k)
    return HTTPStatus.OK, answer.to_dict()


def make_handler(pipeline: RAGPipeline) -> Type[BaseHTTPRequestHandler]:
    """Create a request handler class bound to ``pipeline``."""

    class RAGRequestHandler(BaseHTTPRequestHandler):
        server_version = "rag-eval-lab"

        def _send_json(self, status: int, body: Dict[str, Any]) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - name required by BaseHTTPRequestHandler
            if self.path == "/health":
                self._send_json(HTTPStatus.OK, {"status": "ok", "chunks": len(pipeline.chunks)})
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"No route for GET {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/ask":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"No route for POST {self.path}"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Body too large"})
                return
            try:
                payload = json.loads(self.rfile.read(length) or b"null")
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Body is not valid JSON"})
                return
            try:
                status, body = handle_ask(pipeline, payload)
            except Exception as exc:  # surface generator failures as JSON
                status, body = HTTPStatus.BAD_GATEWAY, {"error": str(exc)}
            self._send_json(status, body)

        def log_message(self, format: str, *args: Any) -> None:
            if not getattr(self.server, "quiet", False):
                super().log_message(format, *args)

    return RAGRequestHandler


def create_server(
    pipeline: RAGPipeline, host: str = "127.0.0.1", port: int = 8000, quiet: bool = False
) -> ThreadingHTTPServer:
    """Create (but do not start) a threaded HTTP server for ``pipeline``."""
    server = ThreadingHTTPServer((host, port), make_handler(pipeline))
    server.quiet = quiet  # type: ignore[attr-defined]
    return server


def serve(pipeline: RAGPipeline, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API until interrupted."""
    server = create_server(pipeline, host, port)
    print(f"Serving rag-eval-lab on http://{host}:{server.server_address[1]} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
