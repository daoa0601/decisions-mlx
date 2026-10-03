"""A local ``/v1/systemone`` server over one or more deciders.

    POST /v1/systemone   request body -> response body; ``images`` are data URLs
    GET  /v1/models      the served model names
    GET  /health

A request goes to the decider named by its ``model``. With a single decider loaded every name is
accepted, so clients that send Jev's own model names (``jev-latest``) work unchanged. Requests are
answered one at a time: MLX runs one forward pass at a time on the GPU anyway.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .api import Decider, RequestError, parse_request, respond


def decode_image(value: Any) -> Any:
    """A ``data:image/...;base64,`` URL as a PIL image."""
    if not isinstance(value, str) or not value.startswith("data:image/") or ";base64," not in value:
        raise RequestError("images must be data:image/...;base64, URLs")
    from PIL import Image

    try:
        return Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1], validate=True))).convert("RGB")
    except Exception as error:  # noqa: BLE001 - any decode failure is the client's input
        raise RequestError(f"could not decode image: {error}") from error


def make_server(deciders: Mapping[str, Decider], host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    if not deciders:
        raise ValueError("at least one decider is required")
    lock = threading.Lock()

    def route(name: str) -> Decider:
        if name in deciders:
            return deciders[name]
        if len(deciders) == 1:
            return next(iter(deciders.values()))
        raise RequestError(f"unknown model {name!r}; serving {sorted(deciders)}")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict[str, Any], timing: float | None = None) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if timing is not None:
                self.send_header("Server-Timing", f"model;dur={timing:.1f}")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            path = self.path.rstrip("/")
            if path == "/health":
                self._send(200, {"ok": True})
            elif path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": name, "object": "model"} for name in deciders]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                request = parse_request(body)
                request = replace(request, images=[decode_image(image) for image in request.images])
                decider = route(request.model)
                with lock:
                    start = time.perf_counter()
                    decision = decider.decide(request)
                    elapsed = (time.perf_counter() - start) * 1000
            except json.JSONDecodeError as error:
                self._send(400, {"error": f"invalid JSON: {error}"})
                return
            except ValueError as error:  # RequestError, or an input the model cannot take (too long)
                self._send(400, {"error": str(error)})
                return
            except Exception as error:  # noqa: BLE001 - report instead of dropping the connection
                self._send(500, {"error": f"{type(error).__name__}: {error}"})
                return
            self._send(200, respond(request, decision), elapsed)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return ThreadingHTTPServer((host, port), Handler)
