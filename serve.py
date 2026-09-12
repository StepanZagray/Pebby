"""Serve the LS20 JSON operations, the HostAI runtime protocol and the static UI.

The server is deliberately stateless: it holds no game, no session and no client
identity. Every request carries the level and the actions taken so far, and the
engine replays them. That is what lets the UI undo by dropping an action, and it
is why two browser tabs cannot corrupt each other.

torch is never imported here. A machine with no checkpoint, and no torch at all,
must still be able to open the UI and play LS20 by hand.
"""

import argparse
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from inference import DEFAULT_CHECKPOINT, Engine

MODEL_NAME = "pebby:latest"
RUNTIME_NAME = "pebby"
VERSION = "pebby-0.2.0"
MAX_BODY = 262144
DEFAULT_UI_DIR = Path(__file__).resolve().parent / "ui"
UI_PREFIX = "/ui/"
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".json": "application/json",
}


def make_server(engine, port=11435, ui_dir=DEFAULT_UI_DIR):
    ui_root = Path(ui_dir).resolve()
    path = engine.agent.path or DEFAULT_CHECKPOINT
    # Stat once: the manifest describes the weights this process would serve,
    # not whatever the file becomes later. Stat only, so torch stays unimported.
    checkpoint_stat = path.stat() if path.exists() else None
    started = datetime.now(timezone.utc)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *_args):
            pass  # Do not log submitted levels or UI requests.

        def reply(self, status, body):
            data = (json.dumps(body) + "\n").encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def manifest(self):
            modified = (datetime.fromtimestamp(checkpoint_stat.st_mtime, timezone.utc)
                        if checkpoint_stat else started)
            return {
                "protocol": 1,
                "runtime": RUNTIME_NAME,
                "models": [{
                    "name": MODEL_NAME,
                    "sizeBytes": checkpoint_stat.st_size if checkpoint_stat else 0,
                    "parameterSize": str(engine.agent.parameters or 0),
                    "quantization": "F32",
                    "modifiedAt": modified.isoformat(),
                }],
                "ui": {"entry": UI_PREFIX + "index.html"},
            }

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"version": VERSION, "model": MODEL_NAME})
            elif self.path == "/hostai/manifest":
                self.reply(200, self.manifest())
            elif self.path == "/ui/hostai-bridge.js":
                data = b"/* HostAI supplies this bridge when embedded. */\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path.startswith(UI_PREFIX):
                self.serve_ui(self.path[len(UI_PREFIX):])
            else:
                self.reply(404, {"error": "Unknown endpoint."})

        def serve_ui(self, relative):
            relative = relative.split("?", 1)[0].split("#", 1)[0]
            segments = relative.split("/")
            if not relative or any(not segment or segment.startswith(".") for segment in segments):
                self.reply(404, {"error": "Unknown endpoint."})
                return
            try:
                target = ui_root.joinpath(*segments).resolve()
            except (OSError, ValueError):
                self.reply(404, {"error": "Unknown endpoint."})
                return
            content_type = CONTENT_TYPES.get(target.suffix.lower())
            if ui_root not in target.parents or not target.is_file() or content_type is None:
                self.reply(404, {"error": "Unknown endpoint."})
                return
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path not in ("/predict", "/hostai/infer"):
                self.reply(404, {"error": "Unknown endpoint."})
                return
            if self.headers.get_content_type() != "application/json":
                self.reply(415, {"error": "Content-Type must be application/json."})
                return
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Use Content-Length, not Transfer-Encoding.")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    self.reply(413, {"error": "Body must contain 1 to 262144 bytes."})
                    return
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request body.")
                request = json.loads(raw)
                if self.path == "/hostai/infer":
                    if not isinstance(request, dict) or request.get("model") != MODEL_NAME:
                        raise ValueError(f"model must be {MODEL_NAME}.")
                    if "input" not in request:
                        raise ValueError("Request must contain input.")
                    request = request["input"]
                result = engine.dispatch(request)
            except (ValueError, UnicodeError, RecursionError) as error:
                self.reply(400, {"error": str(error)})
                return
            self.reply(200, result)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help=f"Trained LS20 policy for the agent operation. Default: {DEFAULT_CHECKPOINT}")
    parser.add_argument("--ui-dir", type=str, default=str(DEFAULT_UI_DIR),
                        help="Directory with index.html, style.css, app.js and board.js served under /ui/")
    args = parser.parse_args()
    engine = Engine(args.checkpoint)
    # Load the policy now rather than on the first request, so an unusable
    # checkpoint is named at startup instead of surfacing as a UI badge later.
    # With no checkpoint on disk this returns immediately and torch stays out.
    engine.agent.ensure()
    with make_server(engine, args.port, args.ui_dir) as server:
        print(f"Pebby listening at http://127.0.0.1:{server.server_port}/ui/index.html", flush=True)
        print(f"Agent: {engine.agent.parameters} parameters from {engine.agent.path}"
              if engine.agent.loaded else f"Agent: none. {engine.agent.reason}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
