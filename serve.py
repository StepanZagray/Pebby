"""Serve stateless LS20 operations through the HostAI SDK and the static UI.

Importing this module never loads a checkpoint or imports torch.
"""

import argparse
import json
from http.server import ThreadingHTTPServer
from pathlib import Path

from hostai.http import MAX_BODY_BYTES as MAX_BODY
from hostai.http import UI_PREFIX
from hostai.ui import CONTENT_TYPES

from inference import DEFAULT_CHECKPOINT, Engine
from pebby.hostai_provider import make_provider

MODEL_NAME = "pebby:latest"
RUNTIME_NAME = "pebby"
VERSION = "pebby-0.2.0"
DEFAULT_UI_DIR = Path(__file__).resolve().parent / "ui"
__all__ = [
    "CONTENT_TYPES",
    "DEFAULT_UI_DIR",
    "MAX_BODY",
    "MODEL_NAME",
    "RUNTIME_NAME",
    "UI_PREFIX",
    "VERSION",
    "make_server",
]


def make_server(engine, port=11435, ui_dir=DEFAULT_UI_DIR):
    provider = make_provider(engine, ui_dir)
    provider_handler = provider.handler(infer_aliases={"/predict": MODEL_NAME})

    class Handler(provider_handler):
        def do_GET(self):
            if self.path == "/health":
                data = (json.dumps({"version": VERSION, "model": MODEL_NAME}) + "\n").encode()
                content_type = "application/json"
            elif self.path == "/ui/hostai-bridge.js":
                data = b"/* HostAI supplies this bridge when embedded. */\n"
                content_type = "text/javascript; charset=utf-8"
            else:
                return super().do_GET()

            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=f"Trained LS20 policy for the agent operation. Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--ui-dir",
        type=str,
        default=str(DEFAULT_UI_DIR),
        help="Directory with index.html, style.css, app.js and board.js served under /ui/",
    )
    args = parser.parse_args()
    engine = Engine(args.checkpoint)
    # Load the policy now rather than on the first request, so an unusable
    # checkpoint is named at startup instead of surfacing as a UI badge later.
    # With no checkpoint on disk this returns immediately and torch stays out.
    engine.agent.ensure()
    with make_server(engine, args.port, args.ui_dir) as server:
        print(f"Pebby listening at http://127.0.0.1:{server.server_port}/ui/index.html", flush=True)
        print(
            f"Agent: {engine.agent.parameters} parameters from {engine.agent.path}"
            if engine.agent.loaded
            else f"Agent: none. {engine.agent.reason}",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
