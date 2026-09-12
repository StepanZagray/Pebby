"""The integration uses an installed package without eager model initialization."""

import importlib.metadata
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hostai import Provider
from inference import Engine
from serve import make_server

ROOT = Path(__file__).resolve().parents[1]


class SdkIntegrationTests(unittest.TestCase):
    def test_importing_the_server_does_not_import_torch(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys, serve; assert 'torch' not in sys.modules"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(importlib.metadata.version("hostai"), "0.1.0")

    def test_startup_and_manifest_use_sdk_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Engine(Path(directory) / "absent.pt")
            with patch.object(
                engine.agent, "ensure", side_effect=AssertionError("Eager policy load")
            ):
                with make_server(engine, 0) as server:
                    provider = server.RequestHandlerClass.hostai_provider
                    self.assertIsInstance(provider, Provider)
                    self.assertEqual(
                        server.RequestHandlerClass.__mro__[1].__module__, "hostai.http"
                    )
                    contract = provider.manifest()["models"][0]["interaction"]
                    for operation in ("info", "generate", "shipped", "play", "oracle", "agent"):
                        self.assertIn(operation, contract["instructions"])
                    self.assertEqual(
                        contract["inputSchema"]["properties"]["actions"]["maxItems"], 2048
                    )
                    self.assertFalse(engine.agent.loaded)


if __name__ == "__main__":
    unittest.main()
