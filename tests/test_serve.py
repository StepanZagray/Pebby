import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from hostai import MAX_ASSET_BYTES, ContractError

from inference import MAX_ACTIONS, Engine
from serve import MAX_BODY, make_server

ROOT = Path(__file__).resolve().parents[1]

# One request per operation, all cheap: shipped levels build instantly and the
# oracle for level 1 is cached inside inference after the first plan.
OPERATIONS = {
    "boot": {"op": "boot"},
    "info": {"op": "info"},
    "shipped": {"op": "shipped", "index": 3},
    "generate": {"op": "generate", "seed": 3, "difficulty": 1},
    "play": {"op": "play", "level": {"shipped": 0}, "actions": [3, 3, 3, 1]},
    "play defaults": {"op": "play", "level": {"shipped": 2}},
    "oracle": {"op": "oracle", "level": {"shipped": 0}, "actions": [3]},
    "agent": {"op": "agent", "level": {"shipped": 0}, "actions": []},
}


class ServeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.checkpoint_dir.cleanup)
        cls.engine = Engine(str(Path(cls.checkpoint_dir.name) / "absent.pt"))
        cls.server = make_server(cls.engine, 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.start()
        cls.origin = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def get(self, path):
        with urlopen(self.origin + path, timeout=10) as response:
            return response.status, response.headers, response.read()

    def post(self, path, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        request = Request(self.origin + path, data, {"Content-Type": content_type})
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    def assert_status(self, code, path, body, content_type="application/json"):
        with self.assertRaises(HTTPError) as error:
            self.post(path, body, content_type)
        self.assertEqual(error.exception.code, code)
        self.assertEqual(error.exception.headers.get_content_type(), "application/json")
        return json.load(error.exception)

    def test_health(self):
        status, headers, raw = self.get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(json.loads(raw), {"version": "pebby-0.2.0", "model": "pebby:latest"})

    def test_manifest_describes_the_absent_checkpoint(self):
        status, headers, raw = self.get("/hostai/manifest")
        self.assertEqual(status, 200)
        manifest = json.loads(raw)
        self.assertEqual(set(manifest), {"protocol", "runtime", "models", "ui"})
        self.assertEqual(manifest["protocol"], 1)
        self.assertEqual(manifest["runtime"], "pebby")
        self.assertEqual(manifest["ui"], {"entry": "/ui/index.html"})
        [model] = manifest["models"]
        self.assertEqual(
            set(model),
            {
                "name",
                "sizeBytes",
                "parameterSize",
                "quantization",
                "modifiedAt",
                "capabilities",
                "interaction",
            },
        )
        self.assertEqual(model["capabilities"], {"chat": False, "infer": True})
        self.assertEqual(
            set(model["interaction"]), {"instructions", "inputSchema", "outputSchema", "examples"}
        )
        self.assertTrue(model["interaction"]["instructions"])
        self.assertEqual(model["interaction"]["inputSchema"]["required"], ["op"])
        self.assertEqual(model["name"], "pebby:latest")
        self.assertEqual(model["quantization"], "F32")
        # Nothing is trained, so the manifest advertises weightless zeroes rather
        # than refusing to describe the model at all.
        self.assertFalse(self.engine.agent.loaded)
        self.assertEqual(model["sizeBytes"], 0)
        self.assertEqual(model["parameterSize"], "0")
        self.assertRegex(
            model["modifiedAt"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$"
        )

    def test_every_operation_matches_the_engine_on_both_routes(self):
        for name, request in OPERATIONS.items():
            with self.subTest(op=name):
                expected = json.loads(json.dumps(self.engine.dispatch(request)))
                self.assertEqual(self.post("/predict", request), expected)
                self.assertEqual(
                    self.post("/hostai/infer", {"model": "pebby:latest", "input": request}),
                    expected,
                )

    def test_replies_are_never_cached(self):
        for path in ("/health", "/hostai/manifest", "/ui/index.html", "/ui/hostai-bridge.js"):
            with self.subTest(path=path):
                _, headers, _ = self.get(path)
                self.assertEqual(headers["Cache-Control"], "no-store")
        request = Request(
            self.origin + "/predict",
            json.dumps({"op": "info"}).encode(),
            {"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=10) as response:
            self.assertEqual(response.headers["Cache-Control"], "no-store")
        with self.assertRaises(HTTPError) as error:
            self.post("/predict", {"op": "nope"})
        self.assertEqual(error.exception.headers["Cache-Control"], "no-store")

    def test_both_routes_accept_all_seven_generation_tiers(self):
        expected = {"level": {}, "frame": [], "status": {}}
        for difficulty in range(1, 8):
            body = {"op": "generate", "seed": 930000 + difficulty, "difficulty": difficulty}
            for route, request in (("/predict", body),
                                   ("/hostai/infer", {"model": "pebby:latest", "input": body})):
                with self.subTest(difficulty=difficulty, route=route):
                    with patch.object(self.engine, "dispatch", return_value=expected) as dispatch:
                        self.assertEqual(self.post(route, request), expected)
                        dispatch.assert_called_once_with(body)

    def test_rejects_schema_invalid_inputs_on_both_routes(self):
        cases = [
            {"op": "nope"},
            {"level": {"shipped": 0}},
            [1, 2],
            {"op": "info", "extra": 1},
            {"op": "play"},
            {"op": "shipped", "index": 7},
            {"op": "generate", "seed": -1},
            {"op": "generate", "difficulty": 8},
            {"op": "play", "level": {"shipped": 0}, "actions": [5]},
            "info",
        ]
        for body in cases:
            for path, request in [
                ("/predict", body),
                ("/hostai/infer", {"model": "pebby:latest", "input": body}),
            ]:
                with self.subTest(path=path, body=body):
                    error = self.assert_status(400, path, request)
                    self.assertEqual(set(error), {"error"})
                    self.assertTrue(
                        error["error"].startswith("input does not conform to the input schema:"),
                        error,
                    )

    def test_domain_input_errors_keep_their_message_on_both_routes(self):
        body = {"op": "play", "level": {"shipped": 0, "extra": 1}}
        for path, request in [
            ("/predict", body),
            ("/hostai/infer", {"model": "pebby:latest", "input": body}),
        ]:
            with self.subTest(path=path):
                self.assertEqual(
                    self.assert_status(400, path, request),
                    {"error": 'A shipped level is exactly {"shipped": index}.'},
                )

    def test_rejects_bad_json_and_envelopes(self):
        for path in ("/predict", "/hostai/infer"):
            for body in (b'{"op": "info"', b'{"op": "\xff\xfe"}'):
                with self.subTest(path=path, body=body):
                    self.assertEqual(
                        self.assert_status(400, path, body),
                        {"error": "request body is not valid JSON"},
                    )
        for body in (
            [],
            {"input": {"op": "info"}},
            {"model": "pebby:latest"},
            {"model": "pebby:latest", "input": {"op": "info"}, "extra": True},
        ):
            with self.subTest(body=body):
                self.assertEqual(
                    self.assert_status(400, "/hostai/infer", body),
                    {
                        "error": 'request body must be a JSON object with exactly the keys "model" and "input"'
                    },
                )
        self.assertEqual(
            self.assert_status(400, "/hostai/infer", {"model": "llama3", "input": {"op": "info"}}),
            {"error": "unknown model"},
        )
        self.assertEqual(
            self.assert_status(400, "/hostai/infer", {"model": 1, "input": {"op": "info"}}),
            {"error": '"model" must be a string'},
        )

    def test_implementation_and_output_failures_are_generic_500(self):
        for path, body in [
            ("/predict", {"op": "info"}),
            ("/hostai/infer", {"model": "pebby:latest", "input": {"op": "info"}}),
        ]:
            for options in (
                {"side_effect": RuntimeError("private implementation detail")},
                {"return_value": {}},
                {"return_value": {"bad": float("nan")}},
                {"return_value": {"action": 9}},
            ):
                with self.subTest(path=path, options=options):
                    with patch.object(self.engine, "dispatch", **options):
                        self.assertEqual(
                            self.assert_status(500, path, body), {"error": "inference failed"}
                        )
            with patch.object(self.engine, "dispatch", return_value={"padding": "x" * MAX_BODY}):
                self.assertEqual(
                    self.assert_status(500, path, body),
                    {"error": "provider output exceeds 262144 bytes"},
                )

    def test_unsupported_methods_are_405(self):
        for method, path in [
            ("GET", "/hostai/infer"),
            ("POST", "/hostai/manifest"),
            ("POST", "/ui/index.html"),
            ("PUT", "/predict"),
            ("DELETE", "/hostai/infer"),
            ("PATCH", "/health"),
            ("OPTIONS", "/predict"),
            ("HEAD", "/hostai/manifest"),
        ]:
            request = Request(self.origin + path, method=method)
            with self.subTest(method=method, path=path), self.assertRaises(HTTPError) as error:
                urlopen(request, timeout=10)
            self.assertEqual(error.exception.code, 405)
            self.assertEqual(error.exception.headers.get_content_type(), "application/json")
            if method != "HEAD":
                self.assertEqual(json.load(error.exception), {"error": "method not allowed"})
            error.exception.close()

    def test_query_strings_are_rejected(self):
        for path in ("/hostai/manifest?x=1", "/ui/index.html?x=1"):
            with self.subTest(path=path), self.assertRaises(HTTPError) as error:
                self.get(path)
            self.assertEqual(error.exception.code, 404)
            self.assertEqual(
                json.load(error.exception),
                {"error": "query strings and fragments are not supported"},
            )

    def test_startup_rejects_ui_without_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            ui_dir = Path(directory)
            (ui_dir / "index.html").write_text("<!doctype html><title>No bridge</title>")
            with self.assertRaisesRegex(ContractError, "must load the HostAI bridge"):
                make_server(self.engine, 0, ui_dir)

    def test_rejects_chunked_bodies_with_400(self):
        request = Request(
            self.origin + "/predict",
            json.dumps({"op": "info"}).encode(),
            {"Content-Type": "application/json", "Transfer-Encoding": "chunked"},
        )
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=10)
        self.assertEqual(error.exception.code, 400)
        self.assertEqual(
            json.load(error.exception), {"error": "Transfer-Encoding is not supported"}
        )

    def test_action_history_cap_is_enforced_over_http(self):
        # The last accepted history and the first rejected one, both against a
        # real socket, so the cap is proved where it actually protects the CPU.
        level = {"shipped": 0}
        accepted = self.post(
            "/predict", {"op": "play", "level": level, "actions": [1, 2] * (MAX_ACTIONS // 2)}
        )
        self.assertEqual(accepted["status"]["state"], "GAME_OVER")
        error = self.assert_status(
            400, "/predict", {"op": "play", "level": level, "actions": [1] * (MAX_ACTIONS + 1)}
        )
        self.assertTrue(error["error"].startswith("input does not conform to the input schema:"))

    def test_agent_without_a_checkpoint_is_an_answer_not_a_500(self):
        for path, body in [
            ("/predict", {"op": "agent", "level": {"shipped": 0}, "actions": []}),
            (
                "/hostai/infer",
                {"model": "pebby:latest", "input": {"op": "agent", "level": {"shipped": 0}}},
            ),
        ]:
            with self.subTest(path=path):
                result = self.post(path, body)
                self.assertFalse(result["loaded"])
                self.assertIsNone(result["action"])
                self.assertIsNone(result["probabilities"])
                self.assertTrue(
                    result["reason"].startswith("No agent checkpoint at "), result["reason"]
                )

    def test_rejects_wrong_content_type_with_415(self):
        for content_type in (
            "text/plain",
            "application/x-ndjson",
            "application/octet-stream",
            "application/json-patch+json",
        ):
            with self.subTest(content_type=content_type):
                error = self.assert_status(415, "/predict", {"op": "info"}, content_type)
                self.assertEqual(error["error"], "expected application/json")

    def test_rejects_empty_and_oversize_bodies(self):
        oversize = {"op": "play", "level": {"shipped": 0}, "padding": "x" * MAX_BODY}
        self.assertGreater(len(json.dumps(oversize)), MAX_BODY)
        for path in ("/predict", "/hostai/infer"):
            with self.subTest(path=path):
                self.assertEqual(
                    self.assert_status(400, path, b""), {"error": "request body is not valid JSON"}
                )
                self.assertEqual(
                    self.assert_status(413, path, oversize),
                    {"error": "request body exceeds 262144 bytes"},
                )
                # Exactly MAX_BODY bytes passes framing and reaches validation.
                body = b"{}" + b" " * (MAX_BODY - 2)
                self.assert_status(400, path, body)

    def test_rejects_unknown_endpoints_with_404(self):
        for path in ("/", "/infer", "/api/chat", "/predict/", "/hostai/infer/x", "/PREDICT"):
            with self.subTest(post=path):
                self.assertEqual(
                    self.assert_status(404, path, {"op": "info"}), {"error": "no such route"}
                )
        for path in ("/", "/predict", "/health/", "/hostai", "/unknown"):
            with self.subTest(get=path), self.assertRaises(HTTPError) as error:
                self.get(path)
            self.assertEqual(error.exception.code, 404)
            self.assertEqual(json.load(error.exception), {"error": "no such route"})

    def test_static_ui_is_served_verbatim(self):
        expectations = {
            "index.html": "text/html; charset=utf-8",
            "app.js": "text/javascript; charset=utf-8",
            "board.js": "text/javascript; charset=utf-8",
            "style.css": "text/css; charset=utf-8",
        }
        for name, content_type in expectations.items():
            with self.subTest(file=name):
                status, headers, raw = self.get("/ui/" + name)
                self.assertEqual(status, 200)
                self.assertEqual(headers["Content-Type"], content_type)
                self.assertEqual(int(headers["Content-Length"]), len(raw))
                self.assertEqual(raw, (ROOT / "ui" / name).read_bytes())
        html = (ROOT / "ui" / "index.html").read_text()
        self.assertIn('<script src="hostai-bridge.js"></script>', html)
        self.assertIn('<script src="app.js"></script>', html)
        self.assertNotRegex(html, r"<script>|\son[a-z]+=|\sstyle=")

    def test_hostai_bridge_is_synthesised_not_shipped(self):
        # HostAI injects the real bridge; a copy on disk would shadow it, so the
        # server answers with an inert placeholder and ships no file.
        self.assertFalse((ROOT / "ui" / "hostai-bridge.js").exists())
        status, headers, raw = self.get("/ui/hostai-bridge.js")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/javascript; charset=utf-8")
        self.assertEqual(int(headers["Content-Length"]), len(raw))
        self.assertEqual(raw, b"/* HostAI supplies this bridge when embedded. */\n")

    def test_static_ui_rejects_traversal_and_unknown_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            ui_dir = Path(directory) / "ui"
            ui_dir.mkdir()
            (ui_dir / "index.html").write_text(
                '<!doctype html><script src="hostai-bridge.js"></script>'
            )
            (ui_dir / "notes.bin").write_bytes(b"not served")
            with (ui_dir / "large.json").open("wb") as asset:
                asset.truncate(MAX_ASSET_BYTES + 1)
            (Path(directory) / "secret.json").write_text("{}")
            (ui_dir / "escape.json").symlink_to(Path(directory) / "secret.json")
            (ui_dir / ".hidden.json").write_text("{}")
            server = make_server(self.engine, 0, ui_dir)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            origin = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(origin + "/ui/index.html", timeout=10) as response:
                    self.assertEqual(response.status, 200)
                for path in [
                    "/ui/../secret.json",
                    "/ui/sub/../../secret.json",
                    "/ui/%2e%2e/secret.json",
                    "/ui/escape.json",
                    "/ui/.hidden.json",
                    "/ui/notes.bin",
                    "/ui/large.json",
                    "/ui/",
                    "/ui",
                    "/ui//index.html",
                    "/ui/missing.html",
                    "/ui/index.html/",
                    "/hostai/manifest/../../secret.json",
                    "/unknown",
                ]:
                    with self.subTest(path=path), self.assertRaises(HTTPError) as error:
                        urlopen(origin + path, timeout=10)
                    self.assertEqual(error.exception.code, 404)
                    message = "no such asset" if path.startswith("/ui/") else "no such route"
                    self.assertEqual(json.load(error.exception), {"error": message})
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
