from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import collect_workspace_onpolicy_large as large


class LargeWorkspaceCollectorTests(unittest.TestCase):
    def test_large_selection_accepts_512_per_source_and_keeps_sources_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old.jsonl"
            extended = root / "extended.jsonl"
            old.write_text("old\n")
            extended.write_text("extended\n")

            def fake_select(path, count, seed, allowed):
                base = 10_000 if path == old else 20_000
                return [{"seed": base + index, "difficulty": index % 5 + 1}
                        for index in range(count)]

            allowed = set(range(10_000, 10_512)) | set(range(20_000, 20_512))
            with patch.object(large, "_json_lines", return_value=[{}]), \
                 patch.object(large, "_validate_bank", side_effect=[set(range(512)),
                                                                       set(range(512, 1024))]), \
                 patch.object(large, "_load_allowed_seeds", return_value=(allowed, {})), \
                 patch.object(large, "_select", side_effect=fake_select):
                specs, records, _ = large.select_sources(
                    old, extended, root / "combined.npz", root / "eligible.npy",
                    root / "cache", per_source=512)

            self.assertEqual(len(specs), 1024)
            self.assertEqual(len({row["seed"] for row in specs}), 1024)
            self.assertEqual([record["name"] for record in records], ["original", "extended"])
            self.assertEqual(records[0]["selected_difficulties"], {1: 103, 2: 103, 3: 102, 4: 102, 5: 102})
            self.assertEqual(records[1]["selected_difficulties"], {1: 103, 2: 103, 3: 102, 4: 102, 5: 102})

    def test_large_selection_rejects_over_bound_before_reading_sources(self):
        with self.assertRaisesRegex(ValueError, "1..512"):
            large.select_sources(Path("missing-old"), Path("missing-extended"),
                                 Path("missing-combined"), per_source=513)

    def test_source_binding_names_executed_large_module(self):
        self.assertEqual(large._bound_code_paths()[0].resolve(),
                         Path("tools/collect_workspace_onpolicy_large.py").resolve())
        self.assertEqual(large.DEFAULT_OUT, Path("data/ls20-world-workspace-onpolicy-large-train.npz"))
        self.assertEqual(large.DEFAULT_REPORT, Path("artifacts/world-workspace-onpolicy-large.json"))


if __name__ == "__main__":
    unittest.main()
