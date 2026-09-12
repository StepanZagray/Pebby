"""Generated-only checks for the deterministic edge-boundary migration."""

from pathlib import Path
import unittest

from tools.transform_fastplan_edge_boundaries import compress_source_edges


ROOT = Path(__file__).parents[1]
OLD = ROOT / "tests" / "fixtures" / "fastplan_source_edges_before.c"
NEW = ROOT / "tests" / "fixtures" / "fastplan_before_dense_storage.c"


class EdgeBoundaryMigrationTests(unittest.TestCase):
    def test_transform_matches_staged_kernel_exactly(self):
        old = OLD.read_text()
        transformed = compress_source_edges(old)
        self.assertEqual(transformed, NEW.read_text())
        self.assertNotIn("edge_src", transformed)
        search = transformed[transformed.index("int ls20_search"):transformed.index("void ls20_free")]
        self.assertIn("edge_end", search)
        self.assertNotIn("edge_src", search)

    def test_transform_is_fail_closed_on_already_migrated_or_changed_source(self):
        with self.assertRaisesRegex(ValueError, "expected one exact match"):
            compress_source_edges(NEW.read_text())
        changed = OLD.read_text().replace(
            "state_count = 0, state_capacity = 0, edge_count = 0",
            "state_count = 0, state_capacity = 0, edge_count = 1",
            1,
        )
        with self.assertRaisesRegex(ValueError, "search capacities"):
            compress_source_edges(changed)


if __name__ == "__main__":
    unittest.main()
