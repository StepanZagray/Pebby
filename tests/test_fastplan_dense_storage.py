import ctypes
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.transform_fastplan_dense_storage import compact_graph_storage
from tests.test_fastplan_compact import fixture_layout
from pebby.ls20 import fastplan
from pebby.ls20.plan import Oracle


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "pebby" / "ls20" / "_fastplan.c"
BEFORE = ROOT / "tests" / "fixtures" / "fastplan_before_dense_storage.c"


class DenseStorageTransformTests(unittest.TestCase):
    def test_checked_in_source_is_exact_transform(self):
        before = BEFORE.read_text()
        expected = compact_graph_storage(before)
        self.assertEqual(SOURCE.read_text(), expected)

    def test_transform_is_fail_closed_for_already_migrated_source(self):
        with self.assertRaises(ValueError):
            compact_graph_storage(SOURCE.read_text())

    def test_no_accidental_outside_search_mutation(self):
        before = BEFORE.read_text()
        after = SOURCE.read_text()
        start = before.index("int ls20_search(")
        end = before.index("void ls20_free", start)
        new_start = after.index("int ls20_search(")
        new_end = after.index("void ls20_free", new_start)
        # The hash-table helper and search body are the two intentional
        # changes. The ABI, result index, and free function remain stable.
        self.assertEqual(before[end:], after[new_end:])
        self.assertEqual(before[:before.index("/* -- open-addressed")],
                         after[:after.index("/* -- open-addressed")])

    def test_dense_kernel_matches_pretransform_kernel(self):
        """Exercise complete and truncated generated searches through both C builds."""
        signature = [
            ctypes.POINTER(fastplan.Params), ctypes.c_uint64, ctypes.c_int64,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int32),
        ]

        def build(source, output):
            subprocess.run(
                ["cc", "-x", "c", "-O2", "-std=c99", "-fPIC", "-shared",
                 "-Wall", "-Wextra", "-Werror", "-o", str(output), str(source)],
                check=True, capture_output=True, text=True,
            )

        def run(library_path, layout, limit):
            library = ctypes.CDLL(str(library_path))
            library.ls20_search.argtypes = signature
            library.ls20_search.restype = ctypes.c_int
            library.ls20_free.argtypes = [ctypes.c_void_p]
            library.ls20_free.restype = None
            reference = Oracle(layout, limit=limit, engine="reference")
            tables = fastplan.Tables(layout, reference.refills)
            states = ctypes.POINTER(ctypes.c_uint64)()
            distances = ctypes.POINTER(ctypes.c_int32)()
            count = ctypes.c_int64()
            reachable = ctypes.c_int64()
            truncated = ctypes.c_int32()
            status = library.ls20_search(
                ctypes.byref(tables.params), tables.pack(reference.start), limit,
                ctypes.byref(states), ctypes.byref(distances), ctypes.byref(count),
                ctypes.byref(reachable), ctypes.byref(truncated),
            )
            try:
                values = ([int(states[i]) for i in range(count.value)],
                          [int(distances[i]) for i in range(count.value)])
            finally:
                if states:
                    library.ls20_free(states)
                if distances:
                    library.ls20_free(distances)
            return status, *values, reachable.value, truncated.value

        with tempfile.TemporaryDirectory(prefix="pebby-dense-c-") as tmp:
            tmp = Path(tmp)
            old_library, new_library = tmp / "old.so", tmp / "new.so"
            build(BEFORE, old_library)
            build(SOURCE, new_library)
            for fixture in ("tier1", "tier2", "tier3"):
                layout = fixture_layout(fixture)
                for limit in (1, 17, 700, 600_000):
                    with self.subTest(fixture=fixture, limit=limit):
                        self.assertEqual(run(old_library, layout, limit),
                                         run(new_library, layout, limit))


if __name__ == "__main__":
    unittest.main()
