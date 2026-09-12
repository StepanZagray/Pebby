"""Focused generated-only checks for the native result Mapping.

The snapshot intentionally contains the small generated reference fixtures but
not the upstream game checkout.  Reconstructing their logical Layouts keeps
these tests independent of sprites and still exercises the real generated
specifications, the reference Oracle, truncation, and all Mapping methods.
"""

import gc
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import unittest

from pebby.ls20 import fastplan
from pebby.ls20.layout import Layout
from pebby.ls20.plan import Oracle


FIXTURES = Path(__file__).parent / "fixtures" / "ls20_reference"


def fixture_layout(name):
    data = json.loads((FIXTURES / f"{name}.json").read_text())
    goals = [(tuple(goal["cell"]), tuple(goal["triple"])) for goal in data["goals"]]
    cyclers = {tuple(item["cell"]): item["kind"] for item in data["cyclers"]}
    return Layout(
        cols=12, rows=12, walls={tuple(cell) for cell in data["walls"]},
        cyclers=cyclers, refills={tuple(cell) for cell in data["refills"]},
        launchers=[], rails=[tuple(cell) for cell in data.get("rails", [])],
        goals=goals, goal_at={cell: i for i, (cell, _triple) in enumerate(goals)},
        patrollers=[], moving_cyclers=[{} for _ in range(data["tick_period"])],
        tick_tail=0, tick_period=data["tick_period"], tick_span=data["tick_period"],
        start_cell=tuple(data["start"]), start_triple=tuple(data["start_triple"]),
        max_steps=data["step_counter"], step_cost=data["step_cost"],
        match_hint=bool(data.get("verification_match_hint", False)),
        fog=bool(data.get("fog", False)), level_index=data.get("verification_level_index", 0),
    )


class CompactMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not fastplan.available():
            raise unittest.SkipTest(fastplan.load_error())

    def test_full_generated_mapping_and_iteration_match_reference(self):
        for fixture in ("tier1", "tier2", "tier3"):
            with self.subTest(fixture=fixture):
                layout = fixture_layout(fixture)
                reference = Oracle(layout, limit=600_000, engine="reference")
                tables = fastplan.tables_for(layout, reference.refills)
                distance, reachable, truncated = fastplan.search(
                    tables, reference.start, 600_000)
                try:
                    self.assertEqual(reachable, reference._reachable)
                    self.assertEqual(truncated, reference.truncated)
                    self.assertEqual(len(distance), len(reference._distance))
                    self.assertEqual(dict(distance), reference._distance)
                    self.assertEqual(distance, reference._distance)
                    # The C result arrays are in discovery order.  The
                    # reference reverse-BFS dict has a different insertion
                    # order, so compare the native Mapping's order against a
                    # second identical search instead of conflating values
                    # with dict insertion order.
                    native_order = list(distance.items())
                    # Compare with the raw C result order, which is the order
                    # the pre-change dict(zip(keys, values)) exposed.
                    library = fastplan._load()
                    raw_states = fastplan.ctypes.POINTER(fastplan.ctypes.c_uint64)()
                    raw_values = fastplan.ctypes.POINTER(fastplan.ctypes.c_int32)()
                    raw_count = fastplan.ctypes.c_int64()
                    raw_reachable = fastplan.ctypes.c_int64()
                    raw_truncated = fastplan.ctypes.c_int32()
                    status = library.ls20_search(
                        fastplan.ctypes.byref(tables.params), tables.pack(reference.start),
                        600_000, fastplan.ctypes.byref(raw_states),
                        fastplan.ctypes.byref(raw_values), fastplan.ctypes.byref(raw_count),
                        fastplan.ctypes.byref(raw_reachable), fastplan.ctypes.byref(raw_truncated))
                    self.assertEqual(status, 0)
                    try:
                        raw_order = [(tables.unpack(int(raw_states[i])), int(raw_values[i]))
                                     for i in range(raw_count.value)]
                    finally:
                        library.ls20_free(raw_states)
                        library.ls20_free(raw_values)
                    self.assertEqual(native_order, raw_order)
                    repeat, _, _ = fastplan.search(tables, reference.start, 600_000)
                    try:
                        self.assertEqual(list(repeat.items()), native_order)
                    finally:
                        repeat.close()
                    self.assertNotIn(("not", "a", "state"), distance)
                    self.assertIsNone(distance.get(("not", "a", "state")))
                    with self.assertRaises(KeyError):
                        distance[("not", "a", "state")]
                finally:
                    distance.close()

    def test_truncation_and_foreign_states_match_reference(self):
        layout = fixture_layout("tier2")
        for limit in (1, 2, 17, 700):
            with self.subTest(limit=limit):
                reference = Oracle(layout, limit=limit, engine="reference")
                tables = fastplan.tables_for(layout, reference.refills)
                distance, reachable, truncated = fastplan.search(
                    tables, reference.start, limit)
                try:
                    self.assertEqual((reachable, truncated),
                                     (reference._reachable, reference.truncated))
                    self.assertEqual(list(distance.items()), list(reference._distance.items()))
                    foreign = (reference.start[0], 99, *reference.start[2:])
                    self.assertNotIn(foreign, distance)
                    self.assertEqual(distance.get(foreign, "missing"), "missing")
                finally:
                    distance.close()

    def test_result_owns_native_buffers_and_repeated_close_is_safe(self):
        layout = fixture_layout("tier1")
        reference = Oracle(layout, limit=600_000, engine="reference")
        tables = fastplan.tables_for(layout, reference.refills)
        for _ in range(12):
            distance, _reachable, _truncated = fastplan.search(tables, reference.start, 600_000)
            self.assertFalse(hasattr(distance, "_table"))
            self.assertGreaterEqual(distance._capacity, max(2, 2 * len(distance)))
            distance.close()
            distance.close()
            self.assertEqual(len(distance), 0)
            del distance
            gc.collect()

    def test_concurrent_readers_agree_and_close_waits_for_them(self):
        layout = fixture_layout("tier1")
        reference = Oracle(layout, limit=600_000, engine="reference")
        tables = fastplan.tables_for(layout, reference.refills)
        distance, _reachable, _truncated = fastplan.search(tables, reference.start, 600_000)
        states = list(reference._distance)[:256]
        expected = [reference._distance[state] for state in states]
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(lambda: [distance[state] for state in states])
                           for _ in range(4)]
                self.assertEqual([future.result() for future in futures], [expected] * 4)
            # close() is idempotent and no native reader is left in flight.
            distance.close()
            self.assertEqual(distance.get(reference.start), None)
        finally:
            distance.close()

    def test_close_racing_reader_is_serialized(self):
        layout = fixture_layout("tier1")
        reference = Oracle(layout, limit=600_000, engine="reference")
        tables = fastplan.tables_for(layout, reference.refills)
        for _ in range(16):
            distance, _reachable, _truncated = fastplan.search(tables, reference.start, 600_000)
            expected = reference._distance[reference.start]

            def read_once():
                try:
                    return ("value", distance[reference.start])
                except KeyError:
                    return ("closed", None)

            with ThreadPoolExecutor(max_workers=2) as pool:
                # Gate both operations on the same lock, then release it so
                # either legal serialization is exercised without a dangling
                # native value pointer.
                distance._lock.acquire()
                reader = pool.submit(read_once)
                closer = pool.submit(distance.close)
                distance._lock.release()
                result = reader.result()
                closer.result()
            self.assertIn(result, (("value", expected), ("closed", None)))

    def test_index_rejects_unrepresentable_count_without_dereferencing_keys(self):
        library = fastplan._load()
        capacity = fastplan.ctypes.c_size_t()
        null_keys = fastplan.ctypes.POINTER(fastplan.ctypes.c_uint64)()
        index = library.ls20_index_build(
            null_keys, fastplan.ctypes.c_size_t(2 ** 31), fastplan.ctypes.byref(capacity))
        self.assertFalse(index)


if __name__ == "__main__":
    unittest.main()
