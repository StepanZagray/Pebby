"""Generated-only context regressions and independent small search checks."""

from collections import deque
import json
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pebby.ls20 import generate, names
from pebby.ls20.bank import load
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.layout import extract
from pebby.ls20.plan import Oracle, advance
from tools.extend_curriculum_bank import check_row
from tools.merge_world_data import _record_from_validity
from tools.repair_context_bank import repair_bank, repair_row, repair_mismatched_bank
from tests.test_bank_extension import verified


class ContextProofTests(unittest.TestCase):
    def test_old_generated_context_zero_solution_fails_and_repaired_one_wins(self):
        spec = json.loads((Path(__file__).parent / "fixtures" /
                           "generator_context_seed628.json").read_text())
        env = Ls20Scenario(generate.build_level(spec), spec["seed"] % 7)
        for action in spec["solution"]:
            result = env.perform(action)
        self.assertFalse(result.won)
        self.assertLess(env.lives(), 3)
        fixed = repair_row(spec)
        self.assertEqual(fixed["optimal_actions"], 48)
        self.assertEqual(fixed["context_optimal_actions"], len(fixed["solution"]))
        self.assertEqual(fixed["context_solution"], fixed["solution"])
        self.assertEqual(fixed["verification_level_index"], 5)
        self.assertFalse(fixed["verification_match_hint"])
        self.assertEqual(fixed, repair_row(spec))

    def test_base_rejects_solvable_but_truncated_search_before_export(self):
        spec = generate.generate_legacy_level(1, 1)
        with patch("pebby.ls20.generate.Oracle") as oracle:
            oracle.return_value.truncated = True
            oracle.return_value.solvable = True
            self.assertIsNone(generate._verify(spec, 0))
            oracle.return_value.solution.assert_not_called()

    def test_hard_base_tier_completes_under_original_search_cap(self):
        for seed in (5, 7, 13):
            spec = generate.generate_legacy_level(seed, 5)
            self.assertFalse(spec["search_truncated"])
            self.assertLess(spec["reachable_states"], 600_000)
            self.assertEqual(spec["context_index"], seed % 7)
            self.assertEqual(len(spec["solution"]), spec["context_optimal_actions"])

    def test_seeded_ties_preserve_exact_distance_and_vary_optimal_routes(self):
        # A generated empty square gives many shortest routes. The independent
        # forward BFS below never consults Oracle's reverse distance table.
        free = {(x, y) for x in range(2, 5) for y in range(2, 5)}
        spec = {"start": (2, 2), "start_triple": [0, 0, 0],
                "goals": [{"cell": (4, 4), "triple": [0, 0, 0]}],
                "walls": sorted({(x, y) for x in range(12) for y in range(12)} - free),
                "cyclers": [], "refills": [], "step_counter": 42,
                "step_cost": 1, "fog": False}
        oracle = Oracle(extract(Ls20Scenario(generate.build_level(spec), 3)))
        queue, seen = deque([(oracle.start, 0)]), {oracle.start}
        while queue:
            state, distance = queue.popleft()
            if state[4] == oracle.full_mask:
                break
            for action in range(4):
                nxt = advance(oracle.layout, state, action, oracle.refills)
                if nxt is not None and nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, distance + 1))
        self.assertEqual(distance, 4)
        self.assertEqual(oracle.optimal_actions, distance)
        reference = Oracle(oracle.layout, engine="reference")
        self.assertEqual(reference.optimal_actions, distance)
        paths = {tuple(oracle.solution(seed=seed)) for seed in range(20)}
        self.assertGreater(len(paths), 1)
        self.assertEqual({len(path) for path in paths}, {distance})
        self.assertEqual(oracle.solution(seed=4), oracle.solution(seed=4))
        self.assertEqual(oracle.solution(), oracle.solution())
        for path in paths:
            env = Ls20Scenario(generate.build_level(spec), 3)
            for action in path:
                result = env.perform(action)
            self.assertTrue(result.won)

    def test_extender_rejects_stale_context_distances_and_missing_proofs(self):
        row = verified((10000, 1))
        for change in ({"context_optimal_actions": 4}, {"context_engine_verified": False},
                       {"context_solution": [4, 4, 4]}, {"verification_level_index": 0},
                       {"generator_version": 2}, {"search_truncated": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                check_row({**row, **change}, 10000, 1)

    def test_merge_never_promotes_status_alone_or_incomplete_search(self):
        for status in ("verified", "accepted", "complete", "completed"):
            record = _record_from_validity({"seed": 10, "status": status,
                                             "context_index": 3})
            self.assertFalse(record["accepted"])
            self.assertNotIn("context_engine_verified", record["proof"])
        summary = {"seed": 10, "context_index": 3, "context_validity": True,
                   "engine_win": True, "replay_lives": 3, "levels_completed": 1}
        self.assertFalse(_record_from_validity(summary)["accepted"])
        self.assertTrue(_record_from_validity({**summary, "search_truncated": False})["accepted"])

    def test_repair_is_copy_only_and_fails_closed(self):
        spec = generate.generate_legacy_level(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            source, target = (Path(directory) / name for name in ("source.jsonl", "copy.jsonl"))
            original = json.dumps(spec) + "\n"
            source.write_text(original)
            report = repair_bank(source, target)
            self.assertTrue(report["published"])
            self.assertEqual(report["reverified"], 1)
            self.assertEqual(source.read_text(), original)
            with self.assertRaisesRegex(ValueError, "new path"):
                repair_bank(source, target)
            failed = Path(directory) / "failed.jsonl"
            report = repair_bank(source, failed, search_limit=1)
            self.assertFalse(report["published"])
            self.assertEqual(len(report["failures"]), 1)
            self.assertFalse(failed.exists())
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["copy.jsonl", "source.jsonl"])

    def test_legacy_collection_replans_old_bank_in_actual_training_context(self):
        from pebby.agent.data import episode_from_spec
        path = Path(__file__).parent / "fixtures" / "generator_context_seed628.json"
        spec = json.loads(path.read_text())
        row = episode_from_spec(spec, 0, random.Random(0), deviations=0, pads=0)
        self.assertTrue(row["completed"])
        self.assertEqual(row["oracle_length"], 48)
        self.assertEqual(int(row["to_go"][0]), 48)

    def test_bank_read_accepts_historical_geometry_but_refuses_unknown_version(self):
        spec = generate.generate_legacy_level(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.jsonl"
            path.write_text(json.dumps({**spec, "generator_version": 2}) + "\n")
            self.assertEqual(load(path)[0]["generator_version"], 2)
            path.write_text(json.dumps({**spec, "generator_version": 999}) + "\n")
            with self.assertRaises(ValueError):
                load(path)

    def test_targeted_repair_preserves_untouched_rows_without_certifying_them(self):
        stale = json.loads((Path(__file__).parent / "fixtures" /
                            "generator_context_seed628.json").read_text())
        untouched = generate.generate_legacy_level(1, 1)
        untouched.pop("context_engine_verified")
        with tempfile.TemporaryDirectory() as directory:
            source, target = (Path(directory) / name for name in ("old.jsonl", "fixed.jsonl"))
            first = json.dumps(untouched, separators=(", ", ": ")) + "\n"
            original = first + json.dumps(stale) + "\n"
            source.write_text(original)
            report = repair_mismatched_bank(source, target)
            self.assertTrue(report["published"])
            self.assertEqual(report["repaired"], 1)
            self.assertEqual(report["old_routes_failed_replay"], 1)
            self.assertEqual(report["untouched_rows"], 1)
            self.assertEqual(target.read_text().splitlines(keepends=True)[0], first)
            self.assertEqual(source.read_text(), original)
            rows = load(target)
            self.assertNotIn("context_engine_verified", rows[0])
            self.assertEqual(rows[1]["optimal_actions"], 48)
            with self.assertRaises(ValueError):
                repair_mismatched_bank(source, target)


if __name__ == "__main__":
    unittest.main()
