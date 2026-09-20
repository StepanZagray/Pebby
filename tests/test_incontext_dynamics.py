"""Tests for the in-context transition predictor on LS20 rule-variant games.

All data here comes from a toy rule simulator written below that emits the exact NPZ data
contract of ``tools/collect_variant_games.py`` (random action permutation, walls and the
grid edge block, cycler tiles advance the glyph, a short step budget per life).  It is not LS20
and reads nothing from the engine.  Runs on the CPU in well under 90 s.
"""
from contextlib import redirect_stdout
import io
import itertools
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from pebby.agent import incontext_dynamics as icd
from pebby.agent import variant_metrics as vm
from tools import train_incontext_dynamics as cli

# Tiny models on tiny batches: more intra-op threads only add contention on a shared machine.
torch.set_num_threads(min(4, torch.get_num_threads()))

PERMUTATIONS = list(itertools.permutations(range(4)))
ENGINE_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))  # up, down, left, right in (dx, dy)
GRID = 12
BUDGET = 24  # toy step budget per life; shorter than the real 42 so resets occur


# ------------------------------------------------------------ toy simulator (NPZ contract)
def make_level(rng):
    tiles = np.zeros((GRID, GRID), dtype=np.int8)
    tiles[rng.random((GRID, GRID)) < 0.14] = vm.TILE_WALL
    for cls, count in ((vm.TILE_CYCLER_SHAPE, 4), (vm.TILE_CYCLER_COLOR, 4), (vm.TILE_CYCLER_ROTATION, 4),
                       (vm.TILE_REFILL, 1), (vm.TILE_GOAL, 1), (vm.TILE_RAIL, 2), (vm.TILE_LAUNCHER, 1)):
        for _ in range(count):
            x, y = rng.integers(0, GRID, size=2)
            tiles[y, x] = cls
    while True:
        x, y = rng.integers(0, GRID, size=2)
        if tiles[y, x] == vm.TILE_FREE:
            break
    return {"tiles": tiles, "start": (int(x), int(y)),
            "glyph": [int(rng.integers(0, 6)), int(rng.integers(0, 4)), int(rng.integers(0, 4))],
            "goals": int(rng.integers(0, 1 << 6))}


def neighbours_of(tiles, x, y):
    out = []
    for dx, dy in ENGINE_DELTAS:
        nx, ny = x + dx, y + dy
        out.append(vm.TILE_OUTSIDE if not (0 <= nx < GRID and 0 <= ny < GRID) else int(tiles[ny, nx]))
    return out


def simulate_game(seed, variant_id, levels=7, steps_per_level=(15, 35), game_id=None):
    rng = np.random.default_rng(seed)
    perm = PERMUTATIONS[variant_id]
    rows = {key: [] for key in (
        "level_index", "step_in_level", "before_player_x", "before_player_y", "before_shape", "before_color",
        "before_rotation", "before_steps_left", "before_lives", "before_goals_mask", "neighbours", "agent_action",
        "engine_action", "after_player_x", "after_player_y", "after_shape", "after_color", "after_rotation",
        "after_steps_left", "after_lives", "after_goals_mask", "life_lost", "level_changed", "reset", "won",
        "finished")}
    level = make_level(rng)
    lives = 3
    for level_index in range(levels):
        x, y = level["start"]
        shape, color, rotation = level["glyph"]
        steps_left = BUDGET
        length = int(rng.integers(steps_per_level[0], steps_per_level[1] + 1))
        for step in range(length):
            action = int(rng.integers(0, 4))
            engine = perm[action]
            before = (x, y, shape, color, rotation, steps_left, lives)
            before_neighbours = neighbours_of(level["tiles"], x, y)
            dx, dy = ENGINE_DELTAS[engine]
            nx, ny = x + dx, y + dy
            tile = vm.TILE_OUTSIDE if not (0 <= nx < GRID and 0 <= ny < GRID) else int(level["tiles"][ny, nx])
            if tile not in vm.BLOCKING_TILES:
                x, y = nx, ny
                if tile == vm.TILE_CYCLER_SHAPE:
                    shape = (shape + 1) % 6
                elif tile == vm.TILE_CYCLER_COLOR:
                    color = (color + 1) % 4
                elif tile == vm.TILE_CYCLER_ROTATION:
                    rotation = (rotation + 1) % 4
            steps_left -= 1
            life_lost = reset = False
            if steps_left == 0:
                life_lost = reset = True
                lives = lives - 1 if lives > 1 else 3
                x, y = level["start"]
                shape, color, rotation = level["glyph"]
                steps_left = BUDGET
            last_step = step == length - 1
            level_changed = last_step and level_index < levels - 1
            if level_changed:
                level = make_level(rng)
                x, y = level["start"]
                shape, color, rotation = level["glyph"]
                steps_left = BUDGET
            rows["level_index"].append(level_index)
            rows["step_in_level"].append(step)
            for key, value in zip(("before_player_x", "before_player_y", "before_shape", "before_color",
                                   "before_rotation", "before_steps_left", "before_lives"), before):
                rows[key].append(value)
            rows["before_goals_mask"].append(level["goals"])
            rows["neighbours"].append(before_neighbours)
            rows["agent_action"].append(action)
            rows["engine_action"].append(engine)
            for key, value in zip(("after_player_x", "after_player_y", "after_shape", "after_color",
                                   "after_rotation", "after_steps_left", "after_lives"),
                                  (x, y, shape, color, rotation, steps_left, lives)):
                rows[key].append(value)
            rows["after_goals_mask"].append(level["goals"])
            rows["life_lost"].append(life_lost)
            rows["level_changed"].append(level_changed)
            rows["reset"].append(reset)
            rows["won"].append(last_step and level_index == levels - 1)
            rows["finished"].append(last_step and level_index == levels - 1)
    return rows_to_arrays(rows, variant_id, perm, game_id or f"toy-{seed}-{variant_id}")


def rows_to_arrays(rows, variant_id, perm, game_id):
    dtypes = {"level_index": np.int8, "step_in_level": np.int16, "before_goals_mask": np.int16,
              "after_goals_mask": np.int16, "neighbours": np.int8, "agent_action": np.int8,
              "engine_action": np.int8}
    arrays = {}
    for key, values in rows.items():
        if key in ("life_lost", "level_changed", "reset", "won", "finished"):
            arrays[key] = np.asarray(values, dtype=bool)
        else:
            arrays[key] = np.asarray(values, dtype=dtypes.get(key, np.int16))
    arrays["game_id"] = np.asarray(game_id)
    arrays["variant_id"] = np.asarray(variant_id, dtype=np.int16)
    arrays["action_map"] = np.asarray(perm, dtype=np.int8)
    arrays["tier_seeds"] = np.zeros(7, dtype=np.int64)
    arrays["truncated"] = np.asarray(False)
    arrays["levels_completed"] = np.asarray(7, dtype=np.int8)
    return arrays


class ToySimulatorTest(unittest.TestCase):
    def test_contract_shapes(self):
        arrays = simulate_game(0, 5)
        steps = arrays["agent_action"].shape[0]
        self.assertEqual(arrays["neighbours"].shape, (steps, 4))
        self.assertEqual(arrays["action_map"].tolist(), list(PERMUTATIONS[5]))
        self.assertGreater(steps, 7 * 15 - 1)
        self.assertEqual(int(arrays["level_changed"].sum()), 6)
        self.assertTrue(arrays["life_lost"].any())


# ------------------------------------------------------------ featurize / model
class FeaturizeTest(unittest.TestCase):
    def test_shapes_and_no_label_leakage(self):
        arrays = simulate_game(1, 7)
        steps = arrays["agent_action"].shape[0]
        features = icd.featurize(arrays)
        self.assertEqual(tuple(features["position"].shape), (steps, 2))
        self.assertEqual(tuple(features["glyph"].shape), (steps, 3))
        self.assertEqual(tuple(features["neighbours"].shape), (steps, 4))
        self.assertEqual(tuple(features["scalars"].shape), (steps, 1 + icd.GOAL_BITS))
        for key in ("lives", "level", "boundary", "action", "prev_action", "prev_movement"):
            self.assertEqual(tuple(features[key].shape), (steps,), key)
        for name in vm.FIELDS:
            self.assertEqual(tuple(features["targets"][name].shape), (steps,))
            self.assertEqual(tuple(features["masks"][name].shape), (steps,))
            self.assertLess(int(features["targets"][name].max()), vm.FIELD_SIZES[name])
        self.assertEqual(int(features["prev_action"][0]), icd.PREV_ACTION_NONE)
        self.assertEqual(int(features["prev_movement"][0]), icd.PREV_MOVEMENT_UNKNOWN)
        # Leakage check: scrambling the labels must not change any input tensor.
        scrambled = dict(arrays)
        scrambled["engine_action"] = (arrays["engine_action"] + 1) % 4
        scrambled["action_map"] = np.asarray([3, 2, 1, 0], dtype=np.int8)
        scrambled["variant_id"] = np.asarray(23, dtype=np.int16)
        other = icd.featurize(scrambled)
        for key in icd.INPUT_KEYS:
            self.assertTrue(torch.equal(features[key], other[key]), key)
        for name in vm.FIELDS:
            self.assertTrue(torch.equal(features["targets"][name], other["targets"][name]))
        # Level boundaries and resets hide the previous-step outcome.
        boundary = features["boundary"].bool()
        self.assertTrue((features["prev_movement"][boundary] == icd.PREV_MOVEMENT_UNKNOWN).all())
        self.assertFalse(features["masks"]["movement"][arrays["level_changed"] | arrays["life_lost"]].any())
        self.assertTrue(features["masks"]["life_lost"].all())

    def test_movement_targets(self):
        arrays = simulate_game(2, 0)  # identity permutation: engine dir == agent action
        targets = vm.targets_from_arrays(arrays)
        mask = vm.transition_mask(arrays)
        moved = (targets["movement"] > 0) & mask
        self.assertTrue((targets["movement"][moved] - 1 == arrays["agent_action"][moved]).all())


class ModelTest(unittest.TestCase):
    def test_parameter_budget(self):
        model = icd.InContextDynamics()
        self.assertLess(model.parameter_count(), 300_000)
        self.assertGreater(model.parameter_count(), 50_000)
        baseline = icd.MemorylessBaseline()
        self.assertLess(baseline.parameter_count(), 300_000)
        self.assertFalse(baseline.embed.include_previous)

    def test_causal_mask(self):
        torch.manual_seed(0)
        arrays = simulate_game(3, 11)
        features = icd.featurize(arrays)
        window = icd.slice_features(features, 0, 80)
        model = icd.InContextDynamics(max_steps=128).eval()
        with torch.no_grad():
            base = model(icd.collate([window]))
            changed = icd.slice_features(features, 0, 80)
            changed["action"] = changed["action"].clone()
            changed["action"][50:] = (changed["action"][50:] + 1) % 4
            changed["position"] = changed["position"].clone()
            changed["position"][50:] = (changed["position"][50:] + 3) % 12
            after = model(icd.collate([changed]))
        for name in vm.FIELDS:
            self.assertTrue(torch.allclose(base[name][0, :50], after[name][0, :50], atol=1e-5), name)
            self.assertFalse(torch.allclose(base[name][0, 50:], after[name][0, 50:], atol=1e-5), name)

    def test_padding_does_not_change_predictions(self):
        torch.manual_seed(0)
        a = icd.slice_features(icd.featurize(simulate_game(4, 3)), 0, 40)
        b = icd.slice_features(icd.featurize(simulate_game(5, 9)), 0, 70)
        model = icd.InContextDynamics(max_steps=128).eval()
        with torch.no_grad():
            alone = model(icd.collate([a]))["movement"][0]
            padded = model(icd.collate([a, b]))["movement"][0, :40]
        self.assertTrue(torch.allclose(alone, padded, atol=1e-5))

    def test_windows_and_streaming(self):
        self.assertEqual(icd._windows_for(50, 64, None), [(0, 50, 0)])
        windows = icd._windows_for(150, 64, 16)
        covered = []
        for start, end, keep in windows:
            self.assertLessEqual(end - start, 64)
            covered.extend(range(start + keep, end))
        self.assertEqual(covered, list(range(150)))
        arrays = simulate_game(6, 2)
        model = icd.InContextDynamics(max_steps=64).eval()
        report = icd.evaluate_game(model, arrays, max_steps=64, stride=16)
        self.assertEqual(report["steps"], arrays["agent_action"].shape[0])


# ------------------------------------------------------------ metrics
class MetricsTest(unittest.TestCase):
    def test_steps_to_stable(self):
        self.assertIsNone(vm.steps_to_stable(np.zeros(100, dtype=bool)))
        self.assertEqual(vm.steps_to_stable(np.ones(100, dtype=bool)), 0)
        correct = np.ones(100, dtype=bool)
        correct[:10] = False
        self.assertEqual(vm.steps_to_stable(correct), 10)
        correct = np.ones(100, dtype=bool)
        correct[15] = False
        correct[45] = False
        self.assertEqual(vm.steps_to_stable(correct), 16)
        self.assertIsNone(vm.steps_to_stable(np.ones(19, dtype=bool)))
        # Unscored steps are skipped, not counted as wrong or as progress.
        correct = np.ones(30, dtype=bool)
        correct[5] = False
        scored = np.ones(30, dtype=bool)
        scored[5] = False
        self.assertEqual(vm.steps_to_stable(correct, scored), 0)
        scored[:] = True
        scored[10:20] = False
        self.assertEqual(vm.steps_to_stable(np.ones(30, dtype=bool), scored), 0)
        self.assertIsNone(vm.steps_to_stable(np.ones(29, dtype=bool), scored[:29]))

    def test_bin_edges(self):
        edges = vm.bin_edges(230)
        self.assertEqual(edges[0], (0, 10))
        self.assertEqual(edges[19], (190, 200))
        self.assertEqual(edges[20], (200, 250))
        self.assertEqual(len(edges), 21)
        self.assertEqual(vm.bin_edges(5), [(0, 10)])

    def test_score_predictions_and_aggregate_format(self):
        arrays = simulate_game(7, 13)
        perfect = vm.targets_from_arrays(arrays)
        report = vm.score_predictions(perfect, arrays)
        steps = arrays["agent_action"].shape[0]
        self.assertEqual(report["steps"], steps)
        self.assertEqual(report["joint"]["accuracy"], 1.0)
        self.assertEqual(report["movement_accuracy"], 1.0)
        self.assertEqual(report["steps_to_stable"], 0)
        self.assertEqual(report["movement_steps_to_stable"], 0)
        self.assertEqual(len(report["curve"]["bins"]), len(vm.bin_edges(steps)))
        self.assertEqual(len(report["curve"]["joint"]), len(report["curve"]["bins"]))
        for name in vm.FIELDS:
            self.assertEqual(report["fields"][name]["accuracy"], 1.0)
        self.assertEqual(report["fields"]["movement"]["count"], report["scored_steps"])
        self.assertEqual(report["fields"]["life_lost"]["count"], steps)
        # Break movement on the first 30 scored steps: joint and movement stable move to >= 30.
        wrong = {k: v.copy() for k, v in perfect.items()}
        wrong["movement"][:30] = (wrong["movement"][:30] + 1) % 5
        report = vm.score_predictions(wrong, arrays)
        self.assertLess(report["movement_accuracy"], 1.0)
        self.assertEqual(report["fields"]["shape"]["accuracy"], 1.0)
        scored = vm.transition_mask(arrays)
        expected = int(np.flatnonzero(scored[30:])[0]) + 30   # first scored movement step >= 30
        self.assertEqual(report["movement_steps_to_stable"], expected)
        self.assertGreaterEqual(report["steps_to_stable"], 29)  # a masked step may open the joint run
        self.assertLessEqual(report["steps_to_stable"], expected)
        self.assertEqual(report["curve"]["movement"][0]["correct"], 0)
        # Report must be JSON serialisable and aggregate must pool by bin.
        json.dumps(report)
        other = vm.score_predictions(vm.targets_from_arrays(simulate_game(8, 1)), simulate_game(8, 1))
        agg = vm.aggregate([report, other])
        json.dumps(agg)
        self.assertEqual(agg["games"], 2)
        self.assertEqual(agg["steps_to_stable"]["reached"], 2)
        self.assertEqual(agg["steps_to_stable"]["values"][1], 0)
        self.assertEqual(len(agg["curve"]["joint"]), max(len(report["curve"]["joint"]), len(other["curve"]["joint"])))
        self.assertEqual(agg["curve"]["movement"][0]["games"], 2)
        self.assertEqual(agg["curve"]["movement"][0]["correct"], other["curve"]["movement"][0]["correct"])
        self.assertEqual(agg["fields"]["movement"]["count"],
                         report["fields"]["movement"]["count"] + other["fields"]["movement"]["count"])
        self.assertIn("mean_game_accuracy", agg["joint"])

    def test_identity_baseline(self):
        identity = simulate_game(9, 0)
        report = vm.score_predictions(vm.identity_baseline_predictions(identity), identity)
        self.assertEqual(report["movement_accuracy"], 1.0)
        for name in ("shape", "color", "rotation"):
            self.assertEqual(report["fields"][name]["accuracy"], 1.0)
        reversed_map = PERMUTATIONS.index((1, 0, 3, 2))
        swapped = simulate_game(9, reversed_map)
        report = vm.score_predictions(vm.identity_baseline_predictions(swapped), swapped)
        self.assertLess(report["movement_accuracy"], 0.6)


# ------------------------------------------------------------ learning
def games_for(variants, per_variant, seed_base):
    games = []
    for variant in variants:
        for k in range(per_variant):
            games.append(simulate_game(seed_base + 100 * variant + k, variant))
    return games


def fit(model, games, updates, batch_size=16, lr=3e-3, max_steps=256, seed=0):
    rng = np.random.default_rng(seed)
    features = [icd.featurize(g) for g in games]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cli.lr_lambda(30, updates))
    model.train()
    for _ in range(updates):
        windows = []
        for index in rng.integers(0, len(features), size=batch_size):
            f = features[int(index)]
            start, end = icd.sample_window(int(f["action"].shape[0]), max_steps, rng)
            windows.append(icd.slice_features(f, start, end))
        batch = icd.collate(windows)
        loss, _ = icd.prediction_loss(model(batch), batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    model.eval()
    return model


def held_out_movement(model, games):
    reports = [icd.evaluate_game(model, g, max_steps=256) for g in games]
    return vm.aggregate(reports)


class LearningTest(unittest.TestCase):
    TRAIN_VARIANTS = [v for v in range(24) if v % 4 != 3]   # 18 permutations
    TEST_VARIANTS = [v for v in range(24) if v % 4 == 3]    # 6 held-out permutations

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.train_games = games_for(cls.TRAIN_VARIANTS, 3, 1000)
        cls.test_games = games_for(cls.TEST_VARIANTS, 2, 5000)
        cls.identity = vm.aggregate(vm.score_predictions(vm.identity_baseline_predictions(g), g)
                                    for g in cls.test_games)

    def test_transformer_infers_held_out_permutations(self):
        torch.manual_seed(0)
        started = time.time()
        model = fit(icd.InContextDynamics(d_model=64, layers=2, heads=4, max_steps=256),
                    self.train_games, updates=300)
        summary = held_out_movement(model, self.test_games)
        elapsed = time.time() - started
        print(f"\ntransformer: held-out movement {summary['movement_accuracy']:.3f} joint "
              f"{summary['joint']['accuracy']:.3f} identity {self.identity['movement_accuracy']:.3f} "
              f"stable {summary['steps_to_stable']['reached']}/{summary['steps_to_stable']['games']} "
              f"({elapsed:.1f}s, {model.parameter_count():,} params)")
        self.assertGreater(summary["movement_accuracy"], 0.9)
        self.assertGreater(summary["movement_accuracy"], self.identity["movement_accuracy"] + 0.2)
        # Accuracy must rise with step index: the last fine bins beat the first bin.
        curve = [c["accuracy"] for c in summary["curve"]["movement"][:10] if c["count"]]
        self.assertGreater(np.mean(curve[5:]), curve[0])

    def test_memoryless_stays_near_identity_baseline(self):
        torch.manual_seed(0)
        model = fit(icd.MemorylessBaseline(d_model=64, layers=2), self.train_games, updates=300)
        summary = held_out_movement(model, self.test_games)
        print(f"\nmemoryless: held-out movement {summary['movement_accuracy']:.3f} identity "
              f"{self.identity['movement_accuracy']:.3f}")
        self.assertLess(summary["movement_accuracy"], 0.7)
        self.assertLess(abs(summary["movement_accuracy"] - self.identity["movement_accuracy"]), 0.3)


# ------------------------------------------------------------ CLI
class CliTest(unittest.TestCase):
    def test_smoke_run_writes_report_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "games").mkdir()
            for variant in (0, 1, 2, 5, 9):
                for k in range(2):
                    arrays = simulate_game(200 + 10 * variant + k, variant, game_id=f"g{variant}-{k}")
                    np.savez(root / "games" / f"g{variant}-{k}.npz", **arrays)
            out = root / "run"
            with redirect_stdout(io.StringIO()):
                report = cli.main(["--data-dir", str(root), "--train-variants", "0", "1", "2",
                                   "--test-variants", "5", "9", "--out-dir", str(out), "--updates", "6",
                                   "--eval-every", "3", "--batch-size", "4", "--d-model", "32", "--layers", "1",
                                   "--max-steps", "128", "--device", "cpu", "--seed", "1"])
            self.assertTrue((out / "report.json").exists())
            self.assertTrue((out / "best.pt").exists())
            self.assertTrue((out / "final.pt").exists())
            saved = json.loads((out / "report.json").read_text())
            self.assertTrue(saved["finished"])
            self.assertEqual(len(saved["evaluations"]), 2)
            self.assertEqual(saved["splits"]["held_out"]["variants"], [5, 9])
            self.assertEqual(saved["splits"]["held_in"]["variants"] and True, True)
            self.assertEqual(len(saved["splits"]["train"]["games"]) + len(saved["splits"]["held_in"]["games"]), 6)
            self.assertIn("held_out", saved["identity_baseline"])
            self.assertIn("curve", saved["evaluations"][-1]["held_out"])
            self.assertIn("steps_to_stable", saved["evaluations"][-1]["held_out"])
            self.assertEqual(saved["best"]["split"], "held_out")
            self.assertEqual(saved["config"]["model"], "InContextDynamics")
            model, payload = icd.load_checkpoint(out / "final.pt")
            self.assertEqual(payload["update"], 6)
            self.assertIsInstance(model, icd.InContextDynamics)
            with redirect_stdout(io.StringIO()):
                report = cli.main(["--data-dir", str(root), "--test-variants", "5", "9", "--out-dir",
                                   str(root / "mem"), "--updates", "3", "--eval-every", "3", "--batch-size", "4",
                                   "--d-model", "32", "--layers", "1", "--device", "cpu", "--history", "0"])
            self.assertEqual(report["config"]["model"], "MemorylessBaseline")
            self.assertEqual(report["splits"]["train"]["variants"] + report["splits"]["held_in"]["variants"]
                             and sorted(set(report["splits"]["train"]["variants"]) |
                                        set(report["splits"]["held_in"]["variants"])), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
