"""Explicit-inference predictor for the LS20 rule-variant experiment.

Covers: the local-rule predictor identifies the true permutation within a
few informative steps and its predictions are then 100 percent correct on
non-abstained steps; a permutation that never moves (all walls) keeps every
hypothesis alive; prior weighting changes the plurality tie-break but never
changes elimination; no leakage (scrambling engine_action/action_map/variant_id
never changes the predictor's behaviour, because it never reads them). The
engine rule model gets one real-game test, skipped with a clear message
unless ``pebby.variants`` is importable (it is not, as of this test's
introduction -- see the module docstring in hypothesis_dynamics.py for why).

All games here are synthetic, built with a toy simulator in this file that
implements the exact row contract documented in hypothesis_dynamics.py:
neighbours in engine order (up/down/left/right), agent_action 0..3, and a
fixed permutation mapping agent_action -> engine direction for the whole game.
CPU only, sub-second per test.
"""

import random
import time
import unittest

import numpy as np

from pebby.agent.hypothesis_dynamics import (
    DEFAULT_BANK_PATH,
    PERMUTATIONS,
    NEI_CYCLER_COLOR,
    NEI_CYCLER_ROTATION,
    NEI_CYCLER_SHAPE,
    NEI_FREE,
    NEI_WALL,
    PUBLIC_STEP_FIELDS,
    EngineRuleModel,
    HypothesisPredictor,
    LocalRuleModel,
    movement_class,
    run_game,
    score_predictions,
)


# --- a toy simulator that satisfies the exact row contract --------------------

class ToySim:
    """A minimal grid world with cyclers and walls, playable in one dimension
    per step (only one of the four neighbour tiles is ever anything but free,
    which is all a step needs to be informative or not for a single field).

    ``variant_id`` fixes the whole game's action permutation. ``perm[agent_action]``
    is the engine direction (0..3, up/down/left/right) actually applied.
    """

    SHAPE_COUNT, COLOR_COUNT, ROTATION_COUNT = 6, 4, 4

    def __init__(self, variant_id, neighbour_plan, rng):
        self.perm = PERMUTATIONS[variant_id]
        self.neighbour_plan = neighbour_plan  # list of 4-tuples, one per step
        self.rng = rng
        self.x, self.y = 0, 0
        self.shape, self.color, self.rotation = 0, 0, 0
        self.lives = 3

    def step(self, step_index, agent_action):
        neighbours = self.neighbour_plan[step_index]
        direction = self.perm[agent_action]
        tile = neighbours[direction]
        before = dict(x=self.x, y=self.y, shape=self.shape, color=self.color, rotation=self.rotation)
        moved = tile not in (NEI_WALL,)
        if moved:
            dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[direction]
            self.x += dx
            self.y += dy
        if tile == NEI_CYCLER_SHAPE:
            self.shape = (self.shape + 1) % self.SHAPE_COUNT
        elif tile == NEI_CYCLER_COLOR:
            self.color = (self.color + 1) % self.COLOR_COUNT
        elif tile == NEI_CYCLER_ROTATION:
            self.rotation = (self.rotation + 1) % self.ROTATION_COUNT
        after = dict(x=self.x, y=self.y, shape=self.shape, color=self.color, rotation=self.rotation)
        return neighbours, before, after


def build_game(variant_id, steps, rng, all_walls=False):
    """One synthetic game's arrays, in the exact NPZ row contract (minus the
    label columns the predictor never sees; those are appended separately so
    tests can exercise both the scrambled and unscrambled dict).
    """
    if all_walls:
        neighbour_plan = [(NEI_WALL, NEI_WALL, NEI_WALL, NEI_WALL) for _ in range(steps)]
    else:
        cycler_tiles = (NEI_CYCLER_SHAPE, NEI_CYCLER_COLOR, NEI_CYCLER_ROTATION, NEI_FREE, NEI_WALL)
        neighbour_plan = [tuple(rng.choice(cycler_tiles) for _ in range(4)) for _ in range(steps)]
    sim = ToySim(variant_id, neighbour_plan, rng)
    agent_actions = [rng.randrange(4) for _ in range(steps)]
    rows = {
        "level_index": [], "step_in_level": [],
        "before_player_x": [], "before_player_y": [], "before_shape": [], "before_color": [],
        "before_rotation": [], "before_steps_left": [], "before_lives": [], "before_goals_mask": [],
        "neighbours": [], "agent_action": [], "engine_action": [],
        "after_player_x": [], "after_player_y": [], "after_shape": [], "after_color": [],
        "after_rotation": [], "after_steps_left": [], "after_lives": [], "after_goals_mask": [],
        "life_lost": [], "level_changed": [], "reset": [], "won": [], "finished": [],
    }
    for t in range(steps):
        neighbours, before, after = sim.step(t, agent_actions[t])
        rows["level_index"].append(0)
        rows["step_in_level"].append(t)
        rows["before_player_x"].append(before["x"])
        rows["before_player_y"].append(before["y"])
        rows["before_shape"].append(before["shape"])
        rows["before_color"].append(before["color"])
        rows["before_rotation"].append(before["rotation"])
        rows["before_steps_left"].append(steps - t)
        rows["before_lives"].append(3)
        rows["before_goals_mask"].append(0)
        rows["neighbours"].append(neighbours)
        rows["agent_action"].append(agent_actions[t])
        rows["engine_action"].append(sim.perm[agent_actions[t]] + 1)
        rows["after_player_x"].append(after["x"])
        rows["after_player_y"].append(after["y"])
        rows["after_shape"].append(after["shape"])
        rows["after_color"].append(after["color"])
        rows["after_rotation"].append(after["rotation"])
        rows["after_steps_left"].append(steps - t - 1)
        rows["after_lives"].append(3)
        rows["after_goals_mask"].append(0)
        rows["life_lost"].append(False)
        rows["level_changed"].append(False)
        rows["reset"].append(False)
        rows["won"].append(False)
        rows["finished"].append(False)
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    arrays["neighbours"] = np.asarray(rows["neighbours"], dtype=np.int8)
    arrays["variant_id"] = np.asarray(variant_id)
    arrays["action_map"] = np.asarray(list(sim.perm))
    arrays["tier_seeds"] = np.zeros(7, dtype=np.int64)
    return arrays


class LocalRuleIdentificationTests(unittest.TestCase):
    def test_identifies_true_permutation_and_then_predicts_perfectly(self):
        rng = random.Random(0)
        arrays = build_game(variant_id=7, steps=60, rng=rng)
        predictor = HypothesisPredictor(LocalRuleModel())
        pred_arrays, diagnostics = run_game(predictor, arrays)

        self.assertIsNotNone(diagnostics["steps_until_unique"])
        self.assertLess(diagnostics["steps_until_unique"], 15)
        self.assertEqual(diagnostics["final_surviving"], 1)
        self.assertIn(7, predictor.surviving)

        # Once unique, every non-abstained prediction from then on is exact.
        start = diagnostics["steps_until_unique"]
        for t in range(start, len(arrays["agent_action"])):
            if pred_arrays["abstained"][t]:
                continue
            self.assertEqual(int(pred_arrays["movement"][t]),
                              _observed_movement(arrays, t), msg=f"step {t}")
            self.assertEqual(int(pred_arrays["shape"][t]), int(arrays["after_shape"][t]))
            self.assertEqual(int(pred_arrays["color"][t]), int(arrays["after_color"][t]))
            self.assertEqual(int(pred_arrays["rotation"][t]), int(arrays["after_rotation"][t]))

        report = score_predictions(pred_arrays, arrays)
        scored_tail = report["scored_steps"]
        self.assertGreater(scored_tail, 0)

    def test_all_walls_keeps_every_hypothesis_alive(self):
        rng = random.Random(1)
        arrays = build_game(variant_id=3, steps=40, rng=rng, all_walls=True)
        predictor = HypothesisPredictor(LocalRuleModel())
        _, diagnostics = run_game(predictor, arrays)
        self.assertEqual(diagnostics["final_surviving"], 24)
        self.assertIsNone(diagnostics["steps_until_unique"])
        self.assertEqual(len(predictor.surviving), 24)

    def test_prior_changes_tie_break_not_elimination(self):
        # Two hypotheses disagree about direction for agent_action=0: hA maps
        # it to "up" (free -> moves), hC maps it to "down" (a wall -> stays).
        # With exactly one surviving hypothesis per outcome, a uniform prior
        # is a genuine tie; skewing the prior toward one hypothesis must flip
        # the plurality prediction to that hypothesis's outcome, deterministically.
        h_up = min(h for h, perm in enumerate(PERMUTATIONS) if perm[0] == 0)
        h_down = min(h for h, perm in enumerate(PERMUTATIONS) if perm[0] == 1)
        step_fields = {
            "agent_action": 0,
            "neighbours": np.array([NEI_FREE, NEI_WALL, NEI_FREE, NEI_FREE]),
            "before_shape": 0, "before_color": 0, "before_rotation": 0,
        }

        uniform = HypothesisPredictor(LocalRuleModel())
        uniform.surviving = {h_up, h_down}
        uniform_pred = uniform.predict(step_fields)
        loser = h_down if uniform_pred["movement"] == 1 else h_up  # whichever the tie did NOT favour
        self.assertIn(uniform_pred["movement"], (0, 1))

        skewed_prior = np.ones(24)
        skewed_prior[loser] = 1000.0
        skewed = HypothesisPredictor(LocalRuleModel(), prior=skewed_prior)
        skewed.surviving = {h_up, h_down}
        skewed_pred = skewed.predict(step_fields)

        loser_movement = 1 if loser == h_up else 0
        self.assertEqual(skewed_pred["movement"], loser_movement)
        self.assertNotEqual(skewed_pred["movement"], uniform_pred["movement"],
                             "skewing the prior toward the tie's loser must flip the plurality winner")

        # Elimination never consults the prior: given the same observed
        # after-state (consistent with "up"/moved), both predictors eliminate
        # the same hypothesis regardless of how skewed their prior was.
        after_fields = dict(step_fields)
        after_fields.update({
            "before_player_x": 0, "before_player_y": 0, "after_player_x": 0, "after_player_y": -1,
            "after_shape": 0, "after_color": 0, "after_rotation": 0, "life_lost": False,
        })
        uniform.observe(after_fields)
        skewed.observe(after_fields)
        self.assertEqual(uniform.surviving, skewed.surviving)
        self.assertEqual(uniform.surviving, {h_up})

    def test_no_leakage_from_scrambled_labels(self):
        rng = random.Random(3)
        arrays = build_game(variant_id=11, steps=25, rng=rng)

        baseline = HypothesisPredictor(LocalRuleModel())
        baseline_pred, baseline_diag = run_game(baseline, dict(arrays))

        scrambled = dict(arrays)
        scramble_rng = random.Random(99)
        scrambled["engine_action"] = np.array(
            [scramble_rng.randrange(1, 5) for _ in arrays["engine_action"]])
        scrambled["action_map"] = np.array([scramble_rng.randrange(4) for _ in arrays["action_map"]])
        scrambled["variant_id"] = np.asarray(scramble_rng.randrange(24))

        scrambled_predictor = HypothesisPredictor(LocalRuleModel())
        scrambled_pred, scrambled_diag = run_game(scrambled_predictor, scrambled)

        for key in ("movement", "shape", "color", "rotation", "surviving", "abstained"):
            np.testing.assert_array_equal(baseline_pred[key], scrambled_pred[key])
        self.assertEqual(baseline_diag["steps_until_unique"], scrambled_diag["steps_until_unique"])
        self.assertEqual(set(baseline.surviving), set(scrambled_predictor.surviving))


def _observed_movement(arrays, t):
    from pebby.agent.hypothesis_dynamics import movement_class
    return movement_class(arrays["before_player_x"][t], arrays["before_player_y"][t],
                           arrays["after_player_x"][t], arrays["after_player_y"][t])


def _play_real_variant_game(V, variant_id, tier_seeds, tiers, max_steps, seed):
    """Drive one real, engine-generated game with a fixed variant, recording
    exactly the public per-step fields (plus the tier_seeds/tiers scalars a
    predictor's reset_game is allowed to see). Mirrors
    tools/collect_variant_games.py's per-step recording, minus its
    oracle-guided policy (plain uniform-random actions here, since this test
    only needs real engine dynamics, not a winning trajectory).
    """
    rng = random.Random(seed)
    specs = V.game_specs(str(DEFAULT_BANK_PATH), tier_seeds, tiers=tiers)
    game = V.VariantGame(specs, variant_id)
    columns = {field: [] for field in PUBLIC_STEP_FIELDS}
    per_level_actions = [0] * game.level_count
    steps = 0
    while not game.finished and steps < max_steps:
        level = game.level_index
        before = game.state()
        neighbours = game.neighbours()
        action = rng.randrange(4)
        after, info = game.step(action)
        per_level_actions[level] += 1
        columns["level_index"].append(level)
        columns["step_in_level"].append(per_level_actions[level])
        for field in ("player_x", "player_y", "shape", "color", "rotation", "steps_left", "lives", "goals_mask"):
            columns[f"before_{field}"].append(before[field])
            columns[f"after_{field}"].append(after[field])
        columns["neighbours"].append(neighbours)
        columns["agent_action"].append(action)
        for flag in ("life_lost", "level_changed", "reset", "won", "finished"):
            columns[flag].append(info[flag])
        steps += 1
    arrays = {key: np.asarray(values) for key, values in columns.items()}
    arrays["neighbours"] = np.asarray(columns["neighbours"], dtype=np.int8).reshape(-1, 4)
    arrays["tier_seeds"] = np.asarray(tier_seeds, dtype=np.int64)
    arrays["tiers"] = np.asarray(tiers, dtype=np.int8)
    return arrays


class EngineRuleModelRealGameTests(unittest.TestCase):
    def test_real_two_level_game(self):
        try:
            from pebby import variants as V
        except ImportError:
            self.skipTest(
                "pebby.variants is not importable yet (it is being written concurrently "
                "by another agent); the engine-rule-model real-game test is skipped until "
                "it lands, per this module's test contract.")

        # Two real, bank-generated levels (tiers 1 and 2), a non-identity true
        # variant, and a uniform-random policy -- enough real engine dynamics
        # (including at least one life loss, a genuine teleport step) to prove
        # elimination and convergence on the actual game, not just the toy sim.
        tier_seeds = [311200, 300384]
        tiers = [1, 2]
        true_variant = 7
        arrays = _play_real_variant_game(V, true_variant, tier_seeds, tiers, max_steps=60, seed=1234)
        self.assertGreater(len(arrays["agent_action"]), 20, "test needs enough real steps to be meaningful")
        self.assertGreater(int(arrays["life_lost"].sum()), 0,
                            "test needs at least one real teleport (life-loss) step")

        predictor = HypothesisPredictor(EngineRuleModel())
        started = time.perf_counter()
        pred_arrays, diagnostics = run_game(predictor, arrays)
        elapsed = time.perf_counter() - started

        self.assertIn(true_variant, predictor.surviving)
        self.assertIsNotNone(diagnostics["steps_until_unique"])
        self.assertEqual(diagnostics["final_surviving"], 1)

        # Exact rule model: once unique, every prediction is exactly correct on
        # every field, on every step, including the teleport (life-loss) step --
        # EngineRuleModel.MODELS_TELEPORTS is True because its clone runs the
        # real VariantGame.step(), which gets teleports right by construction.
        start = diagnostics["steps_until_unique"]
        for t in range(start, len(arrays["agent_action"])):
            self.assertFalse(bool(pred_arrays["abstained"][t]), msg=f"step {t}")
            self.assertEqual(int(pred_arrays["movement"][t]), _observed_movement(arrays, t), msg=f"step {t}")
            self.assertEqual(int(pred_arrays["shape"][t]), int(arrays["after_shape"][t]), msg=f"step {t}")
            self.assertEqual(int(pred_arrays["color"][t]), int(arrays["after_color"][t]), msg=f"step {t}")
            self.assertEqual(int(pred_arrays["rotation"][t]), int(arrays["after_rotation"][t]), msg=f"step {t}")
            self.assertEqual(bool(pred_arrays["life_lost"][t]), bool(arrays["life_lost"][t]), msg=f"step {t}")

        engine_steps = predictor.rule_model.engine_steps
        steps_per_second = engine_steps / elapsed if elapsed > 0 else float("inf")
        print(f"\n[EngineRuleModel] {engine_steps} real clone+step() calls in {elapsed:.3f}s "
              f"= {steps_per_second:.1f} steps/sec "
              f"(converged to the true variant after {start} recorded game steps)")


if __name__ == "__main__":
    unittest.main()
