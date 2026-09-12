"""LS20 rules, layout reading, planning and generation.

The rule tests assert against the real upstream game rather than a
re-implementation, so they pin the semantics Pebby's planner depends on. If
upstream is ever re-vendored and a rule shifts, these fail loudly.
"""

import hashlib
from pathlib import Path
import unittest

from pebby.ls20 import names
from pebby.ls20.env import Ls20Env, Ls20Scenario, UPSTREAM
from pebby.ls20.generate import LEGACY_DIFFICULTIES as DIFFICULTIES, build_level, generate_legacy_level as generate_level
from pebby.ls20.layout import extract
from pebby.ls20.plan import Oracle, Unplannable, oracle_for

ROOT = Path(__file__).resolve().parents[1]
UP, DOWN, LEFT, RIGHT = names.ACTION_IDS

# Recorded from the shipped levels; see third_party/ls20/PROVENANCE.md.
SHIPPED = [
    # `cyclers` counts static and rail-riding cyclers together; `extract` splits
    # them into `layout.cyclers` and `layout.patrollers`.
    # level, goals, cyclers, refills, launchers, rails, max_steps, cost
    (0, 1, 1, 0, 0, 0, 42, 1),
    (1, 1, 1, 2, 0, 0, 42, 2),
    (2, 1, 2, 2, 2, 0, 42, 2),
    (3, 1, 2, 2, 8, 0, 42, 1),
    (4, 1, 3, 3, 8, 1, 42, 2),
    (5, 2, 3, 3, 2, 3, 42, 1),
    (6, 1, 3, 6, 3, 1, 42, 2),
]


class VendoredSourceTests(unittest.TestCase):
    def test_upstream_copy_is_unmodified_and_carries_its_licence(self):
        text = UPSTREAM.read_bytes()
        self.assertEqual(hashlib.sha256(text).hexdigest(),
                         "298c810da2850d557c95d92a2cbd846df29a45d7134e20888617bedf5dafcd92")
        head = text.decode().split("\n", 21)
        self.assertIn("MIT License", head[0])
        self.assertIn("Copyright (c) 2026 ARC Prize Foundation", head[2])
        for name in ("LICENSE", "LICENSE.arcengine", "PROVENANCE.md"):
            self.assertTrue((ROOT / "third_party" / "ls20" / name).is_file(), name)


class RuleTests(unittest.TestCase):
    """Each test drives the real game and asserts one LS20 rule."""

    def setUp(self):
        self.env = Ls20Env()

    def test_shipped_levels_have_the_recorded_structure(self):
        for index, goals, cyclers, refills, launchers, rails, steps, cost in SHIPPED:
            with self.subTest(level=index + 1):
                self.env.set_level(index)
                layout = extract(self.env)
                self.assertEqual(len(layout.goals), goals)
                self.assertEqual(len(layout.cyclers) + len(layout.patrollers), cyclers)
                self.assertEqual(len(layout.refills), refills)
                self.assertEqual(len(layout.launchers), launchers)
                self.assertEqual(len(layout.rails), rails)
                self.assertEqual((layout.max_steps, layout.step_cost), (steps, cost))

    def test_frame_is_64x64_of_palette_indices(self):
        frame = self.env.render()
        self.assertEqual(len(frame), names.FRAME_SIZE)
        self.assertTrue(all(len(row) == names.FRAME_SIZE for row in frame))
        self.assertTrue(all(0 <= cell <= 15 for row in frame for cell in row))

    def test_a_move_into_a_wall_changes_nothing_but_still_costs_budget(self):
        layout = extract(self.env)
        before, steps = self.env.player_cell(), self.env.steps_left()
        blocked = next(action for action, (dx, dy) in zip(names.ACTION_IDS, names.ACTION_DELTAS)
                       if (before[0] + dx, before[1] + dy) in layout.walls)
        self.env.perform(blocked)
        self.assertEqual(self.env.player_cell(), before)
        self.assertEqual(self.env.steps_left(), steps - layout.step_cost)

    def test_a_cycler_advances_exactly_one_attribute(self):
        # Level 1 has a single rotation cycler, so the walk to it is short.
        self.env.set_level(0)
        layout = extract(self.env)
        cell, kind = next(iter(layout.cyclers.items()))
        self.assertEqual(kind, "rotation")
        oracle = Oracle(layout)
        before = self.env.triple()
        for action in oracle.solution():
            self.env.perform(action)
            if self.env.player_cell() == cell:
                break
        after = self.env.triple()
        self.assertEqual(after[0], before[0])
        self.assertEqual(after[1], before[1])
        self.assertEqual(after[2], (before[2] + 1) % names.ROTATION_COUNT)

    def test_a_goal_pad_blocks_a_wrong_triple_and_the_bump_costs_no_budget(self):
        # Level 1's start column runs straight up to the goal, so the player can
        # reach the pad without passing the rotation cycler and still mismatch.
        self.env.set_level(0)
        goal_cell, goal_triple = extract(self.env).goals[0]
        for _ in range(6):
            self.env.perform(UP)
        approach = self.env.player_cell()
        self.assertEqual(approach, (goal_cell[0], goal_cell[1] + 1))
        self.assertNotEqual(self.env.triple(), goal_triple)

        steps = self.env.steps_left()
        for _ in range(5):
            self.env.perform(UP)
        self.assertEqual(self.env.player_cell(), approach, "a mismatched pad must block")
        self.assertEqual(self.env.steps_left(), steps, "a rejected bump must be free")

    def test_running_the_budget_out_three_times_ends_the_game(self):
        # Level 1 is walled below the spawn, so DOWN is always blocked -- and a
        # blocked move still charges budget, unlike a rejected goal bump. Three
        # exhausted budgets spend all three lives.
        lives, actions = self.env.lives(), 0
        self.assertEqual(lives, 3)
        for _ in range(400):
            observation = self.env.perform(DOWN)
            actions += 1
            if observation.finished:
                break
        self.assertTrue(observation.finished)
        self.assertFalse(observation.won)
        self.assertEqual(actions, 3 * 43, "42 charged moves per life, plus the fatal one")
        self.assertEqual(self.env.perform(UP).frames, [],
                         "a finished game must absorb further actions")


class PlannerTests(unittest.TestCase):
    def test_planner_solves_the_exact_shipped_levels_and_the_game_agrees(self):
        # Levels 1-4 carry only mechanics the planner models; 5-7 add rail-riding
        # cyclers. Optimal counts are well under the published human baselines
        # of 22 / 123 / 73 / 84 actions.
        for index, optimal in ((0, 13), (1, 45), (2, 39), (3, 43)):  # 5-7 are slower; see tools/
            with self.subTest(level=index + 1):
                env = Ls20Env()
                env.set_level(index)
                oracle = oracle_for(env)
                self.assertTrue(oracle.solvable)
                self.assertEqual(oracle.optimal_actions, optimal)
                replay = Ls20Env()
                replay.set_level(index)
                for action in oracle.solution():
                    observation = replay.perform(action)
                self.assertEqual(replay.levels_completed, 1)
                self.assertEqual(replay.level_index, index + 1)

    def test_transition_model_matches_the_real_engine_on_every_shipped_level(self):
        """The regression gate for the whole rule model.

        `tools/differential.py` runs this exhaustively; this is the cheap version
        that runs in the suite. Any drift between Pebby's model and upstream's
        own code shows up here first.
        """
        import random

        from pebby.ls20.plan import _refill_order, simulate
        for index in range(7):
            with self.subTest(level=index + 1):
                env = Ls20Env()
                env.set_level(index)
                layout = extract(env)
                self.assertTrue(layout.exact, "every shipped mechanic must be modelled")
                refills = _refill_order(layout)
                state = (layout.start_cell, *layout.start_triple, 0, 0, layout.max_steps, 0)
                rng = random.Random(index)
                for _ in range(40):
                    action = rng.randrange(4)
                    state, outcome = simulate(layout, state, action, refills)
                    observation = env.perform(names.ACTION_IDS[action])
                    if outcome in ("died", "won") or observation.finished:
                        break
                    self.assertEqual(env.player_cell(), state[0])
                    self.assertEqual(env.triple(), state[1:4])
                    self.assertEqual(env.steps_left(), state[6])

    def test_optimal_action_is_available_off_the_solution_path(self):
        env = Ls20Env()
        env.set_level(0)
        oracle = oracle_for(env)
        start = oracle.distance_for(oracle.state_of(env))
        env.perform(oracle.action_at(env))
        self.assertEqual(oracle.distance_for(oracle.state_of(env)), start - 1)


class GeneratorTests(unittest.TestCase):
    def test_every_generated_level_is_completable_in_the_real_game(self):
        """The generator's whole contract, checked by the rules' own code."""
        for difficulty in DIFFICULTIES:
            with self.subTest(difficulty=difficulty):
                spec = generate_level(100 + difficulty, difficulty)
                env = Ls20Scenario(build_level(spec), spec["seed"] % 7)
                for action in spec["solution"]:
                    observation = env.perform(action)
                self.assertTrue(observation.won)
                self.assertEqual(env.levels_completed, 1)

    def test_generation_is_deterministic(self):
        first = generate_level(11, 3)
        second = generate_level(11, 3)
        self.assertEqual(first, second)

    def test_generated_levels_are_walled_in_and_leave_budget_headroom(self):
        for seed in range(4):
            with self.subTest(seed=seed):
                spec = generate_level(seed, 2)
                walls = {tuple(cell) for cell in spec["walls"]}
                border = {(c, r) for c in range(names.GRID_COLS) for r in range(names.GRID_ROWS)
                          if c in (0, names.GRID_COLS - 1) or r in (0, names.GRID_ROWS - 1)}
                self.assertTrue(border <= walls, "player could walk off the lattice")
                self.assertGreaterEqual(spec["slack_moves"], 8)
                self.assertGreaterEqual(spec["optimal_actions"], 6)

    def test_generated_levels_need_the_attributes_they_claim_to_need(self):
        spec = generate_level(5, 3)
        self.assertNotEqual(spec["start_triple"], spec["goals"][0]["triple"])
        kinds = {entry["kind"] for entry in spec["cyclers"]}
        start, goal = spec["start_triple"], spec["goals"][0]["triple"]
        for kind, index in (("shape", 0), ("color", 1), ("rotation", 2)):
            if start[index] != goal[index]:
                self.assertIn(kind, kinds, f"{kind} must change but no {kind} cycler exists")


if __name__ == "__main__":
    unittest.main()
