"""Real-engine provenance checks for generated transition supervision."""

import json
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from arcengine import GameState

from pebby.agent import data, world_data
from pebby.ls20 import generate, names
from pebby.ls20.curriculum import generate_legacy_level as curriculum_level
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.layout import extract
from pebby.ls20.plan import Oracle


def independent_route(spec, context):
    """A fresh engine and complete oracle built by the test, not the collector."""
    env = Ls20Scenario(generate.build_level(spec), context)
    oracle = Oracle(extract(env))
    if oracle.truncated or not oracle.solvable:
        return env, None
    return env, oracle.solution(seed=spec["seed"])


def route_windows(spec, context, solution, history):
    """Public history at every route state, replayed on a fresh engine."""
    env = Ls20Scenario(generate.build_level(spec), context)
    frames, actions, windows = [env.render()], [-1], []
    result = None
    for action in solution:
        windows.append(world_data.history_arrays(frames, actions, history))
        result = env.perform(action)
        frames.append(result.frame)
        actions.append(names.ACTION_IDS.index(action))
    assert result is not None and result.won and env.lives() == 3
    return windows


def rows_matching(rows, window):
    observed, valid, previous = window
    return [row for row in rows
            if np.array_equal(row["frames"], observed) and np.array_equal(row["history_valid"], valid)
            and np.array_equal(row["previous_actions"], previous)]


def counted_performs(spec, **kwargs):
    original = Ls20Scenario.perform
    calls = []

    def counted(env, action):
        calls.append(action)
        return original(env, action)

    with patch.object(Ls20Scenario, "perform", counted):
        rows, meta = world_data.collect_level(spec, **kwargs)
    return rows, meta, len(calls)


class WorldDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = generate.generate_legacy_level(0, 1)

    def test_context_preserves_later_level_rules_across_resets(self):
        for context in (0, 3, 6):
            env = Ls20Scenario(generate.build_level(self.spec), context)
            for _ in range(2):
                env.reset()
                self.assertEqual(env.level_count, 1)
                self.assertEqual(env.level_index, context)
                oracle = Oracle(extract(env))
                self.assertFalse(oracle.truncated)
                self.assertEqual(oracle.layout.match_hint, context == 0)
                for action in oracle.solution():
                    outcome = env.perform(action)
                self.assertTrue(outcome.won)
                self.assertEqual(env.levels_completed, 1)

    def test_counterfactual_successors_equal_independent_engine_actions(self):
        rows, meta = world_data.collect_level(self.spec, history=4, samples=2, epsilon=0., context_index=3)
        self.assertEqual(len(rows), 2)
        self.assertFalse(meta["search_truncated"])
        self.assertEqual(rows[0]["history_valid"].tolist(), [False, False, False, True])
        self.assertEqual(rows[0]["previous_actions"].tolist(), [-1, -1, -1, -1])
        self.assertEqual(rows[1]["history_valid"].tolist(), [False, False, True, True])
        np.testing.assert_array_equal(rows[0]["frames"][-1], rows[1]["frames"][-2])
        for index, action in enumerate(names.ACTION_IDS):
            env = Ls20Scenario(generate.build_level(self.spec), 3)
            result = env.perform(action)
            np.testing.assert_array_equal(rows[0]["next_frames"][index], result.frame)
        # Targets must actually distinguish actions; copying the input isn't enough.
        self.assertGreater(len({frame.tobytes() for frame in rows[0]["next_frames"]}), 1)

    def test_truncated_oracle_is_never_called_an_exact_action_set(self):
        class Incomplete:
            truncated = True
        self.assertEqual(data._optimal_mask(Incomplete(), None, 2), (0b0010, False))
        rows, meta = world_data.collect_level({**self.spec, "search_truncated": True})
        self.assertEqual(rows, [])
        self.assertIn("truncated", meta["excluded"])
        rows, meta = world_data.collect_level(self.spec, search_limit=1)
        self.assertEqual(rows, [])
        self.assertIn("incomplete", meta["excluded"])

    def test_state_labels_align_with_current_and_all_four_real_engine_branches(self):
        rows, _ = world_data.collect_level(self.spec, history=4, samples=4, epsilon=0.,
                                          context_index=3)
        env = Ls20Scenario(generate.build_level(self.spec), 3)
        for row_index, row in enumerate(rows):
            if row_index:
                # The final action slot records the action that produced this
                # row, allowing independent replay of the sampled trajectory.
                action_index = int(row["previous_actions"][-1])
                env.perform(names.ACTION_IDS[action_index])
            np.testing.assert_array_equal(row["frames"][-1], env.render())
            np.testing.assert_array_equal(row["player_cell"], env.player_cell())
            np.testing.assert_array_equal(row["current_triple"], env.triple())
            self.assertEqual(row["current_steps"], env.steps_left())
            self.assertEqual(row["current_lives"], env.lives())
            for index, action in enumerate(names.ACTION_IDS):
                # Rebuild and replay from scratch, independent of clone_env.
                branch = Ls20Scenario(generate.build_level(self.spec), 3)
                for earlier_row in rows[1:row_index + 1]:
                    branch.perform(names.ACTION_IDS[int(earlier_row["previous_actions"][-1])])
                result = branch.perform(action)
                np.testing.assert_array_equal(row["next_frames"][index], result.frame)
                np.testing.assert_array_equal(row["next_player_cell"][index], branch.player_cell())
                np.testing.assert_array_equal(row["next_triple"][index], branch.triple())
                self.assertEqual(row["next_steps"][index], branch.steps_left())
                self.assertEqual(row["next_lives"][index], branch.lives())

    def test_state_supervision_shapes_and_provenance_survive_save(self):
        arrays = world_data.build([self.spec], samples=2, history=3, context_index=3,
                                  epsilon=0., workers=1)
        shapes = {"player_cell": (2, 2), "next_player_cell": (2, 4, 2),
                  "current_triple": (2, 3), "next_triple": (2, 4, 3),
                  "current_steps": (2,), "next_steps": (2, 4),
                  "current_lives": (2,), "next_lives": (2, 4)}
        with tempfile.TemporaryDirectory(prefix="pebby-world-state-labels-") as directory:
            path = Path(directory) / "transitions.npz"
            world_data.save(path, arrays)
            with np.load(path, allow_pickle=False) as saved:
                metadata = json.loads(str(saved["meta"]))
                self.assertEqual(metadata["state_supervision_version"], 1)
                self.assertEqual(metadata["source"], "generated_only")
                for key, shape in shapes.items():
                    self.assertEqual(saved[key].shape, shape)
                    self.assertEqual(saved[key].dtype, np.int16)
                    np.testing.assert_array_equal(saved[key], arrays[key])

    def test_successor_labels_use_post_death_reset_and_terminal_state(self):
        free = {(3, 3), (4, 3), (5, 3)}
        spec = {**self.spec, "start": (3, 3), "start_triple": [0, 0, 0],
                "walls": sorted({(x, y) for x in range(names.GRID_COLS)
                                 for y in range(names.GRID_ROWS)} - free),
                "goals": [{"cell": (5, 3), "triple": [0, 0, 0]}],
                "cyclers": [], "launchers": [], "refills": [],
                "step_counter": 1, "step_cost": 1}
        rows, _ = world_data.collect_level(spec, samples=2, epsilon=0., context_index=3)
        self.assertEqual(len(rows), 2)
        row = rows[1]
        self.assertEqual(row["current_steps"], 0)
        for index, action in enumerate(names.ACTION_IDS):
            env = Ls20Scenario(generate.build_level(spec), 3)
            env.perform(4)
            result = env.perform(action)
            np.testing.assert_array_equal(row["next_player_cell"][index], env.player_cell())
            np.testing.assert_array_equal(row["next_triple"][index], env.triple())
            self.assertEqual(row["next_steps"][index], env.steps_left())
            self.assertEqual(row["next_lives"][index], env.lives())
            self.assertEqual(row["won"][index], result.won)
        # Winning is checked before exhaustion in the engine, so this success
        # legitimately has -1 steps. Other directions die and restore the start.
        self.assertTrue(row["won"][3])
        self.assertEqual(row["next_steps"][3], -1)
        self.assertEqual(row["next_lives"].tolist(), [2, 2, 2, 3])
        np.testing.assert_array_equal(row["next_player_cell"][:3], [[3, 3]] * 3)

    def test_branch_copy_shares_only_unplayed_templates_and_isolates_runtime(self):
        env = Ls20Scenario(generate.build_level(self.spec), 3)
        source_frame, source_state = env.render(), env.snapshot()
        fast = world_data.clone_env(env)
        reference = copy.deepcopy(env, {id(env.module): env.module})
        self.assertIsNot(fast.game._levels, env.game._levels)
        self.assertIsNot(fast.game.current_level, env.game.current_level)
        self.assertIs(fast.game._clean_levels[0], env.game._clean_levels[0])
        self.assertIs(fast.game._levels[0], env.game._levels[0])
        for action in (1, 2, 3, 4, 0, 4):
            actual, expected = fast.perform(action), reference.perform(action)
            self.assertEqual(actual.frames, expected.frames)
            self.assertEqual(fast.snapshot(), reference.snapshot())
        self.assertEqual(env.render(), source_frame)
        self.assertEqual(env.snapshot(), source_state)
        np.testing.assert_array_equal(fast.reset(), reference.reset())

    def test_collection_reuses_the_chosen_counterfactual_engine_action(self):
        original = Ls20Scenario.perform
        calls = []

        def counted(env, action):
            calls.append(action)
            return original(env, action)

        with patch.object(Ls20Scenario, "perform", counted):
            rows, meta = world_data.collect_level(self.spec, samples=3, epsilon=0., context_index=3)
        self.assertEqual(len(rows), 3)
        self.assertTrue(meta["context_engine_verified"])
        self.assertEqual(len(calls), 4 * len(rows) + meta["context_optimal_actions"])

    def test_context_verification_rejects_a_nonwinning_oracle_route(self):
        with patch.object(world_data.Oracle, "solution", return_value=[1]):
            env, oracle, proof = world_data.verified_context(self.spec, context_index=3)
        self.assertIsNone(env)
        self.assertIsNone(oracle)
        self.assertEqual(proof["excluded"], "contextual solution failed real-engine replay")

    def test_context_zero_launcher_is_excluded_before_claiming_exact_supervision(self):
        spec = {**self.spec, "launchers": [{"cell": [1, 1], "delta": [1, 0]}]}
        with patch.object(world_data, "Oracle") as oracle:
            rows, proof = world_data.collect_level(spec, context_index=0)
        self.assertEqual(rows, [])
        self.assertIn("pending hint", proof["excluded"])
        oracle.assert_not_called()

    def test_fast_clone_preserves_moving_cyclers_fog_and_refill_state(self):
        from pebby.ls20.curriculum import generate_legacy_level as generate_level

        for difficulty in (3, 4, 5):
            with self.subTest(difficulty=difficulty):
                spec = generate_level(difficulty, difficulty)
                source = Ls20Scenario(generate.build_level(spec), 6)
                frame, state = source.render(), source.snapshot()
                branch = world_data.clone_env(source)
                reference = copy.deepcopy(source, {id(source.module): source.module})
                for action in [1, 2, 3, 4, 0, *spec["solution"]]:
                    actual, expected = branch.perform(action), reference.perform(action)
                    self.assertEqual(actual.frames, expected.frames)
                    self.assertEqual(branch.snapshot(), reference.snapshot())
                self.assertEqual(source.render(), frame)
                self.assertEqual(source.snapshot(), state)

    # -- mixed coverage ---------------------------------------------------------

    def _check_branches(self, spec, context, prefix_actions, row):
        """All four successors must equal fresh from-scratch engine replays."""
        for action_index, action in enumerate(names.ACTION_IDS):
            branch = Ls20Scenario(generate.build_level(spec), context)
            for earlier in prefix_actions:
                branch.perform(earlier)
            result = branch.perform(action)
            np.testing.assert_array_equal(row["next_frames"][action_index], result.frame)
            self.assertEqual(bool(row["won"][action_index]), result.won)
            self.assertEqual(bool(row["terminal"][action_index]), result.finished)
            np.testing.assert_array_equal(row["next_player_cell"][action_index], branch.player_cell())
            np.testing.assert_array_equal(row["next_triple"][action_index], branch.triple())
            self.assertEqual(row["next_steps"][action_index], branch.steps_left())
            self.assertEqual(row["next_lives"][action_index], branch.lives())

    def test_spread_indices_keep_the_final_state_and_stay_distinct(self):
        self.assertEqual(world_data.spread_indices(list(range(10)), 4), [0, 3, 6, 9])
        self.assertEqual(world_data.spread_indices([5, 6, 7], 1), [7])
        self.assertEqual(world_data.spread_indices([2, 3], 5), [2, 3])
        self.assertEqual(world_data.spread_indices([], 3), [])
        self.assertEqual(world_data.spread_indices([4, 5], 0), [])
        for length, count in ((20, 7), (13, 12), (30, 16), (7, 2)):
            picks = world_data.spread_indices(list(range(3, 3 + length)), count)
            self.assertEqual(len(picks), count)
            self.assertEqual(len(set(picks)), count)
            self.assertEqual(picks[-1], 3 + length - 1)
            self.assertEqual(picks, sorted(picks))

    def test_mixed_coverage_keeps_distinct_teacher_states_with_aliased_observations(self):
        # Fog can make histories identical while hidden rail clocks differ.
        # Even with identical public keys, retain a different predictive state.
        with patch.object(world_data, "history_key", return_value=b"same public history"):
            rows, meta = world_data.collect_level(self.spec, samples=2, context_index=3,
                                                  coverage="mixed")
        self.assertEqual(len(rows), 2)
        self.assertTrue(any(row["won"].any() for row in rows))
        self.assertTrue(meta["win_covered"])

    def test_mixed_coverage_replays_long_level_endings_independently(self):
        """Rows, not metadata, must prove the ending: every expert row and its
        four branches are rebuilt from scratch, and the final row's winning
        branch is the route's actual last action on a fresh engine."""
        history = 4
        cases = [(self.spec, 3)] + [(curriculum_level(d, d), d) for d in (3, 4, 5)]
        checked = []
        for spec, context in cases:
            with self.subTest(seed=spec["seed"], difficulty=spec.get("difficulty")):
                _, solution = independent_route(spec, context)
                if solution is None:
                    rows, meta = world_data.collect_level(spec, samples=2, context_index=context,
                                                          coverage="mixed")
                    self.assertEqual(rows, [])
                    self.assertIn("excluded", meta)
                    continue
                length = len(solution)
                samples = max(2, length // 2)  # strictly fewer rows than route states
                rows, meta = world_data.collect_level(spec, history=history, samples=samples,
                                                      context_index=context, coverage="mixed")
                if not rows:
                    self.assertIn("excluded", meta)
                    continue
                checked.append(spec["seed"])
                self.assertGreater(length, samples)
                self.assertEqual(meta["coverage"], "mixed")
                self.assertEqual(meta["context_optimal_actions"], length)
                self.assertEqual(len(rows), samples)
                self.assertEqual(meta["samples"], samples)
                self.assertEqual(meta["explore_samples"] + meta["expert_samples"], samples)
                self.assertLessEqual(meta["explore_samples"], samples // 2)
                self.assertGreaterEqual(meta["expert_samples"], samples - samples // 2)
                self.assertEqual(meta["expert_indices"][-1], length - 1)
                self.assertEqual(meta["expert_indices"], sorted(set(meta["expert_indices"])))
                self.assertTrue(meta["win_covered"])
                self.assertGreaterEqual(meta["win_rows"], 1)
                keys = {world_data.history_key(row["frames"], row["history_valid"],
                                               row["previous_actions"]) for row in rows}
                self.assertEqual(len(keys), len(rows), "rows must not repeat an input history")
                # Explorer rows come first and chain through their recorded actions.
                explorer, chain = rows[:meta["explore_samples"]], []
                env = Ls20Scenario(generate.build_level(spec), context)
                for row_index, row in enumerate(explorer):
                    if row_index:
                        action_index = int(row["previous_actions"][-1])
                        if action_index < 0:
                            break  # a life was lost; the history legitimately restarted
                        chain.append(names.ACTION_IDS[action_index])
                        env.perform(chain[-1])
                    np.testing.assert_array_equal(row["frames"][-1], env.render())
                    self._check_branches(spec, context, chain, row)
                # Expert rows sit exactly at the recorded route indices.
                windows = route_windows(spec, context, solution, history)
                expert = rows[meta["explore_samples"]:]
                self.assertEqual(len(expert), len(meta["expert_indices"]))
                for row, index in zip(expert, meta["expert_indices"]):
                    self.assertEqual(len(rows_matching([row], windows[index])), 1)
                    self.assertTrue(row["history_valid"][-1])
                    self._check_branches(spec, context, solution[:index], row)
                    winning = names.ACTION_IDS.index(solution[index])
                    self.assertTrue(row["optimal"] & (1 << winning))
                    self.assertEqual(row["distances"][winning], length - index - 1)
                final, winning = expert[-1], names.ACTION_IDS.index(solution[-1])
                self.assertTrue(final["won"][winning])
                self.assertTrue(final["terminal"][winning])
                self.assertEqual(final["next_lives"][winning], 3)
                self.assertEqual(final["current_lives"], 3)
                self.assertEqual(int(final["previous_actions"][-1]),
                                 names.ACTION_IDS.index(solution[-2]))
                self.assertEqual(sum(bool(row["won"].any()) for row in rows), meta["win_rows"])
                # Late route states are represented, not just the ending.
                late = [index for index in meta["expert_indices"] if index >= length // 2]
                self.assertGreaterEqual(len(late), 2 if meta["expert_samples"] >= 3 else 1)
        self.assertIn(self.spec["seed"], checked)

    def test_mixed_coverage_single_sample_is_the_real_winning_state(self):
        _, solution = independent_route(self.spec, 3)
        rows, meta = world_data.collect_level(self.spec, history=3, samples=1, context_index=3,
                                              coverage="mixed")
        self.assertEqual(len(rows), 1)
        self.assertEqual((meta["explore_samples"], meta["expert_samples"]), (0, 1))
        self.assertEqual(meta["expert_indices"], [len(solution) - 1])
        window = route_windows(self.spec, 3, solution, 3)[-1]
        self.assertEqual(len(rows_matching(rows, window)), 1)
        winning = names.ACTION_IDS.index(solution[-1])
        self.assertTrue(rows[0]["won"][winning])
        self.assertEqual(int(rows[0]["optimal"]) & (1 << winning), 1 << winning)
        self._check_branches(self.spec, 3, solution[:-1], rows[0])

    def test_mixed_coverage_branches_only_selected_expert_states(self):
        _, solution = independent_route(self.spec, 3)
        length, samples = len(solution), 4
        rows, meta, calls = counted_performs(self.spec, samples=samples, epsilon=0.,
                                             context_index=3, coverage="mixed")
        explore, expert = meta["explore_samples"], meta["expert_samples"]
        self.assertEqual(len(rows), samples)
        self.assertEqual(explore, samples // 2)
        # verification replay + four branches per explorer row + one engine
        # action per unselected route step + four per selected route state.
        self.assertEqual(calls, length + 4 * explore + (length - expert) + 4 * expert)
        self.assertLess(calls, length + 4 * explore + 4 * length)
        # Prefix accounting is unchanged.
        rows, meta, calls = counted_performs(self.spec, samples=samples, epsilon=0., context_index=3)
        self.assertEqual(meta["coverage"], "prefix")
        self.assertEqual(calls, length + 4 * len(rows))
        self.assertEqual(meta["expert_indices"], [])

    def test_mixed_coverage_metadata_survives_build_and_save(self):
        arrays = world_data.build([self.spec], samples=4, history=3, context_index=3,
                                  epsilon=0., workers=1, coverage="mixed")
        meta = arrays["meta"]
        self.assertEqual(meta["coverage"], "mixed")
        self.assertEqual(meta["samples_per_level"], 4)
        self.assertEqual(meta["accepted_levels"], 1)
        self.assertEqual(meta["win_covered_levels"], 1)
        self.assertGreaterEqual(meta["win_rows"], 1)
        self.assertEqual(meta["samples"], len(arrays["optimal"]))
        self.assertTrue(arrays["won"].any())
        level = meta["levels"][0]
        self.assertEqual(level["coverage"], "mixed")
        self.assertTrue(level["context_engine_verified"])
        self.assertEqual(level["context_index"], 3)
        with tempfile.TemporaryDirectory(prefix="pebby-world-mixed-") as directory:
            path = Path(directory) / "transitions.npz"
            world_data.save(path, arrays)
            with np.load(path, allow_pickle=False) as saved:
                loaded = json.loads(str(saved["meta"]))
                self.assertEqual(loaded["coverage"], "mixed")
                self.assertEqual(loaded["levels"][0]["expert_indices"], level["expert_indices"])
        default = world_data.build([self.spec], samples=2, history=3, context_index=3,
                                   epsilon=0., workers=1)
        self.assertEqual(default["meta"]["coverage"], "prefix")
        self.assertEqual(default["meta"]["levels"][0]["coverage"], "prefix")
        with self.assertRaises(ValueError):
            world_data.collect_level(self.spec, context_index=3, coverage="everything")
        with self.assertRaises(ValueError):
            world_data.collect_level(self.spec, samples=0, context_index=3, coverage="mixed")

    def test_mixed_coverage_fails_closed_on_teacher_or_engine_disagreement(self):
        _, solution = independent_route(self.spec, 3)
        length = len(solution)
        # The expert route is checked against the complete teacher at every step.
        with patch.object(world_data.Oracle, "state_of", return_value=None):
            with self.assertRaisesRegex(ValueError, "disagrees with the complete teacher"):
                world_data.collect_level(self.spec, samples=1, context_index=3, coverage="mixed")
        # A final action that the real engine does not report as a WIN is refused.
        original = Ls20Scenario.perform
        calls = []

        def faked(env, action):
            calls.append(action)
            result = original(env, action)
            if len(calls) > length and result.won:
                result.state = GameState.GAME_OVER
            return result

        with patch.object(Ls20Scenario, "perform", faked):
            with self.assertRaises(ValueError):
                world_data.collect_level(self.spec, samples=1, context_index=3, coverage="mixed")
        self.assertGreater(len(calls), length)

    def test_mixed_coverage_matches_prefix_engine_cost_on_a_handful_of_levels(self):
        """Engine actions (the CPU cost driver) for mixed stay within one
        route length of the prefix collector and well below branching
        the whole route four ways."""
        samples = 8
        for spec, context in [(self.spec, 3)] + [(curriculum_level(d, d), d) for d in (1, 2)]:
            with self.subTest(seed=spec["seed"], difficulty=spec.get("difficulty")):
                _, solution = independent_route(spec, context)
                length = len(solution)
                _, prefix_meta, prefix_calls = counted_performs(
                    spec, samples=samples, context_index=context)
                rows, meta, mixed_calls = counted_performs(
                    spec, samples=samples, context_index=context, coverage="mixed")
                if not rows:
                    continue
                self.assertEqual(prefix_calls, length + 4 * prefix_meta["samples"])
                self.assertLessEqual(mixed_calls, length + 4 * samples + length)
                self.assertLess(mixed_calls, length + 4 * meta["explore_samples"] + 4 * length)
                self.assertTrue(meta["win_covered"])


if __name__ == "__main__":
    unittest.main()
