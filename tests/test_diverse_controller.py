"""Unit tests for the runtime decision heuristics in `pebby.agent.diverse_controller`.

Nothing here exercises learning: every test feeds a fixed-logit stub through the
controller and checks the stall mask, the penalties and the per-life sampling.
"""

import unittest

import torch

from pebby.agent.diverse_controller import (DEFAULTS, OPPOSITE, DiverseLivesController,
                                            derive_seed, frame_digest)
from pebby.agent.history import PolicyHistory

UP, DOWN, LEFT, RIGHT = range(4)


class FixedPolicy:
    """Return the same four logits every call (no `config`: single-frame path)."""

    def __init__(self, logits):
        self.logits = list(logits)
        self.calls = 0

    def __call__(self, frames):
        self.calls += 1
        return torch.tensor([self.logits], dtype=torch.float32)


class SequencePolicy:
    """Return one logit row per call, repeating the last row afterwards."""

    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.calls = 0

    def __call__(self, frames):
        row = self.rows[min(self.calls, len(self.rows) - 1)]
        self.calls += 1
        return torch.tensor([row], dtype=torch.float32)


class StructuredPolicy:
    """Declare a history so `for_policy` builds one; record how it is called."""

    def __init__(self):
        self.kwargs = []

    def config(self):
        return {"architecture": "structured", "history": 3}

    def __call__(self, frames, history_valid=None, previous_actions=None):
        self.kwargs.append(dict(frames=frames.shape, history_valid=history_valid,
                                previous_actions=previous_actions))
        return torch.tensor([[0., 1., 0., 0.]])


def frame(value):
    return [[value, value], [value, value]]


class DiverseLivesControllerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_opposites_and_seed_derivation(self):
        self.assertEqual(OPPOSITE, (DOWN, UP, RIGHT, LEFT))
        self.assertEqual(derive_seed(0, 0, 1), derive_seed(0, 0, 1))
        self.assertNotEqual(derive_seed(0, 0, 1), derive_seed(0, 0, 2))
        self.assertNotEqual(derive_seed(0, 0, 1), derive_seed(1, 0, 1))
        self.assertNotEqual(derive_seed(0, 0, 1), derive_seed(0, 1, 1))
        self.assertLess(derive_seed(3, 4, 5), 1 << 63)

    def test_knobs_are_validated_and_reported(self):
        for bad in (dict(reversal_penalty=-1), dict(repeat_penalty=-.1), dict(temperature=-1),
                    dict(base_seed=1.5), dict(base_seed=True)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                DiverseLivesController(FixedPolicy([0, 0, 0, 0]), **bad)
        metadata = DiverseLivesController(FixedPolicy([0, 0, 0, 0])).metadata
        self.assertFalse(metadata["learned"])
        self.assertTrue(metadata["runtime_heuristics"])
        self.assertFalse(metadata["emits_reset"])
        self.assertEqual(metadata["controller"], "diverse_lives")
        for knob, value in DEFAULTS.items():
            self.assertEqual(metadata[knob], value)
        self.assertEqual(DEFAULTS["reversal_penalty"], .5)
        self.assertEqual(metadata["penalty_gating"], "diversity_index_above_zero_only")
        self.assertEqual(metadata["first_life_selection"], "strict_argmax_with_stall_mask_no_penalties")
        self.assertEqual(DiverseLivesController(FixedPolicy([0] * 4), temperature=0)
                         .metadata["later_life_selection"], "strict_argmax")

    def test_stall_mask_blocks_the_repeated_action_until_the_frame_moves(self):
        controller = DiverseLivesController(FixedPolicy([0., 3., 2., 1.]))
        controller.start(frame(0))
        self.assertEqual(controller.decide(frame(0)), DOWN)
        controller.observe(frame(0), DOWN)  # DOWN did nothing: masked out.
        self.assertEqual(controller.decide(frame(0)), LEFT)
        self.assertEqual(controller.last["blocked"], [DOWN])
        self.assertTrue(controller.last["mask_applied"])
        controller.observe(frame(0), LEFT)  # LEFT did nothing too: both masked.
        self.assertEqual(controller.decide(frame(0)), RIGHT)
        self.assertEqual(controller.last["blocked"], [DOWN, LEFT])
        controller.observe(frame(1), RIGHT)  # The frame moved: the mask clears.
        self.assertEqual(controller.decide(frame(1)), DOWN)
        self.assertEqual(controller.last["blocked"], [])
        self.assertFalse(controller.last["mask_applied"])

    def test_exhausted_mask_falls_back_to_the_best_raw_action_and_can_be_disabled(self):
        controller = DiverseLivesController(FixedPolicy([0., 3., 2., 1.]), reversal_penalty=0,
                                            repeat_penalty=0)
        controller.start(frame(0))
        for action in (DOWN, LEFT, RIGHT, UP):
            self.assertEqual(controller.decide(frame(0)), action)
            controller.observe(frame(0), action)
        self.assertEqual(controller.decide(frame(0)), DOWN)  # Nothing left to mask.
        self.assertTrue(controller.last["mask_exhausted"])
        self.assertFalse(controller.last["mask_applied"])
        unmasked = DiverseLivesController(FixedPolicy([0., 3., 2., 1.]), stall_mask=False,
                                          repeat_penalty=0)
        unmasked.start(frame(0))
        unmasked.observe(frame(0), DOWN)
        self.assertEqual(unmasked.decide(frame(0)), DOWN)
        self.assertEqual(unmasked.last["blocked"], [])

    def later_life(self, policy, **knobs):
        """A controller on its second trajectory of level 0 with sampling off."""
        controller = DiverseLivesController(policy, temperature=0, **knobs)
        controller.start(frame(0))
        controller.observe(frame(0), None, reset=True)
        self.assertEqual(controller.diversity_index, 1)
        return controller

    def test_first_life_is_argmax_plus_stall_mask_with_no_penalties(self):
        # UP, then a DOWN/UP near-tie: with no reversal penalty on the first
        # life the policy's own DOWN stands, and the record shows zero penalties.
        controller = DiverseLivesController(SequencePolicy([[2., 0., 0., 0.], [.9, 1., 0., 0.]]))
        controller.start(frame(0))
        self.assertEqual(controller.decide(frame(0)), UP)
        controller.observe(frame(1), UP)
        self.assertEqual(controller.decide(frame(1)), DOWN)
        self.assertEqual(controller.last["penalties"], [0., 0., 0., 0.])
        self.assertFalse(controller.last["sampled"])
        # Repeating DOWN from one unchanged frame costs nothing either, so only
        # the stall mask can move the first life off the policy's argmax.
        repeat = DiverseLivesController(FixedPolicy([0., 1.2, 0., 0.]), stall_mask=False)
        repeat.start(frame(0))
        for _ in range(4):
            self.assertEqual(repeat.decide(frame(0)), DOWN)
            self.assertEqual(repeat.last["penalties"], [0., 0., 0., 0.])
            repeat.observe(frame(0), DOWN)
        masked = DiverseLivesController(FixedPolicy([0., 1.2, 0., 0.]))
        masked.start(frame(0))
        masked.observe(frame(0), DOWN)
        self.assertEqual(masked.decide(frame(0)), UP)
        self.assertEqual(masked.last["blocked"], [DOWN])
        self.assertEqual(masked.last["penalties"], [0., 0., 0., 0.])

    def test_reversal_penalty_flips_a_near_tie_only_after_a_frame_changing_move(self):
        # First call prefers UP outright; afterwards DOWN edges UP by 0.1.
        policy = SequencePolicy([[2., 0., 0., 0.], [.9, 1., 0., 0.]])
        controller = self.later_life(policy, repeat_penalty=0)
        self.assertEqual(controller.decide(frame(0)), UP)
        controller.observe(frame(1), UP)  # UP changed the frame: undoing it is penalised.
        self.assertEqual(controller.decide(frame(1)), UP)
        self.assertEqual(controller.last["penalties"], [0., .5, 0., 0.])
        # The same near-tie with no penalty resolves to DOWN.
        bare = self.later_life(SequencePolicy(policy.rows), reversal_penalty=0, repeat_penalty=0)
        bare.decide(frame(0))
        bare.observe(frame(1), UP)
        self.assertEqual(bare.decide(frame(1)), DOWN)
        # A move that left the frame unchanged is not worth undoing: no penalty.
        stalled = self.later_life(SequencePolicy(policy.rows), stall_mask=False, repeat_penalty=0)
        stalled.decide(frame(0))
        stalled.observe(frame(0), UP)
        self.assertEqual(stalled.decide(frame(0)), DOWN)
        self.assertEqual(stalled.last["penalties"], [0., 0., 0., 0.])

    def test_repeat_penalty_grows_per_attempt_from_the_same_frame_and_resets_per_life(self):
        controller = self.later_life(FixedPolicy([0., 1.2, 0., 0.]), stall_mask=False,
                                     reversal_penalty=0)
        for attempt in range(3):  # Adjusted DOWN: 1.2, 0.7, 0.2 -- then -0.3.
            self.assertEqual(controller.decide(frame(0)), DOWN)
            self.assertEqual(controller.last["penalties"][DOWN], .5 * attempt)
            controller.observe(frame(0), DOWN)
        self.assertEqual(controller.decide(frame(0)), UP)  # 1.2 - 1.5 < 0: a loop pays.
        self.assertEqual(controller.last["penalties"], [0., 1.5, 0., 0.])
        # The count is per frame digest: a new frame starts at zero.
        controller.observe(frame(1), UP)
        controller.decide(frame(1))
        self.assertEqual(controller.last["penalties"], [0., 0., 0., 0.])
        # Returning to the earlier frame remembers it within the life ...
        controller.observe(frame(0), DOWN)
        controller.decide(frame(0))
        self.assertEqual(controller.last["penalties"][DOWN], 1.5)
        # ... and a lost life forgets it.
        controller.observe(frame(0), DOWN, life_lost=True)
        self.assertEqual(controller.diversity_index, 2)
        self.assertEqual(controller.decide(frame(0)), DOWN)
        self.assertEqual(controller.last["penalties"], [0., 0., 0., 0.])

    def test_first_life_is_strict_argmax_even_with_a_temperature(self):
        controller = DiverseLivesController(FixedPolicy([0., .1, 0., 0.]), temperature=5.)
        controller.start(frame(0))
        for step in range(20):
            self.assertEqual(controller.decide(frame(step)), DOWN)
            self.assertFalse(controller.last["sampled"])
            self.assertEqual(controller.last["diversity_index"], 0)
            controller.observe(frame(step + 1), DOWN)

    def sampled_life(self, base_seed, lives_lost, steps=12, temperature=.5):
        controller = DiverseLivesController(FixedPolicy([0., 0., 0., 0.]), base_seed=base_seed,
                                            temperature=temperature)
        controller.start(frame(0))
        for life in range(lives_lost):
            controller.observe(frame(100 + life), DOWN, life_lost=True)
        choices = []
        for step in range(steps):
            choices.append(controller.decide(frame(step)))
            self.assertTrue(controller.last["sampled"])
            self.assertEqual(controller.last["diversity_index"], lives_lost)
            controller.observe(frame(step + 1), choices[-1])
        return choices

    def test_later_lives_sample_reproducibly_and_differ_across_lives_and_seeds(self):
        second = self.sampled_life(0, 1)
        self.assertEqual(second, self.sampled_life(0, 1))
        self.assertTrue(all(choice in range(4) for choice in second))
        self.assertNotEqual(second, self.sampled_life(0, 2))
        self.assertNotEqual(second, self.sampled_life(1, 1))
        self.assertGreater(len(set(second)), 1)  # Uniform logits: not one action forever.

    def test_temperature_zero_keeps_every_life_argmax(self):
        controller = DiverseLivesController(FixedPolicy([0., 0., 2., 0.]), temperature=0)
        controller.start(frame(0))
        controller.observe(frame(1), LEFT, life_lost=True)
        self.assertEqual(controller.decide(frame(1)), LEFT)
        self.assertFalse(controller.last["sampled"])

    def test_reset_and_level_change_move_the_diversity_index(self):
        controller = DiverseLivesController(FixedPolicy([0., 0., 0., 0.]))
        controller.start(frame(0))
        controller.observe(frame(1), None, reset=True)  # The caller's RESET.
        controller.decide(frame(1))
        self.assertEqual((controller.level_index, controller.diversity_index), (0, 1))
        self.assertTrue(controller.last["sampled"])
        controller.observe(frame(2), RIGHT, level_changed=True)  # A new level: argmax again.
        controller.decide(frame(2))
        self.assertEqual((controller.level_index, controller.diversity_index), (1, 0))
        self.assertFalse(controller.last["sampled"])
        controller.observe(frame(3), RIGHT, life_lost=True)
        self.assertIn(controller.decide(frame(3), level_index=1), range(4))
        self.assertEqual((controller.level_index, controller.diversity_index), (1, 1))
        self.assertTrue(controller.last["sampled"])
        # An explicit later level index from the caller starts that level's first life.
        controller.decide(frame(3), level_index=2)
        self.assertEqual((controller.level_index, controller.diversity_index), (2, 0))

    def test_never_emits_reset_and_refuses_finished_games(self):
        controller = DiverseLivesController(FixedPolicy([5., -5., 0., 0.]))
        controller.start(frame(0))
        for step in range(8):
            choice = controller.decide(frame(step), lives=3, state="NOT_FINISHED")
            self.assertIn(choice, range(4))
            controller.observe(frame(step), choice)  # Every move stalls: still 0..3.
        for state in ("WIN", "GAME_OVER"):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "RESET"):
                controller.decide(frame(0), state=state)
        with self.assertRaises(ValueError):
            controller.observe(frame(0), 4)
        with self.assertRaises(ValueError):
            controller.observe(frame(0), -1)

    def test_bad_logits_are_refused(self):
        for logits in ([0., 1., 2.], [0., float("nan"), 0., 0.]):
            controller = DiverseLivesController(FixedPolicy(logits))
            controller.start(frame(0))
            with self.subTest(logits=logits), self.assertRaisesRegex(ValueError, "four finite"):
                controller.decide(frame(0))

    def test_history_policies_go_through_for_policy_with_boundary_resets(self):
        policy = StructuredPolicy()
        controller = DiverseLivesController(policy)
        controller.start(frame(0))
        self.assertIsInstance(controller.history, PolicyHistory)
        self.assertEqual(controller.decide(frame(0)), DOWN)
        self.assertEqual(policy.kwargs[-1]["frames"], (1, 3, 2, 2))
        self.assertEqual(policy.kwargs[-1]["previous_actions"].tolist(), [[-1, -1, -1]])
        controller.observe(frame(1), DOWN)
        controller.decide(frame(1))
        self.assertEqual(policy.kwargs[-1]["previous_actions"].tolist(), [[-1, -1, DOWN]])
        self.assertEqual(policy.kwargs[-1]["history_valid"].tolist(), [[False, True, True]])
        controller.observe(frame(2), DOWN, life_lost=True)
        controller.decide(frame(2))
        self.assertEqual(policy.kwargs[-1]["previous_actions"].tolist(), [[-1, -1, -1]])
        self.assertEqual(controller.history.frames, [frame(2)])

    def test_last_decision_record_is_json_friendly(self):
        import json
        controller = DiverseLivesController(FixedPolicy([0., 1., 0., 0.]))
        controller.start(frame(0))
        controller.decide(frame(0))
        json.dumps(controller.last)
        json.dumps(controller.metadata)
        self.assertEqual(frame_digest(frame(0)), frame_digest([[0, 0], [0, 0]]))


if __name__ == "__main__":
    unittest.main()
