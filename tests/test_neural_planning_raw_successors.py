"""Raw public successor targets, using small generated real-engine branches."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pebby.agent.world_data import clone_env
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.generate import build_level
from tools import collect_neural_planning_sequences as collector


def spec(budget=2):
    return dict(walls=[(1, 3), (0, 2), (1, 1)], start=(1, 2),
                start_triple=(0, 0, 0), goals=[dict(cell=(2, 2), triple=(0, 0, 0))],
                cyclers=[], refills=[], launchers=[], step_counter=budget, step_cost=1, fog=False,
                seed=5, difficulty=1, difficulty_version='ls20-reference-v1',
                context_solution=[4], context_optimal_actions=1)


class RecordingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, frames, valid, actions):
        self.inputs.append(tuple(value[0].numpy().copy() for value in (frames, valid, actions)))
        return torch.zeros(1, 148, 96)

    def metadata(self):
        return {}


def policy():
    # Deliberately permuted planner roots test canonical raw-frame alignment.
    roots = torch.tensor([[3, 1, 0, 2]])
    return SimpleNamespace(encoder=RecordingEncoder(), planner=SimpleNamespace(
        imagine=lambda field, horizon: dict(root_actions=roots,
                                            imagined_actions=roots[:, :, None].expand(1, 4, 4)),
        continuation_logits=lambda field: torch.tensor([[0., 0., 0., 1.]]),
        cfg=SimpleNamespace(horizon=4),
        dynamics=SimpleNamespace(state_dict=lambda: {}, config=lambda: {}),
        continuation=SimpleNamespace(state_dict=lambda: {})))


class RawSuccessorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def collect(self, budget=2, prefix=0):
        env = Ls20Scenario(build_level(spec(budget)), 0)
        history = ([env.reset()], [-1])
        for _ in range(prefix):
            old_lives = env.lives()
            observation = env.perform(2)
            self.assertFalse(observation.finished)
            history = collector.observe(history, observation.frame, 1, reset=env.lives() < old_lives)
        model = policy()
        row = collector.collect_root(model, env, history, raw_successor_public=True)
        return env, history, model, row

    def assert_reconstructed_inputs(self, model, row):
        # Deliberately omit every privileged label and frozen feature: neither
        # can influence reconstruction or the encoder's public input tuple.
        public = {key: row[key] for key in ('frames', 'history_valid', 'previous_actions',
                                           'actions', 'next_frames', 'next_frame_valid', 'next_history_reset')}
        expected = iter(model.encoder.inputs[1:])
        for branch in range(4):
            for step in range(4):
                reconstructed = collector.reconstruct_successor_history(public, branch, step)
                if row['next_frame_valid'][branch, step]:
                    for actual, original in zip(reconstructed, next(expected)):
                        np.testing.assert_array_equal(actual, original)
                else:
                    self.assertIsNone(reconstructed)
        self.assertIsNone(next(expected, None))

    def test_real_life_loss_and_emitted_win_frame_reconstruct_exact_h8(self):
        env, history, model, row = self.collect()
        self.assertEqual(env.lives(), 3)  # Cloned branches cannot mutate the root.
        self.assertEqual(env.steps_left(), 2)
        self.assertEqual(row['next_frames'].shape, (4, 4, 64, 64))
        self.assertEqual(row['next_frames'].dtype, np.uint8)
        np.testing.assert_array_equal(row['next_frame_valid'], row['next_field_valid'])
        np.testing.assert_array_equal(row['next_history_reset'], row['lost_life'])
        self.assertTrue(row['lost_life'][:3, 2].all())
        self.assertTrue(row['won'][3, 0])
        self.assertTrue(row['next_frame_valid'][3, 0])  # Actual engine emits the winning frame.
        self.assertFalse(row['transition_valid'][3, 1:].any())
        self.assertFalse(row['next_history_reset'][3, 1:].any())
        self.assertFalse(row['next_frames'][~row['next_frame_valid']].any())
        self.assert_reconstructed_inputs(model, row)
        reset = collector.reconstruct_successor_history(row, 0, 2)
        self.assertEqual(int(reset[1].sum()), 1)
        self.assertTrue((reset[2] == -1).all())
        following = collector.reconstruct_successor_history(row, 0, 3)
        self.assertEqual(int(following[1].sum()), 2)
        self.assertEqual(int(following[2][-1]), 0)

    def test_full_history_truncation_and_default_arrays_are_unchanged(self):
        env, history, model, row = self.collect(budget=40, prefix=9)
        self.assertTrue(row['history_valid'].all())
        self.assert_reconstructed_inputs(model, row)
        baseline = collector.collect_root(policy(), env, history)
        self.assertEqual(set(row) - set(baseline), {'next_frames', 'next_frame_valid', 'next_history_reset'})
        for key in baseline:
            np.testing.assert_array_equal(row[key], baseline[key])
        before = copy.deepcopy(history)
        row['next_frames'].fill(15)
        np.testing.assert_array_equal(history[0], before[0])
        self.assertEqual(history[1], before[1])

    def test_real_game_over_clears_terminal_history_but_not_into_padding(self):
        _, _, model, row = self.collect(prefix=8)
        self.assertEqual(int(row['lives']), 1)
        self.assertTrue(row['terminal'][:3, 0].all())
        self.assertTrue(row['lost_life'][:3, 0].all())
        self.assertFalse(row['next_history_reset'][:, 1:].any())
        self.assert_reconstructed_inputs(model, row)

    def test_missing_terminal_frame_has_no_reconstructed_target(self):
        class WithoutTerminalFrame:
            """Exercise the supported empty terminal transport on a real win."""
            def __init__(self, env):
                self.env = env

            def __getattr__(self, name):
                return getattr(self.env, name)

            def perform(self, action):
                observation = self.env.perform(action)
                return SimpleNamespace(frame=None if observation.finished else observation.frame,
                                       finished=observation.finished, won=observation.won)

        with patch.object(collector, 'clone_env', side_effect=lambda env: WithoutTerminalFrame(clone_env(env))):
            _, _, model, row = self.collect()
        self.assertTrue(row['won'][3, 0])
        self.assertTrue(row['transition_valid'][3, 0])
        self.assertFalse(row['next_frame_valid'][3].any())
        self.assertFalse(row['next_field_valid'][3].any())
        self.assert_reconstructed_inputs(model, row)

    def test_manifest_and_both_collection_modes_opt_in_without_model_loading(self):
        class Guard:
            def __init__(self, *args, **kwargs):
                self.started = time.monotonic()

            def __call__(self):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'fake.pt'
            checkpoint.write_bytes(b'CPU test model is supplied by the fixture')
            bank = root / 'bank.jsonl'
            bank.write_text('[]')
            for teacher in (False, True):
                for raw in (False, True):
                    with self.subTest(teacher=teacher, raw=raw):
                        model = policy()
                        model.encoder.world_checkpoint_path = checkpoint
                        model.encoder.visibility_checkpoint_path = checkpoint
                        # This is seven tiny independent synthetic real-engine
                        # fixtures, not a production bank or trained checkpoint.
                        specs = [dict(spec(), seed=tier, difficulty=tier) for tier in range(1, 8)]
                        out = root / f'{teacher}-{raw}'
                        argv = ['--checkpoint', str(checkpoint), '--checkpoint-sha256',
                                hashlib.sha256(checkpoint.read_bytes()).hexdigest(), '--bank', str(bank),
                                '--out', str(out), '--levels', '7', '--root-steps', '0',
                                '--behavior-actions', '1', '--max-roots', '1', '--random-fraction', '0']
                        if teacher:
                            argv += ['--teacher-endings-only']
                        if raw:
                            argv += ['--raw-successor-public']
                        with patch.object(collector, 'Budget', Guard), \
                             patch.object(collector, 'checked_specs', return_value=specs), \
                             patch.object(collector, 'load_checkpoint', return_value=(model, {})):
                            result = collector.main(argv)
                        self.assertEqual(result['status'], 'complete')
                        self.assertEqual(result['format'], f'pebby.neural-planning-sequences.v{2 if raw else 1}')
                        self.assertEqual('raw_successor_public' in result, raw)
                        self.assertEqual(json.loads((out / 'manifest.json').read_text())['format'], result['format'])
                        with np.load(out / 'sequences.npz') as arrays:
                            self.assertEqual('next_frames' in arrays, raw)
                            if raw:
                                self.assertEqual(arrays['next_frames'].shape, (7, 4, 4, 64, 64))
                                self.assertEqual(result['arrays']['next_frames']['dtype'], 'uint8')
                                for index in range(7):
                                    row = {key: arrays[key][index] for key in arrays.files}
                                    self.assertIsNotNone(collector.reconstruct_successor_history(row, 3, 0))


if __name__ == '__main__':
    unittest.main()
