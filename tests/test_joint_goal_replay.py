"""Bounded real-engine K4, semantic-mask and immutable replay tests; CPU only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pebby.agent import joint_goal_replay as replay
from pebby.agent.joint_goal_data import qualification_specs, state_targets
from pebby.agent.world_data import clone_env, verified_context
from pebby.ls20 import names, rails
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.generate import build_level
from pebby.ls20.provenance import generated_context
from tools.audit_cell_appearance_dynamics import sprite_cells


class PublicBehavior:
    def __init__(self):
        self.calls = []

    def encode(self, frames, history_valid, previous_actions):
        assert frames.shape == (1, 8, 64, 64)
        assert history_valid.shape == previous_actions.shape == (1, 8)
        self.calls.append('encode')
        return torch.zeros(1, 148, 96)

    def imagine(self, field, horizon):
        assert horizon == 4 and field.shape == (1, 148, 96)
        self.calls.append('imagine')
        return {'imagined_actions': torch.arange(4)[None, :, None].expand(1, 4, 4)}

    def __call__(self, frames, history_valid, previous_actions):
        assert frames.shape == (1, 8, 64, 64)
        return torch.tensor([[1., 0., 0., 0.]])


def bank_spec(tier, split='train'):
    with Path(f'data/ls20-reference-unequal-v1/{split}.jsonl').open() as stream:
        for line in stream:
            spec = json.loads(line)
            if spec['difficulty'] == tier:
                return spec
    raise AssertionError('missing test bank tier')


class JointGoalReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, index):
        spec = qualification_specs()[index]
        env, oracle, proof = verified_context(spec, search_limit=50_000)
        self.assertIsNotNone(env, proof)
        self.assertEqual(oracle.engine, 'fast')
        self.addCleanup(oracle._distance.close)
        return spec, env, oracle, ([env.render()], [-1])

    def test_actions_chosen_once_before_all_actual_branches_and_exact_k4(self):
        spec, env, oracle, history = self.fixture(3)
        before = copy.deepcopy(history)
        model = PublicBehavior()
        row = replay.collect_root(model, spec, env, oracle, history)
        self.assertEqual(model.calls, ['encode', 'imagine'])
        self.assertFalse(any('field' in key for key in row))
        for branch in range(4):
            actual = clone_env(env)
            for step in range(4):
                observation = actual.perform(names.ACTION_IDS[branch])
                self.assertTrue(row['transition_valid'][branch, step])
                self.assertEqual(row['next_player_cell'][branch, step].tolist(), list(actual.player_cell()))
                self.assertEqual(row['next_triple'][branch, step].tolist(), list(actual.triple()))
                self.assertEqual(int(row['next_steps'][branch, step]), actual.steps_left())
                public = replay.successor_history(row, branch, step)
                if observation.frame is not None:
                    np.testing.assert_array_equal(row['next_frames'][branch, step], observation.frame)
                    np.testing.assert_array_equal(public['frames'][-1], observation.frame)
                else:
                    self.assertIsNone(public)
                if observation.finished:
                    self.assertFalse(row['transition_valid'][branch, step + 1:].any())
                    break
        np.testing.assert_array_equal(history[0], before[0])
        self.assertEqual(history[1], before[1])

    def test_life_reset_missing_frame_terminal_padding_and_known_unreachable(self):
        spec, env, oracle, history = self.fixture(6)
        for _ in range(3):
            observation = env.perform(names.ACTION_IDS[0])
            history = replay.observe(history, observation.frame, 0)
        row = replay.collect_root(PublicBehavior(), spec, env, oracle, history)
        self.assertEqual(int(row['current_distance']), -1)
        self.assertTrue(row['current_distance_valid'])
        self.assertEqual(int(row['optimal']), 0)
        self.assertTrue(row['lost_life'][0, 0])
        self.assertTrue(row['transition_valid'][0, 1])
        public = replay.successor_history(row, 0, 0)
        self.assertEqual(int(public['history_valid'].sum()), 1)
        self.assertTrue((public['previous_actions'] == -1).all())
        # Continue real wall exhaustion to the final life, then branch.
        while env.lives() > 1 or env.steps_left() > 0:
            lives = env.lives()
            observation = env.perform(names.ACTION_IDS[0])
            history = replay.observe(history, observation.frame, 0, env.lives() < lives)
        row = replay.collect_root(PublicBehavior(), spec, env, oracle, history)
        self.assertTrue(row['terminal'][0, 0])
        self.assertFalse(row['transition_valid'][0, 1:].any())
        self.assertFalse(row['next_distance_valid'][0, 0])
        # Emitted terminal pixels are retained; missing frames get no history.
        if not row['next_frame_valid'][0, 0]:
            self.assertIsNone(replay.successor_history(row, 0, 0))
            self.assertFalse(row['next_support'][0, 0].any())

    def test_live_rail_roles_rewind_launcher_landing_and_fog_masks(self):
        spec = bank_spec(5)
        env = Ls20Scenario(build_level(spec), generated_context(spec))
        moved = blocked = False
        for action_id in spec['context_solution'][:12]:
            current = rails.live_states(env.game)
            for action in range(4):
                branch = clone_env(env)
                cell = branch.player_cell()
                observation = branch.perform(names.ACTION_IDS[action])
                target = state_targets(spec, branch, observation.frame)
                for tag, kind in names.CYCLER_TAGS.items():
                    column = {'shape': 2, 'color': 3, 'rotation': 4}[kind]
                    expected = {y * 12 + x for x, y in sprite_cells(branch, tag)}
                    self.assertEqual(set(np.flatnonzero(target['roles'][:, column])), expected)
                if branch.player_cell() == cell:
                    self.assertEqual(rails.live_states(branch.game), current)
                    blocked = True
                elif rails.live_states(branch.game) != current:
                    moved = True
            env.perform(action_id)
        self.assertTrue(moved and blocked)
        spec = bank_spec(3)
        env = Ls20Scenario(build_level(spec), generated_context(spec))
        launched = False
        for action in spec['context_solution']:
            before = env.player_cell()
            observation = env.perform(action)
            after = env.player_cell()
            target = state_targets(spec, env, observation.frame)
            self.assertEqual(set(np.flatnonzero(target['roles'][:, 5])), {y * 12 + x for x, y in (p['cell'] for p in spec['launchers'])})
            launched |= sum(abs(a - b) for a, b in zip(before, after)) > 1
            if observation.finished:
                break
        self.assertTrue(launched)
        spec = bank_spec(7)
        env = Ls20Scenario(build_level(spec), generated_context(spec))
        target = state_targets(spec, env, env.render())
        self.assertTrue((~target['visible']).any())
        self.assertFalse((target['support'] & ~target['visible']).any())
        self.assertFalse((target['goal_attribute_valid'] & ~target['support']).any())
        self.assertFalse(target['goal_attribute_valid'][~target['visible']].any())

    def test_solved_static_goal_covered_surface_and_consumed_refill(self):
        spec, env, oracle, history = self.fixture(2)
        row = replay.collect_root(PublicBehavior(), spec, env, oracle, history)
        solved = 6 * 12 + 4
        self.assertTrue(row['next_goal_presence'][3, 1, solved])
        self.assertTrue(row['next_goal_solved'][3, 1, solved])
        self.assertFalse(row['next_roles'][3, 1, solved, 1])
        self.assertFalse(row['next_goal_attribute_valid'][3, 1, solved])
        spec, env, oracle, history = self.fixture(6)
        row = replay.collect_root(PublicBehavior(), spec, env, oracle, history)
        self.assertTrue(row['roles'][7 * 12 + 4, 6])
        self.assertFalse(row['next_roles'][3, 1, 7 * 12 + 4, 6])

    def test_bounded_native_level_and_reader_alignment_hashes_masks_mutation_close(self):
        spec = bank_spec(1)
        arrays, proof = replay.collect_level(PublicBehavior(), spec, 'train', max_roots=128)
        self.assertLessEqual(len(arrays['seeds']), 128)
        self.assertTrue(proof['teacher_replay_won'])
        self.assertTrue(proof['full_teacher_route_roots'])
        np.testing.assert_array_equal(arrays['trajectory_step'][arrays['collection'] == 0], np.arange(len(proof['teacher_actions'])))
        self.assertTrue(arrays['won'].any())
        self.assertTrue(arrays['lost_life'].any())
        self.assertTrue((arrays['collection'] == replay.COLLECTIONS['on_policy']).any())
        self.assertFalse(proof['on_policy_teacher_steering'])
        self.assertTrue(all(entry['action'] == 0 for entry in proof['actual_on_policy_actions']))
        resets = arrays['collection'] == replay.COLLECTIONS['post_loss']
        np.testing.assert_array_equal(arrays['lives'][resets], [2, 1])
        self.assertTrue(arrays['optimal_valid'][resets].all())
        self.assertTrue((arrays['history_valid'][resets].sum(1) == 1).all())
        self.assertTrue((arrays['previous_actions'][resets] == -1).all())
        # Replaying each recorded exhaustion path proves these are actual
        # lower-life observations, with causal histories cleared by real loss.
        for index in np.flatnonzero(resets):
            actual = Ls20Scenario(build_level(spec), generated_context(spec))
            for action in proof['origins'][index]['actions']:
                observation = actual.perform(names.ACTION_IDS[action])
            self.assertEqual(actual.lives(), int(arrays['lives'][index]))
            np.testing.assert_array_equal(arrays['frames'][index, -1], observation.frame)
        # The learner's own post-life-loss reset states are labelled too: real
        # closed-loop roots with cleared H8 history and a fresh oracle mask.
        learner_resets = [index for index, origin in enumerate(proof['origins'])
                          if 'actual_policy_post_loss' in origin['reasons']]
        self.assertEqual(len(learner_resets), proof['actual_on_policy_post_loss_roots'])
        self.assertGreaterEqual(proof['actual_on_policy_post_loss_roots'], 1)
        self.assertLessEqual(proof['actual_on_policy_post_loss_roots'], proof['actual_on_policy_life_losses'])
        self.assertLessEqual(proof['on_policy_roots'], proof['on_policy_root_quota'])
        self.assertEqual(proof['on_policy_roots'], proof['collection_counts']['on_policy'])
        for index in learner_resets:
            self.assertEqual(int(arrays['collection'][index]), replay.COLLECTIONS['on_policy'])
            self.assertIn(int(arrays['lives'][index]), (1, 2))
            self.assertEqual(int(arrays['history_valid'][index].sum()), 1)
            self.assertTrue((arrays['previous_actions'][index] == -1).all())
            self.assertTrue(bool(arrays['optimal_valid'][index]))
            # Replaying only the learner's own actions reaches exactly this state.
            actual = Ls20Scenario(build_level(spec), generated_context(spec))
            for action in proof['origins'][index]['actions']:
                observation = actual.perform(names.ACTION_IDS[action])
            self.assertEqual(actual.lives(), int(arrays['lives'][index]))
            np.testing.assert_array_equal(arrays['frames'][index, -1], observation.frame)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / 'shards' / str(spec['seed'])
            shard.mkdir(parents=True)
            replay.write_json(root / 'bank.json', [spec])
            producer = root / 'original_producer.py'
            producer.write_text('original producer bytes\n')
            sources = {str(producer): replay.digest(producer)}
            archive = root / 'sources' / replay.digest(producer)
            archive.parent.mkdir()
            archive.write_bytes(producer.read_bytes())
            descriptors = {}
            for key, value in arrays.items():
                path = shard / (key + '.npy')
                np.save(path, value)
                descriptors[key] = dict(sha256=replay.digest(path), bytes=path.stat().st_size)
            replay.write_json(shard / 'manifest.json', dict(format=replay.FORMAT, status='complete', arrays=descriptors,
                                                          producer_bindings_sha256=replay.spec_digest(sources), **proof))
            manifest = dict(format=replay.FORMAT, status='complete', split='train', sources=sources,
                            source_archive_contract={'version': 1},
                            bank_sha256=replay.digest(root / 'bank.json'), shards=[dict(
                                seed=spec['seed'], roots=len(arrays['seeds']), spec_sha256=replay.spec_digest(spec),
                                directory=str(shard.relative_to(root)), manifest_sha256=replay.digest(shard / 'manifest.json'))])
            replay.write_json(root / 'manifest.json', manifest)
            producer.write_text('new checkout source is intentionally different\n')
            with replay.JointGoalReplay(root, 'train') as bank:
                public, targets = bank.batch([len(bank) - 1, 0, 0])
                self.assertEqual(set(public), set(replay.PUBLIC_KEYS))
                for key, values in {**public, **targets}.items():
                    np.testing.assert_array_equal(values.numpy(), arrays[key][[-1, 0, 0]])
                public['frames'].zero_()
                following, _ = bank.batch([0])
                np.testing.assert_array_equal(following['frames'][0], arrays['frames'][0])
                indices = bank.sample(100, np.random.default_rng(7), event_fraction=1)
                self.assertTrue(np.isin(indices, bank._events[0]).all())
                self.assertIn(str((root / 'manifest.json').resolve()), bank.source_bindings)
                self.assertIn(str((shard / 'manifest.json').resolve()), bank.source_bindings)
                self.assertIn(str(archive.resolve()), bank.source_bindings)
                self.assertNotIn(str(producer.resolve()), bank.source_bindings)
                handles = [value._mmap for value in bank._shards[0].values()]
                self.assertFalse(bank._shards[0]['frames'].flags.writeable)
            self.assertTrue(all(handle.closed for handle in handles))
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                bank.batch([0])
            archived_bytes = archive.read_bytes()
            archive.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                replay.JointGoalReplay(root)
            archive.unlink()
            with self.assertRaisesRegex(ValueError, 'archive missing'):
                replay.JointGoalReplay(root)
            legacy = dict(manifest)
            legacy.pop('source_archive_contract')
            replay.write_json(root / 'manifest.json', legacy)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                replay.JointGoalReplay(root)  # No archive: changed live source must fail closed.
            replay.write_json(root / 'manifest.json', manifest)
            archive.write_bytes(archived_bytes)
            for key in ('next_history_reset', 'transition_valid', 'next_distance_valid', 'actions'):
                changed = {name: value.copy() for name, value in arrays.items()}
                changed[key].flat[0] = 9 if key == 'actions' else not changed[key].flat[0]
                with self.subTest(key=key), self.assertRaises(ValueError):
                    replay.validate_arrays(changed, spec)
            path = shard / 'frames.npy'
            with path.open('r+b') as stream:
                stream.seek(-1, 2)
                value = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([value[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                replay.JointGoalReplay(root)

    def test_provenance_and_incomplete_native_proof_rejected(self):
        spec = bank_spec(1)
        for changed in ({'split': 'validation'}, {'source': 'shipped'}, {'search_truncated': True},
                        {'context_index': 4}, {'generator_version': 4}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                replay.check_spec(spec | changed, 'train')
        spec, env, oracle, history = self.fixture(0)
        oracle.truncated = True
        with self.assertRaisesRegex(ValueError, 'complete native'):
            replay.collect_root(PublicBehavior(), spec, env, oracle, history)


if __name__ == '__main__':
    unittest.main()
