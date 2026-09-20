"""Full-route collector: every route state, learner states, uncapped recoveries, unwinnable rows kept."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pebby.agent import full_route_data as frd
from pebby.agent import world_data as wd
from pebby.ls20 import names

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / 'data/ls20-reference-unequal-v1/validation.jsonl'


def tiny_levels(count=2, lines=40):
    """The first difficulty-1 records of the validation bank, without loading it all."""
    specs = []
    with BANK.open() as stream:
        for index, line in enumerate(stream):
            if index >= lines:
                break
            spec = json.loads(line)
            if spec['difficulty'] == 1:
                specs.append(spec)
            if len(specs) == count:
                break
    if len(specs) < count:
        raise unittest.SkipTest('validation bank lacks enough tiny levels')
    return specs


class FixedPolicy(torch.nn.Module):
    """Always prefers one action; records how many public decisions it made."""

    def __init__(self, action, history=8):
        super().__init__()
        self.action, self.length, self.calls = action, history, 0

    def config(self):
        return {'architecture': 'world', 'history': self.length}

    def forward(self, frames, history_valid=None, previous_actions=None):
        self.calls += 1
        logits = torch.zeros(frames.size(0), 4)
        logits[:, self.action] = 4.
        return logits


class AlternatingPolicy(FixedPolicy):
    """Bounces between two actions so budget burns even when one is a free bump."""

    def forward(self, frames, history_valid=None, previous_actions=None):
        self.calls += 1
        logits = torch.zeros(frames.size(0), 4)
        logits[:, self.action if self.calls % 2 else (self.action + 1) % 4] = 4.
        return logits


class FullRouteDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.specs = tiny_levels(2)
        cls.up = names.ACTION_NAMES.index('up')
        cls.results = {}
        for spec in cls.specs:
            policy = FixedPolicy(cls.up)
            rows, proof = frd.collect_level(spec, policy, learner_cap=120)
            cls.results[spec['seed']] = (rows, proof, policy)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_route_rows_cover_the_whole_solution_with_oracle_actions_optimal(self):
        for spec in self.specs:
            rows, proof, _ = self.results[spec['seed']]
            route = [r for r in rows if int(r['row_kind']) == 0]
            self.assertEqual(len(route), proof['route_length'])
            self.assertEqual(len(route), len(spec['context_solution']))
            self.assertEqual([int(r['step']) for r in route], list(range(len(route))))
            for row in route:
                choice = int(row['chosen_action'])
                self.assertTrue(int(row['optimal']) & (1 << choice))
                self.assertTrue(bool(row['solvable']))
                self.assertFalse(bool(row['lost_life'][choice]))
            self.assertTrue(bool(route[-1]['won'][int(route[-1]['chosen_action'])]))
            self.assertEqual(int(route[-1]['distances'][int(route[-1]['chosen_action'])]), 0)
            self.assertEqual(proof['oracle_backend'], 'fast')
            self.assertIs(proof['search_truncated'], False)

    def test_learner_rollout_and_uncapped_recovery_to_win(self):
        for spec in self.specs:
            rows, proof, policy = self.results[spec['seed']]
            learner = [r for r in rows if int(r['row_kind']) == 1]
            self.assertGreater(len(learner), 0)
            self.assertGreater(proof['learner_actions'], 0)
            self.assertEqual(policy.calls, proof['learner_actions'])
            self.assertIn(proof['learner_stop'], ('won', 'game_over', 'cap', 'exact_attractor_repeat', 'lives_exhausted'))
            self.assertTrue(all(int(r['chosen_action']) == self.up for r in learner))
            recoveries = [t for t in proof['trajectories'] if t['kind'] == 'recovery']
            self.assertGreaterEqual(len(recoveries), 1)
            self.assertTrue(all(t['stop'] == 'won' for t in recoveries))
            for trajectory in recoveries:
                samples = [r for r in rows if int(r['trajectory_id']) == trajectory['trajectory_id']]
                # Every recovery state is labelled once per level: the mistake
                # state is already a learner row and a recovery that rejoins the
                # oracle route with an identical public history is already a route row.
                self.assertEqual(len(samples), trajectory['rows'])
                self.assertEqual(len(samples) + trajectory['duplicate_states_skipped'], len(trajectory['actions']))
                self.assertGreaterEqual(trajectory['duplicate_states_skipped'], 1)
                self.assertTrue(all(int(r['row_kind']) == 2 for r in samples))
                for row in samples:
                    choice = int(row['chosen_action'])
                    self.assertTrue(int(row['optimal']) & (1 << choice))
                if samples:
                    self.assertTrue(bool(samples[-1]['won'][int(samples[-1]['chosen_action'])])
                                    or trajectory['duplicate_states_skipped'] > 1)
            self.assertEqual(proof['recoveries_won'], len(recoveries))
            self.assertEqual(proof['counts_by_kind']['recovery'], sum(t['rows'] for t in recoveries))
            self.assertEqual(proof['counts_by_kind']['learner'],
                             proof['learner_actions'] - proof['duplicate_states_skipped']['learner'])

    def test_unwinnable_rows_are_retained_with_zero_optimal(self):
        unwinnable = []
        for spec in self.specs:
            rows, proof, _ = self.results[spec['seed']]
            unwinnable.extend(r for r in rows if not bool(r['solvable']))
            self.assertEqual(proof['unwinnable_rows'], sum(not bool(r['solvable']) for r in rows))
        if not unwinnable:
            spec = self.specs[0]
            rows, proof = frd.collect_level(spec, AlternatingPolicy(self.up), learner_cap=120)
            unwinnable = [r for r in rows if not bool(r['solvable'])]
        self.assertGreater(len(unwinnable), 0)
        for row in unwinnable:
            self.assertEqual(int(row['optimal']), 0)
            self.assertEqual(int(row['row_kind']), 1)
            live = ~row['lost_life'] & ~row['terminal']
            self.assertTrue((row['distances'][live] == -1).all())

    def test_row_contract_matches_world_data(self):
        spec = self.specs[0]
        rows, _, _ = self.results[spec['seed']]
        initial, oracle, proof = frd.native_context(spec)
        env = wd.clone_env(initial)
        targets, _, _, _ = wd._expand(env, oracle, oracle.distance_for(oracle.state_of(env)), spec['seed'], 0)
        observed, valid, previous = wd.history_arrays([env.render()], [-1], 8)
        reference = wd._row(targets, observed, valid, previous, spec['seed'], proof['context_index'])
        self.assertEqual(set(rows[0]), set(reference) | set(frd.EXTRA_KEYS))
        for key, value in reference.items():
            self.assertEqual(np.asarray(rows[0][key]).dtype, np.asarray(value).dtype, key)
            self.assertEqual(np.asarray(rows[0][key]).shape, np.asarray(value).shape, key)
        self.assertEqual(rows[0]['frames'].shape, (8, 64, 64))
        self.assertEqual(rows[0]['frames'].dtype, np.uint8)
        for key, dtype in (('row_kind', np.uint8), ('trajectory_id', np.int64), ('step', np.int64),
                           ('chosen_action', np.int8), ('solvable', np.bool_)):
            self.assertEqual(np.asarray(rows[0][key]).dtype, dtype, key)
        keys = set()
        for row in rows:
            key = (row['frames'].tobytes(), row['history_valid'].tobytes(), row['previous_actions'].tobytes(),
                   row['player_cell'].tobytes(), row['current_triple'].tobytes(), int(row['current_steps']),
                   int(row['current_lives']))
            keys.add(key)
        self.assertEqual(len(keys), len(rows), 'exact public-history duplicates must be emitted once')

    def test_write_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            total = 0
            for spec in self.specs:
                rows, proof, _ = self.results[spec['seed']]
                array, proof_path = frd.write_level(directory, spec, rows, proof)
                self.assertTrue(frd.level_complete(directory, spec))
                record = json.loads(proof_path.read_text())
                self.assertEqual(record['status'], 'complete')
                self.assertEqual(record['array_sha256'], frd.digest(array))
                total += len(rows)
            loaded = frd.load_rows(directory)
            self.assertEqual(len(loaded['optimal']), total)
            self.assertEqual(len(loaded['seed']), total)
            self.assertEqual(loaded['seed'].dtype, np.int64)
            np.testing.assert_array_equal(loaded['seed'], loaded['seeds'].astype(np.int64))
            self.assertEqual(set(loaded), set(self.results[self.specs[0]['seed']][0][0]) | {'seed'})
            first = self.results[self.specs[0]['seed']][0]
            spec_seed = self.specs[0]['seed']
            selected = loaded['seed'] == spec_seed
            np.testing.assert_array_equal(loaded['frames'][selected], np.stack([r['frames'] for r in first]))
            self.assertFalse(frd.level_complete(directory, {**self.specs[0], 'seed': 424242}))

    def test_route_only_without_policy(self):
        spec = self.specs[0]
        rows, proof = frd.collect_level(spec, None)
        self.assertEqual(proof['counts_by_kind'], {'route': len(rows), 'learner': 0, 'recovery': 0})
        self.assertIsNone(proof['learner_stop'])
        self.assertEqual(proof['recoveries'], 0)


if __name__ == '__main__':
    unittest.main()
