"""Generated-only extended curriculum: contextual validity and bounded rejection."""
import random
import unittest
from unittest.mock import patch

from pebby.ls20 import extended_curriculum as extended
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.generate import RESERVED, _connected, build_level
from pebby.ls20.layout import extract
from pebby.ls20.plan import Oracle
from tools.audit_world_contexts import gameplay_hash


class ExtendedCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = {d: extended.generate_level(seed, d, attempts=16)[0]
                     for d, seed in ((1, 20001), (3, 20008), (5, 20005))}
        assert all(cls.specs.values())

    def test_deterministic_separate_namespace_and_larger_connected_layouts(self):
        self.assertEqual(self.specs[3], extended.generate_level(20008, 3)[0])
        for spec in self.specs.values():
            self.assertEqual(spec['extended_curriculum_version'], 2)
            self.assertEqual(spec['generation_namespace'], extended.NAMESPACE)
            self.assertNotIn('curriculum_version', spec)
            free = {(x, y) for x in range(12) for y in range(12)} - set(spec['walls'])
            self.assertEqual(len(_connected(free, tuple(spec['start']))), len(free))
            self.assertGreaterEqual(spec['room_width'], 6)
            self.assertGreaterEqual(spec['room_height'], 6)
            self.assertEqual(len(free), spec['free_cells'])
            self.assertEqual(spec['gameplay_sha256'], gameplay_hash(spec))
        for seed in (0, 10000, 1000000, 1019999, 2000000):
            with self.assertRaises(ValueError):
                extended.generate_level(seed)

    def test_actual_context_complete_oracle_real_win_and_two_moving_types(self):
        for d, spec in self.specs.items():
            env = Ls20Scenario(build_level(spec), spec['seed'] % 7)
            oracle = Oracle(extract(env), engine='fast', limit=600000)
            self.assertFalse(oracle.truncated)
            self.assertTrue(oracle.solvable)
            self.assertEqual(oracle.solution(), spec['solution'])
            self.assertEqual(spec['training_context_index'], spec['seed'] % 7)
            self.assertEqual(spec['random_transitions_checked'], 8)
            for action in spec['solution']:
                result = env.perform(action)
            self.assertTrue(result.won)
            self.assertEqual(env.lives(), 3)
            if d in (3, 5):
                expected = 1 if d == 3 else 2
                self.assertEqual(spec['patroller_count'], expected)
                self.assertEqual(len(spec['solution_mechanics']['moving_types']), expected)
                self.assertEqual(spec['solution_mechanics']['used_patroller_count'], expected)
        goals = self.specs[5]['goals']
        self.assertEqual(len(goals), 2)
        self.assertNotEqual(goals[0]['triple'], goals[1]['triple'])

    def test_drafts_vary_resources_launchers_and_keep_goal_attributes(self):
        rows = [spec for seed in range(80)
                if (spec := extended.draft(random.Random(seed), seed % 5 + 1))]
        self.assertEqual({s['step_cost'] for s in rows}, {1, 2})
        self.assertEqual({s['step_counter'] for s in rows}, {42})
        self.assertTrue(any(s['launchers'] for s in rows))
        for spec in rows:
            required = 1 if spec['difficulty'] == 1 else 2 if spec['difficulty'] in (2, 3) else 3
            self.assertGreater(len(spec['cyclers']), required)
            for goal in spec['goals']:
                self.assertNotEqual(goal['triple'], spec['start_triple'])
                self.assertEqual(sum(a != b for a, b in zip(goal['triple'], spec['start_triple'])), required)

    def test_row_ten_can_hold_free_cells_and_entities_without_using_reserved_cell(self):
        rows = [spec for seed in range(80)
                if (spec := extended.draft(random.Random(seed), seed % 5 + 1))]
        self.assertTrue(any((2, 10) not in set(spec['walls']) for spec in rows))
        entities = [tuple(cell) for spec in rows
                    for cell in ([spec['start']] + spec['refills'] +
                                 [entry['cell'] for entry in spec['goals'] + spec['cyclers']])]
        self.assertTrue(any(cell[1] == 10 for cell in entities))
        for spec in rows:
            self.assertTrue(RESERVED <= set(spec['walls']))
        self.assertFalse(RESERVED & set(entities))

    def test_accepted_routes_have_eight_moves_and_real_unused_cyclers(self):
        from pebby.ls20 import names
        from pebby.ls20.plan import simulate
        for spec in self.specs.values():
            env = Ls20Scenario(build_level(spec), spec['seed'] % 7)
            layout = extract(env)
            oracle = Oracle(layout, engine='fast', limit=600000)
            state = oracle.start
            minimum = state[6] // layout.step_cost
            contacted = set()
            for action in spec['solution']:
                dx, dy = names.ACTION_DELTAS[names.ACTION_IDS.index(action)]
                target = (state[0][0] + dx, state[0][1] + dy)
                if target in layout.cyclers:
                    contacted.add(target)
                state, outcome = simulate(layout, state, names.ACTION_IDS.index(action), oracle.refills)
                if outcome == 'launched' and state[0] in layout.cyclers:
                    contacted.add(state[0])
                minimum = min(minimum, state[6] // layout.step_cost)
            unused = set(layout.cyclers) - contacted
            self.assertGreaterEqual(minimum, 8)
            self.assertGreaterEqual(spec['slack_moves'], 8)
            self.assertEqual(spec['minimum_slack_moves'], minimum)
            self.assertGreaterEqual(len(unused), 1)
            self.assertEqual(spec['distractor_count'], len(unused))
            required = {extended.KINDS[i] for i in range(3)
                        if any(g['triple'][i] != spec['start_triple'][i] for g in spec['goals'])}
            non_required = sum(layout.cyclers[cell] not in required for cell in unused)
            self.assertEqual(spec['non_required_distractor_count'], non_required)
            self.assertEqual(spec['changing_attributes'], len(required))
            if len(required) < 3:
                self.assertGreaterEqual(non_required, 1)
            else:
                self.assertEqual(non_required, 0)

    def test_rejects_low_final_and_pre_refill_budget_even_after_winning(self):
        # A corridor whose late refill hides the exhausted approach.
        path = ([(x, 1) for x in range(1, 11)] + [(10, 2)] +
                [(x, 3) for x in range(10, 3, -1)])
        free = set(path) | {(1, 2)}
        spec = {**self.specs[1], 'seed': 1020001,
                'walls': sorted({(x, y) for x in range(12) for y in range(12)} - free),
                'start': path[0], 'start_triple': [0, 0, 0],
                'goals': [{'cell': path[-1], 'triple': [0, 0, 1]}],
                'cyclers': [{'cell': (2, 1), 'kind': 'rotation'},
                           {'cell': (1, 2), 'kind': 'color'}],
                'rails': [], 'launchers': [], 'refills': [path[13]],
                'step_counter': 12, 'step_cost': 1}
        self.assertEqual(extended.verify(spec), (None, 'under_eight_route_slack_moves'))
        spec.update(refills=[], step_counter=24)
        self.assertEqual(extended.verify(spec), (None, 'under_eight_final_slack_moves'))

    def test_rejects_routes_without_actual_distractors(self):
        original = self.specs[1]
        unused = set(original['distractor_cells'])
        spec = {**original, 'cyclers': [cycler for cycler in original['cyclers']
                                      if tuple(cycler['cell']) not in unused]}
        self.assertEqual(extended.verify(spec), (None, 'no_unused_distractor_cycler'))

    def test_challenge_profile_retains_tight_routes_without_relaxing_learning(self):
        challenge, _ = extended.generate_level(20001, 1, quality_profile='challenge')
        self.assertIsNotNone(challenge)
        self.assertEqual(challenge['step_counter'], 42)
        self.assertEqual(challenge['step_cost'], 2)
        self.assertEqual(challenge['budget_floor'], 0)
        self.assertGreaterEqual(challenge['minimum_slack_moves'], 0)
        self.assertLess(challenge['minimum_slack_moves'], 8)
        self.assertTrue(challenge['solution_mechanics']['launcher'])
        self.assertGreater(challenge['solution_mechanics']['used_launcher_count'], 0)
        accepted, reason = extended.verify({**challenge, 'quality_profile': 'learning'})
        self.assertIsNone(accepted)
        self.assertIn(reason, ('under_eight_final_slack_moves', 'under_eight_route_slack_moves'))

    def test_challenge_drafts_cover_transfer_features_and_non_required_decoys(self):
        rows = [spec for seed in range(120)
                if (spec := extended.draft(random.Random(seed), seed % 5 + 1, 'challenge'))]
        self.assertEqual({len(s['refills']) for s in rows}, set(range(7)))
        self.assertEqual(max(len(s['launchers']) for s in rows), 8)
        self.assertEqual({len(r['cells']) for s in rows for r in s['rails']}, set(range(2, 7)))
        self.assertTrue(any(len(s['rails']) == 3 for s in rows))
        for spec in rows:
            required = {extended.KINDS[i] for i in range(3)
                        if any(g['triple'][i] != spec['start_triple'][i] for g in spec['goals'])}
            if len(required) < 3:
                self.assertTrue(any(c['kind'] not in required for c in spec['cyclers']))
            if spec['difficulty'] == 5:
                self.assertEqual(len(required), 3)
                self.assertEqual(len(spec['rails']), 3)
            self.assertEqual(spec['step_counter'], 42)
            self.assertIn(spec['step_cost'], (1, 2))
            self.assertEqual(spec['budget_floor'], 0)

    def test_geometry_holdout_is_translation_invariant_and_separated(self):
        from pebby.ls20.generation_quality import geometry_partition
        train = self.specs[1]
        validation, _ = extended.generate_level(1020001, 1)
        self.assertIsNotNone(validation)
        for spec, split in ((train, 'train'), (validation, 'validation')):
            fingerprint, partition = geometry_partition(spec)
            self.assertEqual(partition, split)
            self.assertEqual(spec['geometry_sha256'], fingerprint)
            self.assertEqual(spec['geometry_split'], split)
        self.assertNotEqual(train['geometry_sha256'], validation['geometry_sha256'])
        self.assertEqual(extended.verify({**train, 'seed': 1020001}),
                         (None, 'geometry_split_mismatch'))
        board = {(x, y) for x in range(12) for y in range(12)}
        shape = {(x, y) for x in range(3, 7) for y in range(2, 6)} - {(3, 2)}
        original = {'walls': sorted(board - shape)}
        translated = {'walls': sorted(board - {(x + 1, y + 2) for x, y in shape})}
        self.assertEqual(geometry_partition(original), geometry_partition(translated))

    def test_truncated_search_and_launcher_context_zero_fail_closed(self):
        spec = self.specs[1]
        with patch.object(extended, 'Oracle') as oracle:
            oracle.return_value.truncated = True
            accepted, reason = extended.verify(spec)
            self.assertIsNone(accepted)
            self.assertEqual(reason, 'search_truncated')
            oracle.return_value.solution.assert_not_called()
        launcher = {**spec, 'seed': 20006, 'launchers': [{'cell': [1, 1], 'delta': [1, 0]}]}
        self.assertEqual(launcher['seed'] % 7, 0)
        self.assertEqual(extended.verify(launcher), (None, 'context_zero_launcher_pending_hint'))
        accepted, reasons = extended.generate_level(20001, 1, attempts=2, search_limit=1)
        self.assertIsNone(accepted)
        self.assertEqual(reasons.get('search_truncated'), 2)

    def test_transition_disagreement_stops_instead_of_discarding_draft(self):
        real = extended.simulate
        def incorrect(*args):
            state, outcome = real(*args)
            changed = list(state)
            changed[6] += 1
            return tuple(changed), outcome
        with patch.object(extended, 'simulate', side_effect=incorrect):
            with self.assertRaises(extended.ContractMismatch):
                extended.verify(self.specs[3])


if __name__ == '__main__':
    unittest.main()
