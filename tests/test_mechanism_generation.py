"""Training quality gates exercised through the real contextual engine/oracle."""
import unittest

from pebby.ls20.env import Ls20Scenario
from pebby.ls20.generate import build_level, _connected
from pebby.ls20.generate import FORMAT, GENERATOR_VERSION
from pebby.ls20.generation_quality import budget_floor, geometry_partition, route_budget_slack
from pebby.ls20.layout import extract
from pebby.ls20.plan import Oracle, simulate
from tools import generate_mechanism_pilot as mechanism


def budget_fixture(budget=12, refills=True):
    # A 17-action bent corridor; the refill is reached with zero budget,
    # but the final state misleadingly has eight moves left.
    path = ([(x, 1) for x in range(1, 11)] + [(10, 2)]
            + [(x, 3) for x in range(10, 3, -1)])
    free = set(path) | {(1, 2)}
    return dict(format=FORMAT, generator_version=GENERATOR_VERSION,
                seed=700001, difficulty=2, size=64, pilot_mode='two_attributes',
                walls=sorted({(x, y) for x in range(12) for y in range(12)} - free),
                start=path[0], start_triple=[0, 0, 0],
                goals=[{'cell': path[-1], 'triple': [0, 0, 1]}],
                cyclers=[{'cell': (2, 1), 'kind': 'rotation'},
                         {'cell': (1, 2), 'kind': 'color'}], rails=[], launchers=[],
                refills=[path[13]] if refills else [], step_counter=budget,
                step_cost=1, fog=False)


class MechanismQualityTests(unittest.TestCase):
    def test_zero_margin_before_refill_is_rejected_even_with_final_slack(self):
        accepted, reason = mechanism.verify(budget_fixture(), 180000)
        self.assertIsNone(accepted)
        self.assertEqual(reason, 'under_eight_route_slack_moves')

    def test_refills_need_not_be_used_to_accept_a_good_training_route(self):
        accepted, reason = mechanism.verify(budget_fixture(42, False), 180000)
        self.assertIsNone(reason)
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted['slack_moves'], 25)
        self.assertEqual(accepted['minimum_slack_moves'], 25)
        self.assertEqual(accepted['distractor_count'], 1)

    def test_spawn_satisfied_goals_fail_before_search(self):
        spec = budget_fixture(42, False)
        spec['goals'][0]['triple'] = spec['start_triple'].copy()
        self.assertEqual(mechanism.verify(spec, 180000), (None, 'goal_satisfied_at_spawn'))

    def test_drafts_never_include_spawn_satisfied_goals(self):
        for mode in mechanism.MODES:
            for attempt in range(12):
                spec = mechanism.candidate(700001, mode, attempt)
                if spec is None:
                    continue
                self.assertTrue(all(g['triple'] != spec['start_triple'] for g in spec['goals']), mode)

    def test_all_modes_produce_real_wins_with_verified_quality_and_composition(self):
        accepted = []
        for mode in mechanism.MODES:
            with self.subTest(mode=mode):
                row, _ = mechanism.generate_one(
                    700001, mode, attempts=40, limit=600000,
                    seen_seeds=set(), seen_specs=set(), record_rejection=lambda reason: None)
                self.assertIsNotNone(row, mode)
                self.assertGreaterEqual(row['optimal_actions'], 72 if mode == 'long_route' else 10)
                self.assertGreaterEqual(row['minimum_slack_moves'], budget_floor(row))
                self.assertGreaterEqual(row['slack_moves'], budget_floor(row))
                self.assertGreaterEqual(row['distractor_count'], 1)
                free = {(x, y) for x in range(12) for y in range(12)} - set(row['walls'])
                self.assertEqual(_connected(free, row['start']), free)
                if mode not in ('one_attribute', 'long_route', 'refill_chain'):
                    self.assertGreaterEqual(row['changing_attributes'], 2)
                if mode == 'three_rails':
                    self.assertEqual(row['solution_mechanics']['distinct_moving_cyclers'], 3)
                if mode == 'refill_chain':
                    self.assertGreaterEqual(row['solution_mechanics']['refills_consumed'], 3)
                    self.assertEqual(row['step_cost'], 2)
                if mode in ('three_launchers', 'launcher_network'):
                    self.assertGreaterEqual(row['solution_mechanics']['distinct_launchers'], 3)
                if row['changing_attributes'] < 3:
                    self.assertGreaterEqual(row['nonrequired_distractor_count'], 1)
                self.assertEqual(row['step_counter'], 42)
                self.assertIn(row['step_cost'], (1, 2))
                env = Ls20Scenario(build_level(row), row['seed'] % 7)
                for action in row['solution']:
                    result = env.perform(action)
                self.assertTrue(result.won)
                self.assertEqual(env.lives(), 3)
                accepted.append(row)
        self.assertEqual({r['changing_attributes'] for r in accepted}, {1, 2, 3})
        self.assertEqual({r['difficulty'] for r in accepted}, {1, 2, 3, 4, 5})
        self.assertGreater(len({r['topology'] for r in accepted}), 1)

    def test_trivial_route_and_cost_scaled_zero_margin_are_rejected(self):
        spec = budget_fixture(42, False)
        spec['goals'][0]['cell'] = (4, 1)
        self.assertEqual(mechanism.verify(spec, 180000), (None, 'route_under_10_actions'))
        spec = budget_fixture()
        spec.update(step_counter=24, step_cost=2)
        self.assertEqual(mechanism.verify(spec, 180000), (None, 'under_eight_route_slack_moves'))

    def test_drafts_cover_row_ten_and_vary_rails_without_geometry_leakage(self):
        geometries = {'train': set(), 'validation': set()}
        rail_shapes = set()
        row_ten_entities = 0
        for seed in (700001, 8000001):
            for attempt in range(40):
                spec = mechanism.candidate(seed, 'three_rails', attempt)
                if spec is None:
                    continue
                fingerprint, split = geometry_partition(spec)
                geometries[split].add(fingerprint)
                self.assertEqual(spec['geometry_split'], split)
                self.assertIn((1, 10), spec['walls'])
                cells = ([spec['start']] + spec['refills']
                         + [g['cell'] for g in spec['goals']]
                         + [c['cell'] for c in spec['cyclers']])
                row_ten_entities += sum(cell[1] == 10 for cell in cells)
                rail_cells = [cell for rail in spec['rails'] for cell in rail['cells']]
                x0, y0 = min(x for x, _ in rail_cells), min(y for _, y in rail_cells)
                rail_shapes.add(tuple(sorted(tuple(sorted((x-x0, y-y0) for x, y in rail['cells']))
                                             for rail in spec['rails'])))
        self.assertTrue(all(geometries.values()))
        self.assertFalse(geometries['train'] & geometries['validation'])
        self.assertGreater(row_ten_entities, 0)
        self.assertGreater(len(rail_shapes), 10)

    def test_hint_on_context_is_supported_without_launchers(self):
        accepted = None
        for attempt in range(40):
            spec = mechanism.candidate(700000, 'one_attribute', attempt)
            if spec is not None:
                accepted, _ = mechanism.verify(spec, 600000)
            if accepted is not None:
                break
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted['training_context_index'], 0)
        self.assertGreater(accepted['solution_mechanics']['hint'], 0)

    def test_verifier_rejects_forged_geometry_proof(self):
        spec = budget_fixture(42, False)
        spec.update(geometry_split='train', geometry_sha256='wrong', split='train')
        self.assertEqual(mechanism.verify(spec, 180000), (None, 'geometry_split_mismatch'))

    def test_launcher_landing_refill_does_not_hide_the_charged_move(self):
        spec = budget_fixture(42, False)
        spec.update(start=(1, 2), refills=[(10, 1)],
                    launchers=[{'cell': (1, 1), 'delta': (1, 0)}])
        spec['cyclers'] = [spec['cyclers'][0]]
        layout = extract(Ls20Scenario(build_level(spec), spec['seed'] % 7))
        oracle = Oracle(layout)
        before = list(oracle.start)
        before[6] = 8
        after, outcome = simulate(layout, tuple(before), 0, oracle.refills)
        self.assertEqual(outcome, 'launched')
        self.assertEqual(after[6], 42)
        self.assertEqual(route_budget_slack(layout, tuple(before), after, action=0, outcome=outcome), 7)

    def test_long_route_drafts_randomize_the_navigation_problem(self):
        specs = [mechanism.candidate(700001, 'long_route', attempt) for attempt in range(40)]
        specs = [s for s in specs if s is not None]
        self.assertGreater(len({s['geometry_sha256'] for s in specs}), 25)
        self.assertGreater(len({s['start'] for s in specs}), 10)
        self.assertGreater(len({s['goals'][0]['cell'] for s in specs}), 10)
        self.assertGreater(len({tuple(sorted(s['refills'])) for s in specs}), 25)
        self.assertTrue(all(s['required_cycler_presses'] >= 2 for s in specs))


if __name__ == '__main__':
    unittest.main()
