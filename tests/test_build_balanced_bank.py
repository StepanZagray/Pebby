"""Balanced generator-v4 bank builder: ordering proof, selection, and a small real build."""
import json
from pathlib import Path
import tempfile
import time
import unittest

from pebby.ls20.reference_generator_v2 import seed_split
from tools import build_balanced_bank as builder


def _corridor_spec(free, goals, cyclers, start, budget):
    """A hand-built two-goal level: every cell outside ``free`` is a wall."""
    free = {tuple(cell) for cell in free}
    return dict(walls=sorted({(x, y) for x in range(12) for y in range(12)} - free),
                start=tuple(start), start_triple=(0, 0, 0), goals=goals, cyclers=cyclers,
                refills=[], launchers=[], rails=[], step_counter=budget, step_cost=1, fog=False,
                seed=1, difficulty=6, training_context_index=0)


def forced_spec():
    """start -> rotation -> goal B -> rotation -> goal A (dead end).

    Goal A (index 0) can only be entered through B's pad, which rejects the
    player until B's triple matches; matching solves B on entry. So A is never
    cleared first: goals [A, B] with A behind B, like shipped level 6.
    """
    row = 5
    return _corridor_spec(free=[(x, row) for x in range(2, 7)],
                          goals=[dict(cell=(6, row), triple=(0, 0, 2)), dict(cell=(4, row), triple=(0, 0, 1))],
                          cyclers=[dict(cell=(3, row), kind='rotation'), dict(cell=(5, row), kind='rotation')],
                          start=(2, row), budget=20)


def free_order_spec():
    """goal A <- rotation <- start -> rotation -> goal B: either goal can go first."""
    row = 5
    return _corridor_spec(free=[(x, row) for x in range(2, 7)],
                          goals=[dict(cell=(2, row), triple=(0, 0, 1)), dict(cell=(6, row), triple=(0, 0, 2))],
                          cyclers=[dict(cell=(3, row), kind='rotation'), dict(cell=(5, row), kind='rotation')],
                          start=(4, row), budget=30)


class ForcedGoalOrderTests(unittest.TestCase):
    def test_goal_behind_the_other_is_a_proved_ordering_on_both_backends(self):
        for engine in ('fast', 'reference'):
            with self.subTest(engine=engine):
                analysis = builder.analyze_spec_goal_order(forced_spec(), context_index=0, search_limit=200_000, engine=engine)
                self.assertEqual(analysis['goal_count'], 2)
                self.assertTrue(analysis['forced_goal_order'])
                self.assertEqual(analysis['forced_first_goal'], [4, 5])
                # Winning routes pass only through "nothing", "B solved" and "both solved".
                self.assertEqual(analysis['winning_goal_masks'], [0, 2, 3])

    def test_independently_reachable_goals_are_not_forced(self):
        for engine in ('fast', 'reference'):
            with self.subTest(engine=engine):
                analysis = builder.analyze_spec_goal_order(free_order_spec(), context_index=0, search_limit=200_000, engine=engine)
                self.assertFalse(analysis['forced_goal_order'])
                self.assertIsNone(analysis['forced_first_goal'])
                self.assertEqual(analysis['winning_goal_masks'], [0, 1, 2, 3])

    def test_unsolvable_spec_is_refused(self):
        spec = forced_spec()
        spec['goals'][0]['triple'] = (1, 0, 2)  # needs a shape change, and the level has no shape cycler
        with self.assertRaises(ValueError):
            builder.analyze_spec_goal_order(spec, context_index=0, search_limit=200_000)
        with self.assertRaises(ValueError):  # a truncated search proves nothing
            builder.analyze_spec_goal_order(forced_spec(), context_index=0, search_limit=4)


class SelectionTests(unittest.TestCase):
    def test_seed_ranges_are_disjoint_per_split_and_tier(self):
        seen = set()
        for split in ('train', 'validation', 'test'):
            for tier in range(1, 8):
                for index in (0, 1, builder.TIER_STRIDE - 1):
                    seed = builder.tier_seed(split, tier, index)
                    self.assertEqual(seed_split(seed), split)
                    self.assertNotIn(seed, seen)
                    seen.add(seed)
        with self.assertRaises(ValueError):
            builder.tier_seed('train', 1, builder.TIER_STRIDE)

    def test_forced_quota_only_applies_to_two_goal_tiers(self):
        self.assertEqual(builder.forced_needed(6, 10, 0.3), 3)
        self.assertEqual(builder.forced_needed(6, 10, 0.0), 0)
        self.assertEqual([builder.forced_needed(t, 10, 0.3) for t in (1, 2, 3, 4, 5, 7)], [0] * 6)

    def test_select_rows_swaps_forced_levels_in_deterministically(self):
        rows = [dict(seed=s, forced_goal_order=(s % 4 == 3)) for s in range(12)]
        base = builder.select_rows(rows, 4, 0)
        self.assertEqual([r['seed'] for r in base], [0, 1, 2, 3])
        chosen = builder.select_rows(rows, 4, 2)
        self.assertEqual([r['seed'] for r in chosen], [0, 1, 3, 7])
        self.assertTrue(builder.satisfied(rows, 4, 2))
        self.assertFalse(builder.satisfied(rows, 4, 4))
        self.assertEqual(builder.select_rows(rows, 4, 4), builder.select_rows(rows[::-1], 4, 4))


class SmallBuildTests(unittest.TestCase):
    def test_small_end_to_end_build_writes_disjoint_audited_banks(self):
        started = time.perf_counter()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'bank'
            argv = ['--out-dir', str(out), '--train-per-tier', '2', '--validation-per-tier', '2',
                    '--test-per-tier', '2', '--tiers', '1', '2', '3', '--workers', '2',
                    '--generator-version', '4', '--reserve-gib', '0.5']
            self.assertEqual(builder.main(argv), 0)
            elapsed = time.perf_counter() - started
            report = json.loads((out / 'generation-report.json').read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['generator_version'], 4)
            self.assertEqual(report['quotas'], dict(train=2, validation=2, test=2))
            self.assertEqual(sorted(report['timing_per_tier']), ['1', '2', '3'])
            self.assertTrue(all(t['seconds_per_accepted'] > 0 for t in report['timing_per_tier'].values()))
            self.assertEqual(report['forced_order_count'], dict(train=0, validation=0, test=0))
            audit = json.loads((out / 'audit-report.json').read_text())
            self.assertEqual(audit['status'], 'complete', audit['errors'])
            self.assertTrue(all(not any(v.values()) for v in audit['pairwise_overlap'].values()))
            seeds = {}
            for split in ('train', 'validation', 'test'):
                rows = [json.loads(line) for line in (out / f'{split}.jsonl').read_text().splitlines()]
                self.assertEqual([r['difficulty'] for r in rows], [1, 1, 2, 2, 3, 3])
                self.assertTrue(all(r['generator_version'] == 4 and r['split'] == split for r in rows))
                self.assertTrue(all(seed_split(r['seed']) == split for r in rows))
                self.assertTrue(all(r['goal_count'] == 1 and r['forced_goal_order'] is None for r in rows))
                seeds[split] = {r['seed'] for r in rows}
            self.assertFalse(seeds['train'] & seeds['validation'] or seeds['train'] & seeds['test'])
            # Resuming a complete build replays the progress log and dispatches nothing.
            report2, audit2 = builder.build(builder.parse_args(argv + ['--skip-audit']))
            self.assertEqual(report2['status'], 'complete')
            self.assertIsNone(audit2)
            self.assertEqual({s: t['seeds_tried'] for s, t in report2['timing_per_tier'].items()},
                             {s: t['seeds_tried'] for s, t in report['timing_per_tier'].items()})
        self.assertLess(elapsed, 90.0, f'small build took {elapsed:.1f}s')


if __name__ == '__main__':
    unittest.main()
