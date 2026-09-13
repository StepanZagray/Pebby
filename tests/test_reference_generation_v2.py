"""Versioned procedural mechanics, real synthetic engine replay, no shipped game."""
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from pebby.ls20 import bank, names
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.generate import build_level, generate_level
from pebby.ls20.extended_curriculum import gameplay_hash
from pebby.ls20.reference_generator import draft as legacy_draft
from pebby.ls20.reference_generator_v2 import draft, seed_split, geometry_partition, MECHANICS_VERSION
from pebby.ls20.reference_profiles import profile_errors


def first_draft(seed, difficulty=6):
    rng = random.Random(seed)
    for _ in range(1000):
        result = draft(rng, difficulty)
        if result is not None:
            return result
    raise AssertionError('bounded deterministic draft fixture unavailable')


def small_spec(vanishing=True):
    free = {(x, y) for x in range(2, 6) for y in range(2, 5)}
    goals = [dict(cell=(3, 2), triple=[0, 0, 0]), dict(cell=(5, 4), triple=[0, 0, 1])]
    if vanishing:
        goals[0]['vanishing_ring'] = True
    return dict(generator_version=4, walls=sorted({(x, y) for x in range(12) for y in range(12)} - free),
                start=(2, 2), start_triple=[0, 0, 0], goals=goals,
                cyclers=[dict(cell=(4, 3), kind='rotation')], refills=[],
                step_counter=42, step_cost=1, fog=False)


class ReferenceGenerationV2Tests(unittest.TestCase):
    def test_legacy_drafts_keep_their_prechange_seed_identity(self):
        expected = {1: (43, '7cdfcda93b5b56e411d494a2788f59dcc5b8cead53b4562398a768917d21410e'),
                    6: (1, 'c1d981805176e9948a8647a8b90a31095aecab9de9f7b9093b646957b8c602af')}
        for tier, (attempts, digest) in expected.items():
            rng = random.Random(f'ls20-reference-v1:720000:{tier}')
            for _ in range(attempts):
                result = legacy_draft(rng, tier)
            self.assertIsNotNone(result)
            payload = json.dumps(result, sort_keys=True, separators=(',', ':')).encode()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)

    def test_later_goals_can_restore_every_component_without_initial_glyph_goal(self):
        restored = set()
        ring_states = set()
        for seed in range(32):
            result = first_draft(seed)
            self.assertEqual(result, first_draft(seed))
            self.assertEqual(result['generator_version'], 4)
            self.assertEqual(result['mechanics_version'], MECHANICS_VERSION)
            self.assertEqual(profile_errors(result, require_proof=False), [])
            initial = result['start_triple']
            self.assertTrue(all(a != b for a, b in zip(result['goals'][0]['triple'], initial)))
            self.assertEqual(len({tuple(g['triple']) for g in result['goals']}), 2)
            for goal in result['goals']:
                self.assertNotEqual(goal['triple'], initial)
                restored.update(i for i, (a, b) in enumerate(zip(goal['triple'], initial)) if a == b)
                ring_states.add(goal.get('vanishing_ring', False))
                if 'vanishing_ring' in goal:
                    self.assertIs(goal['vanishing_ring'], True)
        self.assertEqual(restored, {0, 1, 2})
        self.assertEqual(ring_states, {False, True})

    def test_vanishing_ring_and_hint_disappear_when_first_goal_clears(self):
        for vanishing in (False, True):
            spec = small_spec(vanishing)
            env = Ls20Scenario(build_level(spec), 5)
            x, y = names.cell_to_pixel(3, 2)
            ring = env.game.current_level.get_sprite_at(x - 1, y - 1, names.TAG_GOAL_RING)
            hint = env.game.current_level.get_sprite_at(x - 1, y - 1, names.TAG_GOAL_HINT_FRAME)
            self.assertTrue(ring.is_visible)
            result = env.perform(4)
            self.assertFalse(result.won)
            self.assertEqual(env.goals_solved(), [True, False])
            self.assertEqual(ring.is_visible, not vanishing)
            if vanishing:
                self.assertFalse(hint.is_visible)
            self.assertEqual(env.lives(), 3)
        self.assertNotEqual(gameplay_hash(small_spec(False)), gameplay_hash(small_spec(True)))

    def test_legacy_plain_goals_render_identically_and_flag_is_explicitly_versioned(self):
        plain = small_spec(False)
        legacy = {**plain, 'generator_version': 3}
        self.assertEqual(Ls20Scenario(build_level(legacy), 5).render(), Ls20Scenario(build_level(plain), 5).render())
        for version, flag in ((3, True), (4, 1), (4, 'yes')):
            bad = small_spec()
            bad['generator_version'] = version
            bad['goals'][0]['vanishing_ring'] = flag
            with self.assertRaises(ValueError):
                build_level(bad)

    def test_seed_ranges_and_d4_geometry_are_distinct_three_way_holdouts(self):
        for seed, expected in ((0, 'train'), (999999, 'train'), (1000000, 'validation'),
                               (1999999, 'validation'), (2000000, 'test'), (2999999, 'test'),
                               (3000000, 'train'), (8000000, 'validation')):
            self.assertEqual(seed_split(seed), expected)
        for bad in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                seed_split(bad)
        observed = set()
        for seed in range(16):
            row = first_draft(seed, 1)
            expected = geometry_partition(row)
            observed.add(expected[1])
            for swap in (False, True):
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        free = {(x, y) for x in range(12) for y in range(12)} - set(map(tuple, row['walls']))
                        transformed = {((11 if sx < 0 else 0) + sx * (y if swap else x),
                                        (11 if sy < 0 else 0) + sy * (x if swap else y)) for x, y in free}
                        other = {**row, 'walls': sorted({(x, y) for x in range(12) for y in range(12)} - transformed)}
                        self.assertEqual(geometry_partition(other), expected)
        self.assertEqual(observed, {'train', 'validation', 'test'})

    def test_opt_in_dispatch_and_legacy_test_bank_fail_before_pool_creation(self):
        with patch('pebby.ls20.reference_generator.generate_level', return_value='legacy') as old:
            self.assertEqual(generate_level(12), 'legacy')
            old.assert_called_once()
        with patch('pebby.ls20.reference_generator_v2.generate_level', return_value='new') as new:
            self.assertEqual(generate_level(2000000, generator_version=4), 'new')
            new.assert_called_once()
        with patch('pebby.ls20.bank.Pool') as pool:
            with self.assertRaisesRegex(ValueError, 'generator_version=4'):
                bank.build(1, split='test')
            pool.assert_not_called()
            pool.return_value.__enter__.return_value.map.return_value = []
            bank.build(2, split='test', difficulties=(1,), generator_version=4)
            jobs = pool.return_value.__enter__.return_value.map.call_args.args[1]
            self.assertEqual(jobs, [(2000000, 1, 'test', 4), (2000001, 1, 'test', 4)])
        with patch('pebby.ls20.bank.generate_level', return_value='row') as generator:
            self.assertEqual(bank._one((2000000, 1, 'test', 4)), 'row')
            generator.assert_called_once_with(2000000, 1, split='test', generator_version=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'versions.jsonl'
            bank.save([{'generator_version': version} for version in (2, 3, 4)], path)
            self.assertEqual(len(bank.load(path)), 3)

    def test_seed_split_mismatch_is_rejected_before_any_generation(self):
        with patch('pebby.ls20.reference_generator_v2.draft') as generator:
            for seed, split in ((2000000, 'train'), (1000000, 'test'), (12, 'validation')):
                with self.assertRaisesRegex(ValueError, 'seed range'):
                    generate_level(seed, generator_version=4, split=split)
            generator.assert_not_called()

    def test_v4_test_row_has_real_complete_proof_and_test_geometry(self):
        row = generate_level(2000000, 1, attempts=80, generator_version=4)
        self.assertEqual(row['split'], 'test')
        self.assertEqual(row['geometry_split'], 'test')
        self.assertEqual(row['geometry_version'], 'dihedral-three-way-v2')
        self.assertEqual(row['proof']['generator_version'], 4)
        self.assertEqual(profile_errors(row), [])
        self.assertFalse(row['search_truncated'])
        env = Ls20Scenario(build_level(row), 0)
        for action in row['solution']:
            result = env.perform(action)
        self.assertTrue(result.won)
        self.assertEqual(env.lives(), 3)


if __name__ == '__main__':
    unittest.main()
