"""Action-permutation variant games over the real LS20 engine, and their collector."""

import json
from pathlib import Path
import random
import tempfile
import unittest

from arcengine import GameState
import numpy as np

from pebby import variants as V
from pebby.ls20 import names
from pebby.ls20.env import Ls20Env
from pebby.ls20.generate import build_level
from pebby.ls20.layout import extract
from pebby.ls20.plan import Oracle
from tools import collect_variant_games as collector

BANK = Path('data/ls20-reference-unequal-v1/train.jsonl')


def bank_specs():
    """A few specs per tier 1..3, read straight out of the reference bank."""
    wanted = {1: 3, 2: 3, 3: 3}
    chosen = {tier: [] for tier in wanted}
    with BANK.open() as handle:
        for line in handle:
            spec = json.loads(line)
            tier = spec['difficulty']
            if tier in wanted and len(chosen[tier]) < wanted[tier]:
                chosen[tier].append(spec)
            if all(len(chosen[t]) >= wanted[t] for t in wanted):
                break
    return chosen


def border_spec():
    """Player starts on the left edge at (0, 5); a wall above, free below, a goal far right."""
    return dict(walls=[(0, 4)], start=(0, 5), start_triple=(0, 0, 0),
                goals=[dict(cell=(3, 5), triple=(0, 0, 1))], cyclers=[dict(cell=(1, 5), kind='rotation')],
                refills=[], launchers=[], step_counter=42, step_cost=1, fog=False)


def drive_raw(env, engine_actions):
    """Play engine actions on a bare Ls20Env with competition GAME_OVER handling; yield factored states."""
    from types import MethodType
    env.game.handle_reset = MethodType(lambda game: game.level_reset(), env.game)
    for action in engine_actions:
        env.perform(names.ACTION_IDS[action])
        if env.state == GameState.GAME_OVER:
            env.perform(0)
        yield V.factored_state(env)
        if env.state == GameState.WIN:
            return


class PermutationTableTests(unittest.TestCase):
    def test_24_permutations_with_identity_first(self):
        self.assertEqual(len(V.PERMUTATIONS), 24)
        self.assertEqual(V.PERMUTATIONS[0], (0, 1, 2, 3))
        self.assertEqual(len(set(V.PERMUTATIONS)), 24)
        for v, perm in enumerate(V.PERMUTATIONS):
            for e in range(4):
                self.assertEqual(perm[V.INVERSE_PERMUTATIONS[v][e]], e)
        self.assertEqual(len(V.TILE_CLASSES), 10)
        self.assertEqual(V.TILE_CLASSES.index('outside'), V.OUTSIDE)


class VariantGameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = bank_specs()

    def test_identity_variant_reproduces_ls20env_step_for_step(self):
        specs = [self.specs[1][0], self.specs[1][1]]
        rng = random.Random(3)
        actions = [rng.randrange(4) for _ in range(400)]
        game = V.VariantGame(specs, 0)
        raw = Ls20Env(levels=[build_level(s) for s in specs])
        raw.reset()
        self.assertEqual(game.state(), V.factored_state(raw))
        seen_life_loss = seen_reset = False
        for action, expected in zip(actions, drive_raw(raw, actions)):
            state, info = game.step(action)
            self.assertEqual(info['engine_action'], action)
            self.assertEqual(state, expected)
            seen_life_loss |= info['life_lost']
            seen_reset |= info['reset']
            if info['finished']:
                break
        self.assertTrue(seen_life_loss, 'random play should lose a life within 400 actions')
        self.assertTrue(seen_reset, 'random play should hit GAME_OVER and level-reset within 400 actions')
        self.assertEqual(game.level_index, raw.level_index)
        with self.assertRaises(RuntimeError):
            game.env.game.full_reset()

    def test_non_identity_variant_matches_permuted_engine_action(self):
        specs = [self.specs[1][2], self.specs[2][0]]
        for v in (5, 23):
            perm = V.PERMUTATIONS[v]
            rng = random.Random(v)
            agent_actions = [rng.randrange(4) for _ in range(120)]
            game = V.VariantGame(specs, v)
            raw = Ls20Env(levels=[build_level(s) for s in specs])
            raw.reset()
            engine_actions = [perm[a] for a in agent_actions]
            for a, expected in zip(agent_actions, drive_raw(raw, engine_actions)):
                state, info = game.step(a)
                self.assertEqual(info['engine_action'], perm[a])
                self.assertEqual(state, expected)
                if info['finished']:
                    break

    def test_tile_map_matches_layout_extract_on_tier3(self):
        spec = self.specs[3][0]
        game = V.VariantGame([spec], 0)
        tiles = game.tile_map()
        self.assertEqual(tiles.shape, (12, 12))
        layout = extract(game.env)
        for (x, y) in layout.walls:
            self.assertEqual(tiles[y, x], V.WALL)
        for (x, y), kind in layout.cyclers.items():
            self.assertEqual(tiles[y, x], V.TILE_CLASSES.index(f'cycler_{kind}'))
        for (x, y) in layout.refills:
            self.assertEqual(tiles[y, x], V.REFILL)
        for (x, y), _ in layout.goals:
            self.assertEqual(tiles[y, x], V.GOAL)
        for launcher in layout.launchers:
            x, y = launcher['cell']
            self.assertEqual(tiles[y, x], V.LAUNCHER)
        self.assertGreater(len(layout.launchers), 0)
        self.assertGreater(len(layout.cyclers), 0)
        self.assertGreater(len(layout.refills), 0)
        specials = len(layout.walls) + len(layout.cyclers) + len(layout.refills) + len(layout.goals) \
            + len(layout.launchers)
        self.assertEqual(int((tiles != V.FREE).sum()), specials)
        self.assertEqual(game.goal_triples(), [tuple(g['triple']) for g in spec['goals']])
        self.assertEqual(game.frame().shape, (64, 64))

    def test_neighbours_report_outside_at_border(self):
        game = V.VariantGame([border_spec()], 0)
        self.assertEqual(game.state()['player_x'], 0)
        self.assertEqual(game.state()['player_y'], 5)
        # engine order: up, down, left, right
        self.assertEqual(game.neighbours(), [V.WALL, V.FREE, V.OUTSIDE, V.CYCLER_ROTATION])
        state, info = game.step(1)          # down
        self.assertEqual((state['player_x'], state['player_y']), (0, 6))
        self.assertEqual(game.neighbours()[2], V.OUTSIDE)
        game.step(0)                        # back up
        state, info = game.step(3)          # right onto the rotation cycler
        self.assertEqual(state['rotation'], 1)
        self.assertEqual(game.neighbours(), [V.FREE, V.FREE, V.FREE, V.FREE])

    def test_oracle_agent_action_inverts_the_permutation(self):
        spec = self.specs[1][0]
        game = V.VariantGame([spec], 0)
        oracle = Oracle(game.layout(), engine='auto')
        state = oracle.state_of(game.env)
        engine_action = oracle.action_for(state)
        self.assertIsNotNone(engine_action)
        for v in range(24):
            agent_action = V.oracle_agent_action(oracle, state, v)
            self.assertEqual(V.PERMUTATIONS[v][agent_action], engine_action)
        # Following the oracle through a scrambled variant wins the level.
        v = 17
        game = V.VariantGame([spec], v)
        oracle = Oracle(game.layout(), engine='auto')
        for _ in range(60):
            a = V.oracle_agent_action(oracle, oracle.state_of(game.env), v)
            self.assertIsNotNone(a)
            state, info = game.step(a)
            if info['won']:
                break
        self.assertTrue(game.won)
        self.assertEqual(game.steps, spec['optimal_actions'])
        self.assertIsNone(V.oracle_agent_action(oracle, ('nowhere',), v))

    def test_sampling_helpers(self):
        flat = [s for tier in (1, 2, 3) for s in self.specs[tier]]
        rng = random.Random(0)
        pool = V.pool_specs(flat, 2, rng, tiers=(1, 2, 3))
        self.assertEqual(len(pool), 6)
        games = V.sample_games(pool, 3, [0, 7], rng, tiers=(1, 2, 3))
        self.assertEqual([g[0] for g in games], [0, 0, 0, 7, 7, 7])
        for variant, seeds in games:
            specs = V.game_specs(pool, seeds, tiers=(1, 2, 3))
            self.assertEqual([s['difficulty'] for s in specs], [1, 2, 3])
            self.assertEqual([s['seed'] for s in specs], seeds)
        with self.assertRaises(KeyError):
            V.game_specs(pool, [1, 2, 3], tiers=(1, 2, 3))
        with self.assertRaises(ValueError):
            V.VariantGame([flat[0]], 24)


class CollectorTests(unittest.TestCase):
    def test_cli_end_to_end_two_games_and_chained_states(self):
        specs = bank_specs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bank = tmp / 'bank.jsonl'
            with bank.open('w') as handle:
                for tier in (1, 2, 3):
                    for spec in specs[tier]:
                        handle.write(json.dumps(spec) + '\n')
            argv = ['--bank', str(bank), '--out-dir', str(tmp / 'out'), '--games-per-variant', '1',
                    '--variants', '0', '9', '--tiers', '1', '2', '3', '--levels-per-tier', '2',
                    '--epsilon', '0.5', '--max-actions-per-level', '30', '--workers', '1', '--seed', '4',
                    '--quiet']
            manifest = collector.main(argv)
            self.assertEqual(manifest['games_complete'], 2)
            self.assertEqual(manifest['per_variant_games'], {'0': 1, '9': 1})
            self.assertEqual(manifest['tiers'], [1, 2, 3])
            self.assertTrue(manifest['bank_sha256'])
            self.assertEqual(sum(manifest['levels_completed_histogram'].values()), 2)
            games_dir = tmp / 'out' / 'games'
            total = 0
            for game_id in (0, 1):
                data = np.load(games_dir / f'{game_id:06d}.npz')
                record = json.loads((games_dir / f'{game_id:06d}.json').read_text())
                self.assertTrue(record['complete'])
                for key in collector.SCALAR_KEYS:
                    self.assertIn(key, data.files)
                T = int(data['level_index'].shape[0])
                self.assertEqual(T, record['steps'])
                self.assertGreater(T, 0)
                total += T
                for key, dtype in collector.STEP_KEYS:
                    self.assertIn(key, data.files)
                    self.assertEqual(data[key].dtype, np.dtype(dtype), key)
                    self.assertEqual(data[key].shape[0], T, key)
                self.assertEqual(data['neighbours'].shape, (T, 4))
                self.assertEqual(data['action_map'].tolist(), list(V.PERMUTATIONS[int(data['variant_id'])]))
                self.assertEqual(data['tier_seeds'].shape, (3,))
                self.assertEqual(int(data['variant_id']), record['variant_id'])
                perm = V.PERMUTATIONS[int(data['variant_id'])]
                np.testing.assert_array_equal(data['engine_action'],
                                              np.asarray([perm[a] for a in data['agent_action']], dtype=np.int8))
                # Level of step t+1 is the level after step t.
                after_level = np.where(data['level_changed'], data['level_index'] + 1, data['level_index'])
                np.testing.assert_array_equal(after_level[:-1], data['level_index'][1:])
                # Within a level, after(t) == before(t+1); resets/life losses also chain because the
                # recorded after-state is the restored level.
                for field in collector.STEP_INT16_FIELDS:
                    same_level = ~data['level_changed'][:-1]
                    np.testing.assert_array_equal(data[f'after_{field}'][:-1][same_level],
                                                  data[f'before_{field}'][1:][same_level], field)
                # step_in_level counts actions per level and never exceeds the cap.
                self.assertLessEqual(int(data['step_in_level'].max()), 30)
                self.assertEqual(sum(record['per_level_actions']), T)
                self.assertTrue(bool(data['truncated']) or bool(data['game_won']))
                if bool(data['game_won']):
                    self.assertTrue(bool(data['finished'][-1]))
                    self.assertEqual(int(data['levels_completed']), 3)
                else:
                    self.assertEqual(record['per_level_actions'][int(data['levels_completed'])], 30)
                self.assertTrue(set(np.unique(data['action_source'])) <= {0, 1, 2})
                self.assertTrue(((data['oracle_agent_action'] >= -1) & (data['oracle_agent_action'] < 4)).all())
            self.assertEqual(manifest['total_steps'], total)
            # Resume: a second run plays nothing new and reproduces the manifest counts.
            again = collector.main(argv)
            self.assertEqual(again['games_complete'], 2)
            self.assertEqual(again['total_steps'], total)
            self.assertEqual([g['game_id'] for g in again['games']], [0, 1])


if __name__ == '__main__':
    unittest.main()
