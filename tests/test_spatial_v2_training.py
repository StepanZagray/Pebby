"""CPU tests for spatial v2 training: data contract, sampling, weights, plateau, selection, real path."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from pebby.agent import spatial_v2_training as training
from pebby.agent.neural_outcome_planner import EVENT_NAMES, VALUE_BINS
from pebby.agent.spatial_outcome_objective import training_weights
from pebby.agent.world_grounding import SIZES

ROOT = Path(__file__).resolve().parents[1]
V1_CHECKPOINT = ROOT / 'artifacts/spatial-recovery-v1/quality-fit/recovery.pt'
HAVE_V2 = importlib.util.find_spec('pebby.agent.spatial_v2_policy') is not None


def synthetic_level(rng, seed, rows, *, kinds=(0, 1, 2), optimal_zero_fraction=.25):
    """One level's rows in the exact world_data._row contract (frames random)."""
    frames = rng.integers(0, 16, size=(rows, 8, 64, 64), dtype=np.uint8)
    frames[:, :, 61:63, 13:55] = 0  # a blank HUD keeps decode_hud in range
    lengths = rng.integers(1, 9, size=rows)
    history_valid = np.arange(8)[None] >= (8 - lengths)[:, None]
    previous_actions = np.where(history_valid, rng.integers(0, 4, size=(rows, 8)), -1).astype(np.int64)
    previous_actions[:, 0] = np.where(lengths == 8, previous_actions[:, 0], -1)
    current_triple = rng.integers(0, [6, 4, 4], size=(rows, 3)).astype(np.int64)
    next_triple = np.repeat(current_triple[:, None], 4, axis=1).copy()
    change = rng.random((rows, 4)) < .3
    next_triple[change] = rng.integers(0, [6, 4, 4], size=(int(change.sum()), 3))
    optimal = rng.integers(1, 16, size=rows).astype(np.uint8)
    optimal[rng.random(rows) < optimal_zero_fraction] = 0
    return dict(frames=frames, history_valid=history_valid, previous_actions=previous_actions,
                next_player_cell=rng.integers(0, 12, size=(rows, 4, 2)).astype(np.int64),
                next_triple=next_triple,
                next_steps=rng.integers(-1, SIZES[4] - 1, size=(rows, 4)).astype(np.int64),
                next_lives=rng.integers(0, SIZES[5], size=(rows, 4)).astype(np.int64),
                distances=rng.integers(-1, 140, size=(rows, 4)).astype(np.int64),
                lost_life=rng.random((rows, 4)) < .2, terminal=rng.random((rows, 4)) < .1,
                won=rng.random((rows, 4)) < .05, optimal=optimal,
                player_cell=rng.integers(0, 12, size=(rows, 2)).astype(np.int64),
                current_triple=current_triple,
                current_steps=rng.integers(0, SIZES[4] - 1, size=rows).astype(np.int64),
                current_lives=rng.integers(1, SIZES[5], size=rows).astype(np.int64),
                seeds=np.full(rows, seed, dtype=np.int32), context_index=np.zeros(rows, dtype=np.int8),
                row_kind=rng.choice(kinds, size=rows).astype(np.uint8),
                trajectory_id=np.zeros(rows, dtype=np.int64), step=np.arange(rows, dtype=np.int64),
                chosen_action=rng.integers(-1, 4, size=rows).astype(np.int64),
                solvable=np.ones(rows, dtype=bool))


def write_levels(directory, rng, levels, rows, **kwargs):
    (Path(directory) / 'levels').mkdir(parents=True)
    for index in range(levels):
        seed = 1000 + index
        np.savez(Path(directory) / 'levels' / f'{seed}.npz', **synthetic_level(rng, seed, rows, **kwargs))


class FakePlanner(nn.Module):
    """Learnable per-action logits with the real comparator shape; no encoder needed."""

    def __init__(self):
        super().__init__()
        self.fields = nn.ParameterList([nn.Parameter(torch.zeros(4, size)) for size in SIZES])
        self.value = nn.Parameter(torch.zeros(4, VALUE_BINS))
        self.events = nn.Parameter(torch.zeros(4, len(EVENT_NAMES)))
        width = sum(SIZES) + VALUE_BINS + len(EVENT_NAMES)
        self.outcome_projection = nn.Sequential(nn.Linear(width, 16), nn.GELU())
        self.comparator = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1, bias=False))

    def score_outcomes(self, field_logits, value_logits, event_logits):
        probabilities = torch.cat([*[x.float().softmax(-1) for x in field_logits],
                                   value_logits.float().softmax(-1), event_logits.float().sigmoid()], -1)
        branches = self.outcome_projection(probabilities)
        pooled = branches.mean(1, keepdim=True).expand_as(branches)
        return self.comparator(torch.cat((branches, pooled), -1)).squeeze(-1)

    def forward(self, batch):
        fields = tuple(p[None].expand(batch, -1, -1) for p in self.fields)
        value, events = self.value[None].expand(batch, -1, -1), self.events[None].expand(batch, -1, -1)
        return dict(action_logits=self.score_outcomes(fields, value, events), field_logits=fields,
                    value_logits=value, event_logits=events)


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.planner = FakePlanner()
        self.encoder = nn.Linear(1, 1)
        self.encoder_mode = 'frozen'
        self.encoder.requires_grad_(False)

    def config(self):
        return dict(architecture='world', history=8)

    def predict(self, frames, history_valid=None, previous_actions=None):
        frames = frames if torch.is_tensor(frames) else torch.as_tensor(frames)
        return self.planner(frames.shape[0])

    def forward(self, frames, history_valid=None, previous_actions=None):
        return self.predict(frames, history_valid, previous_actions)['action_logits']


def fake_saver(path, policy, **metadata):
    json.dumps(metadata, allow_nan=False)
    torch.save(dict(format='fake', weights=policy.state_dict(), metadata=metadata), path)


def fake_sequential(levels_completed):
    def sequential(policy, device, guard=lambda: None, per_level_cap=300):
        return dict(levels_completed=levels_completed, completed=levels_completed == 7, actions=10,
                    per_level_actions=[10, 0, 0, 0, 0, 0, 0], per_level_caps=[per_level_cap] * 7,
                    lives_left=3, ending='per_level_action_cap', resets=0, protocol_identity='fake',
                    execution_device=str(device), ledger=[])
    return sequential


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(1)
        cls.train_dir, cls.validation_dir = Path(cls.temp.name) / 'train', Path(cls.temp.name) / 'validation'
        write_levels(cls.train_dir, rng, levels=6, rows=5)
        write_levels(cls.validation_dir, rng, levels=2, rows=4)
        cls.store = training.LevelStore(cls.train_dir, frame_cache_bytes=1 << 20)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_store_concatenates_and_fetches_frames_lazily(self):
        store = self.store
        self.assertEqual(len(store), 30)
        self.assertEqual(store.level_count, 6)
        self.assertEqual(store.arrays['seeds'].tolist(), sorted(store.arrays['seeds'].tolist()))
        indices = np.array([29, 0, 7, 12])
        frames = store.frames(indices)
        self.assertEqual(frames.shape, (4, 8, 64, 64))
        expected = np.load(store.paths[5])['frames'][4]
        np.testing.assert_array_equal(frames[0], expected)
        items = store.batch(indices, 'cpu')
        self.assertEqual(items['frames'].dtype, torch.uint8)
        self.assertEqual(tuple(items['distances'].shape), (4, 4))
        self.assertEqual(len(store.manifest['files']), 6)
        self.assertEqual(len(store.manifest['sha256']), 64)

    def test_missing_optional_columns_are_defaulted(self):
        with tempfile.TemporaryDirectory() as temp:
            rng = np.random.default_rng(3)
            level = synthetic_level(rng, 5, 3)
            for key in ('row_kind', 'trajectory_id', 'solvable'):
                level.pop(key)
            (Path(temp) / 'levels').mkdir()
            np.savez(Path(temp) / 'levels' / '5.npz', **level)
            store = training.LevelStore(temp, hash_files=False)
            self.assertEqual(store.missing_keys, {'row_kind', 'trajectory_id', 'solvable'})
            self.assertEqual(store.arrays['row_kind'].tolist(), [0, 0, 0])

    def test_level_first_sampling_is_distinct_and_weighted(self):
        sampler = training.LevelSampler(self.store, batch_size=6, rows_per_level=1, seed=0)
        indices, draw = sampler.sample()
        self.assertEqual(draw['distinct_levels'], 6)
        self.assertEqual(len(np.unique(self.store.level_of_row[indices])), 6)
        sampler = training.LevelSampler(self.store, batch_size=4, rows_per_level=2, seed=0)
        indices, draw = sampler.sample()
        self.assertEqual(draw['distinct_levels'], 2)
        self.assertEqual(len(indices), 4)
        only_route = training.LevelSampler(self.store, batch_size=4, kind_weights=(1., 0., 0.), seed=0)
        for _ in range(5):
            indices, _ = only_route.sample()
            self.assertTrue((self.store.arrays['row_kind'][indices] == 0).all())

    def test_objective_weights_match_hand_computation_on_used_rows(self):
        weights, rows = training.objective_weights(self.store, (1., 0., 1.))
        arrays = self.store.arrays
        used = np.isin(arrays['row_kind'], (0, 2))
        self.assertEqual(rows, int(used.sum()))
        branches = 4 * int(used.sum())
        changed = (arrays['next_triple'][used] != arrays['current_triple'][used][:, None]).sum(axis=(0, 1))
        self.assertEqual(weights['train_branches'], branches)
        self.assertEqual(weights['glyph_changed_counts'], changed.tolist())
        for index in range(3):
            self.assertAlmostEqual(weights['glyph_change_weights'][index][0], branches / (2 * (branches - changed[index])))
            self.assertAlmostEqual(weights['glyph_change_weights'][index][1], branches / (2 * changed[index]))
        for index, name in enumerate(EVENT_NAMES):
            positives = int(arrays[name][used].sum())
            self.assertEqual(weights['event_positive_counts'][index], positives)
            self.assertAlmostEqual(weights['event_positive_weights'][index], min((branches - positives) / positives, 50.))
        # Unfiltered weights differ, so the filtered subset must not reuse them.
        self.assertNotEqual(weights, training_weights({k: arrays[k] for k in training.WEIGHT_KEYS}))

    def test_optimal_zero_rows_stay_in_batches_with_finite_losses(self):
        self.assertGreater(int((self.store.arrays['optimal'] == 0).sum()), 0)
        sampler = training.LevelSampler(self.store, batch_size=30, rows_per_level=5, seed=0)
        indices, draw = sampler.sample()
        self.assertGreater(draw['optimal_zero_rows'], 0)
        weights, _ = training.objective_weights(self.store)
        record = training.compute_losses(FakePolicy(), self.store.batch(indices, 'cpu'), weights)
        self.assertTrue(torch.isfinite(record['total']))
        self.assertEqual(float(record['diagnostics']['policy_valid_count']), 30 - draw['optimal_zero_rows'])

    def test_validation_covers_all_rows(self):
        store = training.LevelStore(self.validation_dir, hash_files=False)
        weights, _ = training.objective_weights(self.store)
        result = training.validate(FakePolicy(), store, weights, batch_size=3, device='cpu')
        self.assertEqual(result['rows'], 8)
        self.assertTrue(np.isfinite(result['total']))
        self.assertIn('policy', result['losses'])


class RuleTests(unittest.TestCase):
    def test_plateau_stopper(self):
        stopper = training.PlateauStopper(patience=2, min_delta=.1)
        self.assertTrue(stopper.update(1.))
        self.assertFalse(stopper.update(.95))  # within min_delta: stale
        self.assertFalse(stopper.should_stop)
        self.assertFalse(stopper.update(.96))
        self.assertTrue(stopper.should_stop)
        self.assertTrue(stopper.update(.5))
        self.assertFalse(stopper.should_stop)
        self.assertEqual(stopper.state()['best'], .5)
        never = training.PlateauStopper(patience=0)
        for value in (3., 3., 3., 3.):
            never.update(value)
        self.assertFalse(never.should_stop)

    def test_selection_rule(self):
        def evaluation(levels, wins, loss):
            gameplay = None if levels is None else dict(sequential=dict(levels_completed=levels),
                                                        generated=None if wins is None else dict(wins=wins))
            return dict(validation=dict(total=loss), gameplay=gameplay)
        candidates = [evaluation(1, 5, .1), evaluation(2, 0, .9), evaluation(2, 3, .8), evaluation(2, 3, .7),
                      evaluation(None, None, .01)]
        self.assertIs(training.select_best(candidates), candidates[3])
        self.assertEqual(training.selection_key(candidates[4]), (0, 0, -.01))
        self.assertIs(training.select_best(candidates[:1] + candidates[4:]), candidates[0])

    def test_lr_schedule(self):
        config = training.TrainConfig(updates=10, warmup=2, schedule='cosine', batch_size=2, eval_every=1)
        with tempfile.TemporaryDirectory() as temp:
            write_levels(temp, np.random.default_rng(0), levels=2, rows=3)
            trainer = training.Trainer(FakePolicy(), training.LevelStore(temp, hash_files=False), config, 'cpu')
        self.assertAlmostEqual(trainer.lr_scale(1), .5)
        self.assertAlmostEqual(trainer.lr_scale(2), 1.)
        self.assertAlmostEqual(trainer.lr_scale(10), 0.)
        self.assertEqual({g['name'] for g in trainer.optimizer.param_groups}, {'planner'})


class RunTests(unittest.TestCase):
    def test_run_with_fake_policy_writes_report_and_selects_best(self):
        with tempfile.TemporaryDirectory() as temp:
            rng = np.random.default_rng(2)
            train_dir, validation_dir, out = Path(temp) / 'train', Path(temp) / 'val', Path(temp) / 'out'
            write_levels(train_dir, rng, levels=4, rows=4)
            write_levels(validation_dir, rng, levels=2, rows=3)
            config = training.TrainConfig(updates=6, batch_size=4, eval_every=2, gameplay_every=4, patience=0,
                                          warmup=1, lr=1e-2, log_every=1)
            calls = iter([1, 2])
            def sequential(policy, device, guard=lambda: None, per_level_cap=300):
                return fake_sequential(next(calls))(policy, device, guard, per_level_cap)
            report = training.run(FakePolicy(), training.LevelStore(train_dir, hash_files=False),
                                  training.LevelStore(validation_dir, hash_files=False), config, out, device='cpu',
                                  sequential=sequential, saver=fake_saver, argv=['test'])
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['stop_reason'], 'max_updates')
            self.assertEqual([e['update'] for e in report['evaluations']], [2, 4, 6])
            self.assertEqual([e['gameplay'] is not None for e in report['evaluations']], [False, True, True])
            self.assertEqual(report['best']['update'], 6)
            self.assertEqual(report['best']['selection_key'][0], 2)
            self.assertTrue((out / 'best.pt').exists() and (out / 'final.pt').exists())
            self.assertTrue((out / 'gameplay-000004.json').exists())
            written = json.loads((out / 'report.json').read_text())
            self.assertEqual(written['objective_weights'], report['objective_weights'])
            self.assertEqual(written['data']['weights_source'],
                             'training_weights over exactly the sampler-eligible train rows')
            saved = torch.load(out / 'best.pt', weights_only=False)['metadata']
            self.assertEqual(saved['update'], 6)
            self.assertEqual(saved['data_manifests']['train_sha256'], report['data']['train']['sha256'])
            self.assertIn('head', saved['git'])

    def test_plateau_stops_training_early(self):
        with tempfile.TemporaryDirectory() as temp:
            rng = np.random.default_rng(4)
            train_dir, validation_dir, out = Path(temp) / 'train', Path(temp) / 'val', Path(temp) / 'out'
            write_levels(train_dir, rng, levels=3, rows=3)
            write_levels(validation_dir, rng, levels=1, rows=3)
            config = training.TrainConfig(updates=50, batch_size=3, eval_every=1, gameplay_every=0, patience=2,
                                          min_delta=10., warmup=0, lr=1e-3)
            report = training.run(FakePolicy(), training.LevelStore(train_dir, hash_files=False),
                                  training.LevelStore(validation_dir, hash_files=False), config, out, device='cpu',
                                  sequential=fake_sequential(0), saver=fake_saver, argv=['test'])
            self.assertEqual(report['stop_reason'], 'plateau')
            self.assertEqual(report['updates'], 3)
            evaluations = report['evaluations']
            self.assertEqual([e['update'] for e in evaluations], [1, 2, 3])  # one record per evaluated update
            # Sub-min_delta improvements count as stale, yet every evaluation is a candidate
            # without periodic gameplay, so the lowest validation loss wins.
            self.assertEqual([e['improved_validation'] for e in evaluations], [True, False, False])
            lowest = min(evaluations, key=lambda e: e['validation']['total'])
            self.assertEqual(report['best']['update'], lowest['update'])
            self.assertEqual([e['gameplay'] is not None for e in evaluations], [False, False, True])
            self.assertEqual(report['final']['update'], 3)


@unittest.skipUnless(HAVE_V2 and V1_CHECKPOINT.exists(), 'spatial_v2_policy or the v1 checkpoint is absent')
class RealPolicyTests(unittest.TestCase):
    def test_from_v1_warm_start_trains_and_best_loads(self):
        from pebby.agent import gameplay_gate, spatial_v2_policy
        from pebby.agent.competition import CompetitionSession
        from pebby.agent.model import load_checkpoint
        from pebby.ls20.generate import build_level
        torch.set_num_threads(2)
        warm = getattr(spatial_v2_policy.SpatialOutcomePolicyV2, 'from_v1_checkpoint', None) or spatial_v2_policy.from_v1_checkpoint
        policy = warm(V1_CHECKPOINT, encoder_mode='frozen', hud_scalars=4)
        self.assertEqual(policy.config()['architecture'], 'world')
        self.assertEqual(policy.config()['history'], 8)
        self.assertFalse(any(p.requires_grad for p in policy.encoder.parameters()))
        level = build_level(dict(walls=[(1, 3), (0, 2), (1, 1)], start=(1, 2), start_triple=(0, 0, 0),
                                 goals=[dict(cell=(2, 2), triple=(0, 0, 0))], cyclers=[], refills=[],
                                 launchers=[], step_counter=2, step_cost=1, fog=False))
        with tempfile.TemporaryDirectory() as temp:
            rng = np.random.default_rng(5)
            train_dir, validation_dir, out = Path(temp) / 'train', Path(temp) / 'val', Path(temp) / 'out'
            write_levels(train_dir, rng, levels=4, rows=2)
            write_levels(validation_dir, rng, levels=1, rows=2)
            config = training.TrainConfig(updates=2, batch_size=4, eval_every=2, gameplay_every=2, patience=0,
                                          warmup=0, per_level_cap=3, validation_batch_size=2)
            session = CompetitionSession([level for _ in range(7)])
            with patch.object(gameplay_gate.competition, 'CompetitionSession', return_value=session):
                report = training.run(policy, training.LevelStore(train_dir, hash_files=False),
                                      training.LevelStore(validation_dir, hash_files=False), config, out,
                                      device='cpu', argv=['test'])
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(len(report['train_log']), 2)
            self.assertTrue(all(np.isfinite(row['loss']) for row in report['train_log']))
            self.assertTrue(np.isfinite(report['evaluations'][-1]['validation']['total']))
            self.assertIn('levels_completed', report['evaluations'][-1]['gameplay']['sequential'])
            self.assertTrue((out / 'report.json').exists())
            loaded, data = load_checkpoint(out / 'best.pt', 'cpu')
            self.assertEqual(data['format'], spatial_v2_policy.FORMAT)
            self.assertEqual(data['metadata']['update'], 2)
            self.assertEqual(data['metadata']['objective_weights'], report['objective_weights'])
            self.assertEqual(loaded.encoder_mode, 'frozen')
            for name, value in policy.planner.state_dict().items():
                torch.testing.assert_close(loaded.planner.state_dict()[name], value)

    def test_finetune_mode_gives_encoder_its_own_learning_rate(self):
        from pebby.agent import spatial_v2_policy
        warm = getattr(spatial_v2_policy.SpatialOutcomePolicyV2, 'from_v1_checkpoint', None) or spatial_v2_policy.from_v1_checkpoint
        policy = warm(V1_CHECKPOINT, encoder_mode='finetune', hud_scalars=4)
        config = training.TrainConfig(lr=1e-4, encoder_lr=1e-5, batch_size=2)
        groups = training.parameter_groups(policy, config)
        rates = {group['name']: group['base_lr'] for group in groups}
        self.assertEqual(rates, {'encoder': 1e-5, 'planner': 1e-4})
        self.assertTrue(any(p.requires_grad for p in policy.encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
