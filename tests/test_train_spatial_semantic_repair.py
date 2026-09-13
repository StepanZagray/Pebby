"""CPU contracts for matched semantic replay and public model inputs."""
from pathlib import Path
from types import SimpleNamespace
import math
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from tools.cache_reference_outcome_inputs import SCHEMA
from tools import train_spatial_semantic_repair as trainer
from tools.train_spatial_semantic_repair import add_semantic, matched_items, build_model, forward, input_reliance
from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner


def arrays(seeds):
    result = {name: np.zeros((len(seeds), *shape), dtype=dtype) for name, (dtype, shape) in SCHEMA.items()}
    result['seeds'][:] = seeds
    result['optimal'][:] = 1
    result['semantic'] = np.full((len(seeds), 144, 22), .5, np.float32)
    for start, size in ((8, 6), (14, 4), (18, 4)):
        result['semantic'][..., start:start+size] = 1 / size
    return result


class SemanticTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        assert not torch.cuda.is_initialized()

    def test_replay_replaces_semantics_and_targets_together_without_aliasing(self):
        base, recent = arrays([11, 22]), arrays([22])
        recent['semantic'][..., 0] = .99
        recent['optimal'][0] = 0
        chosen = dict(seeds=np.array([11, 22]), base_rows=np.array([0, 1]), recent_rows=np.array([-1, 0]))
        items = matched_items(base, recent, chosen, 'cpu')
        np.testing.assert_array_equal(items['semantic'][1], recent['semantic'][0])
        self.assertEqual(items['optimal'].tolist(), [1, 0])
        items['semantic'].zero_()
        self.assertEqual(float(base['semantic'][0, 0, 0]), .5)
        self.assertAlmostEqual(float(recent['semantic'][0, 0, 0]), .99)
        chosen['seeds'][1] = 999
        with self.assertRaisesRegex(ValueError, 'base rows'):
            matched_items(base, recent, chosen, 'cpu')

    def test_cache_alignment_checks_row_and_seed_identity(self):
        group = arrays([11, 22])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name in ('rows', 'seeds', 'semantic'):
                np.save(path / (name + '.npy'), group[name])
            add_semantic(group, path)
            self.assertIsInstance(group['semantic'], np.memmap)
            np.save(path / 'seeds.npy', [22, 11])
            with self.assertRaisesRegex(ValueError, 'alignment'):
                add_semantic(group, path)
            group['semantic']._mmap.close()

    def models(self):
        torch.manual_seed(3)
        parent = SpatialOutcomePlanner(channels=8, width=8, hud_width=8, summary=16, comparator_hidden=16)
        envelope = dict(planner_config=parent.config(), planner_weights=parent.state_dict())
        return parent, build_model(envelope, 'control', 'cpu'), build_model(envelope, 'actor', 'cpu')

    def test_arms_share_initial_scene_path_and_exact_parent_behavior(self):
        parent, control, actor = self.models()
        for name, value in control.state_dict().items():
            torch.testing.assert_close(value, actor.state_dict()[name], atol=0, rtol=0)
        features = [torch.randn(2, 160, 8), torch.randn(2, 160, 8), torch.randn(2, 14), torch.randn(2, 144).softmax(-1)]
        semantic = torch.from_numpy(arrays([1, 2])['semantic'])
        with torch.no_grad():
            expected = parent(*features)
            for model in (control, actor):
                actual = model(*features, semantic)
                for name in ('action_logits', 'event_logits', 'value_logits'):
                    torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        self.assertIsNone(control.actor_readout)
        self.assertTrue(all(p.requires_grad for p in actor.parameters()))

    def test_training_forward_has_only_five_public_inputs(self):
        items = {name: torch.from_numpy(value) for name, value in arrays([1, 2]).items()}
        player = torch.ones(2, 144) / 144
        model = mock.Mock(return_value={'predicted': 'public'})
        with mock.patch('tools.train_spatial_semantic_repair.player_probabilities', return_value=player), \
             mock.patch('tools.train_spatial_semantic_repair.spatial_outcome_losses', return_value='loss') as loss:
            self.assertEqual(forward(model, items, {}, {}, 'float32'), 'loss')
        self.assertEqual(len(model.call_args.args), 5)
        for actual, expected in zip(model.call_args.args, (items['raw'], items['state'], items['glyph'], player, items['semantic'])):
            self.assertIs(actual, expected)
        self.assertIs(loss.call_args.args[2], items)


class ProspectiveSelectionTests(unittest.TestCase):
    def record(self, count, valid, policy=2., physical=3.):
        return dict(losses=dict(policy=torch.tensor(policy), teacher_policy=torch.tensor(policy + 1),
                                physical=torch.tensor(physical), value=torch.tensor(4.), events=torch.tensor(5.)),
            diagnostics=dict(policy_valid_count=torch.tensor(valid), sample_metric=torch.tensor(float(count))),
            diagnostic_weights=dict(policy_valid_count=torch.tensor(1), sample_metric=torch.tensor(count)))

    def test_fixed_interval_includes_final_once_and_never_adds_zero(self):
        self.assertEqual(trainer.validation_steps(250, 100), (100, 200, 250))
        self.assertEqual(trainer.validation_steps(200, 100), (100, 200))
        self.assertEqual(trainer.validation_steps(50, 100), (50,))
        for args in [(0, 10), (10, 0), (True, 1)]:
            with self.assertRaises(ValueError): trainer.validation_steps(*args)

    def test_schedule_limits_are_checked_before_allocation_or_runtime_setup(self):
        with mock.patch.object(trainer, 'range', side_effect=AssertionError('schedule allocated'), create=True):
            for steps, every in [(10**100, 1), (1_000_001, 1_000_000), (1_000_000, 1)]:
                with self.assertRaises(ValueError): trainer.validation_steps(steps, every)
            with mock.patch.object(trainer, 'gpu_available') as gpu, mock.patch.object(trainer, 'guard') as guard, \
                 mock.patch('sys.stderr'):
                with self.assertRaises(SystemExit) as error:
                    trainer.main(['--out', '/unused-invalid-schedule', '--qualify', '--steps', '1000000', '--validation-every', '1'])
                self.assertEqual(error.exception.code, 2)
                gpu.assert_not_called(); guard.assert_not_called()
        schedule = trainer.validation_steps(1_000_000, 100)
        self.assertEqual(len(schedule), 10_000)
        self.assertEqual(schedule[-1], 1_000_000)

    def test_tier_balanced_set_nll_and_policy_loss_use_correct_denominators(self):
        overall = trainer.ValidationStats(); tiers = {tier: trainer.ValidationStats() for tier in range(1, 8)}
        first_scores = torch.zeros(3, 4); first_masks = torch.tensor([3, 1, 0])
        second_scores = torch.tensor([[0., math.log(4.), 0., 0.]])
        for accumulator in (overall, tiers[1]):
            accumulator.update(first_scores, first_masks, self.record(3, 2, policy=2., physical=3.))
        for accumulator in (overall, tiers[2]):
            accumulator.update(second_scores, torch.tensor([1]), self.record(1, 1, policy=5., physical=7.))
        result = trainer.validation_summary(overall, tiers)
        tier1 = (math.log(2) + math.log(4)) / 2
        self.assertAlmostEqual(result['tier_balanced_policy_set_nll'], (tier1 + math.log(7)) / 2, places=6)
        self.assertAlmostEqual(result['optimal_set_nll'], (math.log(2) + math.log(4) + math.log(7)) / 3, places=6)
        self.assertAlmostEqual(result['losses']['policy'], 3.)
        self.assertAlmostEqual(result['losses']['physical'], 4.)
        self.assertEqual(result['policy_undefined_roots'], 1)
        self.assertEqual(result['included_tiers'], ['1', '2'])
        self.assertEqual(result['excluded_tiers'], ['3', '4', '5', '6', '7'])
        self.assertIsNone(result['tiers']['3']['optimal_set_nll'])
        self.assertIsNone(result['tiers']['3']['total_loss'])
        undefined = trainer.ValidationStats(); undefined.update(torch.zeros(1, 4), torch.tensor([0]), self.record(1, 0))
        empty = trainer.validation_summary(undefined, {1: undefined})
        self.assertIsNone(empty['tier_balanced_policy_set_nll'])
        self.assertIsNone(empty['losses']['policy'])
        self.assertEqual(empty['losses']['physical'], 3.)
        self.assertEqual(empty['total_loss'], 12.)

    def test_full_evaluation_visits_each_row_once_restores_mode_and_rng(self):
        data = arrays([11, 22, 33, 44]); data['glyph'][:, 0] = np.arange(4)
        data['optimal'][:] = [1, 0, 3, 2]
        seen = []
        class Model(torch.nn.Module):
            def forward(self, raw, state, glyph, player, semantic):
                self_outer.assertFalse(self.training)
                self_outer.assertFalse(torch.is_grad_enabled())
                seen.extend(glyph[:, 0].long().tolist())
                return {'action_logits': torch.zeros(len(raw), 4)}
        self_outer = self; model = Model().train()
        def losses(current, predicted, items, weights):
            return self.record(len(items['optimal']), int((items['optimal'] != 0).sum()))
        rng = torch.get_rng_state().clone()
        with mock.patch.object(trainer, 'guard'), mock.patch.object(trainer, 'player_probabilities', side_effect=lambda items, _: torch.ones(len(items['raw']), 144) / 144), \
             mock.patch.object(trainer, 'spatial_outcome_losses', side_effect=losses):
            result = trainer.evaluate(model, data, {}, {}, {11: 2, 22: 1, 33: 2, 44: 7}, size=1, device='cpu')
        self.assertEqual(sorted(seen), [0, 1, 2, 3]); self.assertEqual(len(seen), 4)
        self.assertTrue(model.training); self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(result['rows'], 4); self.assertEqual(result['policy_defined_roots'], 3)
        self.assertEqual(result['tiers']['2']['rows'], 2)

    def test_first_minimum_selected_final_preserved_without_aliasing(self):
        model = torch.nn.Linear(1, 1, bias=False); selector = trainer.CheckpointSelection('validation-policy')
        scores = [.5, .2, .2]
        provenance = dict(sampling_sha256='prefix', tier_counts={'1': 10})
        for step, score in enumerate(scores, 1):
            with torch.no_grad(): model.weight.fill_(step)
            selector.observe(model, step, 3, dict(tier_balanced_policy_set_nll=score, included_tiers=['1'], excluded_tiers=[]), provenance)
        self.assertEqual(selector.selected['step'], 2); self.assertEqual(selector.final['step'], 3)
        provenance['tier_counts']['1'] = 999
        with torch.no_grad(): model.weight.zero_()
        self.assertEqual(float(selector.selected['weights']['weight']), 2.)
        self.assertEqual(float(selector.final['weights']['weight']), 3.)
        self.assertEqual(selector.selected['provenance']['tier_counts']['1'], 10)
        selector.final['weights']['weight'].zero_()
        self.assertEqual(float(selector.selected['weights']['weight']), 2.)
        meta = trainer.selection_metadata(SimpleNamespace(steps=3, selection='validation-policy', validation_every=1), selector.selected, 'selected')
        self.assertEqual(meta['selected_step'], 2); self.assertEqual(meta['completed_run_steps'], 3)
        self.assertFalse(meta['exposed_validation_is_fresh'])
        final_meta = trainer.selection_metadata(SimpleNamespace(steps=3, selection='validation-policy', validation_every=1), selector.final, 'final')
        self.assertEqual(final_meta['criterion'], 'fixed final optimizer step')
        self.assertEqual(final_meta['criterion_value'], 3)
        with self.assertRaises(ValueError):
            trainer.selection_metadata(SimpleNamespace(steps=3, selection='validation-policy', validation_every=1), selector.selected, 'final')

    def test_final_mode_never_chooses_earlier_better_loss_and_none_is_explicit(self):
        model = torch.nn.Linear(1, 1, bias=False); selector = trainer.CheckpointSelection('final')
        selector.observe(model, 1, 2, {'tier_balanced_policy_set_nll': .01}, {})
        self.assertIsNone(selector.selected)
        selector.observe(model, 2, 2, {'tier_balanced_policy_set_nll': .5}, {})
        self.assertEqual(selector.selected['step'], 2)
        empty = trainer.CheckpointSelection('validation-policy')
        empty.observe(model, 2, 2, {'tier_balanced_policy_set_nll': None}, {})
        self.assertIsNone(empty.selected); self.assertEqual(empty.final['step'], 2)

    def test_checkpoint_actual_steps_and_sampling_prefix_match_saved_weights(self):
        parent = SpatialOutcomePlanner(channels=8, width=8, hud_width=8, summary=16, comparator_hidden=16)
        model = build_model(dict(planner_config=parent.config(), planner_weights=parent.state_dict()), 'control', 'cpu')
        snapshot = dict(step=3, weights={k: v.clone() for k, v in model.state_dict().items()}, criterion_value=.5,
            validation=dict(included_tiers=['1'], excluded_tiers=[]), provenance=dict(sampling_sha256='three-step-prefix'))
        args = SimpleNamespace(steps=10, selection='validation-policy', validation_every=2, lr=.001, tier_counts=[64, 0, 0, 0, 0, 0, 0],
                               precision='float32', seed=42, cache=Path('base'), recent=Path('recent'), semantics=Path('semantic'))
        report = dict(source_sha256={}, sampling={}, quality_row_file_sha256={}, base_manifest_sha256='base', recent_manifest_sha256='recent',
                      objective_weights={}, qualification_sha256='qualified', semantic_manifest_sha256='semantic')
        with mock.patch.object(trainer, 'load_perceptor'), mock.patch.object(trainer, 'checkpoint_from_parent', return_value={}):
            saved = trainer.checkpoint({}, model, 'control', args, report, snapshot, role='selected')
            self.assertEqual(saved['optimizer_steps'], 3)
            self.assertEqual(saved['checkpoint_selection']['selected_step'], 3)
            self.assertEqual(saved['planner_sampling_sha256'], 'three-step-prefix')
            with torch.no_grad(): next(model.parameters()).add_(1.)
            with self.assertRaisesRegex(ValueError, 'snapshot'):
                trainer.checkpoint({}, model, 'control', args, report, snapshot, role='selected')

    def test_gradient_norm_return_is_preclip_and_stored_gradient_is_postclip(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad(): model.weight.fill_(1.)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        with mock.patch.object(trainer, 'forward', side_effect=lambda *args: {'total': 100 * model.weight.square().sum()}):
            _, norm = trainer.fit_step(model, optimizer, {}, {}, {}, 'float32')
        self.assertAlmostEqual(float(norm), 200., places=5)
        self.assertAlmostEqual(float(model.weight.grad.norm()), 1., places=5)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
