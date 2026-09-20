"""CPU checks for native decision gradients into an otherwise frozen value head."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
from tools import train_spatial_value_decisions as repair


def fixture():
    torch.manual_seed(42)
    model = SpatialOutcomePlanner(channels=4, width=8, hud_width=8, summary=8, comparator_hidden=8).eval()
    items = dict(raw=torch.randn(3, 160, 4), state=torch.randn(3, 160, 4), glyph=torch.randn(3, 14),
                 distances=torch.tensor([[1, 3, -1, 5], [-1, 4, 3, 1], [5, 8, 3, -1]]),
                 optimal=torch.tensor([1, 0, 4], dtype=torch.uint8))
    player = {'player_head.weight': torch.randn(1, 4), 'player_head.bias': torch.randn(1)}
    summary = []
    hook = model.summary_head.register_forward_hook(lambda _module, _args, output: summary.append(output))
    with torch.no_grad():
        prediction = model(items['raw'], items['state'], items['glyph'], repair.player_probabilities(items, player))
    hook.remove()
    fixed = torch.cat([*[scores.softmax(-1) for scores in prediction['field_logits']], prediction['event_logits'].sigmoid()], -1)
    return model, items, player, prediction, (summary[0], fixed, items['distances'], items['optimal'])


class SpatialValueDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        assert not torch.cuda.is_initialized()

    def test_native_score_reconstruction_and_action_permutation(self):
        model, _items, _player, prediction, payload = fixture()
        values = model.value_head(payload[0])
        actual = repair.native_scores(model, values, payload[1])
        torch.testing.assert_close(actual, prediction['action_logits'], atol=0, rtol=0)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = repair.native_scores(model, values[:, permutation], payload[1][:, permutation])
        torch.testing.assert_close(permuted, actual[:, permutation])

    def test_policy_only_gradient_reaches_value_head_and_no_other_parameters(self):
        model, _items, _player, _prediction, payload = fixture()
        repair.value_repair.freeze_except_value_head(model)
        record = repair.decision_loss(model, *payload, 1.)
        record['native_policy_cross_entropy'].backward()
        for name, parameter in model.named_parameters():
            if name.startswith('value_head.'):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
            else:
                self.assertIsNone(parameter.grad)

    def test_zero_optimal_rows_keep_value_loss_and_have_zero_policy_gradient(self):
        model, _items, _player, _prediction, payload = fixture()
        repair.value_repair.freeze_except_value_head(model)
        values = model.value_head(payload[0]).detach().requires_grad_()
        scores = repair.native_scores(model, values, payload[1])
        policy = repair.masked_optimal_set_cross_entropy(scores, payload[3])
        policy.backward()
        torch.testing.assert_close(values.grad[1], torch.zeros(4, 130), atol=0, rtol=0)
        self.assertGreater(float(values.grad[[0, 2]].abs().sum()), 0)
        zero = torch.zeros_like(payload[3])
        record = repair.decision_loss(model, *payload[:3], zero, 1.)
        self.assertEqual(float(record['native_policy_cross_entropy'].detach()), 0)
        record['total'].backward()
        self.assertGreater(float(model.value_head.weight.grad.abs().sum()), 0)

    def test_control_is_exact_full_value_ce_and_treatment_adds_only_native_policy(self):
        model, _items, _player, _prediction, payload = fixture()
        control, decision = (repair.decision_loss(model, *payload, weight) for weight in (0., 1.))
        values = model.value_head(payload[0])
        expected = F.cross_entropy(values.flatten(0, 1), repair.value_targets(values, payload[2]).flatten())
        torch.testing.assert_close(control['total'], expected, atol=0, rtol=0)
        torch.testing.assert_close(decision['value_cross_entropy'], control['value_cross_entropy'], atol=0, rtol=0)
        torch.testing.assert_close(decision['total'], expected + decision['native_policy_cross_entropy'], atol=0, rtol=0)
        self.assertEqual(set(decision), {'total', 'value_cross_entropy', 'native_policy_cross_entropy'})

    def test_training_detaches_public_cache_and_preserves_frozen_native_outputs(self):
        model, items, player, prediction, payload = fixture()
        repair.value_repair.freeze_except_value_head(model)
        before = repair.value_repair.frozen_digest(model)
        summary, fixed = (tensor.detach().requires_grad_() for tensor in payload[:2])
        record, diagnostic = repair.fit_step(model, repair.optimizer_for(model, .01),
                                             (summary, fixed, *payload[2:]), 1., probe_policy_gradient=True)
        self.assertTrue(torch.isfinite(record['total']))
        self.assertTrue(diagnostic['native_policy_gradient_reaches_value_head'])
        self.assertIsNone(summary.grad)
        self.assertIsNone(fixed.grad)
        self.assertEqual(repair.value_repair.frozen_digest(model), before)
        check = repair.value_repair.verify_frozen_predictions(model, items, player, prediction)
        self.assertTrue(check['physical_and_event_predictions_bitwise_unchanged'])
        self.assertTrue(check['fixed_input_comparator_bitwise_unchanged'])

    def test_public_cache_covers_unsupported_rows_and_roundtrips_native_scores(self):
        model, items, player, prediction, _payload = fixture()
        arrays = {key: value.numpy() for key, value in items.items()}
        arrays['seeds'] = np.array([11, 12, 13])
        with tempfile.TemporaryDirectory() as directory, patch.object(repair, 'guard', return_value=10 * 2**30):
            cache = repair.cache_public_inputs(model, arrays, player, Path(directory) / 'cache', device='cpu', size=3)
            self.assertEqual(cache['summary'].shape, (3, 4, 8))
            self.assertEqual(cache['nonvalue'].shape, (3, 4, 209))
            self.assertGreater(float(np.abs(cache['summary'][1]).sum()), 0)
            self.assertGreater(float(cache['nonvalue'][1].sum()), 0)
            payload = repair.selected_items(arrays, cache, np.arange(3), 'cpu')
            scores = repair.native_scores(model, model.value_head(payload[0]), payload[1])
            torch.testing.assert_close(scores, prediction['action_logits'], atol=0, rtol=0)
            self.assertEqual(payload[-1].tolist(), [1, 0, 4])
            repair.release(cache, close=True)
        self.assertFalse(model.summary_head._forward_hooks)

    def test_native_policy_ce_uses_supported_root_denominator(self):
        model, _items, _player, _prediction, payload = fixture()
        all_rows = repair.decision_loss(model, *payload, 1.)
        keep = torch.tensor([0, 2])
        supported = repair.decision_loss(model, *(value[keep] for value in payload), 1.)
        torch.testing.assert_close(all_rows['native_policy_cross_entropy'], supported['native_policy_cross_entropy'])

    def test_treatment_changes_update_while_start_and_nonvalue_weights_match(self):
        first, _items, _player, _prediction, payload = fixture()
        second = copy.deepcopy(first)
        for model in (first, second):
            repair.value_repair.freeze_except_value_head(model)
        for model, weight in ((first, 0.), (second, 1.)):
            repair.fit_step(model, torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1), payload, weight)
        self.assertFalse(torch.equal(first.value_head.weight, second.value_head.weight))
        self.assertEqual(repair.value_repair.frozen_digest(first), repair.value_repair.frozen_digest(second))

    def test_checkpoint_truthfully_replaces_ordinal_metadata_and_preserves_inference(self):
        model, _items, _player, _prediction, _payload = fixture()
        parent = dict(planner_weights=copy.deepcopy(model.state_dict()), planner_config=model.config(),
                      encoder_weights={'test': torch.tensor(1)}, score_weights={'planner': 1., 'direct': 0.},
                      format='unchanged', encoder_frozen=True, learned_voluntary_reset=False,
                      persistent_game_memory=False, actual_outcome_comparator_auxiliary_training=True,
                      quality_filter={'required_optimal_nonzero': True}, objective_weights={'old': True})
        args = SimpleNamespace(checkpoint_sha256='parent', steps=1000, lr=.001, seed=42, cache=Path('base'))
        report = dict(initial_planner_weights_sha256='initial', source_sha256={'source': 'hash'},
                      train_levels=10000, cache_manifest_sha256='cache', sampling='all original roots',
                      trainable_names=repair.value_repair.freeze_except_value_head(model),
                      frozen_weights_sha256=repair.value_repair.frozen_digest(model))
        for arm, weight in repair.ARMS.items():
            output = repair.checkpoint_for(parent, model, arm, args, report)
            self.assertNotIn('value_repair_continuation', output)
            self.assertNotIn('quality_filter', output)
            self.assertEqual(output['continuation_arm'], arm)
            self.assertEqual(output['objective_weights']['native_policy_cross_entropy'], weight)
            self.assertEqual(output['objective_weights']['finite_conditional_cdf_mse'], 0)
            self.assertFalse(output['actual_outcome_comparator_auxiliary_training'])
            self.assertFalse(output['value_decision_continuation']['gameplay_gain_established'])
            self.assertTrue(output['value_decision_continuation']['public_nonvalue_cache_covers_all_roots'])
            for key in ('planner_config', 'score_weights', 'format', 'encoder_frozen', 'learned_voluntary_reset', 'persistent_game_memory'):
                self.assertEqual(output[key], parent[key])


if __name__ == '__main__':
    unittest.main()
