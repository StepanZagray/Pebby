"""Spatial outcome policy v2 contracts: exact HUD scalars, warm start, tuning, IO.

Synthetic CPU encoders cover the wrapper contracts; the retained v1 artifact
(when present) proves the planner change keeps v1 loading and scoring exactly.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from pebby.agent.history import PolicyHistory, for_policy
from pebby.agent.hud_decoder import HUD_SCALARS, decode_hud
from pebby.agent.model import load_checkpoint as load_any_checkpoint
from pebby.agent.neural_outcome_policy import ENCODER_RUNTIME as V1_RUNTIME, PARENT_SHA, weights_sha256
from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner, SpatialOutcomePlannerConfig
from pebby.agent.spatial_outcome_policy import FORMAT as V1_FORMAT, SpatialOutcomePolicy
from pebby.agent import spatial_v2_policy
from pebby.agent.spatial_v2_policy import (FORMAT, SpatialOutcomePolicyV2, from_v1_checkpoint,
                                           load_checkpoint, save_checkpoint, warm_start_planner)
from pebby.agent.world_model import WorldModelConfig, WorldPolicy

RETAINED = Path(__file__).resolve().parents[1] / 'artifacts/spatial-recovery-v1/quality-fit/recovery.pt'
# Captured from the retained checkpoint with the planner source before it gained
# hud_scalars, single-threaded (the encoder's float32 sums depend on the thread
# count), on torch.randint(0, 16, (3, 8, 64, 64), generator=torch.Generator().manual_seed(2024)).
RETAINED_LOGITS = torch.tensor([[-5.4288334846, -4.5101108551, -3.7247643471, -2.6314065456],
                                [2.8841428757, 4.9216699600, 4.7158050537, 6.2239260674],
                                [0.4915953875, 2.2512085438, 2.8495302200, 2.8849740028]])
RETAINED_PLANNER_PARAMETERS = 261188


def encoder():
    return WorldPolicy(WorldModelConfig(channels=16, blocks=1, heads=4, expansion=2, loops=1,
        history=8, temporal_layers=1, hud_channels=8, hud_tokens=16, latent=16, reduce=2,
        predictor_blocks=1, predictor_hidden=16, value_hidden=16, max_distance=16,
        lookahead_depth=1, summary=4, readout_hidden=8, ranker_hidden=8,
        sigreg_projections=8, sigreg_knots=3, state_recall=True, glyph_recall=True, query_readout=True))


def planner(hud_scalars=HUD_SCALARS):
    return SpatialOutcomePlanner(channels=16, width=8, hud_width=8, summary=8, comparator_hidden=8,
                                 hud_scalars=hud_scalars)


def policy(encoder_mode='frozen', hud_scalars=HUD_SCALARS):
    return SpatialOutcomePolicyV2(encoder(), planner(hud_scalars), encoder_mode=encoder_mode)


def public_inputs():
    return (torch.randint(0, 16, (2, 8, 64, 64)), torch.tensor([[False] * 4 + [True] * 4, [True] * 8]),
            torch.tensor([[-1] * 4 + [0, 1, 2, 3], [-1, 0, 1, 2, 3, 0, 1, 2]]))


def retained_frames():
    return torch.randint(0, 16, (3, 8, 64, 64), generator=torch.Generator().manual_seed(2024))


def v1_checkpoint(v1):
    weights = {key: value.clone() for key, value in v1.encoder.state_dict().items()}
    return dict(format=V1_FORMAT, encoder_parent_sha256=PARENT_SHA, encoder_frozen=True,
                official_training_inputs=False, encoder_runtime=dict(V1_RUNTIME),
                encoder_weights=weights, encoder_weights_sha256=weights_sha256(weights),
                encoder_config=v1.encoder.config(), planner_config=v1.planner.config(),
                planner_weights=v1.planner.state_dict(),
                score_weights=dict(direct=v1.direct_weight, planner=v1.planner_weight))


class PlannerHudScalarTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.raw, self.state = torch.randn(2, 160, 16), torch.randn(2, 160, 16)
        self.glyph, self.player = torch.randn(2, 14), torch.randn(2, 144).softmax(-1)
        self.scalars = torch.tensor([[.5, 0., 1., 0.], [1., 0., 0., 1.]])

    def test_zero_keeps_v1_parameters_config_and_signature(self):
        old, new = planner(0), planner(4)
        self.assertNotIn('hud_scalars', old.config())
        self.assertEqual(SpatialOutcomePlanner(old.config()).config(), old.config())
        self.assertEqual(new.config()['hud_scalars'], 4)
        self.assertEqual(SpatialOutcomePlanner(new.config()).config(), new.config())
        self.assertEqual(SpatialOutcomePlannerConfig.from_dict(old.config()).hud_scalars, 0)
        self.assertEqual(new.parameter_count() - old.parameter_count(), 4 * (3 * 8 + 8))
        self.assertEqual(SpatialOutcomePlanner().parameter_count(), RETAINED_PLANNER_PARAMETERS)
        old(self.raw, self.state, self.glyph, self.player)
        with self.assertRaisesRegex(ValueError, 'without hud_scalars'):
            old(self.raw, self.state, self.glyph, self.player, hud_scalars=self.scalars)
        with self.assertRaises(ValueError):
            SpatialOutcomePlannerConfig(hud_scalars=-1)

    def test_scalars_required_validated_and_used(self):
        new = planner(4)
        for bad in (None, self.scalars[:1], self.scalars.long(), self.scalars[:, :3], self.scalars * float('nan')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                new(self.raw, self.state, self.glyph, self.player, hud_scalars=bad)
        first = new(self.raw, self.state, self.glyph, self.player, hud_scalars=self.scalars)['action_logits']
        second = new(self.raw, self.state, self.glyph, self.player, hud_scalars=self.scalars * 0)['action_logits']
        self.assertGreater(float((first - second).detach().abs().max()), 1e-6)

    def test_warm_start_with_zeroed_scalar_columns_matches_v1_exactly(self):
        old = planner(0)
        new, report = warm_start_planner(old, 4)
        self.assertEqual(report['fresh'], [])
        self.assertEqual(sorted(report['widened']), sorted(spatial_v2_policy._WIDENED))
        self.assertEqual(len(report['copied']) + 4, len(old.state_dict()))
        expected = old(self.raw, self.state, self.glyph, self.player)
        actual = new(self.raw, self.state, self.glyph, self.player, hud_scalars=self.scalars)
        torch.testing.assert_close(actual['action_logits'], expected['action_logits'], atol=0, rtol=0)
        torch.testing.assert_close(actual['value_logits'], expected['value_logits'], atol=0, rtol=0)
        # The scalar columns are live: a gradient reaches them from the loss.
        actual['action_logits'].square().sum().backward()
        for block in new.blocks:
            self.assertGreater(float(block.condition.weight.grad[:, 22:26].abs().sum()), 0.)
        with self.assertRaises(ValueError):
            warm_start_planner(new, 4)

    def test_direct_readout_is_optional_zero_initialised_and_trainable(self):
        old = planner(0)
        self.assertNotIn('direct_readout', old.config())
        self.assertIsNone(old.direct_head)
        new, report = warm_start_planner(old, 4, direct_readout=True)
        self.assertEqual(report['fresh'], ['direct_head.weight'])
        self.assertTrue(new.config()['direct_readout'])
        self.assertEqual(SpatialOutcomePlanner(new.config()).config(), new.config())
        self.assertEqual(new.parameter_count() - old.parameter_count(), 4 * (3 * 8 + 8) + new.cfg.summary)
        expected = old(self.raw, self.state, self.glyph, self.player)['action_logits']
        actual = new(self.raw, self.state, self.glyph, self.player, hud_scalars=self.scalars)['action_logits']
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.square().sum().backward()
        self.assertGreater(float(new.direct_head.weight.grad.abs().sum()), 0.)
        with self.assertRaises(ValueError):
            SpatialOutcomePlannerConfig(direct_readout=1)
        with self.assertRaises(ValueError):
            warm_start_planner(new, 4)


class SpatialOutcomePolicyV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        if torch.cuda.is_initialized():
            raise AssertionError('CPU wrapper tests must not initialize CUDA')

    def setUp(self):
        torch.manual_seed(73)

    def test_construction_contract_and_config(self):
        model = policy()
        config = model.config()
        self.assertEqual(config['architecture'], 'world')
        self.assertEqual(config['history'], 8)
        self.assertEqual(config['decision_architecture'], 'spatial_outcomes_v2')
        self.assertEqual(config['encoder_mode'], 'frozen')
        self.assertEqual(config['hud_scalars'], HUD_SCALARS)
        self.assertEqual(policy(hud_scalars=0).config()['hud_scalars'], 0)
        with self.assertRaises(ValueError):
            SpatialOutcomePolicyV2(encoder(), planner(), encoder_mode='thawed')
        with self.assertRaises(ValueError):
            SpatialOutcomePolicyV2(encoder(), planner(3))
        with self.assertRaises(ValueError):
            SpatialOutcomePolicyV2(encoder(), SpatialOutcomePlanner(channels=8, width=8, hud_width=8, summary=8, comparator_hidden=8))

    def test_predict_uses_decoded_hud_of_the_current_frame(self):
        model, inputs = policy(), public_inputs()
        frames = inputs[0].clone()
        frames[:, -1, 61:63, 13:55] = 3
        frames[:, -1, 61:63, 13 + 42 - 21:55] = 11  # 21 steps left
        frames[:, -1, 61:63, 56:64] = 3
        frames[:, -1, 61:63, 59:61] = 8  # a single pip -> lives are counted, not positions
        frames[0, -1, 61:63, 62:64] = 8
        inputs = (frames, *inputs[1:])
        seen = []
        original = model.planner.forward

        def spy(*args, **kwargs):
            seen.append(kwargs['hud_scalars'])
            return original(*args, **kwargs)

        with patch.object(model.planner, 'forward', side_effect=spy), torch.no_grad():
            result = model.predict(*inputs)
            logits = model(*inputs)
        self.assertEqual(set(result), {'action_logits', 'field_logits', 'value_logits', 'event_logits'})
        torch.testing.assert_close(logits, result['action_logits'], atol=0, rtol=0)
        torch.testing.assert_close(seen[0], decode_hud(frames), atol=0, rtol=0)
        torch.testing.assert_close(seen[0], torch.tensor([[.5, 0., 1., 0.], [.5, 1., 0., 0.]]), atol=0, rtol=0)
        self.assertEqual(logits.shape, (2, 4))
        self.assertTrue(torch.isfinite(logits).all())
        # Changing only the HUD bar changes the scores through the exact scalars.
        changed = frames.clone()
        changed[:, -1, 61:63, 13:55] = 3
        with torch.no_grad():
            self.assertGreater(float((model(changed, *inputs[1:]) - logits).abs().max()), 0.)

    def test_frozen_and_finetune_eligibility_and_gradients(self):
        frozen, tuned = policy('frozen').train(), policy('finetune').train()
        self.assertFalse(frozen.encoder.training)
        self.assertFalse(tuned.encoder.training)
        self.assertEqual(frozen.parameter_count(), tuned.parameter_count())
        self.assertGreater(tuned.trainable_parameter_count(), frozen.trainable_parameter_count())
        self.assertEqual(frozen.trainable_parameter_count(),
                         frozen.planner.parameter_count() - frozen.planner.player_head.bias.numel())
        self.assertFalse(tuned.encoder.player_head.bias.requires_grad)
        outside = [n for n, p in tuned.encoder.named_parameters()
                   if n.split('.')[0] not in spatial_v2_policy._ENCODER_FEATURES]
        self.assertTrue(outside and all(not dict(tuned.encoder.named_parameters())[n].requires_grad for n in outside))
        self.assertTrue(tuned.encoder.stem[0].weight.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in frozen.encoder.parameters()))
        inputs = public_inputs()
        for model in (frozen, tuned):
            model(*inputs).square().sum().backward()
        self.assertTrue(all(p.grad is None for p in frozen.encoder.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in frozen.planner.parameters()))
        active = dict(tuned.encoder.named_parameters())
        for prefix in ('stem.0.weight', 'core.0', 'player_head.weight', 'glyph_encoder.'):
            gradients = [p.grad for n, p in active.items() if n.startswith(prefix)]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in gradients), prefix)
        self.assertTrue(all(p.grad is None for p in active.values() if not p.requires_grad))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in tuned.parameters()))
        self.assertEqual([g['name'] for g in tuned.parameter_groups(1e-4, 1e-3)], ['encoder', 'planner'])
        self.assertEqual([g['name'] for g in frozen.parameter_groups(1e-4, 1e-3)], ['planner'])

    def test_checkpoint_round_trip_through_generic_loader_and_history(self):
        model, inputs = policy('finetune'), public_inputs()
        model.provenance = dict(parent_path='parent.pt', parent_sha256='a' * 64)
        with torch.no_grad():
            expected = model.eval()(*inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'v2.pt'
            saved = save_checkpoint(path, model, note='synthetic', steps=3)
            self.assertEqual(saved['metadata'], dict(note='synthetic', steps=3))
            for loader in (load_checkpoint, load_any_checkpoint):
                loaded, data = loader(path)
                self.assertIsInstance(loaded, SpatialOutcomePolicyV2)
                self.assertFalse(loaded.training)
                self.assertEqual(data['format'], FORMAT)
                self.assertEqual(data['encoder_mode'], 'finetune')
                self.assertEqual(data['hud_scalars'], HUD_SCALARS)
                self.assertEqual(data['parent_sha256'], 'a' * 64)
                self.assertEqual(loaded.config(), model.config())
                with torch.no_grad():
                    torch.testing.assert_close(loaded(*inputs), expected, atol=0, rtol=0)
            history = for_policy(loaded, inputs[0][0, -1].numpy())
            self.assertIsInstance(history, PolicyHistory)
            self.assertEqual(history.length, 8)
            for index in range(3):
                history.observe(inputs[0][0, index].numpy(), index)
            with torch.no_grad():
                scores = history.scores()
            self.assertEqual(scores.shape, (4,))
            self.assertTrue(torch.isfinite(scores).all())
            base = torch.load(path, weights_only=True)
            for change in (dict(format=V1_FORMAT), dict(encoder_weights_sha256='0' * 64),
                           dict(planner_weights_sha256='0' * 64), dict(hud_scalars=0),
                           dict(encoder_runtime=dict(V1_RUNTIME))):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    torch.save({**base, **change}, path)
                    load_checkpoint(path)
            with self.assertRaises(ValueError):
                save_checkpoint(path, model, bad=float('nan'))

    def test_from_v1_warm_start_on_synthetic_checkpoint(self):
        v1 = SpatialOutcomePolicy(encoder(), planner(0)).eval()
        inputs = public_inputs()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'v1.pt'
            torch.save(v1_checkpoint(v1), path)
            v2 = from_v1_checkpoint(path, encoder_mode='finetune')
        self.assertEqual(v2.provenance['fresh'], [])
        self.assertEqual(v2.provenance['parent_path'], str(path))
        self.assertEqual(len(v2.provenance['parent_sha256']), 64)
        with torch.no_grad(), spatial_v2_policy._same_attention_path():
            expected = v1(*inputs)  # Same attention kernel as v2; scalars enter zeroed columns.
            # The widened condition Linear reduces over four extra zero columns,
            # which only changes float summation order (observed 1.9e-9).
            torch.testing.assert_close(v2.eval()(*inputs), expected, atol=1e-7, rtol=1e-6)


@unittest.skipUnless(RETAINED.exists(), 'retained spatial recovery checkpoint not present')
class RetainedCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.v1, cls.data = load_any_checkpoint(RETAINED)
        cls.frames = retained_frames()
        with torch.no_grad():
            cls.v1_logits = cls.v1(cls.frames)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_v1_artifact_loads_strictly_with_identical_parameters_and_logits(self):
        self.assertIsInstance(self.v1, SpatialOutcomePolicy)
        self.assertNotIn('hud_scalars', self.data['planner_config'])
        self.assertEqual(self.v1.planner.cfg.hud_scalars, 0)
        self.assertEqual(self.v1.planner.parameter_count(), RETAINED_PLANNER_PARAMETERS)
        self.assertEqual(self.v1.planner.parameter_count(), self.data['planner_parameters'])
        self.assertEqual(self.v1.planner.config(), self.data['planner_config'])
        torch.testing.assert_close(self.v1_logits, RETAINED_LOGITS, atol=0, rtol=0)

    def test_warm_start_copies_everything_and_reproduces_v1_scores(self):
        v2 = from_v1_checkpoint(RETAINED)
        self.assertEqual(v2.provenance['fresh'], [])
        self.assertEqual(len(v2.provenance['copied']) + len(v2.provenance['widened']), len(self.v1.planner.state_dict()))
        self.assertEqual(v2.provenance['parent_path'], str(RETAINED))
        self.assertEqual(v2.planner.parameter_count(), RETAINED_PLANNER_PARAMETERS + HUD_SCALARS * (3 * 48 + 64))
        with torch.no_grad():
            logits = v2(self.frames)
        self.assertTrue(torch.isfinite(logits).all())
        # v2 pins the differentiable attention kernel; the frozen v1 arm used the
        # eval/no_grad fastpath, so equality is up to kernel rounding...
        torch.testing.assert_close(logits, self.v1_logits, atol=5e-5, rtol=1e-5)
        # ...and exact once v1 is evaluated with the same kernel.
        with torch.no_grad(), spatial_v2_policy._same_attention_path():
            torch.testing.assert_close(logits, self.v1(self.frames), atol=0, rtol=0)
        self.assertGreater(from_v1_checkpoint(RETAINED, encoder_mode='finetune').trainable_parameter_count(),
                           v2.trainable_parameter_count())


if __name__ == '__main__':
    unittest.main()
