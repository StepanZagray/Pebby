"""CPU parity and execution-boundary checks for the source-pinned adapter."""
import copy
import unittest

import torch

from pebby.agent import world_features
from pebby.agent.structured_factored_policy import state_digest
from pebby.agent.world_model import WorldModelConfig, WorldPolicy
from tests.test_spatial_outcome_policy import make_policy, public_inputs


def encoder(glyph=True):
    return WorldPolicy(WorldModelConfig(channels=16, blocks=1, heads=4, expansion=2, loops=1,
        history=8, temporal_layers=1, hud_channels=8, hud_tokens=16, latent=16, reduce=2,
        predictor_blocks=1, predictor_hidden=16, value_hidden=16, max_distance=16,
        lookahead_depth=1, summary=4, readout_hidden=8, ranker_hidden=8,
        sigreg_projections=8, sigreg_knots=3, state_recall=True, glyph_recall=glyph,
        query_readout=glyph)).eval()


def inputs(history, batch=3):
    frames = torch.randint(0, 16, (batch, history, 64, 64))
    valid = torch.ones(batch, history, dtype=torch.bool)
    valid[0, :history // 2] = False
    actions = torch.randint(-1, 4, (batch, history))
    actions[~valid] = -1
    return frames, valid, actions


class WorldFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(97)

    def assert_features_equal(self, original, actual):
        self.assertEqual(set(actual), {'raw', 'state', 'cells', 'glyph'})
        for key in actual:
            if original[key] is None:
                self.assertIsNone(actual[key])
            else:
                torch.testing.assert_close(actual[key], original[key], atol=0, rtol=0)

    def test_exact_features_across_histories_chunking_glyph_and_loop_overrides(self):
        for glyph in (False, True):
            model = encoder(glyph)
            for history in (1, 4, 8):
                public = inputs(history)
                for chunk in (0, 1, 2):
                    with self.subTest(glyph=glyph, history=history, chunk=chunk), torch.no_grad():
                        model.encoder_chunk_size = chunk
                        expected = model.encode(*public, loops=2)
                        actual = world_features.encode_features(model, *public, loops=2)
                        self.assert_features_equal(expected, actual)
                        torch.testing.assert_close(model.player_weights(actual['cells'])[1],
                                                   model.player_weights(expected['cells'])[1], atol=0, rtol=0)

    def test_spatial_scores_direct_query_and_zero_projector_reduce_execution(self):
        for direct_weight in (0., .25):
            with self.subTest(direct_weight=direct_weight), torch.no_grad():
                policy = make_policy(direct_weight=direct_weight, planner_weight=.75)
                public = public_inputs()
                encoding = policy.encoder.encode(*public)
                player = policy.encoder.player_weights(encoding['cells'])[1]
                expected = .75 * policy.planner(encoding['raw'], encoding['state'], encoding['glyph'], player)['action_logits']
                if direct_weight:
                    direct = policy.encoder.direct_logits(encoding['cells'])[0]
                    direct += policy.encoder.query_logits(encoding, player)
                    expected += direct_weight * direct
                before = state_digest(policy.state_dict())
                def forbidden(*_):
                    self.fail('unused projection executed')
                handles = [module.register_forward_pre_hook(forbidden)
                           for module in (policy.encoder.reduce, policy.encoder.projector)]
                try:
                    actual = policy(*public)
                finally:
                    for handle in handles:
                        handle.remove()
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertEqual(state_digest(policy.state_dict()), before)

    def test_checkpointed_features_and_gradients_match_full_encoding(self):
        model = encoder()
        model.checkpoint_encoder = True
        model.checkpoint_loops = True
        model.encoder_chunk_size = 2
        original = copy.deepcopy(model)
        public = inputs(4)
        expected = original.encode(*public)
        actual = world_features.encode_features(model, *public)
        self.assert_features_equal(expected, actual)
        expected['state'].square().mean().backward()
        actual['state'].square().mean().backward()
        for name, parameter in model.named_parameters():
            other = dict(original.named_parameters())[name]
            if other.grad is None:
                self.assertIsNone(parameter.grad, name)
            else:
                torch.testing.assert_close(parameter.grad, other.grad, atol=0, rtol=0, msg=name)

    def test_public_input_and_loop_validation_remain_effective(self):
        model = encoder()
        public = inputs(4)
        for loops in (True, 0, -1, 1.5):
            with self.subTest(loops=loops), self.assertRaisesRegex(ValueError, 'positive integer'):
                world_features.encode_features(model, *public, loops=loops)
        with self.assertRaisesRegex(ValueError, 'history length'):
            world_features.encode_features(model, *inputs(9))
        frames, valid, actions = public
        valid[:, -1] = False
        with self.assertRaisesRegex(ValueError, 'current'):
            world_features.encode_features(model, frames, valid, actions)

    def test_single_frame_default_history_matches_full_encoding(self):
        model = encoder()
        frames = torch.randint(0, 16, (2, 64, 64))
        with torch.no_grad():
            self.assert_features_equal(model.encode(frames), world_features.encode_features(model, frames))

    def test_source_contract_rejects_changed_method(self):
        def changed_encode(self, frames):
            return frames
        with self.assertRaisesRegex(ValueError, 'source contract changed: encode'):
            world_features._check_contract((('encode', changed_encode, changed_encode.__code__),))

    def test_subclass_cannot_silently_bypass_its_assembly_behavior(self):
        class DifferentWorldPolicy(WorldPolicy):
            pass
        different = DifferentWorldPolicy(encoder().cfg)
        with self.assertRaisesRegex(ValueError, 'unmodified WorldPolicy'):
            world_features.encode_features(different, *inputs(1))


if __name__ == '__main__':
    unittest.main()
