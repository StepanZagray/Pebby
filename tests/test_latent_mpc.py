import unittest

import torch

from pebby.agent.latent_mpc import MPCConfig, choose_action, plan_from_encoding
from tests.test_world_model import make_model, make_synthetic, to_batch
from tools.evaluate_latent_mpc import balanced_indices


class _Cfg:
    max_distance = 4


class FakeWorld:
    """Tiny public-interface model whose latent ids make planning assertions readable."""

    def __init__(self):
        self.cfg = _Cfg()
        self.bin_values = torch.arange(5, dtype=torch.float32)
        self.calls = []

    def encode(self, frames, history_valid=None, previous_actions=None):
        batch = torch.as_tensor(frames).shape[0]
        return {"latent": torch.zeros(batch, 1), "frames_seen": frames}

    def logits_from(self, encoding, successors=None):
        batch = encoding["latent"].shape[0]
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]]).expand(batch, -1).clone()
        return logits, {"successors": successors}

    def predict_successors(self, latent, actions=None):
        latent = torch.as_tensor(latent).float()
        if actions is None:
            actions = torch.arange(4).expand(latent.shape[0], -1)
        actions = torch.as_tensor(actions).long()
        self.calls.append((tuple(latent.shape), tuple(actions.shape)))
        if actions.ndim == 1:
            return latent + actions[:, None].float() + 1
        return latent[:, None, :] + actions[..., None].float() + 1

    def value(self, latent):
        latent = torch.as_tensor(latent).float().flatten()
        distance = torch.full((latent.numel(), 5), -8.0)
        ids = latent.long().clamp(0, 4)
        distance.scatter_(1, ids[:, None], 8.0)
        terminal = torch.full((latent.numel(),), -8.0)
        won = torch.full((latent.numel(),), -8.0)
        return distance, terminal, won


class LatentMpcTests(unittest.TestCase):
    def test_prior_only_returns_exact_legacy_logits(self):
        model = FakeWorld()
        frames = torch.zeros(1, 4, 64, 64, dtype=torch.long)
        valid = torch.ones(1, 4, dtype=torch.bool)
        actions = torch.full((1, 4), -1, dtype=torch.long)
        encoding = model.encode(frames, valid, actions)
        expected, _ = model.logits_from(encoding)

        decision = choose_action(model, frames, valid, actions,
                                MPCConfig(prior_only=True))

        torch.testing.assert_close(decision.logits, expected[0])
        self.assertEqual(decision.action, int(expected.argmax()))
        self.assertEqual(decision.prior_action, decision.action)
        self.assertEqual(decision.root_action_rank, 1)
        self.assertFalse(decision.overrode_prior)

    def test_prior_only_matches_real_world_policy_forward(self):
        model = make_model().eval()
        batch = to_batch(make_synthetic(seed=13, levels=1, steps=4), slice(0, 1))
        with torch.inference_mode():
            legacy = model(batch["frames"], batch["history_valid"], batch["previous_actions"])
            decision = choose_action(model, batch["frames"], batch["history_valid"],
                                    batch["previous_actions"], MPCConfig(prior_only=True))
        torch.testing.assert_close(decision.logits, legacy[0])

    def test_mpc_expands_batched_successors_and_returns_root_scores(self):
        model = FakeWorld()
        encoding = model.encode(torch.zeros(1, 64, 64, dtype=torch.long))
        decision = plan_from_encoding(model, encoding,
                                     MPCConfig(horizon=2, beam_width=4,
                                               prior_weight=0.25))

        self.assertIn(decision.action, range(4))
        self.assertEqual(decision.action_scores.shape, (4,))
        self.assertEqual(decision.root_action_rank, 1)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0][0], (1, 1))
        self.assertEqual(model.calls[0][1], (1, 4))
        self.assertGreaterEqual(decision.beam_nodes, 1)

    def test_unreachable_penalty_can_override_prior_without_game_rules(self):
        class Risky(FakeWorld):
            def value(self, latent):
                latent = torch.as_tensor(latent).float().flatten()
                distance = torch.full((latent.numel(), 5), -8.0)
                terminal = torch.full((latent.numel(),), -8.0)
                won = torch.full((latent.numel(),), -8.0)
                for row, value in enumerate(latent.long().tolist()):
                    if value == 1:  # prior-favoured action 0: unreachable
                        distance[row, -1] = 8.0
                    else:          # finite alternatives
                        distance[row, 0] = 8.0
                return distance, terminal, won

        model = Risky()
        encoding = model.encode(torch.zeros(1, 64, 64, dtype=torch.long))
        decision = plan_from_encoding(
            model, encoding,
            MPCConfig(horizon=1, beam_width=4, prior_weight=0.1,
                      unreachable_weight=4.0, distance_weight=0.5))

        self.assertEqual(decision.prior_action, 0)
        self.assertNotEqual(decision.action, 0)
        self.assertTrue(decision.overrode_prior)

    def test_full_horizon_leaf_can_downgrade_good_looking_one_step_prefix(self):
        class DelayedRisk(FakeWorld):
            def predict_successors(self, latent, actions=None):
                latent = torch.as_tensor(latent).float()
                if actions is None:
                    actions = torch.arange(4).expand(latent.shape[0], -1)
                actions = torch.as_tensor(actions).long()
                self.calls.append((tuple(latent.shape), tuple(actions.shape)))
                return latent[:, None, :] * 10 + actions[..., None].float() + 1

            def value(self, latent):
                latent = torch.as_tensor(latent).float().flatten()
                distance = torch.full((latent.numel(), 5), -8.0)
                terminal = torch.full((latent.numel(),), -8.0)
                won = torch.full((latent.numel(),), -8.0)
                for row, value in enumerate(latent.long().tolist()):
                    if value >= 11 and value <= 14:  # all h2 children below root action 0
                        distance[row, -1] = 8.0
                    else:
                        distance[row, 0] = 8.0
                return distance, terminal, won

        model = DelayedRisk()
        encoding = model.encode(torch.zeros(1, 64, 64, dtype=torch.long))
        decision = plan_from_encoding(
            model, encoding,
            MPCConfig(horizon=2, beam_width=4, prior_weight=0.1,
                      unreachable_weight=4.0, distance_weight=0.5))

        self.assertEqual(decision.prior_action, 0)
        self.assertNotEqual(decision.action, 0)
        self.assertEqual(decision.expanded_depth, 2)

    def test_config_rejects_invalid_search_parameters(self):
        with self.assertRaises(ValueError):
            MPCConfig(horizon=0)
        with self.assertRaises(ValueError):
            MPCConfig(beam_width=0)
        with self.assertRaises(ValueError):
            MPCConfig(distance_weight=-1)

    def test_pilot_selection_is_balanced_and_seeded(self):
        specs = [{"difficulty": difficulty, "seed": difficulty * 100 + index}
                 for difficulty in range(1, 6) for index in range(5)]
        first = balanced_indices(specs, per_difficulty=2, seed=42)
        second = balanced_indices(specs, per_difficulty=2, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual([sum(specs[i]["difficulty"] == d for i in first)
                          for d in range(1, 6)], [2] * 5)


if __name__ == "__main__":
    unittest.main()
