"""CPU behavioral contracts for the goal-only auxiliary intervention."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from pebby.agent import world_goal_objective as goals, world_training_objectives as base
from tests.test_world_model import make_model, make_synthetic, to_batch
from tests.test_world_glyph import wake


class PixelTeacher(nn.Module):
    def __init__(self, confident=True):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(30. if confident else 0.))
        self.seen = []

    def forward(self, frames):
        self.seen.append(frames.clone())
        output = torch.zeros(len(frames), 144, 22, device=frames.device)
        output[..., [1, 8, 14, 18]] = self.weight
        return output.split((8, 6, 4, 4), -1)


class GoalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self):
        model = wake(make_model()).eval()
        batch = to_batch(make_synthetic(seed=3, levels=1, steps=2))
        b = len(batch['frames'])
        batch.update(goal_seeds=torch.full((b,), 42),
                     next_player_cell=batch['player_cell'][:, None].expand(-1, 4, -1).clone())
        batch['lost_life'] = torch.zeros(b, 4, dtype=torch.bool)
        batch['lost_life'][0, 2] = True
        return model, batch, goals.GoalObjective(PixelTeacher(), {42: False}, channels=16)

    def test_rng_isolated_head(self):
        torch.manual_seed(99)
        before = torch.random.get_rng_state().clone()
        a, b = goals.GoalHead(), goals.GoalHead()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for x, y in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(x, y))

    def test_support_matches_all_pixels_and_rejects_missing_seeds(self):
        players = torch.tensor([[0, 0], [5, 5], [11, 11]])
        got = goals.public_support(players, torch.ones(3, dtype=torch.bool))
        for i, (pc, pr) in enumerate(players.tolist()):
            for row in range(12):
                for col in range(12):
                    x, y = 4 + 5 * col, 5 * row
                    expected = y >= 1 and x + 6 <= 64 and y + 6 <= 52
                    expected &= all((px - (4 + 5 * pc + 1.5)) ** 2 + (py - (5 * pr + 1.5)) ** 2 <= 400
                                    for px in range(x - 1, x + 6) for py in range(y - 1, y + 6))
                    self.assertEqual(bool(got[i, row * 12 + col]), expected)
        model, batch, aux = self.fixture()
        with self.assertRaisesRegex(ValueError, 'missing seed'):
            goals.annotations(batch, {})
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            goals.public_support(torch.tensor([[12, 0]]), torch.tensor([False]))

    def test_aux_gradient_actual_order_resets_and_base_losses_unchanged(self):
        model, batch, aux = self.fixture()
        model.encoder_chunk_size = 2
        model.checkpoint_encoder = True
        expected = base.world_losses(model, batch, loops=1, sigreg_generator=torch.Generator().manual_seed(2))
        original = model.assemble
        actual = aux(model, batch, loops=1, sigreg_generator=torch.Generator().manual_seed(2))
        self.assertEqual(model.assemble, original)
        for key in expected['losses']:
            torch.testing.assert_close(actual['losses'][key], expected['losses'][key], rtol=0, atol=0)
        torch.testing.assert_close(actual['targets'], expected['targets'], rtol=0, atol=0)
        seen = torch.cat(aux.teacher.seen)
        torch.testing.assert_close(seen, torch.cat((batch['frames'][:, -1], batch['next_frames'].flatten(0, 1))))
        actual['losses']['goal_preservation'].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.temporal.parameters()))
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in aux.head.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in aux.teacher.parameters()))
        self.assertFalse(aux.teacher.training)
        with patch.object(base, 'world_losses', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                aux(model, batch)
        self.assertEqual(model.assemble, original)

    def test_empty_mask_zero_and_exact_soft_target_average(self):
        aux = goals.GoalObjective(PixelTeacher(), {}, channels=16)
        states = torch.randn(2, 144, 16, requires_grad=True)
        frames = torch.zeros(2, 64, 64, dtype=torch.long)
        mask = torch.zeros(2, 144, dtype=torch.bool)
        loss, selected = aux.goal_loss(states, frames, mask)
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(selected.sum(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(states.grad).all())
        self.assertEqual(states.grad.abs().sum(), 0)
        mask[0, 13] = mask[1, 44] = True
        loss, selected = aux.goal_loss(states, frames, mask)
        expected = 0
        for prediction, classes in zip(aux.head(states[mask]).split((6, 4, 4), -1), (6, 4, 4)):
            target = torch.zeros(classes); target[0] = 30
            expected += -(target.softmax(-1) * prediction.log_softmax(-1)).sum(-1).mean() / 3
        torch.testing.assert_close(loss, expected)
        self.assertTrue(torch.equal(selected, mask))
