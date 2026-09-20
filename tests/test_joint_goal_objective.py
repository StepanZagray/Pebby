"""CPU contracts for the fresh joint goal planning objective."""

import copy
import unittest

import torch

from pebby.agent.joint_goal_objective import (
    joint_goal_loss,
    policy_loss,
    reconstruction_loss,
    semantic_loss,
)
from pebby.agent.joint_goal_planning import JointGoalConfig, JointGoalPlanning


def _batch():
    """Return one small, valid four-action successor row."""
    batch, branches = 1, 4
    frames = (torch.arange(8 * 64 * 64, dtype=torch.long).reshape(1, 8, 64, 64) % 16).to(torch.uint8)

    current = dict(
        player_cell=torch.tensor([[3, 4]], dtype=torch.long),
        triple=torch.tensor([[1, 2, 3]], dtype=torch.long),
        steps=torch.tensor([5], dtype=torch.long),
        lives=torch.tensor([2], dtype=torch.long),
        roles=torch.zeros(batch, 144, 8, dtype=torch.bool),
        goal_triple=torch.zeros(batch, 144, 3, dtype=torch.long),
        goal_presence=torch.ones(batch, 144, dtype=torch.bool),
        goal_solved=torch.zeros(batch, 144, dtype=torch.bool),
        visible=torch.ones(batch, 144, dtype=torch.bool),
        support=torch.ones(batch, 144, dtype=torch.bool),
        semantic_valid=torch.ones(batch, 144, dtype=torch.bool),
        goal_attribute_valid=torch.ones(batch, 144, dtype=torch.bool),
    )
    successors = {
        "next_player_cell": current["player_cell"][:, None].expand(batch, branches, 2).clone(),
        "next_triple": current["triple"][:, None].expand(batch, branches, 3).clone(),
        "next_steps": current["steps"][:, None].expand(batch, branches).clone(),
        "next_lives": current["lives"][:, None].expand(batch, branches).clone(),
    }
    for name in (
        "roles",
        "goal_triple",
        "goal_presence",
        "goal_solved",
        "visible",
        "support",
        "semantic_valid",
        "goal_attribute_valid",
    ):
        value = current[name]
        successors["next_" + name] = value[:, None].expand(batch, branches, *value.shape[1:]).clone()

    return dict(
        frames=frames,
        history_valid=torch.ones(batch, 8, dtype=torch.bool),
        previous_actions=torch.tensor([[-1, 0, 1, 2, 3, 0, 1, 2]], dtype=torch.long),
        optimal=torch.tensor([1], dtype=torch.long),
        next_optimal=torch.tensor([[1, 2, 4, 8]], dtype=torch.long),
        next_optimal_valid=torch.ones(batch, branches, dtype=torch.bool),
        next_frames=frames[:, -1:, :, :].expand(batch, branches, 64, 64).clone(),
        next_frame_valid=torch.ones(batch, branches, dtype=torch.bool),
        distances=torch.full((batch, branches), 2, dtype=torch.long),
        distance_valid=torch.ones(batch, branches, dtype=torch.bool),
        current_distance=torch.tensor([3], dtype=torch.long),
        current_distance_valid=torch.ones(batch, dtype=torch.bool),
        lost_life=torch.zeros(batch, branches),
        terminal=torch.zeros(batch, branches),
        won=torch.zeros(batch, branches),
        **current,
        **successors,
    )


def _model():
    return JointGoalPlanning(JointGoalConfig(horizon=1, hidden=32, encoder_loops=1, dynamics_loops=1)).cpu()


def _clone_batch(batch):
    return {key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in batch.items()}


class _FixedPixelLogits(torch.nn.Module):
    """Nonuniform fixed logits make target changes observable in reconstruction CE."""

    def pixel_logits(self, field):
        count = len(field)
        board = torch.zeros(count, 144, 7, 7, 16)
        hud = torch.zeros(count, 4, 12, 16, 16)
        board[..., 0] = 3
        hud[..., 0] = 3
        return {"board": board, "hud": hud}


class JointGoalObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("joint goal objective tests must run before CUDA initialization")

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_policy_loss_zero_optimal_rows_have_no_action_gradient(self):
        logits = torch.tensor([[4., -1., 2., 0.], [0., 1., -2., 3.]], requires_grad=True)
        loss = policy_loss(logits, torch.tensor([0, 2], dtype=torch.long))
        loss.backward()
        torch.testing.assert_close(logits.grad[0], torch.zeros(4), atol=0, rtol=0)
        self.assertGreater(float(logits.grad[1].abs().sum()), 0.)

    def test_mixed_valid_reconstruction_masks_invalid_rows(self):
        model = _FixedPixelLogits()
        field = torch.zeros(2, 148, 96)
        frames = torch.zeros(2, 64, 64, dtype=torch.uint8)
        valid = torch.tensor([True, False])

        baseline = reconstruction_loss(model, field, frames, valid)
        invalid_changed = frames.clone()
        invalid_changed[1, 10, 10] = 1
        torch.testing.assert_close(
            baseline,
            reconstruction_loss(model, field, invalid_changed, valid),
            atol=0,
            rtol=0,
        )

        valid_changed = frames.clone()
        valid_changed[0, 10, 10] = 1
        self.assertNotEqual(float(baseline), float(reconstruction_loss(model, field, valid_changed, valid)))

    def test_terminal_missing_frame_masks_only_pixel_target(self):
        batch = _batch()
        batch["next_frame_valid"][0, 2] = False
        model = _model()
        first = joint_goal_loss(model, batch)["total"].detach()

        changed = _clone_batch(batch)
        changed["next_frames"][0, 2].fill_(15)
        second = joint_goal_loss(model, changed)["total"].detach()
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_fog_hidden_visibility_target_still_receives_gradient(self):
        labels = {
            "roles": torch.zeros(1, 144, 8),
            "goal_triple": torch.zeros(1, 144, 3, dtype=torch.long),
            "support": torch.ones(1, 144, dtype=torch.bool),
            "semantic_valid": torch.ones(1, 144, dtype=torch.bool),
            "goal_attribute_valid": torch.ones(1, 144, dtype=torch.bool),
            "visible": torch.ones(1, 144, dtype=torch.bool),
        }
        labels["support"][0, 0] = False
        labels["semantic_valid"][0, 0] = False
        labels["goal_attribute_valid"][0, 0] = False
        labels["visible"][0, 0] = False
        readout = {
            "role_logits": torch.zeros(1, 144, 8, requires_grad=True),
            "goal_shape_logits": torch.zeros(1, 144, 6, requires_grad=True),
            "goal_color_logits": torch.zeros(1, 144, 4, requires_grad=True),
            "goal_rotation_logits": torch.zeros(1, 144, 4, requires_grad=True),
        }
        visibility = torch.zeros(1, 144, requires_grad=True)
        semantic_loss(readout, visibility, labels, torch.ones(1, dtype=torch.bool)).backward()
        self.assertGreater(float(visibility.grad[0, 0].abs()), 0.)

    def test_label_targets_do_not_change_forward_inputs(self):
        first = _batch()
        second = _clone_batch(first)
        second["player_cell"][:] = torch.tensor([[10, 11]])
        second["triple"][:] = torch.tensor([[5, 3, 1]])
        second["steps"].fill_(20)
        second["lives"].fill_(3)
        second["roles"].fill_(True)
        second["goal_triple"].fill_(1)
        second["goal_solved"].fill_(True)
        second["visible"].fill_(False)
        second["optimal"][:] = 8
        second["next_optimal"][:] = torch.tensor([[8, 4, 2, 1]])
        second["distances"].fill_(7)
        second["current_distance"].fill_(4)
        for name in ("next_player_cell", "next_triple", "next_steps", "next_lives"):
            second[name].fill_(1)
        for name in (
            "next_roles",
            "next_goal_triple",
            "next_goal_solved",
            "next_visible",
        ):
            second[name].fill_(True)

        captures = []
        model = _model()

        def capture(_module, args):
            captures.append(tuple(value.detach().clone() for value in args))

        handle = model.encoder.register_forward_pre_hook(capture)
        try:
            joint_goal_loss(model, first)
            joint_goal_loss(model, second)
        finally:
            handle.remove()

        self.assertEqual(len(captures), 2)
        self.assertEqual(len(captures[0]), 3)
        for left, right in zip(captures[0], captures[1]):
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_real_model_shapes_and_gradients_reach_all_planning_paths(self):
        model = _model()
        batch = _batch()
        result = joint_goal_loss(model, batch)
        self.assertEqual(result["total"].ndim, 0)
        self.assertTrue(torch.isfinite(result["total"]))
        self.assertEqual(model(batch["frames"], batch["history_valid"], batch["previous_actions"]).shape, (1, 4))

        result["total"].backward()
        for name in (
            "encoder.cell_projection.weight",
            "dynamics.output.1.weight",
            "continuation.scorer.weight",
            "relation_mlp.0.weight",
            "scorer.0.weight",
        ):
            with self.subTest(parameter=name):
                gradient = dict(model.named_parameters())[name].grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
