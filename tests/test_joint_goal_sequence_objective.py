"""CPU chronology, label isolation and real generated K4 backward contracts."""
import unittest
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

from pebby.agent.joint_goal_data import qualification_specs, state_targets
from pebby.agent.joint_goal_planning import JointGoalConfig, JointGoalPlanning
from pebby.agent.joint_goal_sequence_objective import (
    STATE_KEYS, distance_loss, joint_goal_sequence_loss, validate_sequence_batch,
)
from pebby.agent.world_data import clone_env, history_arrays, successor_optimal_mask, verified_context
from pebby.ls20 import names


def actual_batch():
    """One generated engine root and sixteen actual fixed-action transitions."""
    spec = qualification_specs()[0]
    env, oracle, proof = verified_context(spec, search_limit=50_000)
    assert env is not None and proof['context_engine_verified'] and not oracle.truncated
    frame = env.render()
    root = dict(zip(('frames', 'history_valid', 'previous_actions'), history_arrays([frame], [-1], 8)))
    root.update(state_targets(spec, env, frame))
    root.update(optimal=np.uint8(successor_optimal_mask(oracle, oracle.state_of(env))),
                optimal_valid=np.bool_(True), current_distance=np.int16(oracle.distance_for(oracle.state_of(env))),
                current_distance_valid=np.bool_(True))
    actions = np.full((4, 4), 3, np.int64)
    actions[:, 0] = np.arange(4)
    rows = []
    for branch_index in range(4):
        branch = clone_env(env)
        trace = []
        for action in actions[branch_index]:
            before = branch.lives()
            observation = branch.perform(names.ACTION_IDS[action])
            state = oracle.state_of(branch)
            targets = state_targets(spec, branch, observation.frame)
            trace.append(dict(**{'next_' + key: value for key, value in targets.items()},
                next_frames=np.asarray(observation.frame, np.uint8), next_frame_valid=np.bool_(True),
                transition_valid=np.bool_(True), next_history_reset=np.bool_(branch.lives() < before),
                lost_life=np.bool_(branch.lives() < before), terminal=np.bool_(observation.finished), won=np.bool_(observation.won),
                next_distance=np.int16(0 if observation.won else oracle.distance_for(state)),
                next_distance_valid=np.bool_(True),
                next_optimal=np.uint8(successor_optimal_mask(oracle, state, terminal=observation.finished)),
                next_optimal_valid=np.bool_(not observation.finished)))
        rows.append(trace)
    root.update({key: np.stack([np.stack([row[key] for row in trace]) for trace in rows]) for key in rows[0][0]})
    root['actions'] = actions
    return {key: torch.as_tensor(value)[None] for key, value in root.items()}


def clone(batch):
    return {key: value.clone() for key, value in batch.items()}


def model():
    torch.manual_seed(72)
    return JointGoalPlanning(JointGoalConfig(horizon=1, hidden=32, encoder_loops=1, dynamics_loops=1)).eval()


class JointGoalSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU-only sequence tests require CUDA to remain uninitialized')
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.batch = actual_batch()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_actual_k4_backward_reaches_all_deployed_components(self):
        net = model()
        result = joint_goal_sequence_loss(net, self.batch)
        self.assertEqual(result['actual_horizon'], 4)
        self.assertEqual(result['valid_transitions_per_horizon'], [4, 4, 4, 4])
        result['total'].backward()
        for name in ('encoder.cell_projection.weight', 'encoder.appearance.network.4.weight',
                     'dynamics.output.1.weight', 'dynamics.budget_feedback.weight',
                     'dynamics.glyph_head.1.weight', 'continuation.scorer.weight',
                     'relation_projection.weight', 'value_head.weight', 'scorer.0.weight',
                     'pixel_decoder.board.2.weight', 'pixel_decoder.hud.2.weight'):
            gradient = dict(net.named_parameters())[name].grad
            with self.subTest(parameter=name):
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0)

    def test_h4_loss_flows_through_own_h1_field_across_life_reset(self):
        batch = clone(self.batch)
        batch['lost_life'][0, 0, 1] = True
        batch['next_history_reset'][0, 0, 1] = True
        batch['next_lives'][0, 0, 1:] = 2
        batch['next_steps'][0, 0, 1] = 10
        net = model()
        inputs, outputs = [], []
        hook = net.dynamics.register_forward_hook(lambda module, args, out: (inputs.append(args[0]), outputs.append(out)) and None)
        with mock.patch.object(net, 'encode_details', wraps=net.encode_details) as encode:
            result = joint_goal_sequence_loss(net, batch)
            self.assertEqual(encode.call_count, 1)
        hook.remove()
        # One independent H1 policy-imagination call, then four supervised calls.
        supervised = outputs[-4:]
        for h in range(1, 4):
            self.assertIs(inputs[-4 + h], supervised[h - 1]['field'])
        h4_readout = supervised[3]['readout']['player_logits']
        h4_only = F.cross_entropy(h4_readout, torch.tensor([7, 8, 9, 10]))
        gradient = torch.autograd.grad(h4_only, supervised[0]['field'], retain_graph=True)[0]
        self.assertGreater(float(gradient[0].abs().sum()), 0)
        first_gradient = torch.autograd.grad(result['losses']['physical'], supervised[0]['field'], retain_graph=True)[0]
        actual_loss_gradient = torch.autograd.grad(result['losses']['physical'], supervised[3]['field'])[0]
        self.assertGreater(float(actual_loss_gradient.abs().sum()), 0)
        altered = clone(batch)
        altered['next_player_cell'][0, 0, 3] = torch.tensor([11, 11])
        changed_outputs = []
        hook = net.dynamics.register_forward_hook(lambda module, args, out: changed_outputs.append(out))
        changed = joint_goal_sequence_loss(net, altered)
        hook.remove()
        changed_gradient = torch.autograd.grad(changed['losses']['physical'], changed_outputs[-4]['field'])[0]
        self.assertGreater(float((first_gradient - changed_gradient).abs().sum()), 0)
        # Changing actual H4 labels changes the backward signal, never the rollout.
        for old, new in zip(supervised, changed_outputs[-4:]):
            torch.testing.assert_close(old['field'], new['field'], rtol=0, atol=0)
        self.assertEqual(result['valid_transitions_per_horizon'], [4, 4, 4, 4])

    def test_postterminal_sentinels_have_no_loss_or_prediction_effect(self):
        batch = clone(self.batch)
        batch['terminal'][0, 0, 0] = True
        batch['transition_valid'][0, 0, 1:] = False
        batch['next_frame_valid'][0, 0, 1:] = False
        poisoned = clone(batch)
        for name in STATE_KEYS:
            values = poisoned['next_' + name]
            values[0, 0, 1:] = True if values.dtype == torch.bool else -99
        poisoned['next_frames'][0, 0, 1:] = 255
        poisoned['actions'][0, 0, 1:] = -99
        poisoned['next_distance'][0, 0, 1:] = 1000
        poisoned['next_distance_valid'][0, 0, 1:] = True
        poisoned['next_optimal'][0, 0, 1:] = 255
        poisoned['next_optimal_valid'][0, 0, 1:] = True
        net = model()
        first = joint_goal_sequence_loss(net, batch)
        second = joint_goal_sequence_loss(net, poisoned)
        for name in first['losses']:
            torch.testing.assert_close(first['losses'][name], second['losses'][name], rtol=0, atol=0)
        self.assertEqual(first['valid_transitions_per_horizon'], [4, 3, 3, 3])

    def test_missing_frame_and_previous_missing_pixels_are_inert(self):
        batch = clone(self.batch)
        batch['next_frame_valid'][0, 1, 1] = False
        poisoned = clone(batch)
        poisoned['next_frames'][0, 1, 1] = 255
        for name in ('roles', 'goal_triple', 'goal_presence', 'goal_solved', 'visible', 'support',
                     'semantic_valid', 'goal_attribute_valid'):
            values = poisoned['next_' + name]
            values[0, 1, 1] = True if values.dtype == torch.bool else -99
        net = model()
        first = joint_goal_sequence_loss(net, batch)
        second = joint_goal_sequence_loss(net, poisoned)
        for name in first['losses']:
            torch.testing.assert_close(first['losses'][name], second['losses'][name], rtol=0, atol=0)
        # Missing image does not erase known physical outcomes.
        poisoned['next_steps'][0, 1, 1] -= 1
        third = joint_goal_sequence_loss(net, poisoned)
        self.assertNotEqual(float(first['losses']['physical'].detach()), float(third['losses']['physical'].detach()))

    def test_terminal_transition_is_taught_but_broken_prefixes_are_rejected(self):
        batch = clone(self.batch)
        batch['terminal'][0, 0, 1] = True
        batch['transition_valid'][0, 0, 2:] = False
        batch['next_frame_valid'][0, 0, 1:] = False
        net = model()
        first = joint_goal_sequence_loss(net, batch)
        batch['next_steps'][0, 0, 1] = -1
        second = joint_goal_sequence_loss(net, batch)
        self.assertNotEqual(float(first['losses']['physical'].detach()), float(second['losses']['physical'].detach()))
        for change in ('resume', 'missing_terminal', 'reset_mismatch', 'first_action'):
            broken = clone(batch)
            if change == 'resume': broken['transition_valid'][0, 0, 3] = True
            elif change == 'missing_terminal': broken['terminal'][0, 0, 1] = False
            elif change == 'reset_mismatch': broken['next_history_reset'][0, 0, 0] = True
            else: broken['actions'][0, 0, 0] = 3
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_sequence_batch(broken)

    def test_value_unknown_mask_and_proved_unreachable_without_clipping(self):
        logits = torch.randn(3, 130, requires_grad=True)
        loss = distance_loss(logits, torch.tensor([-1, 128, 1000]), torch.tensor([True, True, False]))
        expected = F.cross_entropy(logits[:2], torch.tensor([129, 128]))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(logits.grad[2], torch.zeros(130), rtol=0, atol=0)
        for target in (129, 1000, -2):
            with self.subTest(target=target), self.assertRaises(ValueError):
                distance_loss(logits[:1], torch.tensor([target]), torch.tensor([True]))
        batch = clone(self.batch)
        batch['next_distance'][0, 0, 3] = 129
        with self.assertRaisesRegex(ValueError, '0..128'):
            joint_goal_sequence_loss(model(), batch)


if __name__ == '__main__':
    unittest.main()
