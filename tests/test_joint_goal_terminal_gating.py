"""Learned terminal gates preserve event evidence and legacy configuration."""
from pathlib import Path
import tempfile
import types
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from pebby.agent import joint_goal_planning as staged

class ScriptedDynamics(nn.Module):
    """Only H3+ depend on tail; a learned terminal logit is emitted at H2."""
    def __init__(self, terminal=100., lost_life=-100.):
        super().__init__()
        self.tail = nn.Parameter(torch.linspace(-.4, .7, 96))
        self.terminal = nn.Parameter(torch.tensor(float(terminal)))
        self.win = nn.Parameter(torch.tensor(1.5))
        self.lost_life = lost_life
        self.index = 0

    def forward(self, field, actions):
        index = self.index
        self.index += 1
        basis = torch.linspace(-.1, .2, 96, device=field.device)
        following = field + basis * (actions[:, None, None] + 1)
        if index >= 2:
            following = following + self.tail
        count = len(field)
        negative = field.new_full((count,), -100.)
        return dict(field=following, events=dict(
            terminal_logits=self.terminal.expand(count) if index == 1 else negative,
            lost_life_logits=field.new_full((count,), self.lost_life) if index == 1 else negative,
            won_logits=self.win.expand(count) if index == 1 else negative))

def scripted(gating, *, terminal=100., lost_life=-100.):
    torch.manual_seed(52)
    model = staged.JointGoalPlanning(dict(horizon=4, hidden=32, encoder_loops=1,
                                         dynamics_loops=1, terminal_gating=gating)).eval()
    model.dynamics = ScriptedDynamics(terminal, lost_life)
    model._summary = types.MethodType(lambda self, fields: fields.mean(1), model)
    model.continuation_logits = types.MethodType(lambda self, fields: fields.new_zeros(len(fields), 4), model)
    return model

def imagine(model):
    model.dynamics.index = 0
    return model.imagine(torch.zeros(1, 148, 96))

class TerminalGatingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU-only review cannot initialize CUDA')
        torch.set_num_threads(1)

    def test_high_terminal_blocks_later_effect_and_gradient_but_retains_current_evidence(self):
        for enabled in (False, True):
            model = scripted(enabled)
            original = imagine(model)['action_logits']
            gradient = torch.autograd.grad(original.sum(), model.dynamics.tail)[0]
            if enabled:
                torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)
            else:
                self.assertGreater(float(gradient.abs().sum()), 0)
            with torch.no_grad():
                model.dynamics.tail.add_(4.0)
            changed = imagine(model)['action_logits']
            if enabled:
                torch.testing.assert_close(original, changed, rtol=0, atol=0)
            else:
                self.assertGreater(float((original - changed).detach().abs().sum()), 0)
            with torch.no_grad():
                model.dynamics.win.fill_(-1.5)
            retained = imagine(model)['action_logits']
            self.assertGreater(float((changed - retained).detach().abs().sum()), 0)

    def test_life_loss_alone_retains_future_and_soft_terminal_is_differentiable(self):
        model = scripted(True, terminal=-100.0, lost_life=100.0)
        scores = imagine(model)['action_logits']
        gradient = torch.autograd.grad(scores.sum(), model.dynamics.tail)[0]
        self.assertGreater(float(gradient.abs().sum()), 0)
        control = scripted(False, terminal=-100.0, lost_life=100.0)
        torch.testing.assert_close(scores, imagine(control)['action_logits'], rtol=0, atol=0)
        soft = scripted(True, terminal=0.0)
        gradients = torch.autograd.grad(imagine(soft)['action_logits'].sum(), (soft.dynamics.tail, soft.dynamics.terminal))
        self.assertTrue(all((float(value.abs().sum()) > 0 for value in gradients)))

    def test_enabled_roundtrip_and_ablation_gate_follow_repeated_evidence(self):
        model = staged.JointGoalPlanning(dict(terminal_gating=True, encoder_loops=1, dynamics_loops=1)).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.pt'
            staged.save_checkpoint(path, model)
            restored, _ = staged.load_checkpoint(path)
            self.assertTrue(restored.cfg.terminal_gating)
            self.assertEqual(restored.parameter_counts(), model.parameter_counts())
        with self.assertRaises(ValueError):
            staged.JointGoalPlanning(dict(terminal_gating=1))
        experiment = scripted(True)
        original = experiment.imagine(torch.zeros(1, 148, 96), tail_ablation=True)['action_logits']
        with torch.no_grad():
            experiment.dynamics.terminal.fill_(-100.0)
        experiment.dynamics.index = 0
        changed = experiment.imagine(torch.zeros(1, 148, 96), tail_ablation=True)['action_logits']
        torch.testing.assert_close(original, changed, rtol=0, atol=0)

class CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError('terminal gate tests must be CPU only')

    def test_disabled_scores_match_original_ungated_recursion(self):
        torch.manual_seed(43)
        model=staged.JointGoalPlanning(dict(encoder_loops=1,dynamics_loops=1)).eval()
        self.assertFalse(model.cfg.terminal_gating)
        field=torch.randn(1,148,96)
        with torch.inference_mode():
            result=model.imagine(field)
            evidence=[]
            for step in range(4):
                summary=model._summary(result['imagined_fields'][:,:,step].reshape(4,148,96))
                events=result['imagined_event_logits'][:,:,step].reshape(4,3).sigmoid()
                values=result['imagined_value_logits'][:,:,step].reshape(4,130).softmax(-1)
                evidence.append(torch.cat((summary,events,values),-1))
            hidden=field.new_zeros(4,model.cfg.hidden)
            for item in reversed(evidence):
                hidden=model.trajectory(item,hidden)
            branches=hidden.reshape(1,4,-1)
            pooled=branches.mean(1,keepdim=True).expand_as(branches)
            observed=model._summary(field)[:,None].expand(-1,4,-1)
            expected=model.scorer(torch.cat((branches,pooled,observed),-1)).squeeze(-1)
            torch.testing.assert_close(expected,result['action_logits'],rtol=0,atol=0)

    def test_legacy_checkpoint_config_and_parameter_layout(self):
        torch.manual_seed(9)
        old=staged.JointGoalPlanning(dict(encoder_loops=1,dynamics_loops=1)).eval()
        torch.manual_seed(9)
        enabled=staged.JointGoalPlanning(dict(encoder_loops=1,dynamics_loops=1,terminal_gating=True)).eval()
        self.assertEqual(old.parameter_counts(),enabled.parameter_counts())
        for key,value in old.state_dict().items():
            torch.testing.assert_close(value,enabled.state_dict()[key],rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'current.pt'
            payload=staged.save_checkpoint(path,old)
            payload['config'].pop('terminal_gating')
            legacy=Path(folder)/'legacy.pt'
            torch.save(payload,legacy)
            restored,_=staged.load_checkpoint(legacy)
            self.assertFalse(restored.cfg.terminal_gating)
            public=torch.randint(0,16,(1,8,64,64))
            with torch.inference_mode():
                torch.testing.assert_close(old(public),restored(public),rtol=0,atol=0)

    def test_enabled_real_model_preserves_permutation_and_joint_gradient_paths(self):
        torch.manual_seed(27)
        model=staged.JointGoalPlanning(dict(encoder_loops=1,dynamics_loops=1,terminal_gating=True)).eval()
        field=model.encode(torch.randint(0,16,(1,8,64,64)))
        result=model.imagine(field)
        changed=model.imagine(field,root_actions=torch.tensor([[3,1,0,2]]))
        torch.testing.assert_close(result['action_logits'],changed['action_logits'],atol=1e-6,rtol=1e-5)
        self.assertEqual(result['transition_count_per_decision'],16)
        F.cross_entropy(result['action_logits'],torch.tensor([2])).backward()
        for parameter in (model.encoder.cell_projection.weight,model.dynamics.action_embedding.weight,
                          model.dynamics.event_head[-1].weight,model.trajectory.weight_ih):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in model.continuation.parameters()))


if __name__=='__main__':
    unittest.main()
