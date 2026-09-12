"""Exact label contracts, detached distillation and additive diagnostic statistics."""
import unittest
import torch
from pebby.agent.structured_transition import StructuredTransition
from pebby.agent.structured_objective import objective, diagnostics, DEFAULT_WEIGHTS


def fixture():
    torch.manual_seed(741)
    def field():
        value=torch.randn(4,148,96)
        value[:,:144,48:56]=torch.rand(4,144,8)
        for start,stop in ((56,62),(62,66),(66,70),(70,76),(76,80),(80,84)):
            value[...,start:stop]=value[...,start:stop].softmax(-1)
        value[:,:144,84]=torch.rand(4,144)
        value[:,144:,48:70]=0;value[:,144:,84]=1;value[...,85:96]=0
        return value
    labels=dict(player_cell=torch.tensor([[1,1],[2,2],[3,3],[4,4]]),
                next_player_cell=torch.tensor([[1,1],[1,1],[4,3],[4,4]]),
                triple=torch.tensor([[0,1,2],[3,2,1],[5,3,0],[2,0,3]]),
                next_triple=torch.tensor([[1,1,2],[0,0,0],[5,3,0],[2,1,3]]),
                steps=torch.tensor([0,1,42,42]),next_steps=torch.tensor([-1,42,41,42]),
                lives=torch.tensor([1,3,3,3]),next_lives=torch.tensor([1,2,3,3]),
                lost_life=torch.tensor([False,True,False,False]),
                terminal=torch.tensor([True,False,False,True]),won=torch.tensor([False,False,False,True]))
    return field(),field(),torch.tensor([0,1,2,3]),labels,torch.ones(48)


class StructuredObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_finite_gradients_and_frozen_fields_for_all_supervision_paths(self):
        current,target,actions,labels,scale=fixture();current.requires_grad_();target.requires_grad_();scale.requires_grad_()
        model=StructuredTransition();result=objective(model,current,target,actions,labels,scale)
        self.assertEqual(set(result['losses']),set(DEFAULT_WEIGHTS))
        self.assertTrue(all(torch.isfinite(x) for x in result['losses'].values()))
        result['total'].backward()
        self.assertIsNone(current.grad);self.assertIsNone(target.grad);self.assertIsNone(scale.grad)
        for name,p in model.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(float(p.grad.abs().sum()),0,name)
        # Closure directly supervises every output channel, including zero padding.
        self.assertTrue((model.output[-1].weight.grad.abs().sum(-1)>0).all())

    def test_group_balanced_consistency_includes_carried_visibility_and_padding(self):
        current,target,actions,labels,scale=fixture();model=StructuredTransition()
        weights={k:0. for k in DEFAULT_WEIGHTS};weights['field_carried']=1.
        result=objective(model,current,target,actions,labels,scale,weights=weights)
        expected=(result['output']['field'][...,70:84]-target[...,70:84]).square().mean()
        torch.testing.assert_close(result['total'],expected)
        for name,(start,stop) in {'appearance':(48,70),'visibility':(84,85),'padding':(85,96)}.items():
            expected=(result['output']['field'][...,start:stop]-target[...,start:stop]).square().mean()
            torch.testing.assert_close(result['losses']['field_'+name],expected)

    def test_unobserved_goals_and_unchanged_cells_give_finite_zero_not_false_denominators(self):
        current,target,actions,labels,scale=fixture();current[:,:144,84]=0;target=current.clone()
        result=objective(StructuredTransition(),current,target,actions,labels,scale)
        for name in ('readout_goal','readout_roles','field_changed'):
            self.assertEqual(float(result['losses'][name].detach()),0.)
            self.assertTrue(result['losses'][name].requires_grad)
        stats=diagnostics(result,current,target,labels,scale)
        self.assertEqual(stats['counts']['predicted_changed_appearance_mse'],0)
        self.assertEqual(stats['sums']['next_teacher_visible_goal_mass'],0.)
        result['total'].backward()

    def test_statistics_aggregate_across_chunks_with_correct_event_and_moved_counts(self):
        current,target,actions,labels,scale=fixture();model=StructuredTransition().eval()
        with torch.no_grad():
            result=objective(model,current,target,actions,labels,scale)
            full=diagnostics(result,current,target,labels,scale)
            pieces=[]
            for start,stop in ((0,1),(1,4)):
                subset={k:v[start:stop] for k,v in labels.items()}
                part=objective(model,current[start:stop],target[start:stop],actions[start:stop],subset,scale)
                pieces.append(diagnostics(part,current[start:stop],target[start:stop],subset,scale))
        for name,count in full['counts'].items():self.assertEqual(count,sum(p['counts'][name] for p in pieces),name)
        for name,value in full['sums'].items():
            self.assertAlmostEqual(value,sum(p['sums'][name] for p in pieces),delta=max(1e-5,abs(value)*1e-5),msg=name)
        self.assertEqual(full['counts']['predicted_moved_player_accuracy'],2)
        self.assertEqual(full['counts']['predicted_stationary_player_accuracy'],2)
        self.assertEqual(full['counts']['predicted_reset_player_accuracy'],1)
        self.assertEqual(full['sums']['event_lost_life_positive'],1)
        self.assertEqual(full['sums']['event_terminal_positive'],2)
        self.assertEqual(full['sums']['next_steps_underflow'],1)
        for name in result['output']['events']:result['output']['events'][name]=torch.full((4,),-1.)
        unsafe=diagnostics(result,current,target,labels,scale)
        self.assertEqual(unsafe['sums']['event_unsafe_false_safe_rate'],2.)
        self.assertEqual(unsafe['counts']['event_unsafe_false_safe_rate'],2)
        self.assertEqual(unsafe['sums']['event_lost_life_fn'],1.)

    def test_event_positive_weights_capped_and_invalid_contracts_rejected(self):
        current,target,actions,labels,scale=fixture();model=StructuredTransition()
        result=objective(model,current,target,actions,labels,scale,pos_weight=[100,2,1])
        torch.testing.assert_close(result['effective_pos_weight'],torch.tensor([20.,2.,1.]))
        for value in ([0,1,1],[1,2],[float('nan'),1,1]):
            with self.assertRaises(ValueError):objective(model,current,target,actions,labels,scale,pos_weight=value)
        for scale_bad in (torch.zeros(48),torch.ones(47),torch.full((48,),float('nan'))):
            with self.assertRaises(ValueError):objective(model,current,target,actions,labels,scale_bad)
        for name,value in [('next_steps',torch.tensor([43,0,1,2])),('lives',labels['lives'].float()),
                           ('next_triple',torch.tensor([[0,4,0]]*4)),('next_player_cell',torch.tensor([[12,0]]*4))]:
            changed=dict(labels);changed[name]=value
            with self.assertRaises(ValueError):objective(model,current,target,actions,changed,scale)
        changed=dict(labels);changed['terminal']=torch.zeros(4,dtype=torch.bool)
        with self.assertRaises(ValueError):objective(model,current,target,actions,changed,scale)
        changed=dict(labels);changed.pop('steps')
        with self.assertRaises(ValueError):objective(model,current,target,actions,changed,scale)
        for weights in ({'wrong':1},{'events':-1},{'events':float('inf')}):
            with self.assertRaises(ValueError):objective(model,current,target,actions,labels,scale,weights=weights)


if __name__=='__main__':unittest.main()
