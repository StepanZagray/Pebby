import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from tools.diagnose_structured_workspace_trajectory import CompleteTeacher, diagnose_level, rates
from tools.evaluate_structured_workspace_gameplay import checked_bank
from pebby.ls20.env import Ls20Scenario


class PublicStub(torch.nn.Module):
    def __init__(self, trace=None): super().__init__(); self.trace = trace if trace is not None else []; self.inputs=[]
    def config(self): return {'architecture':'structured','history':8}
    def forward(self, frames, history_valid=None, previous_actions=None):
        self.trace.append('policy')
        self.inputs.append((history_valid.clone(),previous_actions.clone()))
        return torch.tensor([[1.,0.,0.,0.]])


class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_continue_unreachable_reset_and_action_before_teacher(self):
        trace=[]
        class Env:
            level_index=0
            def reset(self): self.index=0;return [[0]*64 for _ in range(64)]
            def lives(self): return 3-int(self.index>=3)
            def goal_triples(self): return [(0,0,0)]
            def goals_solved(self): return [False]
            def perform(self, action):
                self.index+=1
                return SimpleNamespace(frame=[[0]*64 for _ in range(64)],finished=False,won=False)
        class Teacher:
            def label(self, env):
                trace.append('teacher')
                reachable=env.index not in (1,2)
                return {'status':'reachable' if reachable else 'unreachable', 'distance':2 if reachable else None,
                        'optimal_mask':(2 if env.index==0 else 1) if reachable else None}
            def before(self,*args):return None
            def after(self,*args):pass
        policy=PublicStub(trace); result=diagnose_level(policy,Env(),Teacher(),max_actions=12)
        self.assertEqual(result['summary']['actions'],12)
        self.assertEqual(result['summary']['unreachable_decisions'],2)
        self.assertEqual(result['summary']['reachable_decisions'],10)
        self.assertEqual(result['summary']['first_mistake'],0)
        self.assertEqual(result['summary']['life_losses'],1)
        self.assertEqual(result['summary']['resets'],1)
        self.assertEqual(trace,['policy','teacher','teacher']*12)
        # Every next policy call occurs despite the previous unknown/dead-end label.
        self.assertEqual(trace.count('policy'),12)
        self.assertEqual(int(policy.inputs[3][0].sum()),1)
        self.assertTrue((policy.inputs[3][1]==-1).all())
        self.assertGreater(result['summary']['decisions_after8_unchanged'],0)

    def test_unknown_denominators_and_truncation(self):
        rows=[{'status':'reachable','optimal':True},{'status':'unreachable','optimal':None},
              {'status':'unknown_truncated','optimal':None},{'status':'unknown_unsupported','optimal':None}]
        result=rates(rows)
        self.assertEqual(result['reachable_optimal_rate'],1)
        self.assertEqual(result['unreachable_fraction_all'],.25)
        self.assertEqual(result['unreachable_fraction_known'],.5)
        self.assertEqual(result['unknown_decisions'],2)
        oracle=SimpleNamespace(engine='fast',_reachable=3,truncated=True)
        with patch('tools.diagnose_structured_workspace_trajectory.extract',return_value=None), patch('tools.diagnose_structured_workspace_trajectory.Oracle',return_value=oracle):
            teacher=CompleteTeacher(None,{'training_context_index':1})
        self.assertEqual(teacher.label(None)['status'],'unknown_truncated')

    def test_missing_distance_requires_coverage_and_selected_mismatch_fails(self):
        teacher=CompleteTeacher.__new__(CompleteTeacher)
        teacher.oracle=SimpleNamespace(state_of=lambda env:'state',distance_for=lambda state:None,start='start')
        teacher.covered=False
        self.assertEqual(teacher.label(None)['status'],'unknown_graph_coverage')
        teacher.covered=True
        self.assertEqual(teacher.label(None)['status'],'unreachable')
        env=SimpleNamespace(lives=lambda:3)
        with self.assertRaisesRegex(ValueError,'transition mismatch'):
            teacher.after(env,SimpleNamespace(won=False,finished=False),('before','wrong','moved',3))

    def test_pruned_changed_rejection_invalidates_coverage_and_reset_restores(self):
        teacher=CompleteTeacher.__new__(CompleteTeacher)
        teacher.oracle=SimpleNamespace(state_of=lambda env:env.state,start='start')
        teacher.covered=True
        env=SimpleNamespace(state='changed',lives=lambda:3)
        teacher.after(env,SimpleNamespace(won=False,finished=False),('old','changed','rejected',3))
        self.assertFalse(teacher.covered)
        env=SimpleNamespace(state='start',lives=lambda:2)
        teacher.after(env,SimpleNamespace(won=False,finished=False),('changed','dead','died',3))
        self.assertTrue(teacher.covered)

    def test_real_generated_complete_teacher_selected_transition(self):
        levels,optima,specs=checked_bank()
        env=Ls20Scenario(levels[0],specs[0]['training_context_index']);env.reset()
        teacher=CompleteTeacher(env,specs[0])
        self.assertTrue(teacher.proof['complete'])
        result=diagnose_level(PublicStub(),env,teacher,max_actions=1)
        self.assertEqual(result['summary']['actions'],1)
        self.assertEqual(result['decisions'][0]['status'],'reachable')
        self.assertEqual(result['decisions'][0]['distance'],optima[0])


if __name__=='__main__':unittest.main()
