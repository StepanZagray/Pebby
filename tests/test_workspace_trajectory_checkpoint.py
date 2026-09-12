import unittest
import torch
from tests.test_policy_history import corridor
from tests.test_structured_workspace_trajectory import PublicStub
from pebby.ls20.generate import build_level
from pebby.ls20.env import Ls20Scenario
from tools.diagnose_structured_workspace_trajectory import diagnose_level as original
from tools.diagnose_structured_workspace_trajectory_checkpoint import CompleteTeacher, diagnose_level, semantic_before


class FlexibleTrajectoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.spec=corridor(budget=2)
        self.spec.update(seed=3,training_context_index=3,context_optimal_actions=3)

    def make(self):
        env=Ls20Scenario(build_level(self.spec),3);env.reset()
        return env,CompleteTeacher(env,self.spec)

    def test_semantics_preserve_full_actor_reset_trajectory(self):
        env,teacher=self.make();baseline=original(PublicStub(),env,teacher,max_actions=12)
        env,teacher=self.make();trace=[];policy=PublicStub(trace)
        label=teacher.label
        def observed(env):trace.append('teacher');return label(env)
        teacher.label=observed
        result=diagnose_level(policy,env,teacher,max_actions=12,spec=self.spec)
        self.assertEqual(result['summary'],baseline['summary'])
        self.assertEqual(trace[0],'policy')
        self.assertEqual(result['summary']['life_losses'],3)
        self.assertEqual(result['summary']['ending'],'game_over')
        self.assertTrue(all(r['semantic']['wall_or_edge_attempt'] for r in result['decisions']))
        self.assertEqual(sum(r['semantic']['life_loss_budget_exhaustion_verified'] for r in result['decisions']),3)
        self.assertEqual(int(policy.inputs[3][0].sum()),1)

    def test_mismatched_goal_facts_do_not_choose_action(self):
        env,teacher=self.make()
        self.spec['goals'][0]={'cell':(4,3),'triple':[1,0,0]}
        facts=semantic_before(env,teacher,3,self.spec)
        self.assertEqual(facts['attempted_neighbor'],(4,3))
        self.assertTrue(facts['goal']['unsolved_mismatch'])
        self.assertEqual(facts['goal']['mismatched_fields'],['shape'])


if __name__=='__main__':unittest.main()
