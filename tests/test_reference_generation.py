"""Seven reference tiers are contracts on actual procedural levels, not labels."""
import copy
import json
from pathlib import Path
import unittest

from pebby.ls20 import names
from pebby.ls20.generate import build_level
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.layout import extract
from pebby.ls20.plan import Oracle
from pebby.ls20.reference_generator import generate_level, patroller_contacts
from pebby.ls20.reference_profiles import PROFILES, profile_errors
from pebby.ls20.generation_quality import geometry_d4_hash

FIXTURES=Path(__file__).parent/'fixtures'/'ls20_reference'


class ReferenceGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows={d:json.loads((FIXTURES/f'tier{d}.json').read_text()) for d in range(1,8)}

    def test_all_seven_profiles_replay_in_corresponding_real_engine_context(self):
        for d,row in self.rows.items():
            with self.subTest(difficulty=d):
                self.assertEqual(profile_errors(row),[])
                self.assertEqual(row['training_context_index'],d-1)
                env=Ls20Scenario(build_level(row),d-1)
                layout=extract(env)
                self.assertEqual(len(layout.patrollers),len(PROFILES[d]['rails']))
                for step,action in enumerate(row['solution']):
                    result=env.perform(action)
                    self.assertEqual(env.lives(),3)
                    self.assertEqual(result.won,step==len(row['solution'])-1)
                self.assertEqual(env.levels_completed,1)

    def test_five_means_ls20_level_five_not_old_two_goal_category(self):
        row=self.rows[5]
        self.assertEqual((len(row['goals']),len(row['launchers']),len(row['refills'])),(1,8,3))
        self.assertEqual(row['step_cost'],2)
        self.assertGreaterEqual(row['solution_mechanics']['used_launcher_count'],4)
        self.assertEqual(row['solution_mechanics']['used_patroller_count'],1)
        self.assertEqual(row['reference_optimal_actions'],44)

    def test_profiles_reject_relabeling_and_structural_collapse(self):
        for d,row in self.rows.items():
            bad=copy.deepcopy(row);bad['difficulty']=d%7+1
            self.assertTrue(profile_errors(bad))
            bad=copy.deepcopy(row);bad['optimal_actions']=5
            self.assertTrue(profile_errors(bad))
        for field,value in [('launchers',[]),('rails',[]),('step_cost',1),('fog',True)]:
            bad=copy.deepcopy(self.rows[5]);bad[field]=value
            self.assertTrue(profile_errors(bad),field)

    def test_new_public_generation_emits_measured_versioned_proof(self):
        a=generate_level(720000,1,attempts=80)
        b=generate_level(720000,1,attempts=80)
        self.assertEqual(a,b)
        self.assertEqual(profile_errors(a),[])
        self.assertEqual(a['proof']['context_index'],0)
        self.assertIs(a['proof']['search_truncated'],False)

    def test_high_tiers_are_not_truncated_into_the_old_small_state_space(self):
        self.assertGreater(self.rows[6]['reachable_states'],600000)
        self.assertGreater(self.rows[7]['reachable_states'],600000)
        self.assertEqual(len(self.rows[6]['goals']),2)
        self.assertEqual(sorted(len(r['cells']) for r in self.rows[6]['rails']),[5,5,8])
        self.assertEqual(len(self.rows[7]['rails'][0]['cells']),6)
        self.assertTrue(self.rows[7]['fog'])

    def test_geometry_holdout_is_invariant_to_rotation_and_mirror(self):
        row=self.rows[3];original=geometry_d4_hash(row)
        for transform in (lambda x,y:(11-y,x),lambda x,y:(11-x,y)):
            changed={**row,'walls':[transform(*cell) for cell in row['walls']]}
            self.assertEqual(geometry_d4_hash(changed),original)

    def test_single_action_ties_vary_without_changing_exact_optimality(self):
        free={(x,y) for x in range(2,5) for y in range(2,5)}
        spec={'walls':sorted({(x,y) for x in range(12) for y in range(12)}-free),
              'start':(2,2),'start_triple':[0,0,0],'goals':[{'cell':(4,4),'triple':[0,0,0]}],
              'cyclers':[],'refills':[],'step_counter':42,'step_cost':1,'fog':False}
        env=Ls20Scenario(build_level(spec),2);oracle=Oracle(extract(env))
        choices={oracle.action_for(oracle.start,seed=seed) for seed in range(32)}
        self.assertEqual(choices,{1,3})
        self.assertEqual(oracle.action_at(env),oracle.action_at(env))
        for seed in range(32):
            action=oracle.action_at(env,seed=seed)
            self.assertIn(names.ACTION_IDS.index(action),choices)

    def test_wall_trigger_launch_contacts_patroller_at_retained_tick(self):
        from pebby.ls20.plan import simulate
        free={(x,y) for x in range(2,6) for y in range(2,5)}
        spec={'walls':sorted({(x,y) for x in range(12) for y in range(12)}-free),
              'start':(2,2),'start_triple':[0,0,0],
              'goals':[{'cell':(4,4),'triple':[0,0,1]}],
              'cyclers':[{'cell':(5,2),'kind':'rotation'}],
              'rails':[{'cells':[(5,2),(5,3)]}],
              'launchers':[{'cell':(2,2),'delta':(1,0)}],
              'refills':[],'step_counter':42,'step_cost':1,'fog':False}
        env=Ls20Scenario(build_level(spec),4);layout=extract(env);oracle=Oracle(layout)
        before=oracle.start
        after,outcome=simulate(layout,before,2,oracle.refills)
        env.perform(3)
        self.assertEqual(outcome,'launched')
        self.assertEqual(oracle.state_of(env),after)
        self.assertEqual(after[7],before[7])
        self.assertEqual(after[1:4],(0,0,1))
        self.assertEqual(patroller_contacts(layout,before,after,2,outcome),{0})

    def test_global_search_ceiling_cannot_exceed_tier_budget(self):
        row=generate_level(720000,1,attempts=80,search_limit=32000000)
        self.assertEqual(row['search_limit'],PROFILES[1]['search_limit'])

    def test_requested_slack_cannot_replace_fixed_profile_floor(self):
        from pebby.ls20.reference_generator import verify
        for requested in (0,16):
            row=generate_level(720000,1,attempts=80,min_slack=requested)
            self.assertEqual(row['budget_floor'],8)
            self.assertGreaterEqual(row['minimum_slack_moves'],max(8,requested))
            self.assertEqual(profile_errors(row),[])
        row,reason=verify(self.rows[2],min_slack=1)
        self.assertIsNone(reason)
        self.assertEqual(row['budget_floor'],0)
        self.assertEqual(profile_errors(row),[])
        row,reason=verify(self.rows[2],min_slack=8)
        self.assertIsNone(row)
        self.assertEqual(reason,'route_budget_floor')
        bad=copy.deepcopy(self.rows[1]);bad['budget_floor']=0
        self.assertTrue(profile_errors(bad))
