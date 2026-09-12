import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from pebby.agent.evaluate import rollout
from pebby.agent import world_data as wd
from tools import diagnose_reference_policy as d


class Scripted(torch.nn.Module):
    def __init__(self,actions):super().__init__();self.actions=actions;self.inputs=[]
    def config(self):return {'architecture':'world','history':8}
    def forward(self,frames,history_valid=None,previous_actions=None):
        self.inputs.append((frames.clone(),history_valid.clone(),previous_actions.clone()))
        logits=torch.zeros((1,4));logits[0,self.actions[(len(self.inputs)-1)%len(self.actions)]]=10
        return logits


class ReferenceDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.spec=json.loads(Path('tests/fixtures/ls20_reference/tier1.json').read_text())

    def test_memory_guard_never_shrinks_search(self):
        with self.assertRaises(MemoryError):d.check_memory(9*d.GIB-1)
        d.check_memory(9*d.GIB)

    def test_scripted_full_lives_matches_existing_rollout_and_history(self):
        spec=self.spec;start=tuple(spec['start']);walls={tuple(c) for c in spec['walls']}
        special={tuple(v['cell']) for name in ('goals','cyclers','launchers') for v in spec.get(name,[])}
        special.update(map(tuple,spec['refills']))
        deltas=[(0,-1),(0,1),(-1,0),(1,0)]
        action=next(i for i,(dx,dy) in enumerate(deltas) if
                    (start[0]+dx,start[1]+dy) not in walls|special
                    and 0<=start[0]+dx<12 and 0<=start[1]+dy<12)
        actor=Scripted([action,action^1]);reference=Scripted([action,action^1])
        original=d.checked_expansion
        def checked(*args):
            self.assertEqual(len(actor.inputs),args[-1]+1) # action was chosen first
            return original(*args)
        with patch.object(d,'memory_available',return_value=10*d.GIB),patch.object(d,'checked_expansion',side_effect=checked):
            result=d.diagnose_level(spec,actor,max_actions=140)
        env,_,_=wd.verified_context(spec)
        expected=rollout(reference,env,140,device='cpu',on_stall='repeat',temperature=0.)
        self.assertEqual(result['ending'],expected['ending'])
        self.assertEqual(result['counts']['actions'],expected['actions'])
        self.assertEqual(result['counts']['stalls'],expected['stalls'])
        self.assertEqual(result['lives_left'],expected['lives_left'])
        self.assertEqual(result['counts']['life_losses'],3)
        self.assertGreater(result['counts']['within_life_unreachable_decisions'],0)
        self.assertEqual(result['branch_checks']['branches'],4*expected['actions'])
        self.assertEqual(len(actor.inputs),len(reference.inputs))
        for actual,other in zip(actor.inputs,reference.inputs):
            for a,b in zip(actual,other):self.assertTrue(torch.equal(a,b))
        for index,step in enumerate(result['steps'][:-1]):
            if step['lost_life']:
                next_step=result['steps'][index+1]
                self.assertEqual(sum(next_step['history_valid']),1)
                self.assertEqual(next_step['previous_actions'],[-1]*8)

    def test_refusal_histories_continue_past_eight(self):
        from collections import deque
        spec=self.spec;start=tuple(spec['start']);goal=tuple(spec['goals'][0]['cell'])
        blocked={tuple(c) for c in spec['walls']}|{tuple(c['cell']) for c in spec['cyclers']}
        directions=[(0,-1),(0,1),(-1,0),(1,0)]
        queue=deque([(start,[])]);seen={start};route=None
        # Tiny144cell fixture path only, to exercise an unmatched-goal refusal.
        while queue:
            cell,path=queue.popleft()
            if cell==goal:route=path;break
            for action,(dx,dy) in enumerate(directions):
                nxt=(cell[0]+dx,cell[1]+dy)
                if nxt not in seen|blocked and 0<=nxt[0]<12 and 0<=nxt[1]<12:
                    seen.add(nxt);queue.append((nxt,path+[action]))
        self.assertIsNotNone(route)
        actor=Scripted(route[:-1]+[route[-1]]*12)
        with patch.object(d,'memory_available',return_value=10*d.GIB):
            result=d.diagnose_level(spec,actor,max_actions=len(route)-1+12)
        self.assertEqual(result['counts']['actions'],len(route)-1+12)
        self.assertEqual(result['ending'],'capped')
        self.assertEqual(result['counts']['stalls'],12)
        self.assertEqual(sum(result['steps'][-1]['history_valid']),8)

    def test_checkpoint_must_be_fresh_and_selected_seeds_training_only(self):
        metadata=dict(data_meta={'source':'generated_only'},curriculum={'difficulty_version':d.DIFFICULTY_VERSION},
                      trained='recorded',epochs=1,train_seeds=[self.spec['seed']],validation_seeds=[])
        d.validate_checkpoint(metadata,[self.spec])
        for change in ({'initialize_checkpoint':'old.pt'},{'validation_seeds':[self.spec['seed']]},
                       {'train_seeds':[]},{'trained':None}):
            with self.assertRaises(ValueError):d.validate_checkpoint(dict(metadata,**change),[self.spec])

    def test_selection_rejects_validation_and_is_tier_balanced(self):
        specs=[]
        for tier in range(1,8):
            for index in range(3):
                value=copy.deepcopy(self.spec);value.update(seed=tier*100+index,difficulty=tier)
                specs.append(value)
        with patch.object(d,'profile_errors',return_value=[]):
            selected=d.select_levels(specs,14)
            self.assertEqual([sum(s['difficulty']==tier for s in selected) for tier in range(1,8)],[2]*7)
            self.assertEqual(selected,d.select_levels(specs,14))
            specs[0]['split']='validation'
            with self.assertRaises(ValueError):d.select_levels(specs,14)

if __name__=='__main__':unittest.main()
