import copy
import hashlib
import unittest

import numpy as np
import torch

from pebby.agent.structured_policy import StructuredPolicyReadout
from pebby.agent.structured_workspace_policy import StructuredWorkspaceReadout
from tools.structured_onpolicy_sampling import PairedStateSampler
from tools.train_structured_onpolicy_comparison import validate_rows, prepare_batch, selection_digest, fresh_head, merge_sources
from tools.preflight_structured_workspace import backward
from tools.structured_policy_batch import prepared_policy_inputs
from tools.train_structured_policy import policy_terms


def fixture():
    base={'seeds':np.arange(2000,2016,dtype=np.int64),'difficulties':np.arange(16)%5+1}
    seeds=np.repeat(base['seeds'][:4],2);n=len(seeds)
    rng=np.random.default_rng(17)
    data={key:rng.normal(size=shape).astype(dtype) for key,shape,dtype in [
        ('fields',(n,148,96),np.float16),('next_fields',(n,4,148,96),np.float16),('imagined_fields',(n,4,148,96),np.float32)]}
    data.update(seeds=seeds,difficulties=np.repeat(base['difficulties'][:4],2),source_rows=np.arange(n),
        on_policy=np.tile([True,False],4),level_seeds=np.unique(seeds),level_offsets=np.arange(0,n+1,2),
        level_rows=np.arange(n),optimal=np.ones(n,np.uint8),distances=np.tile([1,2,3,4],(n,1)),
        current_distance=np.full(n,2),steps=np.full(n,42),next_steps=np.full((n,4),41),
        lives=np.full(n,3),next_lives=np.full((n,4),3),next_optimal=np.ones((n,4),np.uint8),
        player_cell=np.ones((n,2),int),next_player_cell=np.ones((n,4,2),int),
        triple=np.zeros((n,3),int),next_triple=np.zeros((n,4,3),int),
        context_index=seeds%7)
    for key in ('lost_life','terminal','won'):data[key]=np.zeros((n,4),bool)
    for row in range(1,n,2):
        data['terminal'][row,0]=data['won'][row,0]=True
        data['distances'][row,0]=0;data['next_optimal'][row,0]=0;data['current_distance'][row]=1
    proofs=[]
    for i,seed in enumerate(base['seeds'][:4]):
        proofs.append({'seed':int(seed),'context_index':int(seed%7),'context_engine_verified':True,'search_truncated':False,
            'row_start':2*i,'row_count':2,'on_policy_row_count':1,'on_policy_samples':1,'samples':2,
            'branch_checks':{'branches':8,'expansions':2},'win_covered':True})
    meta={'source':'generated_only','oracle_search':'complete_only','history':8,'alternatives_per_state':4,
        'collection_policy':'model_greedy','on_policy_rows':list(range(0,n,2)),'levels':proofs}
    manifest={'rows':n,'levels':4,'on_policy_rows':4,'expert_rows':4,'source_metadata':meta}
    views=[]
    for view in range(2):
        values={'seeds':base['seeds'],'difficulties':base['difficulties'],'source_rows':np.arange(16)+view*100,
            'optimal':np.ones(16,np.uint8)}
        values['next_fields']=rng.normal(size=(16,4,148,96)).astype(np.float16)
        values['imagined_fields']=rng.normal(size=(16,4,148,96)).astype(np.float32)
        views.append(values)
    return base,data,manifest,views


class OnPolicyComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):torch.manual_seed(42)

    def test_group_labels_and_expert_proofs_fail_closed(self):
        base,data,meta,_=fixture();validate_rows(data,meta,base)
        cases=[]
        changed=copy.deepcopy(data);changed['level_rows']=changed['level_rows'][::-1];cases.append(changed)
        changed=copy.deepcopy(data);changed['current_distance'][0]+=1;cases.append(changed)
        changed=copy.deepcopy(data);changed['optimal'][0]=0;cases.append(changed)
        changed=copy.deepcopy(data);changed['imagined_fields']=changed['imagined_fields'].astype(np.float16);cases.append(changed)
        changed=copy.deepcopy(data);changed['on_policy'][0]=False;changed['on_policy'][1]=True;cases.append(changed)
        for changed in cases:
            with self.assertRaises(ValueError):validate_rows(changed,meta,base)
        changed=copy.deepcopy(meta);changed['source_metadata']['levels'][0]['branch_checks']['branches']=4
        with self.assertRaises(ValueError):validate_rows(data,changed,base)

    def test_matched_level_replacement_only_selected_rows_and_no_anchors(self):
        base,data,meta,views=fixture()
        sampler=PairedStateSampler(base['seeds'],base['difficulties'],data['seeds'],data['on_policy'])
        selection=sampler.draw(8,4,.5,np.random.default_rng(42))
        control=prepare_batch(views,data,selection,False);treatment=prepare_batch(views,data,selection,True)
        selected=selection.trajectory_rows>=0
        self.assertEqual(len(np.unique(control['seeds'])),8)
        np.testing.assert_array_equal(control['seeds'],treatment['seeds'])
        for key in control:
            np.testing.assert_array_equal(control[key][~selected],treatment[key][~selected])
            np.testing.assert_array_equal(treatment[key][selected],data[key][selection.trajectory_rows[selected]])
        self.assertEqual(treatment['imagined_fields'].dtype,np.float32)
        data['on_policy'][selection.trajectory_rows[selected][0]]=False
        with self.assertRaisesRegex(ValueError,'expert'):prepare_batch(views,data,selection,True)

    def test_identical_seeded_stream_and_initializer(self):
        base,data,meta,views=fixture();sampler=PairedStateSampler(base['seeds'],base['difficulties'],data['seeds'],data['on_policy'])
        streams=[]
        for arm in range(2):
            rng=np.random.default_rng(42);depth_rng=np.random.default_rng(44);stream=hashlib.sha256()
            for step in range(20):
                selection=sampler.draw(8,4,step/19,rng);depth=int(depth_rng.choice([1,2,4]));selection_digest(stream,selection,depth)
            streams.append(stream.hexdigest())
        self.assertEqual(streams[0],streams[1])
        initial=StructuredWorkspaceReadout.from_readout(StructuredPolicyReadout({'mode':'successors'}),checkpoint_workspace=True)
        a=fresh_head(initial,'cpu');b=fresh_head(initial,'cpu')
        for key in initial.state_dict():
            torch.testing.assert_close(a.state_dict()[key],b.state_dict()[key],atol=0,rtol=0)
        self.assertEqual(a.trainable_parameter_count(),201315)

    def test_split_backward_same_as_joint_objective_on_replacements(self):
        base,data,meta,views=fixture();sampler=PairedStateSampler(base['seeds'],base['difficulties'],data['seeds'],data['on_policy'])
        batch=prepare_batch(views,data,sampler.draw(2,1,.5,np.random.default_rng(2)),True)
        old=StructuredPolicyReadout({'mode':'successors'})
        with torch.no_grad():old.scorer.weight.normal_(std=.1)
        a=StructuredWorkspaceReadout.from_readout(old)
        with torch.no_grad():a.workspace.attention_gate.fill_(.1);a.workspace.mlp_gate.fill_(.1)
        b=fresh_head(a,'cpu');backward(a,batch,'cpu',2)
        inputs,masks=prepared_policy_inputs(batch,'successors','cpu')
        sum(.5*policy_terms(b(x,loops=2),masks)['ce'].mean() for x in inputs.values()).backward()
        for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=1e-7,rtol=1e-5,msg=name)

    def test_conflicting_source_aliases_rejected(self):
        sources={'tools/../tools/file.py':'a'*64}
        with self.assertRaisesRegex(ValueError,'conflicting'):merge_sources(sources,{'tools/file.py':'b'*64})


if __name__=='__main__':unittest.main()
