"""Build and validate source-pinned closing-only sidecar; CPU, no engine/oracle."""
import argparse,hashlib,json,os,resource,time
from pathlib import Path
import numpy as np
import torch
from pebby.agent import world_closing_sequences as closing
from pebby.agent.on_policy_provenance import file_digest
from pebby.agent.world_train import as_tensors

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/ls20-world-onpolicy-aggregate2-train.npz'))
    p.add_argument('--base',type=Path,help='Optional full base bank for default curriculum checks')
    p.add_argument('--attestation',type=Path,default=Path('artifacts/world-closing-actions-batch1.json'))
    p.add_argument('--attestation-sha256',required=True)
    p.add_argument('--out',type=Path,default=Path('data/ls20-world-onpolicy-aggregate2-closing-k4.npz'))
    p.add_argument('--report',type=Path,default=Path('artifacts/world-closing-sidecar-verification.json'))
    args=p.parse_args();torch.set_num_threads(1);started=time.monotonic();print('PID',os.getpid(),flush=True)
    sha=file_digest(args.source);cache=Path('data/world-array-cache')/f'{sha}-07d1d1f252efefe5'
    manifest=json.loads((cache/'manifest.json').read_text());assert manifest['source_sha256']==sha
    data={k:np.load(cache/f'{k}.npy',mmap_mode='r') for k in manifest['arrays'] if k!='meta'}
    with np.load(args.source) as archive:data['meta']=json.loads(str(archive['meta'].item()))
    index=(closing.load_sidecar if args.out.exists() else closing.build_sidecar)(*( (args.out,args.source,data) if args.out.exists() else (data,args.source,args.out) ),attestation=args.attestation,attestation_sha256=args.attestation_sha256)
    loaded=closing.load_sidecar(args.out,args.source,data,attestation=args.attestation,attestation_sha256=args.attestation_sha256)
    assert np.array_equal(index.branch_actions,loaded.branch_actions)
    tensors=as_tensors(data)
    base_data=data;base_tensors=tensors;base_sha=sha
    if args.base:
        base_sha=file_digest(args.base);base_cache=Path('data/world-array-cache')/f'{base_sha}-07d1d1f252efefe5'
        bm=json.loads((base_cache/'manifest.json').read_text());assert bm['source_sha256']==base_sha
        base_data={k:np.load(base_cache/f'{k}.npy',mmap_mode='r') for k in bm['arrays'] if k!='meta'}
        with np.load(args.base) as archive:base_data['meta']=json.loads(str(archive['meta'].item()))
        base_tensors=as_tensors(base_data)
    # Same real source as base gives2048distinct available levels and exercises
    # removal of reserved level identities without touching the full merged bank.
    sampler=closing.ClosingSampler(base_data,data,index,fraction=.5,**({} if args.base else {'start':[.2]*5,'end':[.2]*5}))
    tests=[]
    for progress in (0.,.5,1.):
        indices=sampler.indices(1024,progress,torch.Generator().manual_seed(2026+int(progress*10)))
        bi,pi,order=indices;batch=closing.closing_mixed_batch(base_tensors,tensors,indices,index)
        assert len(set(sampler.last_level_seeds))==1024 and int(batch['rollout_mask'].sum())==512
        positions=torch.argsort(order)[len(bi):]
        branch_rows,branch_actions=index.lookup(pi.numpy())
        for key in closing.TARGETS:
            expected=tensors[key][torch.from_numpy(branch_rows),torch.from_numpy(branch_actions)]
            torch.testing.assert_close(batch[key][positions],expected,atol=0,rtol=0)
        for key in base_tensors:
            torch.testing.assert_close(batch[key][torch.argsort(order)[:len(bi)]],base_tensors[key][bi],atol=0,rtol=0)
        torch.testing.assert_close(batch['frames'][positions],tensors['frames'][pi],atol=0,rtol=0)
        torch.testing.assert_close(batch['optimal'][positions],tensors['optimal'][pi],atol=0,rtol=0)
        assert not batch['terminal'][positions,:3].any() and not batch['lost_life'][positions,:3].any()
        tests.append({'schedule_progress':progress,'difficulty_counts':sampler.last_difficulty_counts,'batch':1024,'distinct_levels':1024,'closing_rows':512,'all_chronological_targets_exact':True,'ordinary_rows_exact':True,'current_history_and_policy_mask_unchanged':True})
        del batch,expected
    rr,aa=index.branch_rows,index.branch_actions
    won=data['won'][rr[:,-1],aa[:,-1]];terminal=data['terminal'][rr[:,-1],aa[:,-1]];lost=data['lost_life'][rr[:,-1],aa[:,-1]];dist=data['distances'][rr[:,-1],aa[:,-1]]
    result={'status':'complete','pid':os.getpid(),'source_sha256':sha,'source_rows':len(data['seeds']),'base_path':str(args.base or args.source),'base_sha256':base_sha,'base_levels':len(np.unique(base_data['seeds'])),'default_curriculum':bool(args.base),'source_levels':len(np.unique(data['seeds'])),'sidecar':str(args.out),'sidecar_sha256':file_digest(args.out),'attestation':str(args.attestation),'attestation_sha256':args.attestation_sha256,'anchors':len(rr),'distinct_levels':len(np.unique(data['seeds'][index.anchor_row])),'closing_outcomes':{'won':int(won.sum()),'game_over':int((terminal&~won).sum()),'life_reset':int((lost&~terminal).sum()),'unreachable':int(((dist<0)&~terminal&~lost).sum()),'live_reachable':int(((dist>=0)&~terminal&~lost).sum())},'real_B1024_batches':tests,'script_sha256':file_digest(__file__),'module_sha256':file_digest(closing.__file__),'cpu_threads':1,'oracle_calls':0,'engine_calls':0,'source_unchanged':file_digest(args.source)==sha,'runtime_seconds':time.monotonic()-started,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'limitations':['Production model/trainer are not changed or integrated.',('Production combined base bank and default curriculum checked.' if args.base else 'Aggregate2 was used as both base and supplemental with uniform curriculum.'),'Actual data has no final life resets; draft loss tests cover generated reset convention separately.','Large preexisting mmap cache files were not rehashed for this bounded check; cache/source manifest binding is verified.']}
    args.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
