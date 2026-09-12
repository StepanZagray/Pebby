"""Score frozen policies on verified generated fatal histories; no controller training.

First invocation replays the original public actor to cache exact actual branches.
Subsequent invocations use only the source-bound cache, without engine or oracle.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import time
from unittest.mock import patch

import numpy as np
import torch

from pebby.agent.on_policy_provenance import file_digest
from tools import probe_world_fatal_choices as fatal
from tools.probe_world_binary_risk import features, metrics

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'artifacts/world-round2-fatal-choices.json'
HISTORIES=ROOT/'data/world-round2-fatal-choice-histories.npz'
BANK=ROOT/'data/ls20-verified-validation-monitor.jsonl'
BEHAVIOR=ROOT/'checkpoints/ls20-world-onpolicy-round2-b1024.epoch1.pt'
SNAPSHOT=ROOT/'artifacts/world_model_fatal_snapshot.py'
CACHE=ROOT/'data/world-fixed-fatal-branches.npz'
FORMAT='pebby.ls20-fixed-fatal-branches.v1'
TARGETS=('next_frames','distances','terminal','won','lost_life','next_optimal')


def load_model(checkpoint):
    spec=importlib.util.spec_from_file_location('pebby.agent._fixed_fatal_scoring',SNAPSHOT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model,meta=module.load_world_checkpoint(checkpoint,'cpu');model.eval();model.requires_grad_(False)
    return model,meta


def reference_records(reference):
    records={}
    for level in reference['levels']:
        for category,event in level['events'].items():
            row=event['history_row']
            if row not in records:records[row]={'seed':level['seed'],'event':event,'categories':[]}
            records[row]['categories'].append(category)
    if sorted(records)!=list(range(len(records))):raise ValueError('reference history indices not contiguous')
    return [records[i] for i in range(len(records))]


def verify_reference_alignment(data,reference,saved):
    records=reference_records(reference)
    if len(data['frames'])!=len(records):raise ValueError('fixed cache event count mismatch')
    for key in ('frames','history_valid','previous_actions','seed','step'):
        if not np.array_equal(data[key],saved[key]):raise ValueError('fixed public history differs from saved reference')
    for i,record in enumerate(records):
        event=record['event']
        if int(data['seed'][i])!=record['seed'] or int(data['step'][i])!=event['step'] or int(data['reference_action'][i])!=event['actor_action']:
            raise ValueError('reference seed/step/action alignment mismatch')
        for target,key in [('distances','true_distances'),('lost_life','true_lost_life'),('terminal','true_terminal'),('won','true_won')]:
            if not np.array_equal(data[target][i],event[key]):raise ValueError('reference branch label mismatch')
        expected='first_reachable_to_unreachable' in record['categories']
        if bool(data['first_unreachable'][i])!=expected:raise ValueError('reference fatal category mismatch')
    return records


def cache_sources():
    return {str(p):file_digest(p) for p in (REFERENCE,HISTORIES,BANK,BEHAVIOR,SNAPSHOT,Path(fatal.__file__),Path(fatal.wd.__file__))}


def verify_cache_digest(path, receipt):
    if receipt.get('cache_sha256') != file_digest(path):
        raise ValueError('fixed branch cache checksum mismatch')


def ensure_cache(path,deadline):
    before=cache_sources();reference=json.loads(REFERENCE.read_text())
    for name,sha in reference['source_hashes'].items():
        if file_digest(name)!=sha:raise ValueError(f'original fatal source changed: {name}')
    with np.load(HISTORIES,allow_pickle=False) as archive:saved={key:archive[key] for key in ('frames','history_valid','previous_actions','seed','step')}
    replayed=False
    if path.exists():
        receipt=json.loads(path.with_suffix('.proof.json').read_text())
        verify_cache_digest(path,receipt)
        if receipt.get('source_hashes')!=before:raise ValueError('cache receipt source binding mismatch')
        with np.load(path,allow_pickle=False) as archive:
            data={k:archive[k] for k in archive.files if k!='meta'};meta=json.loads(str(archive['meta'].item()))
        if meta.get('format')!=FORMAT or meta.get('source')!='generated_only' or meta.get('split')!='validation' or meta.get('source_hashes')!=before:
            raise ValueError('fixed branch cache source binding mismatch')
        if meta.get('exact_replay_verified') is not True:raise ValueError('cache lacks exact replay proof')
    else:
        model,_=load_model(BEHAVIOR)
        specs={s['seed']:s for s in map(json.loads,BANK.read_text().splitlines())}
        if any(not 1_000_000<=s<2_000_000 for s in specs):raise ValueError('only generated validation seeds permitted')
        captured=[];histories=[];original=fatal.event_report
        def capture(model,encoding,logits,extra,targets,frames,actions,choice):
            captured.append({k:np.array(targets[k],copy=True) for k in TARGETS})
            return original(model,encoding,logits,extra,targets,frames,actions,choice)
        with patch.object(fatal,'event_report',side_effect=capture):
            for previous in reference['levels']:
                replay,observed=fatal.probe(specs[previous['seed']],model,deadline)
                if replay['ending']!=previous['ending'] or replay['stats']['checked_actor_transitions']!=previous['stats']['checked_actor_transitions'] or set(replay['events'])!=set(previous['events']):
                    raise ValueError('original public actor replay diverged')
                for category,event in replay['events'].items():
                    for key in ('step','actor_action','true_distances','true_lost_life','true_terminal','true_won'):
                        if event[key]!=previous['events'][category][key]:raise ValueError('replayed event labels/actions diverged')
                histories.extend(observed)
        data={key:np.stack([r[key] for r in histories]) for key in histories[0]}
        data.update({key:np.stack([r[key] for r in captured]) for key in TARGETS})
        records=reference_records(reference)
        data['reference_action']=np.array([r['event']['actor_action'] for r in records],dtype=np.int64)
        data['first_unreachable']=np.array(['first_reachable_to_unreachable' in r['categories'] for r in records],dtype=bool)
        meta={'format':FORMAT,'source':'generated_only','split':'validation','source_hashes':before,'exact_replay_verified':True,'reference_actions_from':'original round2epoch1 public actor, batch1 greedy; never selected using teacher','histories':len(records),'branches':4*len(records)}
        verify_reference_alignment(data,reference,saved)
        if cache_sources()!=before:raise ValueError('cache source changed during replay')
        temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,**data,meta=np.array(json.dumps(meta)))
        try:os.link(temporary,path)
        finally:temporary.unlink(missing_ok=True)
        path.with_suffix('.proof.json').write_text(json.dumps({'cache_sha256':file_digest(path),'source_hashes':before,'exact_replay_verified':True},indent=2)+'\n')
        replayed=True
    verify_reference_alignment(data,reference,saved)
    if cache_sources()!=before:raise ValueError('fixed cache sources changed')
    return data,meta,replayed


def score(model,data):
    with torch.inference_mode():
        # Current policy sees public current history only; actual branches enter
        # the separate diagnostic feature call AFTER logits/action are computed.
        outputs=[];extras=[]
        for row in range(len(data['frames'])):
            encoding=model.encode(torch.from_numpy(data['frames'][row:row+1]).long(),torch.from_numpy(data['history_valid'][row:row+1]),torch.from_numpy(data['previous_actions'][row:row+1]))
            current,parts=model.logits_from(encoding);outputs.append(current);extras.append(parts)
        logits=torch.cat(outputs)
        extra={key:torch.cat([parts[key] for parts in extras]) for key in extras[0]}
        choices=logits.argmax(-1).numpy()
        zs,probabilities=features(model,data)
    unreachable=data['distances']<0;lost=data['lost_life'];terminal_loss=data['terminal']&~data['won']
    safe=~(unreachable|lost|terminal_loss);idx=np.arange(len(choices))
    component_logits={'direct':extra['direct'],'ranker':model.ranker(extra['features']).squeeze(-1),'total':logits}
    if 'query' in extra:
        component_logits.update(query=extra['query'],direct_plus_query=extra['direct']+extra['query'])
    components={k:{'actions':v.argmax(-1).tolist(),'safe_choices':int(safe[idx,v.argmax(-1).numpy()].sum()),'safe_choice_fraction':float(safe[idx,v.argmax(-1).numpy()].mean())} for k,v in component_logits.items()}
    risk={kind:metrics(unreachable,probabilities[kind].numpy()) for kind in probabilities}
    records=[]
    for i in np.flatnonzero(data['first_unreachable']):
        best=int(np.argmax(np.where(safe[i],logits[i].numpy(),-np.inf))) if safe[i].any() else None
        row={'history_row':int(i),'seed':int(data['seed'][i]),'step':int(data['step'][i]),'current_action':int(choices[i]),'reference_action':int(data['reference_action'][i]),'best_safe_policy_action':best,'current_choice_safe':bool(safe[i,choices[i]]),'safe_actions':safe[i].tolist(),'probability_margins':{}}
        for kind,prob in probabilities.items():
            p=prob.numpy().reshape(-1,4)[i]
            row['probability_margins'][kind]={'current_minus_best_safe':float(p[choices[i]]-p[best]) if best is not None else None,'reference_minus_best_safe':float(p[data['reference_action'][i]]-p[best]) if best is not None else None,'current':float(p[choices[i]]),'reference':float(p[data['reference_action'][i]]),'best_safe':float(p[best]) if best is not None else None}
        records.append(row)
    margins={kind:{key:float(np.mean([r['probability_margins'][kind][key] for r in records if r['probability_margins'][kind][key] is not None])) for key in ('current_minus_best_safe','reference_minus_best_safe')} for kind in probabilities}
    return {'histories':len(choices),'branches':int(unreachable.size),'policy_safe_choices':int(safe[idx,choices].sum()),'policy_safe_choice_fraction':float(safe[idx,choices].mean()),'first_unreachable_policy_safe_choices':sum(r['current_choice_safe'] for r in records),'first_unreachable_histories':len(records),'first_unreachable_mean_probability_margins':margins,'unreachable_probability_metrics':risk,'label_counts':{'unreachable':int(unreachable.sum()),'lost_life':int(lost.sum()),'terminal_loss':int(terminal_loss.sum()),'unreachable_and_lost_life':int((unreachable&lost).sum()),'unsafe_union':int((~safe).sum())},'policy_choice_labels':{'unreachable':int(unreachable[idx,choices].sum()),'lost_life':int(lost[idx,choices].sum()),'terminal_loss':int(terminal_loss[idx,choices].sum())},'lost_life_prediction_metrics':None,'lost_life_note':'No explicit learned lost-life head; distance-unreachable probabilities are not mislabeled as life-loss probabilities.','component_preferences':components,'first_unreachable_details':records,'current_logits':logits.tolist()}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cache',type=Path,default=CACHE);args=p.parse_args(argv)
    torch.set_num_threads(1);start=time.monotonic();pid=os.getpid();print('PID',pid,flush=True)
    def timeout(*_):raise TimeoutError('fixed fatal scorer300second deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
    if args.out.exists():raise FileExistsError(args.out)
    dependencies=[ROOT/'pebby/agent/world_readout.py',ROOT/'pebby/agent/glyph_model.py',ROOT/'pebby/agent/world_grounding.py',ROOT/'third_party/ls20/ls20.py',Path(fatal.__file__),Path(fatal.wd.__file__),ROOT/'tools/probe_world_binary_risk.py',Path(__file__)]
    dependency_hashes={str(path):file_digest(path) for path in dependencies}
    cp_sha=file_digest(args.checkpoint);data,meta,replayed=ensure_cache(args.cache,start+290)
    model,cpmeta=load_model(args.checkpoint);score_start=time.monotonic();result=score(model,data)
    if file_digest(args.checkpoint)!=cp_sha:raise ValueError('checkpoint changed')
    if any(file_digest(path)!=sha for path,sha in dependency_hashes.items()):raise ValueError('scorer dependency changed during execution')
    result.update(current_policy_batch_size=1,diagnostic_feature_batch_size=len(data['frames']),dependency_hashes_at_import=dependency_hashes,dependencies_unchanged=True)
    result.update(status='complete',checkpoint=str(args.checkpoint),checkpoint_sha256=cp_sha,checkpoint_epoch=cpmeta.get('epoch',cpmeta.get('best_epoch')),parameters=model.parameter_count(),cache=str(args.cache),cache_sha256=file_digest(args.cache),cache_built_this_call=replayed,cache_proof=meta,model_source_sha256=file_digest(SNAPSHOT),scorer_sha256=file_digest(__file__),feature_helper_sha256=file_digest(ROOT/'tools/probe_world_binary_risk.py'),device='cpu',cpu_threads=1,pid=pid,scoring_seconds=time.monotonic()-score_start,total_seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,oracle_actions_in_policy=0,scoring_engine_or_oracle_calls=0,limitations=['Fixed histories were produced by original round2epoch1 behavior, not the scored policy; safe-choice rates are not rollout completion.','84branches from21histories on19levels are correlated and risk-enriched; no population calibration or significance claim.','Actual successor encodings are diagnostic only and never fed to the current actor.','Unreachable means no completion in current life/budget; life-loss event is separately counted.','No threshold tuning or checkpoint/model training.'])
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');signal.alarm(0)
    print(json.dumps({k:result[k] for k in ('status','policy_safe_choices','first_unreachable_policy_safe_choices','label_counts','unreachable_probability_metrics','cache_built_this_call','scoring_seconds','total_seconds')}),flush=True)
if __name__=='__main__':main()
