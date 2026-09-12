"""Frozen generated held-out fatal-choice diagnostic; teacher never selects actions."""
import argparse
from collections import Counter
import importlib.util
import json,os,resource,time
from pathlib import Path
import numpy as np
import torch
from pebby.agent import world_data as wd
from pebby.agent.on_policy_provenance import file_digest
from pebby.ls20 import names
from pebby.ls20.generate import FORMAT,GENERATOR_VERSION
from pebby.ls20.plan import simulate
from tools.validate_extended_collector import checked_expansion


def value_report(model,z):
    distance,terminal,won=model.value(z)
    probs=distance.softmax(-1)
    return {'expected_distance':(probs*model.bin_values).sum(-1).tolist(),
            'unreachable_probability':probs[...,-1].tolist(),
            'terminal_probability':terminal.sigmoid().tolist(),'won_probability':won.sigmoid().tolist()}


def event_report(model,encoding,logits,extra,targets,frames,actions,choice):
    direct=extra['direct'][0];query=extra.get('query',torch.zeros_like(extra['direct']))[0]
    ranker=model.ranker(extra['features']).squeeze(-1)[0]
    torch.testing.assert_close(logits[0],direct+query+ranker)
    actual=[];valid=[];previous=[]
    for a in range(4):
        history=[targets['next_frames'][a]] if targets['lost_life'][a] else frames+[targets['next_frames'][a]]
        acts=[-1] if targets['lost_life'][a] else actions+[a]
        f,v,p=wd.history_arrays(history,acts,8);actual.append(f);valid.append(v);previous.append(p)
    observed=model.encode(torch.from_numpy(np.stack(actual)).long(),torch.from_numpy(np.stack(valid)),torch.from_numpy(np.stack(previous)))
    safe=(targets['distances']>=0)&~targets['lost_life']&(~targets['terminal']|targets['won'])
    best_safe=int(np.argmax(np.where(safe,logits[0].numpy(),-np.inf))) if safe.any() else None
    margins={key:float(value[choice]-value[best_safe]) for key,value in [('direct',direct),('query',query),('ranker',ranker),('total',logits[0])]} if best_safe is not None else None
    return {'actor_action':choice,'actor_confidence':float(logits[0].softmax(-1)[choice]),'optimal_mask':int(targets['optimal']),
            'logits':logits[0].tolist(),'direct':direct.tolist(),'query':query.tolist(),'ranker':ranker.tolist(),
            'true_distances':targets['distances'].tolist(),'true_reachable':(targets['distances']>=0).tolist(),
            'true_lost_life':targets['lost_life'].tolist(),'true_terminal':targets['terminal'].tolist(),'true_won':targets['won'].tolist(),
            'safe_actions':safe.tolist(),'highest_policy_safe_action':best_safe,'chosen_minus_best_safe':margins,
            'ranker_argmax_safe':bool(safe[int(ranker.argmax())]),'direct_argmax_safe':bool(safe[int(direct.argmax())]),
            'direct_query_argmax_safe':bool(safe[int((direct+query).argmax())]),
            'predicted_value':value_report(model,extra['successors'][0]),'actual_successor_value':value_report(model,observed['latent']),
            'predicted_actual_latent_mse':(extra['successors'][0]-observed['latent']).square().mean(-1).tolist(),
            'actual_value_encoding_limit':'Diagnostic public successor histories; terminal frames encoded for comparison only; not fed to actor.'}


def probe(spec,model,deadline,max_actions=120):
    env,oracle,proof=wd.verified_context(spec,int(spec.get('training_context_index',spec['seed']%7)),search_limit=600000)
    report={'seed':spec['seed'],'difficulty':spec['difficulty'],'fog':bool(spec.get('fog')),'proof':proof,'events':{},'stats':{}}
    if env is None:
        report['ending']='unsupported';return report,[]
    frames,actions=[env.render()],[-1];counts=Counter();saved=[];cache={};ending='capped'
    with torch.inference_mode():
        for step in range(max_actions):
            if time.monotonic()>deadline:raise TimeoutError('fatal diagnostic deadline exceeded')
            f,v,p=wd.history_arrays(frames,actions,8);key=wd.history_key(f,v,p)
            if key not in cache:
                enc=model.encode(torch.from_numpy(f[None]).long(),torch.from_numpy(v[None]),torch.from_numpy(p[None]))
                logits,extra=model.logits_from(enc);cache[key]=(enc,logits,extra);counts['policy_forwards']+=1
            enc,logits,extra=cache[key];choice=int(logits[0].argmax()) # policy BEFORE teacher
            state=oracle.state_of(env);distance=oracle.distance_for(state)
            predicted,outcome=simulate(oracle.layout,state,choice,oracle.refills)
            after_distance=0 if outcome=='won' else None if outcome=='died' else oracle.distance_for(predicted)
            categories=[]
            if distance is not None and outcome not in ('won','died') and after_distance is None and 'first_reachable_to_unreachable' not in report['events']:
                categories.append('first_reachable_to_unreachable')
            if distance is not None and outcome=='died' and 'first_reachable_life_loss' not in report['events']:
                categories.append('first_reachable_life_loss')
            if categories:
                targets,_,_,_=checked_expansion(wd._expand,counts,env,oracle,distance,spec['seed'],step)
                event=event_report(model,enc,logits,extra,targets,frames,actions,choice)
                event.update(step=step,current_distance=distance,lives=env.lives(),history_row=len(saved))
                saved.append({'frames':f,'history_valid':v,'previous_actions':p,'seed':np.int32(spec['seed']),'step':np.int16(step)})
                for category in categories:report['events'][category]=event
            old_lives=env.lives();result=env.perform(names.ACTION_IDS[choice])
            if outcome=='won':agrees=result.won and env.lives()==old_lives
            elif outcome=='died':agrees=not result.won and env.lives()==old_lives-1 and (result.finished or oracle.state_of(env)==oracle.start)
            else:agrees=not result.finished and env.lives()==old_lives and oracle.state_of(env)==predicted
            if not agrees:raise ValueError(f'engine/teacher mismatch seed {spec["seed"]} step {step}')
            counts['checked_actor_transitions']+=1;counts['reachable_actor_states']+=distance is not None
            lost=env.lives()<old_lives;counts['life_losses']+=lost
            if result.finished:ending='won' if result.won else 'game_over';break
            if lost:frames,actions=[result.frame],[-1]
            else:frames,actions=(frames+[result.frame])[-8:],(actions+[choice])[-8:]
    report.update(ending=ending,stats=dict(counts));return report,saved


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path('checkpoints/ls20-world-onpolicy-round2-b1024.epoch1.pt'))
    p.add_argument('--snapshot',type=Path,default=Path('artifacts/world_model_fatal_snapshot.py'))
    p.add_argument('--bank',type=Path,default=Path('data/ls20-verified-validation-monitor.jsonl'))
    p.add_argument('--report',type=Path,default=Path('artifacts/world-round2-fatal-choices.json'))
    p.add_argument('--histories',type=Path,default=Path('data/world-round2-fatal-choice-histories.npz'))
    args=p.parse_args();print('PID',os.getpid(),flush=True);torch.set_num_threads(1);start=time.monotonic()
    if args.report.exists() or args.histories.exists():raise ValueError('refusing overwrite')
    paths=[args.checkpoint,args.snapshot,args.bank,Path(__file__),Path(wd.__file__)]
    hashes={str(path):file_digest(path) for path in paths}
    spec=importlib.util.spec_from_file_location('pebby.agent._fatal_frozen',args.snapshot);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model,_=module.load_world_checkpoint(args.checkpoint);model.requires_grad_(False)
    levels=[json.loads(line) for line in args.bank.read_text().splitlines()]
    if len(levels)!=100 or len({s['seed'] for s in levels})!=100 or any(s.get('format')!=FORMAT or s.get('generator_version')!=GENERATOR_VERSION or not 1_000_000<=s['seed']<2_000_000 for s in levels):raise ValueError('invalid generated validation monitor')
    chosen=[s for d in range(1,6) for s in sorted((s for s in levels if s['difficulty']==d),key=lambda s:s['seed'])[:4]]
    if len(chosen)!=20:raise ValueError('insufficient stratified monitor levels')
    report={'status':'running','pid':os.getpid(),'device':'cpu','torch_threads':1,'source':'generated_only','split':'validation','source_hashes':hashes,'selection':'first four seeds per difficulty; fixed before outcomes','oracle_actions_in_policy':0,'levels':[],
            'limits':['20 stratified generated levels only; not the full100 population.','Reachability is complete single-life task-state reachability, not a learned optimal multi-life policy.','Life reset can return to reachable start; recorded separately from reachable-to-unreachable choices.','Value head has distance/unreachable/terminal/win outputs but no explicitly supervised lost-life probability.','Component preferences are diagnostic comparisons, not causal ablation rollouts.']}
    rows=[]
    def persist():
        report.update(elapsed_seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        for level in chosen:
            result,samples=probe(level,model,start+600)
            for event in result['events'].values():event['history_row']+=len(rows)
            rows+=samples;report['levels'].append(result);persist();print(len(report['levels']),level['seed'],result['ending'],list(result['events']),flush=True)
        summary={}
        for category in ('first_reachable_to_unreachable','first_reachable_life_loss'):
            events=[r['events'][category] for r in report['levels'] if category in r['events']]
            summary[category]={'count':len(events),'not_observed_or_unsupported':20-len(events),'ranker_argmax_safe':sum(e['ranker_argmax_safe'] for e in events),'direct_argmax_safe':sum(e['direct_argmax_safe'] for e in events),'direct_query_argmax_safe':sum(e['direct_query_argmax_safe'] for e in events),'ranker_penalizes_choice_vs_best_safe':sum(e['chosen_minus_best_safe']['ranker']<0 for e in events if e['chosen_minus_best_safe'] is not None),'by_difficulty':dict(Counter(r['difficulty'] for r in report['levels'] if category in r['events'])),'by_fog':dict(Counter(str(r['fog']) for r in report['levels'] if category in r['events']))}
        if any(file_digest(path)!=sha for path,sha in hashes.items()):raise ValueError('source changed')
        metadata={'format':'pebby.ls20-fatal-diagnostic-histories.v1','split':'validation','source':'generated_only','source_hashes':hashes}
        np.savez_compressed(args.histories,**{k:np.stack([r[k] for r in rows]) for k in rows[0]} if rows else {},meta=np.array(json.dumps(metadata)))
        report.update(status='complete',summary=summary,ending_counts=dict(Counter(r['ending'] for r in report['levels'])),histories_path=str(args.histories),histories_sha256=file_digest(args.histories),source_hashes_unchanged=True)
    except Exception as error:report.update(status='failed',error=repr(error));raise
    finally:persist()
if __name__=='__main__':main()
