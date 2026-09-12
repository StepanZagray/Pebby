"""Matched frozen heads on generated policy-prefix rows; no controller updates."""
import json,os,signal,time
from pathlib import Path
import numpy as np
from tools import probe_world_binary_risk as probe
from pebby.agent.on_policy_provenance import file_digest,validate_on_policy_provenance

SOURCE=Path('data/ls20-world-onpolicy-aggregate2-train.npz')
OUT=Path('artifacts/world-frozen-onpolicy-binary-risk-probe.json')
BASELINE=Path('artifacts/world-frozen-binary-risk-probe.json')


def onpolicy_selection(path,count,seed,split):
    with np.load(path,allow_pickle=False) as archive:
        meta=json.loads(str(archive['meta'].item()));seeds=archive['seeds']
        labels={k:archive[k] for k in ('distances','lost_life','terminal','won')}
    validate_on_policy_provenance({'meta':meta,'seeds':seeds})
    if meta.get('oracle_search')!='complete_only' or meta.get('split')!='train':raise ValueError('invalid complete generated training source')
    marked=np.asarray(meta['on_policy_rows'],dtype=np.int64)
    if len(np.unique(seeds[marked]))!=2048:raise ValueError('expected2048distinct policy training levels')
    risk=probe.unsafe(labels).any(-1);rng=np.random.default_rng(185)
    risk_levels=np.unique(seeds[marked[risk[marked]]]);n=min(1024,len(risk_levels))
    enriched=set(map(int,rng.choice(risk_levels,n,replace=False)))
    selected=np.unique(seeds[marked]);rows=[]
    for s in selected:
        pool=marked[seeds[marked]==s]
        if int(s) in enriched:pool=pool[risk[pool]]
        rows.append(int(rng.choice(pool)))
    rows=np.asarray(rows,np.int64);y=probe.unsafe(labels)[rows]
    if len(rows)!=2048 or len(np.unique(seeds[rows]))!=2048 or not set(rows)<=set(marked):raise ValueError('invalid distinct policy-only selection')
    return rows,{'levels':len(rows),'risk_enriched_levels':n,'available_unsafe_levels':len(risk_levels),'rows':rows.tolist(),'seeds':seeds[rows].tolist(),'branches':int(y.size),'unsafe':int(y.sum()),'safe':int((~y).sum()),'unsafe_prevalence':float(y.mean()),'lost_life':int(labels['lost_life'][rows].sum()),'unreachable_distance':int((labels['distances'][rows]<0).sum()),'expert_rows':0,'selection':'up to1024levels choose unsafe true-policy row; remaining levels choose uniform true-policy row; all2048distinct generatedtrainlevels used'}


def main():
    start=time.monotonic();before=file_digest(BASELINE);old=json.loads(BASELINE.read_text())
    if old['status']!='complete':raise ValueError('ordinary baseline incomplete')
    source_sha=file_digest(SOURCE);selected_original=probe.selection
    def select(path,count,seed,split):
        if split=='train':return onpolicy_selection(path,2048,seed,split)
        rows,info=selected_original(path,count,seed,split)
        if rows.tolist()!=old['selection']['validation']['rows']:raise ValueError('held-out panel changed')
        if set(info['seeds'])&set(old['selection']['train']['seeds']):raise ValueError('historical train/validation overlap')
        return rows,info
    def deadline(*_):raise TimeoutError('on-policy frozen probe300second deadline')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(300)
    probe.TRAIN=SOURCE;probe.OUT=OUT;probe.selection=select
    try:probe.main()
    finally:signal.alarm(0)
    result=json.loads(OUT.read_text())
    if result['status']!='complete':raise ValueError('probe incomplete')
    fatal_seeds=set(np.load('data/world-round2-fatal-choice-histories.npz')['seed'].tolist())
    train_seeds=set(result['selection']['train']['seeds']);val_seeds=set(result['selection']['validation']['seeds'])
    assert not train_seeds&val_seeds and not train_seeds&fatal_seeds
    assert file_digest(BASELINE)==before and file_digest(SOURCE)==source_sha
    comparison={}
    for condition in ('actual','imagined'):
        comparison[condition]={}
        for head in ('linear','mlp64'):
            comparison[condition][head]={split:{'ordinary_source':old['conditions'][condition][head][split], 'onpolicy_source':result['conditions'][condition][head][split]} for split in ('validation','fixed_fatal')}
    result.update(script_sha256=file_digest(__file__),historical_ordinary_baseline={'path':str(BASELINE),'sha256':before,'training':{k:old['selection']['train'][k] for k in ('levels','branches','unsafe','unsafe_prevalence','lost_life')}},historical_baseline_not_rerun=True,matched_validation_rows=True,train_validation_and_fatal_levels_disjoint=True,fatal_distinct_levels=len(fatal_seeds),comparison=comparison,
        collection_policy='true policy prefixes only; expert rows excluded',total_elapsed_seconds=time.monotonic()-start,
        process_cleanup={'probe_weights_saved':False,'controller_checkpoint_updated':False,'status':'process exiting'})
    result['limitations'] += ['Training size changes4096→2048levels, and unsafe prevalence also changes; differences cannot be attributed solely to on-policy distribution.','Ordinary held-out1024means the exact previous ordinary-bank risk-enriched panel, not a natural-prevalence sample.','Fatal21histories are diagnostic-only and never used for normalization, fitting, selection, or threshold tuning.','These remain state-only latent risk heads, not an action-conditioned controller architecture.']
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('COMPLETE',json.dumps({'pid':os.getpid(),'elapsed_seconds':result['total_elapsed_seconds'],'train':{k:result['selection']['train'][k] for k in ('levels','unsafe','unsafe_prevalence','lost_life')},'comparison':comparison}),flush=True)
if __name__=='__main__':main()
