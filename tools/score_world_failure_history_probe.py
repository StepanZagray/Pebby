"""Score checkpoints on immutable, generated validation behavior histories only.

No rollout, teacher query, relabeling, optimizer, or training-data loader is used.
CE matches training: cross-entropy against a uniform distribution over optimal
moves, not negative log probability of the entire optimal action set.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F

from pebby.agent.world_model import load_world_checkpoint
from tools.goal_attribute_probes import digest

FORMAT = 'pebby.ls20-validation-history-probe.v1'
FIELDS = ('frames', 'history_valid', 'previous_actions', 'optimal', 'seed', 'step',
          'distance', 'actor_action', 'actor_logits', 'category_bits')


def validate_probe(data):
    meta = data['meta']
    if meta.get('format') != FORMAT or meta.get('split') != 'validation' or meta.get('source') != 'generated_only':
        raise ValueError('requires generated validation history probe, never world training data')
    if 'next_frames' in data:
        raise ValueError('world training transition arrays are not a history probe')
    n = len(data['optimal'])
    shapes = {'frames': (n,8,64,64), 'history_valid': (n,8), 'previous_actions': (n,8),
              'optimal': (n,), 'seed': (n,), 'step': (n,), 'distance': (n,),
              'actor_action': (n,), 'actor_logits': (n,4), 'category_bits': (n,)}
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f'{key} has wrong probe shape')
    for key in ('optimal','seed','step','distance','actor_action','category_bits','previous_actions','frames'):
        if not np.issubdtype(data[key].dtype,np.integer):
            raise ValueError(f'{key} must have integer dtype')
    if data['history_valid'].dtype != np.bool_:
        raise ValueError('history_valid must be boolean')
    if not n or np.any((data['optimal'] < 0) | (data['optimal'] > 15)):
        raise ValueError('invalid probe optimal masks')
    if np.any((data['seed'] < 1_000_000) | (data['seed'] >= 2_000_000)):
        raise ValueError('non-validation seed in probe')
    if np.any((data['actor_action'] < 0) | (data['actor_action'] > 3)) or not np.isfinite(data['actor_logits']).all():
        raise ValueError('invalid actor scores or actions')
    if not np.array_equal(data['actor_logits'].argmax(1),data['actor_action']):
        raise ValueError('stored actor actions disagree with logits')
    categories = meta.get('categories',[])
    if not 1 <= len(categories) <= 8 or len(set(categories)) != len(categories):
        raise ValueError('invalid category schema')
    if np.any(data['category_bits'] == 0) or np.any(data['category_bits'].astype(np.int64) >= 1 << len(categories)):
        raise ValueError('invalid category bits')
    levels = {int(level['seed']):level for level in meta['levels']}
    if set(levels) != set(map(int,data['seed'])):
        raise ValueError('level provenance does not cover probe seeds')
    for seed in levels:
        if np.sum(data['seed']==seed)>8:
            raise ValueError('more than eight histories per level')
    for category_index,category in enumerate(categories):
        expected=[]
        for seed,level in levels.items():
            event=level['events'][category]
            index=event['row_index']
            if index is None:
                if event['status']!='not_observed':
                    raise ValueError('missing row for observed event')
                continue
            if not isinstance(index,int) or not 0<=index<n or int(data['seed'][index])!=seed:
                raise ValueError('event row index does not match level')
            if event['status'] not in ('valid','unreachable','unsupported'):
                raise ValueError('unknown teacher label status')
            if (int(data['optimal'][index])>0)!=(event['status']=='valid'):
                raise ValueError('zero mask/status contradiction')
            if int(data['optimal'][index])!=event['optimal_mask']:
                raise ValueError('event optimal mask contradicts stored mask')
            expected.append(index)
        actual=np.flatnonzero(data['category_bits'].astype(int)&(1<<category_index))
        if sorted(expected)!=actual.tolist():
            raise ValueError('category bits disagree with event records')
    return data


def load_probe(path):
    with np.load(path,allow_pickle=False) as archive:
        if 'next_frames' in archive.files:
            raise ValueError('world training arrays are not a probe')
        data={name:archive[name] for name in FIELDS}
        data['meta']=json.loads(str(archive['meta'].item()))
    return validate_probe(data)


def predict(policy,data,batch_size):
    if not 1<=batch_size<=32:
        raise ValueError('CPU batch must be 1..32')
    if policy.config().get('history')!=8:
        raise ValueError('checkpoint must use public H8 inputs')
    output=[]
    with torch.inference_mode():
        for start in range(0,len(data['optimal']),batch_size):
            sl=slice(start,start+batch_size)
            value=policy(torch.from_numpy(data['frames'][sl]).long(),
                         history_valid=torch.from_numpy(data['history_valid'][sl]),
                         previous_actions=torch.from_numpy(data['previous_actions'][sl])).float()
            if value.shape!=(len(data['frames'][sl]),4) or not torch.isfinite(value).all():
                raise ValueError('checkpoint produced invalid logits')
            output.append(value.cpu())
    return torch.cat(output)


def metrics(logits,masks,actor_actions):
    """Zero masks have no target and never contribute errors or CE."""
    masks=torch.as_tensor(masks,dtype=torch.long)
    valid=masks>0
    result={'rows':len(masks),'valid_rows':int(valid.sum()),'excluded_zero_masks':int((~valid).sum())}
    if not valid.any():
        return dict(result,optimal_set_accuracy=None,uniform_optimal_target_ce=None,corrected_vs_actor=0,
                    regressed_vs_actor=0,actor_optimal_set_accuracy=None,mean_top1_confidence=None,
                    mean_optimal_set_probability=None)
    selected=logits[valid]; chosen=selected.argmax(-1)
    bits=(masks[valid,None]&(1<<torch.arange(4)))!=0
    correct=bits.gather(1,chosen[:,None]).squeeze(1)
    actor=torch.as_tensor(actor_actions,dtype=torch.long)[valid]
    actor_correct=bits.gather(1,actor[:,None]).squeeze(1)
    target=bits.float()/bits.sum(-1,keepdim=True)
    probs=selected.softmax(-1)
    result.update(optimal_set_accuracy=float(correct.float().mean()),
                  uniform_optimal_target_ce=float(-(target*F.log_softmax(selected,dim=-1)).sum(-1).mean()),
                  corrected_vs_actor=int((correct&~actor_correct).sum()),
                  regressed_vs_actor=int((~correct&actor_correct).sum()),
                  actor_optimal_set_accuracy=float(actor_correct.float().mean()),
                  mean_top1_confidence=float(probs.max(-1).values.mean()),
                  mean_optimal_set_probability=float((probs*bits).sum(-1).mean()))
    return result


def summarize(data,logits):
    levels=data['meta']['levels']; summaries={}
    for fog_name,fog in (('all',None),('fog',True),('nonfog',False)):
        group=[level for level in levels if fog is None or bool(level['fog'])==fog]
        result={}
        for category in data['meta']['categories']:
            events=[level['events'][category] for level in group]
            indices=[event['row_index'] for event in events if event['row_index'] is not None]
            index=np.asarray(indices,dtype=np.int64)
            result[category]=metrics(logits[index],data['optimal'][index],data['actor_action'][index])
            result[category]['category_status_counts']=dict(Counter(e['status'] for e in events))
            result[category]['levels_in_group']=len(group)
        summaries[fog_name]=result
    return summaries


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,default=Path('data/ls20-world-query8-validation-history-probe.npz'))
    p.add_argument('--checkpoint',type=Path,action='append',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--batch-size',type=int,choices=range(1,33),default=32)
    args=p.parse_args(argv)
    if args.out.exists():
        raise ValueError('refusing score report overwrite')
    started=time.monotonic();torch.set_num_threads(1)
    data=load_probe(args.probe);meta=data['meta']
    reference=Path(meta['behavior_checkpoint']['path'])
    sources={str(args.probe):digest(args.probe),meta['bank']:meta['bank_sha256'],
             str(reference):meta['behavior_checkpoint']['sha256'],
             __file__:digest(__file__), 'pebby/agent/world_model.py':digest('pebby/agent/world_model.py')}
    for path,expected in sources.items():
        if digest(path)!=expected:
            raise ValueError(f'source hash mismatch: {path}')
    report={'format':'pebby.ls20-fixed-history-scores.v1','status':'running','split':'validation',
            'source':'generated_only','pid':os.getpid(),'device':'cpu','torch_threads':1,
            'probe':str(args.probe),'source_hashes':sources,'models':[],
            'no_rollout_or_relabeling':True,
            'selection_caveat':'First-wrong histories are selected because the behavior actor was wrong; its zero accuracy there is by construction.',
            'distribution_caveat':'These are fixed old-behavior histories, not the evaluated checkpoint own on-policy distribution.',
            'ce_definition':'Cross entropy against uniform optimal-action target, matching world training; not negative log optimal-set probability.'}
    def persist():
        report['elapsed_seconds']=time.monotonic()-started
        report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        args.out.parent.mkdir(parents=True,exist_ok=True)
        temp=args.out.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.out)
    print('PID',os.getpid(),flush=True);persist()
    try:
        paths=list(dict.fromkeys([reference,*args.checkpoint]))
        for path in paths:
            checkpoint_hash=digest(path)
            policy,checkpoint=load_world_checkpoint(path,'cpu');policy.requires_grad_(False)
            logits=predict(policy,data,args.batch_size)
            reference_model=checkpoint_hash==meta['behavior_checkpoint']['sha256']
            fallback=False
            if reference_model and not np.array_equal(logits.argmax(-1).numpy(),data['actor_action']):
                logits=predict(policy,data,1);fallback=True
                if not np.array_equal(logits.argmax(-1).numpy(),data['actor_action']):
                    raise ValueError('reference checkpoint no longer reproduces stored behavior actions, including batch1')
            result={'checkpoint':str(path),'sha256':checkpoint_hash,'parameters':policy.parameter_count(),
                    'config':policy.config(),'epoch':checkpoint.get('epoch'),
                    'batch_size':1 if fallback else args.batch_size,'batch1_reference_fallback':fallback,
                    'reference_model':reference_model,'reference_actions_match':True if reference_model else None,
                    'max_abs_logit_difference_from_stored_actor':float((logits-torch.from_numpy(data['actor_logits'])).abs().max()) if reference_model else None,
                    'summaries':summarize(data,logits)}
            if digest(path)!=checkpoint_hash:
                raise ValueError('checkpoint changed during scoring')
            result['checkpoint_unchanged']=True
            report['models'].append(result);print(str(path),'scored',flush=True);persist()
        for path,expected in sources.items():
            if digest(path)!=expected:
                raise ValueError(f'source changed during scoring: {path}')
        report.update(status='complete',sources_unchanged=True)
    except Exception as error:
        report.update(status='failed',error=repr(error));raise
    finally: persist()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
