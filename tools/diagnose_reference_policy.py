"""TRAIN-only public-H8 greedy diagnosis; labels never choose actions.

Complete native contextual Oracle is required, including on unreachable states.
No optimizer, feature cache, generated level, or official input is produced.
"""
import argparse
import gc
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from unittest.mock import patch

import numpy as np
import torch
from pebby.agent import world_data as wd
from pebby.agent.world_model import load_world_checkpoint
from pebby.ls20.bank import load
from pebby.ls20.plan import Oracle
from pebby.ls20.reference_profiles import DIFFICULTY_VERSION, profile_errors, SEARCH_LIMITS
from tools.validate_extended_collector import checked_expansion

GIB=1024**3


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def memory_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable unavailable')


def check_memory(available):
    if available < 9*GIB:
        raise MemoryError('Need6GiB reserve plus3GiB planner headroom; search budget will not be reduced')


def select_levels(specs,count=14,seed=20260912):
    if type(count) is not int or not 7<=count<=140 or count%7:
        raise ValueError('levels must be a multiple of seven in7..140')
    if len({s['seed'] for s in specs})!=len(specs):raise ValueError('duplicate TRAIN seeds')
    for spec in specs:
        if spec.get('split')!='train' or spec.get('difficulty_version')!=DIFFICULTY_VERSION or profile_errors(spec):
            raise ValueError('verified seven-reference TRAIN bank required')
    rng=np.random.default_rng(seed);selected=[]
    for tier in range(1,8):
        group=sorted((s for s in specs if s['difficulty']==tier),key=lambda s:s['seed'])
        if len(group)<count//7:raise ValueError('insufficient distinct levels in a tier')
        selected.extend(group[i] for i in rng.choice(len(group),count//7,replace=False))
    return selected


def public_choice(policy,observed,valid,previous):
    with torch.inference_mode():
        logits=policy(torch.from_numpy(observed[None]).long(),
            history_valid=torch.from_numpy(valid[None]),previous_actions=torch.from_numpy(previous[None]))
        if logits.shape!=(1,4) or not bool(torch.isfinite(logits).all()):raise ValueError('invalid public policy logits')
        return int(logits[0].argmax()),logits[0].float().softmax(-1).tolist()


def diagnose_level(spec,policy,*,max_actions=300,on_step=None):
    if policy.config().get('history')!=8 or policy.config().get('architecture')!='world':
        raise ValueError('WorldPolicy public H8 required')
    if type(max_actions) is not int or not 1<=max_actions<=300:raise ValueError('action cap must be1..300')
    if spec.get('split')!='train' or spec.get('difficulty_version')!=DIFFICULTY_VERSION:
        raise ValueError('generated reference TRAIN level required')
    check_memory(memory_available())
    # Force native backend before search, so unavailable native storage cannot
    # silently fall back to an unbounded-memory Python distance map.
    def native(*args,**kwargs):return Oracle(*args,**kwargs,engine='fast')
    with patch.object(wd,'Oracle',native):
        initial,oracle,proof=wd.verified_context(spec,search_limit=SEARCH_LIMITS[spec['difficulty']-1])
    if initial is None or oracle is None or oracle.truncated or oracle.engine!='fast':
        raise ValueError(f'complete native contextual teacher unavailable: {proof}')
    env=initial;frames=[env.render()];actions=[-1];checks=Counter();counts=Counter()
    output=dict(seed=spec['seed'],difficulty=spec['difficulty'],fog=spec['fog'],proof=proof,
                steps=[],first_nonoptimal_decision=None,ending='capped')
    if hasattr(policy,'eval'):policy.eval()
    for step in range(max_actions):
        observed,valid,previous=wd.history_arrays(frames,actions,8)
        choice,probabilities=public_choice(policy,observed,valid,previous)
        # All exact labels and branch construction happen AFTER this decision.
        before=oracle.distance_for(oracle.state_of(env))
        targets,branches,results,mask=checked_expansion(wd._expand,checks,env,oracle,before,spec['seed'],step)
        reachable=before is not None
        optimal=bool(mask & (1<<choice)) if reachable else None
        if reachable:
            counts['recoverable_decisions']+=1;counts['recoverable_optimal']+=bool(optimal)
            if not optimal and output['first_nonoptimal_decision'] is None:output['first_nonoptimal_decision']=step
        else:counts['within_life_unreachable_decisions']+=1
        old_lives=env.lives();env,result=branches[choice],results[choice]
        lost=env.lives()<old_lives
        stalled=not result.finished and np.array_equal(frames[-1],result.frame)
        counts['life_losses']+=lost;counts['history_resets']+=lost and not result.finished
        counts['stalls']+=stalled;counts['actions']+=1
        entry=dict(step=step,action_index=choice,action=wd.names.ACTION_IDS[choice],probabilities=probabilities,
            optimal_mask=int(mask),within_life_distance=None if before is None else int(before),
            optimal_choice=optimal,successor_distances=targets['distances'].astype(int).tolist(),
            successor_lost_life=targets['lost_life'].tolist(),successor_won=targets['won'].tolist(),
            successor_terminal=targets['terminal'].tolist(),stalled=bool(stalled),lost_life=bool(lost),
            reset_history=bool(lost and not result.finished),lives_after=env.lives(),history_valid=valid.tolist(),previous_actions=previous.tolist(),
            history_sha256=hashlib.sha256(observed.tobytes()+valid.tobytes()+previous.tobytes()).hexdigest())
        output['steps'].append(entry)
        if on_step:on_step(output)
        if result.finished:
            output['ending']='win' if result.won else 'game_over';break
        if lost:frames,actions=[result.frame],[-1]
        else:frames,actions=(frames+[result.frame])[-8:],(actions+[choice])[-8:]
    output.update(counts=dict(counts),branch_checks=dict(checks),lives_left=env.lives(),
        completed=output['ending']=='win',unknown_decisions=0,
        recoverable_optimal_rate=(counts['recoverable_optimal']/counts['recoverable_decisions']
                                  if counts['recoverable_decisions'] else None),
        within_life_unreachable_fraction=counts['within_life_unreachable_decisions']/counts['actions'])
    return output



def validate_checkpoint(metadata,selected):
    if (metadata.get('data_meta',{}).get('source')!='generated_only'
        or metadata.get('curriculum',{}).get('difficulty_version')!=DIFFICULTY_VERSION
        or not metadata.get('trained') or not isinstance(metadata.get('epochs'),int) or metadata['epochs']<1
        or metadata.get('initialize_checkpoint') is not None or metadata.get('initialize_glyph_checkpoint') is not None
        or metadata.get('cell_source') is not None):
        raise ValueError('completed fresh generated seven-reference WorldPolicy checkpoint required')
    seeds={s['seed'] for s in selected}
    if not seeds <= set(metadata.get('train_seeds',[])) or seeds & set(metadata.get('validation_seeds',[])):
        raise ValueError('diagnostic seeds must belong exclusively to checkpoint TRAIN split')

def write(path,report):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--generation-report',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--levels',type=int,default=14);p.add_argument('--max-actions',type=int,default=300)
    p.add_argument('--seconds',type=int,default=1800);p.add_argument('--seed',type=int,default=20260912)
    a=p.parse_args(argv)
    if a.out.exists():raise FileExistsError(a.out)
    if not 1<=a.seconds<=7200:p.error('seconds must be1..7200')
    if not 1<=a.max_actions<=300:p.error('max-actions must be1..300')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    start=time.monotonic();print('PID',os.getpid(),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(format='pebby.reference-policy-diagnosis.v1',status='running',pid=os.getpid(),split='train',
        generated_only=True,official_inputs_used=False,optimizer_used=False,device='cpu',precision='float32',
        max_actions=a.max_actions,levels=[],limits=[
            'Within-life unreachable means no route with the current remaining budget; a life reset may recover.',
            'Teacher labels are diagnostic only and are computed after the public policy decision.',
            'Four checked branches per visited state; no exhaustive reachable-state engine verification.',
            'Partial/failed runs are not complete population results; no validation or official evaluation.'])
    def persist():report['elapsed_seconds']=time.monotonic()-start;write(a.out,report)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('diagnostic deadline')))
    signal.alarm(a.seconds);persist()
    try:
        files=[a.checkpoint,a.bank,a.generation_report,Path(__file__),Path('tools/validate_extended_collector.py')]
        files+=list(Path('pebby/ls20').glob('*.py'))+list(Path('pebby/ls20').glob('*.c'))
        files += [Path('pebby/agent')/name for name in ('world_model.py','world_data.py','world_grounding.py',
                   'glyph_model.py','history.py','model.py','world_cell_recall.py')]
        files += [Path('third_party/ls20/ls20.py')]
        sources={str(path.resolve()):digest(path) for path in files};report['sources']=sources
        generation=json.loads(a.generation_report.read_text())
        if (generation.get('status')!='complete' or generation.get('config',{}).get('profile')!=DIFFICULTY_VERSION
            or generation.get('banks',{}).get('train',{}).get('sha256')!=sources[str(a.bank.resolve())]):
            raise ValueError('completed reference TRAIN publication required')
        specs=load(a.bank)
        if generation['banks']['train']['levels']!=len(specs):raise ValueError('TRAIN publication count mismatch')
        selected=select_levels(specs,a.levels,a.seed);report['selected_seeds']=[s['seed'] for s in selected]
        policy,metadata=load_world_checkpoint(a.checkpoint);validate_checkpoint(metadata,selected)
        policy=policy.cpu().float().eval()
        if any(digest(path)!=value for path,value in sources.items()):raise ValueError('bound source changed during load')
        report['config']=policy.config();report['parameters']=policy.parameter_count()
        for spec in selected:
            report['active_seed']=spec['seed'];persist()
            def progress(partial):
                report['active_level']=partial
                if len(partial['steps'])%10==0:persist()
            result=diagnose_level(spec,policy,max_actions=a.max_actions,on_step=progress)
            report['levels'].append(result);report.pop('active_level',None);gc.collect();persist()
        if any(digest(path)!=value for path,value in sources.items()):raise ValueError('bound source changed')
        totals=Counter()
        for level in report['levels']:totals.update(level['counts'])
        report.update(status='complete',sources_unchanged=True,counts=dict(totals),
                      completed=sum(level['completed'] for level in report['levels']),
                      recoverable_optimal_rate=(totals['recoverable_optimal']/totals['recoverable_decisions']
                                               if totals['recoverable_decisions'] else None),
                      within_life_unreachable_fraction=totals['within_life_unreachable_decisions']/totals['actions'])
    except BaseException as error:
        report.update(status='failed_partial',error=f'{type(error).__name__}: {error}');raise
    finally:
        signal.alarm(0);report['active_seed']=None;persist()

if __name__=='__main__':main()
