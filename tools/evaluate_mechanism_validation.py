"""Frozen-policy generated mechanism diagnostic; planner calls forbidden."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'artifacts/world-onpolicy-round2-code'
sys.path.insert(0,str(SNAPSHOT))
import torch
from pebby.agent import evaluate as ev
from pebby.agent.model import load_checkpoint
from pebby.ls20 import plan

torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes=json.loads((SNAPSHOT/'source-hashes.json').read_text())
assert Path(ev.__file__).is_relative_to(SNAPSHOT)
assert all(sha(SNAPSHOT/path)==value for path,value in hashes.items())
def forbidden(*args,**kwargs):raise AssertionError('Planner calls forbidden during evaluation')
plan.Oracle=forbidden;plan.oracle_for=forbidden;ev.optimal_actions=forbidden

def main():
    started=time.monotonic();pid=os.getpid();print('PID',pid,flush=True)
    def timeout(*_):raise TimeoutError('six-minute evaluation deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(360)
    bank=ROOT/'data/ls20-mechanism-validation50.jsonl';bank_sha=sha(bank)
    levels,optima,specs=ev.bank_levels(bank)
    assert len(specs)==50 and all(1_900_001<=s['seed']<=1_999_999 for s in specs)
    assert all(s['context_engine_verified'] and not s['search_truncated'] and s['training_context_index']==s['seed']%7 for s in specs)
    reports=[]
    for tag,filename,epoch in [('round2-epoch1','ls20-world-onpolicy-round2-b1024.epoch1.pt',1),('query-epoch8','ls20-world-query-glyph-b1024.epoch8.pt',8)]:
        path=ROOT/'checkpoints'/filename;checkpoint_sha=sha(path)
        policy,metadata=load_checkpoint(path,'cpu')
        assert sum(p.numel() for p in policy.parameters())==403061 and policy.loops==4
        actual=metadata.get('epoch',metadata.get('best_epoch'))
        assert actual==epoch,(filename,actual)
        out=ROOT/f'artifacts/world-mechanism-validation50-{tag}.json'
        if out.exists():raise FileExistsError(out)
        begin=time.monotonic();print('START',tag,checkpoint_sha,flush=True)
        with out.with_suffix('.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            result=ev.completion_rate(policy,levels,200,torch.device('cpu'),optima,'repeat',[s['training_context_index'] for s in specs])
        for run,spec in zip(result['runs'],specs):
            for key in ('seed','pilot_mode','training_context_index','context_optimal_actions'):run[key]=spec[key]
        modes={}
        for mode in sorted({s['pilot_mode'] for s in specs}):
            runs=[r for r in result['runs'] if r['pilot_mode']==mode]
            modes[mode]={'levels':len(runs),'completed':sum(r['completed'] for r in runs),'game_over':sum(r['ending']=='game_over' for r in runs),'capped':sum(r['ending']=='capped' for r in runs),'goals_cleared':sum(r['goals_cleared'] for r in runs),'goals_total':sum(r['goals_total'] for r in runs),'total_actions':sum(r['actions'] for r in runs),'stalled_actions':sum(r['stalls'] for r in runs)}
        assert sha(path)==checkpoint_sha and sha(bank)==bank_sha
        assert all(sha(SNAPSHOT/p)==value for p,value in hashes.items())
        result.update(status='complete',checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=checkpoint_sha,checkpoint_epoch=epoch,
            bank=str(bank.relative_to(ROOT)),bank_sha256=bank_sha,seeds=[s['seed'] for s in specs],per_mode=modes,
            parameters=403061,code_snapshot=str(SNAPSHOT.relative_to(ROOT)),code_hashes_at_import=hashes,
            evaluator_script_sha256=sha(__file__),vendored_engine_sha256=sha(SNAPSHOT/'third_party/ls20/ls20.py'),
            checkpoint_and_sources_unchanged=True,device='cpu',cpu_threads=1,pid=pid,oracle_calls=0,
            protocol='strict greedy repeat; 200 actions per generated isolated level; actual seed%7 context; cached exact optima only',
            temperature=0.,inference_loops=4,total_actions=sum(r['actions'] for r in result['runs']),
            elapsed_seconds=time.monotonic()-begin,
            limitations=['One changing attribute per level, fixed room and board; targeted mechanic transfer only.','Policies are frozen; these levels were not used for training or adaptation.','Tile presence is not proof every tile must be used; generator enforces at least one use per present moving/launcher mechanic and at least one refill.'])
        temp=out.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(out)
        reports.append(str(out.relative_to(ROOT)))
        print('RESULT',json.dumps({k:result[k] for k in ('checkpoint_epoch','completed','game_over','capped','goals_cleared','goals_total','total_actions','per_mode','elapsed_seconds')}),flush=True)
        del policy
    signal.alarm(0)
    print('DONE',json.dumps({'pid':pid,'reports':reports,'runtime_seconds':time.monotonic()-started}),flush=True)

if __name__=='__main__':main()
