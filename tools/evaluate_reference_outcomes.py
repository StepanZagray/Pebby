"""Actual generated-level gameplay for the one-step neural outcome planner.

This isolated generated screen has one native three-life episode per level.
Headline official success uses evaluate_reference_competition instead.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import time

import torch

from pebby.agent import evaluate
from pebby.agent.neural_outcome_policy import load_checkpoint
from pebby.ls20.env import Ls20Scenario
from tools.train_reference_outcomes import sha, write, guard, gpu_available

ROOT = Path(__file__).resolve().parents[1]


def summary(runs):
    return dict(levels=len(runs), completed=sum(r['completed'] for r in runs),
        actions=sum(r['actions'] for r in runs), stalls=sum(r['stalls'] for r in runs),
        goals_cleared=sum(r['goals_cleared'] for r in runs),
        goals_total=sum(r['goals_total'] for r in runs),
        lives_lost=sum(3-r['lives_left'] for r in runs),
        endings=dict(Counter(r['ending'] for r in runs)))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--report-out',type=Path,required=True)
    parser.add_argument('--bank',type=Path,default=ROOT/'artifacts/reference-grounding-repair-v1/validation70.jsonl')
    parser.add_argument('--direct-weight',type=float,default=0.)
    parser.add_argument('--planner-weight',type=float,default=1.)
    parser.add_argument('--max-actions',type=int,default=300)
    parser.add_argument('--max-seconds',type=int,default=600)
    args=parser.parse_args(argv)
    if min(args.max_actions,args.max_seconds)<=0:parser.error('positive bounds required')
    if args.report_out.exists():raise FileExistsError(args.report_out)
    if sha(args.checkpoint)!=args.checkpoint_sha256:raise ValueError('checkpoint hash differs')
    specs=[json.loads(line) for line in args.bank.read_text().splitlines()]
    if not specs or any(s.get('source')!='generated_only' for s in specs):
        raise ValueError('generated-only bank required')
    gpu_available();guard();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    model,metadata=load_checkpoint(args.checkpoint,'cuda',direct_weight=args.direct_weight,
                                   planner_weight=args.planner_weight)
    if metadata.get('official_training_inputs') is not False:
        raise ValueError('checkpoint must declare generated-only training')
    sources=[Path(__file__).resolve(),args.checkpoint.resolve(),args.bank.resolve(),
             *[ROOT/'pebby/agent'/name for name in ('neural_outcome_planner.py','neural_outcome_policy.py',
                 'world_model.py','glyph_model.py','world_readout.py','evaluate.py','history.py')],
             ROOT/'pebby/ls20/env.py']
    bindings={str(p):sha(p) for p in sources}
    args.report_out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        started_local=datetime.now().astimezone().isoformat(),checkpoint_sha256=args.checkpoint_sha256,
        source_sha256=bindings,official_inputs_used=False,training=False,oracle_calls=0,
        protocol=dict(kind='isolated_generated_levels',max_actions=args.max_actions,
                      on_stall='repeat',temperature=0.,native_lives=3,reset_extension=False),
        runtime=dict(device='cuda',precision='FP32',temporal_backend='auto',matmul_tf32=False,cudnn_tf32=True),
        decision=dict(planner_weight=args.planner_weight,direct_weight=args.direct_weight,old_ranker_weight=0.,
                      planner_horizon=1,planner_refinement_loops=1,learned_voluntary_reset=False),runs=[])
    write(args.report_out,report)
    def timeout(*_):raise TimeoutError('bounded generated gameplay deadline reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(args.max_seconds)
    started=time.monotonic()
    try:
        levels,optima,loaded=evaluate.bank_levels(args.bank)
        if loaded!=specs:raise ValueError('bank changed while loading')
        with torch.inference_mode():
            for index,(level,optimal,spec) in enumerate(zip(levels,optima,specs)):
                guard()
                run=evaluate.rollout(model,Ls20Scenario(level,int(spec.get('training_context_index',0))),
                    args.max_actions,torch.device('cuda'),optimal,on_stall='repeat',temperature=0.)
                run.update(run=index,seed=spec['seed'],difficulty=spec['difficulty'])
                report['runs'].append(run)
                if (index+1)%10==0 or index+1==len(specs):
                    report['summary']=summary(report['runs'])
                    report['elapsed_seconds']=time.monotonic()-started
                    write(args.report_out,report)
                    print(json.dumps(dict(event='progress',**report['summary'],elapsed_seconds=report['elapsed_seconds'])),flush=True)
        report['per_tier']={str(tier):summary([r for r in report['runs'] if r['difficulty']==tier]) for tier in range(1,8)}
        if any(sha(p)!=h for p,h in bindings.items()):raise ValueError('source or checkpoint changed during gameplay')
        report.update(status='complete',sources_unchanged=True,elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0);write(args.report_out,report)


if __name__=='__main__':main()
