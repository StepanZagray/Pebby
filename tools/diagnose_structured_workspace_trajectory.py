"""Generated held-out trajectory diagnosis including unreachable decision periods.

Oracle labels are offline diagnostics. 'Unreachable' means no completion before
another life loss, not failure of the remaining multi-life game. Never stop the
actor on a teacher label or repeated history. No four-branch engine expansion.
"""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import resource
import signal
import time

import numpy as np
import torch

from pebby.agent.history import for_policy
from pebby.agent.model import load_checkpoint
from pebby.agent.world_data import clone_env
from pebby.ls20 import names
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.layout import extract
from pebby.ls20.plan import Oracle, Unplannable, simulate, advance
from tools.evaluate_structured_workspace_gameplay import checked_bank, BANK, BANK_SHA
from tools.train_structured_transition import digest, atomic_json


class CompleteTeacher:
    """One exhaustive native graph; coverage tracking distinguishes absence.

    The planner prunes rejected transitions, even rare rejected transitions that
    change state. Following such an edge loses the proof that a missing distance
    means unreachable. Known distance entries and real life resets restore it.
    """
    def __init__(self, env, spec, limit=600000):
        self.oracle = None; self.covered = False
        self.proof = {'status': 'unknown', 'complete': False, 'limit': limit}
        if spec['training_context_index'] == 0 and spec.get('launchers'):
            self.proof.update(status='unknown_unsupported', reason='context0 launcher pending hint outside planner state')
            return
        tick = time.monotonic()
        try:
            oracle = Oracle(extract(env), limit=limit, engine='fast')
        except (Unplannable, RuntimeError) as error:
            self.proof.update(status='unknown_unsupported', reason=str(error), seconds=time.monotonic()-tick)
            return
        self.proof.update(seconds=time.monotonic()-tick, backend=oracle.engine, reachable_states=oracle._reachable,
                          truncated=bool(oracle.truncated))
        if oracle.truncated:
            self.proof.update(status='unknown_truncated', reason='incomplete graph cannot establish optimality or unreachable states')
            return
        if not oracle.solvable:
            raise ValueError('complete initial teacher contradicts verified solvable monitor')
        if oracle.optimal_actions != spec['context_optimal_actions']:
            raise ValueError('contextual teacher optimum contradicts fixed monitor proof')
        replay = clone_env(env); final = None
        for action in oracle.solution(): final = replay.perform(action)
        if final is None or not final.won or replay.lives() != 3 or replay.levels_completed != 1:
            raise ValueError('complete teacher winning replay failed actual engine')
        self.oracle = oracle; self.covered = True
        self.proof.update(status='complete', complete=True, winning_engine_replay=True)

    def label(self, env):
        if self.oracle is None:
            return {'status': self.proof['status'], 'distance': None, 'optimal_mask': None}
        state = self.oracle.state_of(env)
        distance = self.oracle.distance_for(state)
        if distance is not None:
            self.covered = True
            if distance <= 0: raise ValueError('finished teacher state presented as a live decision')
            mask = 0
            for action in range(4):
                nxt = advance(self.oracle.layout, state, action, self.oracle.refills)
                if nxt is not None and self.oracle.distance_for(nxt) == distance - 1: mask |= 1 << action
            if not mask: raise ValueError('complete reachable state has no optimal edge')
            return {'status': 'reachable', 'distance': int(distance), 'optimal_mask': mask}
        return {'status': 'unreachable' if self.covered else 'unknown_graph_coverage', 'distance': None, 'optimal_mask': None}

    def before(self, env, action):
        if self.oracle is None: return None
        state = self.oracle.state_of(env)
        predicted, outcome = simulate(self.oracle.layout, state, action, self.oracle.refills)
        return state, predicted, outcome, env.lives()

    def after(self, env, result, evidence):
        if evidence is None: return
        state, predicted, outcome, old_lives = evidence
        if outcome == 'won':
            agrees = result.won and env.lives() == old_lives
        elif outcome == 'died':
            agrees = not result.won and env.lives() == old_lives - 1 and (result.finished or self.oracle.state_of(env) == self.oracle.start)
        else:
            agrees = not result.finished and env.lives() == old_lives and self.oracle.state_of(env) == predicted
        if not agrees: raise ValueError(f'logical/actual selected transition mismatch: {outcome}')
        if outcome == 'died' and not result.finished:
            self.covered = True
        elif outcome == 'rejected' and predicted != state:
            self.covered = False


def rates(decisions):
    counts = Counter(row['status'] for row in decisions)
    reachable = [row for row in decisions if row['status'] == 'reachable']
    n = len(decisions); known = len(reachable) + counts['unreachable']
    return {'decisions': n, 'reachable_decisions': len(reachable), 'reachable_optimal_choices': sum(row['optimal'] for row in reachable),
            'reachable_optimal_rate': sum(row['optimal'] for row in reachable)/len(reachable) if reachable else None,
            'unreachable_decisions': counts['unreachable'], 'unreachable_fraction_all': counts['unreachable']/n if n else None,
            'unreachable_fraction_known': counts['unreachable']/known if known else None,
            'unknown_decisions': n-known, 'status_counts': dict(counts),
            'life_losses': sum(bool(r.get('lost_life')) for r in decisions),
            'resets': sum(bool(r.get('reset')) for r in decisions),
            'stalls': sum(bool(r.get('stalled')) for r in decisions)}


@torch.inference_mode()
def diagnose_level(policy, env, teacher, max_actions=200, device='cpu', progress=None):
    """Teacher observations follow action choice; labels never alter rollout."""
    frame = env.reset(); history = for_policy(policy, frame, device)
    if history is None: raise ValueError('public H8 structured policy required')
    goals_total = len(env.goal_triples()); initial_lives = env.lives()
    decisions = []; ending = 'capped'; consecutive = 0; life_index = 0; first_by_life = {}
    for step in range(max_actions):
        start = time.perf_counter()
        logits = history.scores()
        if logits.shape != (4,) or not bool(torch.isfinite(logits).all()): raise ValueError('invalid policy logits')
        choice = int(logits.argmax())  # Public argmax committed BEFORE teacher query.
        decision_seconds = time.perf_counter()-start
        label = teacher.label(env)
        evidence = teacher.before(env, choice)
        old_lives = env.lives(); old_level = env.level_index
        row = {'step': step, 'life_index': life_index, 'action_index': choice, **label,
               'optimal': bool(label['optimal_mask'] & (1 << choice)) if label['status'] == 'reachable' else None,
               'preceding_unchanged_frames': consecutive, 'decision_seconds': decision_seconds}
        if row['status'] == 'reachable' and not row['optimal']:
            first_by_life.setdefault(str(life_index), step)
        observation = env.perform(names.ACTION_IDS[choice])
        teacher.after(env, observation, evidence)
        loss = env.lives() < old_lives
        if observation.frame is None: raise ValueError('live action omitted observation')
        history.observe(observation.frame, choice, reset=loss or env.level_index != old_level)
        # Match the existing strict evaluator: terminal actions are not stalls.
        stalled = not observation.finished and observation.frame == frame
        consecutive = consecutive + 1 if stalled and not loss else 0
        row.update(lost_life=loss, reset=loss and not observation.finished, stalled=stalled,
                   terminal=observation.finished, won=observation.won)
        if not observation.finished:
            following = teacher.label(env)
            row['next_status'] = following['status']
        else: row['next_status'] = 'terminal_win' if observation.won else 'terminal_failure'
        decisions.append(row)
        if loss: life_index += 1
        if progress is not None: progress(decisions)
        if observation.finished:
            ending = 'win' if observation.won else 'game_over'; break
        frame = observation.frame
    completed = ending == 'win'
    summary = rates(decisions)
    summary.update(actions=len(decisions), completed=completed, ending=ending,
        goals_total=goals_total, goals_cleared=goals_total if completed else sum(env.goals_solved()),
        lives_left=env.lives(), life_losses=initial_lives-env.lives(), resets=sum(r['reset'] for r in decisions),
        stalls=sum(r['stalled'] for r in decisions), decisions_after8_unchanged=sum(r['preceding_unchanged_frames'] >= 8 for r in decisions),
        first_mistake=next((r['step'] for r in decisions if r['status']=='reachable' and not r['optimal']), None),
        first_mistake_per_life=first_by_life,
        reachable_to_unreachable=sum(r['status']=='reachable' and r['next_status']=='unreachable' for r in decisions))
    return {'summary': summary, 'decisions': decisions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/ls20-structured-workspace-comparison-600-static.pt'))
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--levels', type=int, default=100)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--seconds', type=int, default=1200)
    parser.add_argument('--search-limit', type=int, default=600000)
    args = parser.parse_args()
    if args.report.exists() or not 1 <= args.levels <= 100 or not 1 <= args.seconds <= 1200 or not 1 <= args.search_limit <= 600000:
        parser.error('new report, prefix1..100, deadline<=1200 and search<=600k required')
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    started=time.monotonic(); print('PID',os.getpid(),flush=True)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('trajectory diagnostic deadline')))
    signal.alarm(args.seconds)
    report={'status':'loading','pid':os.getpid(),'source':'generated_only','split':'validation','official_inputs_used':False,
        'training_performed':False,'bank':str(BANK),'bank_sha256':BANK_SHA,'checkpoint':str(args.checkpoint),
        'max_actions':200,'on_stall':'repeat','temperature':0.,'device':args.device,'precision':'FP32, TF32 off',
        'requested_levels':args.levels,'prefix_pilot':args.levels<100,'levels':[],
        'limits':['Unreachable means cannot finish before another life loss, not impossible to win after reset.',
                  'Truncated/unsupported/out-of-graph states are unknown, excluded from reachable optimality.',
                  'One complete native graph per level; only selected actual transitions are checked.',
                  'No early stop for unreachable states or exact repeated histories. No Oracle action feedback.']}
    def persist():
        report['elapsed_seconds']=time.monotonic()-started
        report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        atomic_json(args.report,report)
    persist(); policy=None
    try:
        paths=[args.checkpoint,BANK,Path(__file__),Path('pebby/ls20/plan.py'),Path('pebby/ls20/layout.py'),
               Path('pebby/agent/history.py'),Path('pebby/agent/world_data.py'),Path('pebby/agent/model.py'),
               Path('tools/evaluate_structured_workspace_gameplay.py'),Path('third_party/ls20/ls20.py'),
               Path('pebby/ls20/fastplan.py'),Path('pebby/ls20/_fastplan.c'),Path('pebby/ls20/rails.py'),Path('pebby/ls20/env.py'),
               Path('pebby/ls20/names.py'),Path('pebby/ls20/generate.py'),Path('pebby/agent/structured_workspace_controller.py'),
               Path('pebby/agent/structured_workspace_policy.py'),Path('tools/train_structured_transition.py')]
        sources={str(p):digest(p) for p in paths}
        policy,info=load_checkpoint(args.checkpoint,args.device)
        if digest(args.checkpoint)!=sources[str(args.checkpoint)]:raise ValueError('checkpoint changed while loading')
        if info.get('format')!='pebby.structured-workspace-readout.v1' or info.get('readout_config',{}).get('memory_mode') not in ('static','evolving') or policy.config().get('loops')!=2:
            raise ValueError('expected static/evolving workspace public controller at depth2')
        provenance=info.get('training_provenance',{})
        if provenance.get('smoke') is not False or provenance.get('updates')!=600 or provenance.get('batch_size')!=1024:
            raise ValueError('production600 B1024 static checkpoint required')
        policy.float().eval().requires_grad_(False)
        report['arm']=info['readout_config']['memory_mode']
        report['checkpoint_sha256']=sources[str(args.checkpoint)]
        report['parameters']=policy.parameter_count()
        sources[info['actor_checkpoint']]=info['actor_sha256']
        sources.update(policy.sources['code_hashes']);sources.update({v['path']:v['sha256'] for v in policy.sources['artifacts'].values()})
        levels,optima,specs=checked_bank(); report['selected_seeds']=[s['seed'] for s in specs[:args.levels]]
        report['source_hashes']=sources
        for level,spec in zip(levels[:args.levels],specs[:args.levels]):
            entry={'seed':spec['seed'],'difficulty':spec['difficulty'],'initial_optimal':spec['context_optimal_actions'],
                   'context':spec['training_context_index'],'status':'preparing_teacher'}
            report['levels'].append(entry);persist()
            env=Ls20Scenario(level,spec['training_context_index']);env.reset()
            teacher=CompleteTeacher(env,spec,args.search_limit);entry['teacher']=teacher.proof
            entry['status']='playing';persist()
            def progress(rows):
                entry['partial_summary']=rates(rows)
                entry['decisions']=rows
                if len(rows)%25==0:persist()
            entry.update(diagnose_level(policy,env,teacher,device=args.device,progress=progress))
            entry.pop('partial_summary',None);entry['status']='complete'
            all_decisions=[r for e in report['levels'] for r in e.get('decisions',[])]
            report['aggregate']=rates(all_decisions)
            persist();print(json.dumps({'seed':spec['seed'],**entry['summary']}),flush=True)
            del teacher,env;gc.collect()
        for path,sha in sources.items():
            if digest(path)!=sha:raise ValueError(f'source changed: {path}')
        report.update(status='complete',source_unchanged=True)
    except BaseException as error:
        report.update(status='failed_partial',error=str(error));raise
    finally:
        policy=None;gc.collect()
        if args.device=='cuda' and torch.cuda.is_available():torch.cuda.empty_cache()
        signal.alarm(0);persist()


if __name__=='__main__':main()
