"""Fixed generated monitor comparison after a verified complete workspace fit.

Public causal H8 inference only; stored optimum lengths are reporting metadata.
No planner, training, action masking, retries, or checkpoint/depth selection.
"""
import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.evaluate import bank_levels, rollout
from pebby.agent.model import load_checkpoint
from pebby.ls20.env import Ls20Scenario
from tools.train_structured_transition import digest, atomic_json

BANK = Path('data/ls20-verified-validation-monitor.jsonl')
BANK_SHA = 'd2484bf547d7c624f9e512101828ff62c758715917f678f0c6c9b2e1c4038446'
ACTOR = 'checkpoints/ls20-structured-policy-paired-local-h4-600.pt'
FORMAT = 'pebby.structured-workspace-readout.v1'


def verify_sources(sources):
    if not isinstance(sources, dict) or not sources:
        raise ValueError('nonempty source guards required')
    for path, sha in sources.items():
        if digest(path) != sha: raise ValueError(f'source changed: {path}')


def validate_fit(report):
    """Pure report gate, separately testable without constructing any models."""
    if (report.get('status') != 'complete' or report.get('smoke') is not False
            or report.get('source') != 'generated_only' or report.get('official_inputs_used') is not False
            or report.get('sources_unchanged') is not True or report.get('paired_selections_exact') is not True
            or report.get('primary_depth') != 2 or report.get('trained_depths') != [1, 2, 4]):
        raise ValueError('complete fixed generated-only production fit required')
    args = report.get('args', {})
    if args.get('batch_size') != 1024 or args.get('updates') != 600:
        raise ValueError('production fit must use B1024 and600 updates')
    selections = []
    for name, count in [('static', 108097), ('evolving', 201315)]:
        arm = report.get('arms', {}).get(name, {})
        depths = arm.get('depth_draws', {})
        if (arm.get('status') != 'complete' or arm.get('completed_updates') != 600
                or arm.get('active_parameters') != count or arm.get('parameters') != 201315
                or set(depths) != {'1', '2', '4'} or any(type(v) is not int or v <= 0 for v in depths.values())
                or sum(depths.values()) != 600 or arm.get('checkpoint', {}).get('strict_reload_exact') is not True):
            raise ValueError(f'invalid completed {name} arm')
        selections.append(arm.get('selection_sha256'))
    if (not isinstance(selections[0], str) or len(selections[0]) != 64 or selections[0] != selections[1]
            or report['arms']['static']['depth_draws'] != report['arms']['evolving']['depth_draws']):
        raise ValueError('paired sampling/depth mismatch')


def checked_fit(path):
    path = Path(path); raw = path.read_bytes(); report = json.loads(raw)
    validate_fit(report)
    sources = dict(report['sources']); verify_sources(sources)
    if ACTOR not in sources: raise ValueError('missing baseline actor binding')
    checkpoints = {'actor': {'path': ACTOR, 'sha256': sources[ACTOR]}}
    for name in ('static', 'evolving'):
        item = report['arms'][name]['checkpoint']; file = Path(item['path'])
        payload = file.read_bytes()
        if hashlib.sha256(payload).hexdigest() != item['sha256']: raise ValueError('final checkpoint SHA mismatch')
        saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
        provenance = saved.get('training_provenance', {})
        if (saved.get('format') != FORMAT or saved.get('actor_checkpoint') != ACTOR
                or saved.get('actor_sha256') != sources[ACTOR] or saved.get('sources') != report['sources']
                or saved.get('source_unchanged') is not True or saved.get('official_inputs_used') is not False
                or saved.get('config', {}).get('memory_mode') != name
                or provenance.get('arm') != name or provenance.get('updates') != 600
                or provenance.get('batch_size') != 1024 or provenance.get('smoke') is not False
                or provenance.get('fixed_final') is not True or provenance.get('primary_depth') != 2
                or provenance.get('depths') != [1, 2, 4]
                or provenance.get('depth_draws') != report['arms'][name]['depth_draws']
                or provenance.get('selection_sha256') != report['arms'][name]['selection_sha256']
                or provenance.get('encoder_and_dynamics_frozen') is not True
                or provenance.get('official_inputs_used') is not False or provenance.get('source') != 'generated_only'):
            raise ValueError(f'{name} checkpoint/fit provenance mismatch')
        checkpoints[name] = {'path': str(file), 'sha256': item['sha256']}
        sources[str(file)] = item['sha256']
    expected = hashlib.sha256(raw).hexdigest()
    if digest(path) != expected: raise ValueError('fit report changed during inspection')
    sources[str(path)] = expected
    return report, checkpoints, sources


def checked_bank():
    if digest(BANK) != BANK_SHA: raise ValueError('fixed monitor changed')
    levels, optima, specs = bank_levels(BANK)
    seeds = [s['seed'] for s in specs]
    if len(specs) != 100 or len(set(seeds)) != 100 or any(type(s) is not int or not 1_000_000 <= s < 2_000_000 for s in seeds):
        raise ValueError('expected100 distinct held-out generated monitor levels')
    for spec, optimum in zip(specs, optima):
        if (type(spec.get('training_context_index')) is not int or not 0 <= spec['training_context_index'] <= 6
                or spec.get('context_engine_verified') is not True or spec.get('search_truncated') is not False
                or type(optimum) is not int or optimum <= 0):
            raise ValueError('verified explicit generated context/optimum required')
    return levels, optima, specs


class DecisionTimer:
    """Time public policy forward only; no environment/label access."""
    def __init__(self, policy, device):
        self.device = device; self.durations = []; self.started = None
        self.before = policy.register_forward_pre_hook(self.start)
        self.after = policy.register_forward_hook(self.stop)
    def sync(self):
        if self.device == 'cuda': torch.cuda.synchronize()
    def start(self, module, args):
        self.sync(); self.started = time.perf_counter()
    def stop(self, module, args, output):
        self.sync(); self.durations.append(time.perf_counter() - self.started)
    def close(self):
        self.before.remove(); self.after.remove()


def summarize(rows):
    result = {}
    for name, subset in [('all', rows), ('initial_optimal_ge17', [r for r in rows if r['optimal'] >= 17])]:
        n = len(subset)
        result[name] = {'levels': n, 'completed': sum(r['completed'] for r in subset),
            'goals_cleared': sum(r['goals_cleared'] for r in subset), 'goals_total': sum(r['goals_total'] for r in subset),
            'actions': sum(r['actions'] for r in subset), 'losses': sum(r['losses'] for r in subset),
            'stalls': sum(r['stalls'] for r in subset), 'capped': sum(r['ending'] == 'capped' for r in subset),
            'game_over': sum(r['ending'] == 'game_over' for r in subset)}
    return result


def paired(candidate, baseline):
    if [r['seed'] for r in candidate] != [r['seed'] for r in baseline]: raise ValueError('paired monitor seed order mismatch')
    result = {}
    for group in ('all', 'initial_optimal_ge17'):
        pairs = [(a, b) for a, b in zip(candidate, baseline) if group == 'all' or a['optimal'] >= 17]
        result[group] = {'levels': len(pairs), 'gained_wins': [a['seed'] for a, b in pairs if a['completed'] and not b['completed']],
            'lost_wins': [a['seed'] for a, b in pairs if b['completed'] and not a['completed']],
            'losses_delta': sum(a['losses'] - b['losses'] for a, b in pairs),
            'stalls_delta': sum(a['stalls'] - b['stalls'] for a, b in pairs)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-report', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--seconds', type=int, default=1200)
    args = parser.parse_args()
    if args.report.exists() or not 1 <= args.seconds <= 1200: parser.error('new report and deadline1..1200 required')
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic(); print('PID', os.getpid(), flush=True)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('gameplay deadline')))
    signal.alarm(args.seconds)
    report = {'status': 'validating', 'pid': os.getpid(), 'source': 'generated_only', 'split': 'validation',
        'official_inputs_used': False, 'training_performed': False, 'device': args.device, 'precision': 'FP32; TF32 off',
        'max_actions': 200, 'protocol': 'strict', 'on_stall': 'repeat', 'temperature': 0., 'primary_depth': 2,
        'bank': str(BANK), 'bank_sha256': BANK_SHA, 'arms': {},
        'limits': ['Fixed monitor100 only, not official completion.', 'Static/evolving active parameter counts differ.',
                   'Decision timing measures model forward; excludes public H8 tensor construction.',
                   'Stored exact optima are reporting metadata only; no Oracle is called.']}
    def persist():
        report['elapsed_seconds'] = time.monotonic() - started; atomic_json(args.report, report)
    policy = timer = None
    persist()
    try:
        fit, checkpoints, sources = checked_fit(args.fit_report)
        levels, optima, specs = checked_bank()
        for path in (str(BANK), __file__, 'pebby/agent/evaluate.py', 'pebby/agent/history.py', 'pebby/agent/model.py',
                     'pebby/agent/structured_workspace_controller.py', 'pebby/ls20/env.py', 'pebby/ls20/generate.py',
                     'pebby/ls20/bank.py', 'pebby/ls20/names.py', 'third_party/ls20/ls20.py'):
            sources[str(path)] = digest(path)
        # Namespace guard is supplemented by actual training seed membership.
        train_seeds = np.load('data/structured-field-16384/train/seeds.npy', allow_pickle=False)
        if set(map(int, train_seeds)) & {s['seed'] for s in specs}: raise ValueError('monitor/training overlap')
        report['sources'] = sources; report['fit_report'] = str(args.fit_report)
        for arm, checkpoint in checkpoints.items():
            report['active_arm'] = arm; report['status'] = 'running'; persist()
            if digest(checkpoint['path']) != checkpoint['sha256']: raise ValueError('checkpoint changed before load')
            policy, info = load_checkpoint(checkpoint['path'], args.device)
            policy.float().eval().requires_grad_(False)
            if digest(checkpoint['path']) != checkpoint['sha256']: raise ValueError('checkpoint changed while loading')
            if policy.config().get('loops') != 2: raise ValueError('primary inference requires depth2')
            entry = {'checkpoint': checkpoint, 'parameters': policy.parameter_count(), 'status': 'running', 'runs': []}
            report['arms'][arm] = entry
            timer = DecisionTimer(policy, args.device)
            for index, (level, optimum, spec) in enumerate(zip(levels, optima, specs)):
                first = len(timer.durations)
                entry['incomplete_level'] = {'index': index, 'seed': spec['seed'],
                    'note': 'If interrupted, this level is excluded from completed-run aggregates.'}
                env = Ls20Scenario(level, spec['training_context_index'])
                run = rollout(policy, env, 200, args.device, optimum, 'repeat', temperature=0.)
                elapsed = timer.durations[first:]
                if len(elapsed) != run['actions']: raise ValueError('decision/action census mismatch')
                run.update(seed=spec['seed'], difficulty=spec['difficulty'], context=spec['training_context_index'],
                    losses=3 - run['lives_left'], decision_count=len(elapsed), decision_seconds=sum(elapsed),
                    decision_median_seconds=float(np.median(elapsed)) if elapsed else None)
                entry['runs'].append(run)
                entry.pop('incomplete_level', None)
                entry['summary'] = summarize(entry['runs']); persist()
                if (index + 1) % 10 == 0: print(json.dumps({'arm': arm, 'levels': index + 1, **entry['summary']['all']}), flush=True)
            timer.close(); timer = None
            entry['status'] = 'complete'; entry['decision_seconds'] = sum(r['decision_seconds'] for r in entry['runs'])
            verify_sources(sources); persist()
            del policy; policy = None; gc.collect()
            if args.device == 'cuda': torch.cuda.empty_cache()
        report['paired'] = {'static_vs_actor': paired(report['arms']['static']['runs'], report['arms']['actor']['runs']),
            'evolving_vs_actor': paired(report['arms']['evolving']['runs'], report['arms']['actor']['runs']),
            'evolving_vs_static': paired(report['arms']['evolving']['runs'], report['arms']['static']['runs'])}
        verify_sources(sources); report.update(status='complete', sources_unchanged=True)
    except BaseException as error:
        report.update(status='failed_partial', error=str(error)); raise
    finally:
        if timer is not None: timer.close()
        policy = None; gc.collect()
        if args.device == 'cuda' and torch.cuda.is_available(): torch.cuda.empty_cache()
        signal.alarm(0); persist()


if __name__ == '__main__': main()
