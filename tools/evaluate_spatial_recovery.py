"""Paired generated validation gameplay with passive recovery diagnostics.

No oracle, action mask, reset extension, or planning-depth change is used.
Each checkpoint plays independent native three-life episodes under strict argmax.
"""
import argparse
from collections import Counter
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import numpy as np
import torch

from pebby.agent import evaluate
from pebby.agent.neural_outcome_policy import weights_sha256
from pebby.agent.spatial_outcome_policy import load_checkpoint
from pebby.ls20.env import Ls20Scenario
from tools.evaluate_reference_spatial_outcomes import guard, sha, summary, write

ROOT = Path(__file__).resolve().parents[1]


def input_digest(args, kwargs):
    """Hash the exact public tensor inputs, including validity and action history."""
    digest = hashlib.sha256()
    for name, value in [('frames', args[0]), *sorted(kwargs.items())]:
        digest.update(name.encode())
        if value is None:
            digest.update(b'None')
        else:
            value = value.detach().cpu().contiguous()
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class RecoveryDiagnostics:
    """Observe policy inputs/outputs and real transitions; never change actions."""
    def __init__(self):
        self.pending = None
        self.events = None
        self.fingerprint = None
        self.last_fingerprint = None
        self.last_action = None
        self.last_unchanged = False
        self.seen = set()
        self.refusal_run = 0
        self.counts = Counter()
        self.refusal_win_sum = 0.
        self.refusal_win_max = None
        self.ever_goals = []
        self.all_goals_simultaneously_observed = False

    def pre_policy(self, module, args, kwargs):
        self.fingerprint = input_digest(args, kwargs)

    def planner_output(self, module, args, output):
        self.events = output['event_logits'][0].detach().float().sigmoid().cpu()

    def policy_output(self, module, args, output):
        if output.shape != (1, 4) or not bool(torch.isfinite(output).all()):
            raise ValueError('expected finite four-action policy scores')
        self.pending = int(output[0].argmax())

    def observe_goals(self, goals, won=False):
        goals = list(map(bool, goals))
        if not self.ever_goals:
            self.ever_goals = [False] * len(goals)
        if len(goals) != len(self.ever_goals):
            raise ValueError('isolated level goal census changed')
        self.ever_goals = [a or b or won for a, b in zip(self.ever_goals, goals)]
        self.all_goals_simultaneously_observed |= bool(goals) and (all(goals) or won)

    def transition(self, action_index, before_frame, observation, old_lives, new_lives, goals):
        if action_index != self.pending or self.events is None or self.fingerprint is None:
            raise ValueError('diagnostics must observe the unchanged greedy policy choice')
        unchanged = observation.frame is not None and np.array_equal(before_frame, observation.frame)
        refusal = unchanged and not observation.finished and new_lives == old_lives
        self.counts['unchanged_actions'] += unchanged
        self.counts['lives_lost'] += max(0, old_lives - new_lives)
        if refusal:
            self.counts['refusal_actions'] += 1
            self.counts['refusal_episodes'] += self.refusal_run == 0
            self.refusal_run += 1
            self.counts['max_consecutive_refusals'] = max(self.counts['max_consecutive_refusals'], self.refusal_run)
            win = float(self.events[action_index, 2])
            self.refusal_win_sum += win
            self.refusal_win_max = win if self.refusal_win_max is None else max(win, self.refusal_win_max)
            self.counts['refusal_predicted_win_ge_0_5'] += win >= .5
            self.counts['refusal_predicted_no_win_lt_0_5'] += win < .5
        else:
            self.counts['refusal_episodes_followed_by_change'] += self.refusal_run > 0 and not unchanged
            self.refusal_run = 0
        self.counts['repeated_public_inputs'] += self.fingerprint in self.seen
        self.counts['exact_fixedpoint_steps'] += bool(
            refusal and self.last_unchanged and self.fingerprint == self.last_fingerprint
            and action_index == self.last_action)
        self.seen.add(self.fingerprint)
        self.last_fingerprint, self.last_action, self.last_unchanged = self.fingerprint, action_index, refusal
        self.observe_goals(goals, observation.won)
        self.pending = None

    def report(self):
        names = ('unchanged_actions', 'lives_lost', 'refusal_actions', 'refusal_episodes',
                 'max_consecutive_refusals', 'refusal_predicted_win_ge_0_5',
                 'refusal_predicted_no_win_lt_0_5', 'refusal_episodes_followed_by_change',
                 'repeated_public_inputs', 'exact_fixedpoint_steps')
        return {**{k: self.counts[k] for k in names},
                'refusal_win_output_sum': self.refusal_win_sum,
                'refusal_win_output_mean': self.refusal_win_sum / self.counts['refusal_actions'] if self.counts['refusal_actions'] else None,
                'refusal_win_output_max': self.refusal_win_max,
                'goals_ever_observed_solved': sum(self.ever_goals),
                'all_goals_ever_observed_solved': bool(self.ever_goals) and all(self.ever_goals),
                'all_goals_simultaneously_observed': self.all_goals_simultaneously_observed}


class ObservedEnvironment:
    def __init__(self, env, diagnostics, check):
        self.env, self.diagnostics, self.check = env, diagnostics, check

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        frame = self.env.reset()
        self.frame = frame
        self.diagnostics.observe_goals(self.env.goals_solved())
        return frame

    def perform(self, action):
        from pebby.ls20.names import ACTION_IDS
        self.check()
        old_lives = self.env.lives()
        observation = self.env.perform(action)
        self.diagnostics.transition(ACTION_IDS.index(action), self.frame, observation,
                                    old_lives, self.env.lives(), self.env.goals_solved())
        self.frame = observation.frame
        return observation


def rollout(policy, env, *, max_actions=300, device='cpu', optimal=None, check=lambda: None):
    diagnostics = RecoveryDiagnostics()
    hooks = [policy.register_forward_pre_hook(diagnostics.pre_policy, with_kwargs=True),
             policy.planner.register_forward_hook(diagnostics.planner_output),
             policy.register_forward_hook(diagnostics.policy_output)]
    try:
        run = evaluate.rollout(policy, ObservedEnvironment(env, diagnostics, check), max_actions,
                               torch.device(device), optimal, on_stall='repeat', temperature=0.)
        run['recovery'] = diagnostics.report()
        return run
    finally:
        for hook in hooks:
            hook.remove()


def aggregate(runs):
    result = summary(runs)
    counts = Counter()
    for run in runs:
        for key, value in run['recovery'].items():
            if isinstance(value, (int, bool)) and key != 'max_consecutive_refusals':
                counts[key] += value
    result['recovery'] = dict(counts)
    result['recovery']['max_consecutive_refusals'] = max((r['recovery']['max_consecutive_refusals'] for r in runs), default=0)
    support = counts['refusal_actions']
    result['recovery']['refusal_win_output_mean'] = sum(r['recovery']['refusal_win_output_sum'] for r in runs) / support if support else None
    return result


def paired(reference, candidate):
    """Only completed per-level records shared by both arms enter comparisons."""
    left = {r['seed']: r for r in reference}
    right = {r['seed']: r for r in candidate}
    if len(left) != len(reference) or len(right) != len(candidate):
        raise ValueError('duplicate paired seeds')
    seeds = sorted(left.keys() & right.keys())
    gains = [s for s in seeds if right[s]['completed'] and not left[s]['completed']]
    losses = [s for s in seeds if left[s]['completed'] and not right[s]['completed']]
    return dict(paired_levels=len(seeds), win_gains=gains, win_losses=losses,
                net_wins=len(gains)-len(losses), reference_only=len(left.keys()-right.keys()),
                candidate_only=len(right.keys()-left.keys()))


def validate_specs(specs):
    if not specs or len({s['seed'] for s in specs}) != len(specs):
        raise ValueError('nonempty distinct generated validation seeds required')
    if any(s.get('source') != 'generated_only' or not isinstance(s['seed'], int)
           or s['seed'] < 1_000_000 or s.get('difficulty') not in range(1, 8) for s in specs):
        raise ValueError('only generated validation levels, never TRAIN or official levels')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', nargs=3, metavar=('NAME', 'PATH', 'SHA256'), required=True)
    parser.add_argument('--bank', type=Path, default=ROOT/'artifacts/reference-grounding-repair-v1/validation70.jsonl')
    parser.add_argument('--bank-sha256', required=True)
    parser.add_argument('--report-out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-seconds', type=int, default=600)
    args = parser.parse_args(argv)
    if args.max_seconds < 1 or args.limit is not None and args.limit < 1:
        parser.error('positive deadline and subset limit required')
    names = [arm[0] for arm in args.checkpoint]
    if len(set(names)) != len(names):
        parser.error('checkpoint arm names must be unique')
    for path, expected in [(args.bank, args.bank_sha256), *[(Path(p), h) for _, p, h in args.checkpoint]]:
        if not re.fullmatch('[0-9a-fA-F]{64}', expected) or sha(path) != expected.lower():
            raise ValueError(f'exact hash mismatch: {path}')
    specs = [json.loads(line) for line in args.bank.read_text().splitlines() if line.strip()]
    validate_specs(specs)
    selected = specs[:args.limit] if args.limit is not None else specs
    paths = [Path(__file__), args.bank, *[Path(p) for _, p, _ in args.checkpoint],
             ROOT/'tools/evaluate_reference_spatial_outcomes.py', ROOT/'third_party/ls20/ls20.py',
             *sorted((ROOT/'pebby/agent').glob('*.py')), *sorted((ROOT/'pebby/ls20').glob('*.py'))]
    bindings = {str(p.resolve()): sha(p) for p in paths}
    started = time.monotonic()
    report = dict(status='running', pid=os.getpid(), started_local=datetime.now().astimezone().isoformat(),
        start_ticks=int(Path('/proc/self/stat').read_text().rpartition(')')[2].split()[19]),
        source_sha256=bindings, bank_sha256=args.bank_sha256.lower(), bank_levels=len(specs),
        selected_seeds=[s['seed'] for s in selected], selected_levels=len(selected),
        complete_bank_selected=len(selected)==len(specs), device=args.device, training=False, oracle_calls=0,
        panel='generated_validation70' if len(selected)==70 and Counter(s['difficulty'] for s in selected)==Counter({t:10 for t in range(1,8)}) else 'generated_validation_subset_or_custom_bank',
        max_seconds=args.max_seconds,
        runtime=dict(precision='float32', execution='native_eager', history=8,
                     temporal_backend='auto', matmul_tf32=False, cudnn_tf32=True),
        official_inputs_used=False, protocol=dict(kind='isolated_generated_validation', max_actions=300,
        on_stall='repeat', temperature=0., native_lives=3, planner_horizon=1, planner_refinement_loops=1,
        learned_voluntary_reset=False, direct_weight=0., planner_weight=1.), arms={}, paired={},
        limits=['Generated isolated levels; never an official sequential seven-level game.',
                'Partial reports compare only completed common level records; no unplayed level is a loss.',
                'Diagnostics are passive; a refusal means unchanged public frame with no life loss or terminal event.',
                'Win outputs are positive-weighted model outputs, not calibrated probabilities.',
                'Leaving a refusal episode does not establish useful progress or learned recovery.',
                'Ever-solved goals may accumulate across lives; final completion and simultaneous goal completion remain separate.',
                'H8 history resets at native life loss; persistent game memory and voluntary reset are absent.'])
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open('x') as stream:
        json.dump(report, stream, indent=2)
    def check():
        if time.monotonic()-started > args.max_seconds:
            raise TimeoutError('evaluation deadline reached')
        available = guard()
        report['minimum_memavailable_bytes'] = min(report.get('minimum_memavailable_bytes', available), available)
    old_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError('evaluation deadline reached')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    policy = None
    def refresh():
        for arm in report['arms'].values():
            arm['summary'] = aggregate(arm['runs'])
            arm['per_tier'] = {str(t): aggregate([r for r in arm['runs'] if r['difficulty']==t]) for t in range(1,8)}
        if names[0] in report['arms']:
            report['paired'] = {n: paired(report['arms'][names[0]]['runs'], a['runs'])
                                for n,a in report['arms'].items() if n != names[0]}
            for name, comparison in report['paired'].items():
                comparison['per_tier'] = {
                    str(t): paired([r for r in report['arms'][names[0]]['runs'] if r['difficulty']==t],
                                   [r for r in report['arms'][name]['runs'] if r['difficulty']==t])
                    for t in range(1,8)}
        report['elapsed_seconds'] = time.monotonic()-started
    try:
        check()
        if args.device == 'cuda':
            processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                       check=True, text=True, capture_output=True, timeout=10)
            if any(int(p) != os.getpid() for p in processes.stdout.splitlines() if p.strip()):
                raise RuntimeError('foreign CUDA compute active')
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('highest')
        levels, optima, loaded = evaluate.bank_levels(args.bank)
        if loaded != specs:
            raise ValueError('bank changed during loading')
        encoder_digest = None
        for name, path, expected in args.checkpoint:
            check()
            policy, metadata = load_checkpoint(path, args.device)
            if (policy.config()['history'] != 8 or policy.direct_weight != 0. or policy.planner_weight != 1.
                    or metadata.get('planner_horizon') != 1 or metadata.get('planner_refinement_loops') != 1):
                raise ValueError('H8 planner-only H1 checkpoint required')
            if encoder_digest is not None and encoder_digest != metadata['encoder_weights_sha256']:
                raise ValueError('paired checkpoints must share the frozen encoder')
            encoder_digest = metadata['encoder_weights_sha256']
            before_weights = weights_sha256(policy.state_dict())
            arm = dict(status='running', checkpoint=str(Path(path).resolve()), checkpoint_sha256=expected.lower(), runs=[])
            report['arms'][name] = arm
            for index, (level, optimum, spec) in enumerate(zip(levels, optima, selected)):
                check()
                run = rollout(policy, Ls20Scenario(level, int(spec.get('training_context_index',0))),
                              device=args.device, optimal=optimum, check=check)
                run.update(run=index, seed=spec['seed'], difficulty=spec['difficulty'])
                arm['runs'].append(run)
                refresh(); write(args.report_out, report)
                if (index+1)%10 == 0 or index+1 == len(selected):
                    print(json.dumps(dict(event='progress', arm=name, recorded_levels=len(arm['runs']),
                                          completed=arm['summary']['completed'])), flush=True)
            if weights_sha256(policy.state_dict()) != before_weights:
                raise ValueError('model weights changed during evaluation')
            arm.update(status='complete', weights_unchanged=True)
            del policy
            policy = None
            gc.collect()
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        if any(sha(path) != expected for path, expected in bindings.items()):
            raise ValueError('bound source, bank, or checkpoint changed')
        report.update(status='complete', sources_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        for arm in report['arms'].values():
            if arm['status'] == 'running':
                arm['status'] = 'partial_failed'
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        del policy
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        refresh()
        report.update(finished_local=datetime.now().astimezone().isoformat(),
                      process_cleanup='No persistent children; launcher must verify exact PID exit.')
        write(args.report_out, report)
    return report


if __name__ == '__main__':
    main()
