"""Probe ladder: a development gate for one checkpoint, not a benchmark.

Runs two cheap, engine-verified primitive probes on CPU and judges them
against thresholds:

1. the empty-room navigation probe (``pebby.agent.navigation_diagnostics``,
   smallest configuration: one group per split, development split, all three
   stages) scored by first-action accuracy per case;
2. the cycler spoil probe (``pebby.agent.cycler_probe``) scored by leave
   accuracy and the leave/avoid spoil rate.

Both probes are controlled synthetic primitives with related, non-independent
cases. Passing them says a checkpoint has not lost basic navigation and
cycler discipline; it says nothing about shipped-level scores or held-out
generalization. Use it to reject regressions early, never as evidence of
capability. Thresholds are CLI flags; the exit code is 1 on any failure.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pebby.agent import cycler_probe  # noqa: E402
from pebby.agent.model import load_checkpoint  # noqa: E402
from pebby.agent.navigation_diagnostics import collect_examples, make_cases  # noqa: E402
from tools.train_navigation_probe import evaluate_roots  # noqa: E402

FORMAT = 'pebby.probe-ladder.v1'
DEFAULT_THRESHOLDS = dict(min_empty_room_accuracy=0.75, min_leave_accuracy=0.8, max_spoil_rate=0.1)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def deadline(seconds):
    started = time.monotonic()

    def guard():
        if time.monotonic() - started >= seconds:
            raise TimeoutError('probe ladder wall-clock budget exhausted')
    return guard


def run_navigation(policy, *, seed, groups, device, batch_size, guard):
    """Smallest empty-room probe: one group per split, scored on development."""
    splits = make_cases(seed=seed, groups_per_split=groups, guard=guard)
    cases = splits['development']
    arrays = collect_examples(cases, history=policy.config().get('history', 8), guard=guard)
    roots = evaluate_roots(policy, arrays, cases, device, batch_size, guard)
    per_case = roots['per_case']
    first = float(np.mean([c['first_correct'] for c in per_case])) if per_case else None
    per_stage = {stage: values['first_accuracy'] for stage, values in roots['per_stage'].items()}
    return dict(split='development', groups_per_split=groups, cases=len(cases),
                first_action_accuracy=first, first_action_accuracy_per_stage=per_stage,
                micro_accuracy=roots['micro_accuracy'], random_expected=roots['random_expected'],
                per_case=per_case, limits=roots['limits'])


def judge(navigation, cycler, thresholds):
    """Gate rows; ``value`` None counts as a failure."""
    leave = cycler['per_type']['leave']
    rows = [dict(name='empty_room_first_action_accuracy', value=navigation['first_action_accuracy'],
                 comparison='>=', threshold=thresholds['min_empty_room_accuracy']),
            dict(name='cycler_leave_accuracy', value=leave['accuracy'],
                 comparison='>=', threshold=thresholds['min_leave_accuracy']),
            dict(name='cycler_spoil_rate', value=cycler['overall']['spoil_rate'],
                 comparison='<=', threshold=thresholds['max_spoil_rate'])]
    for row in rows:
        value = row['value']
        row['passed'] = (value is not None and
                         (value >= row['threshold'] if row['comparison'] == '>=' else value <= row['threshold']))
    return rows


def format_table(rows, informative):
    def fmt(value):
        return '   -' if value is None else f'{value:.3f}'
    lines = [f"{'check':<36}{'value':>8}{'':>4}{'threshold':>10}  result",
             '-' * 68]
    for row in rows:
        lines.append(f"{row['name']:<36}{fmt(row['value']):>8}{row['comparison']:>4}"
                     f"{fmt(row['threshold']):>10}  {'PASS' if row['passed'] else 'FAIL'}")
    lines.append('-' * 68)
    for name, value in informative.items():
        lines.append(f"{name:<36}{fmt(value):>8}      (informative)")
    return '\n'.join(lines)


def run_ladder(policy, *, seed=0, count=96, nav_groups=1, device='cpu', batch_size=16,
               thresholds=None, guard=None, kinds=cycler_probe.KINDS):
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    guard = guard or (lambda: None)
    started = time.monotonic()
    navigation = run_navigation(policy, seed=seed, groups=nav_groups, device=device,
                                batch_size=batch_size, guard=guard)
    navigation_seconds = time.monotonic() - started
    cases = cycler_probe.build_cases(seed, count, kinds, guard=guard)
    cycler = cycler_probe.evaluate(policy, cases, device=device, batch_size=batch_size)
    cycler['cases'] = cycler_probe.case_manifest(cases)
    cycler_seconds = time.monotonic() - started - navigation_seconds
    checks = judge(navigation, cycler, thresholds)
    informative = {'cycler_approach_accuracy': cycler['per_type']['approach']['accuracy'],
                   'cycler_avoid_accuracy': cycler['per_type']['avoid']['accuracy'],
                   'cycler_leave_spoil_rate': cycler['per_type']['leave']['spoil_rate'],
                   'cycler_avoid_spoil_rate': cycler['per_type']['avoid']['spoil_rate'],
                   'cycler_mean_spoil_probability': cycler['overall']['mean_spoil_probability'],
                   'empty_room_micro_accuracy': navigation['micro_accuracy']}
    return dict(format=FORMAT, gate='development gate, not a benchmark', thresholds=thresholds,
                checks=checks, passed=all(row['passed'] for row in checks), informative=informative,
                navigation=navigation, cycler=cycler, seed=seed, count=count, nav_groups=nav_groups,
                timings=dict(navigation_seconds=navigation_seconds, cycler_seconds=cycler_seconds,
                             total_seconds=time.monotonic() - started))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--count', type=int, default=96, help='cycler rooms (three cases each)')
    parser.add_argument('--nav-groups', type=int, default=1, help='empty-room groups per split')
    parser.add_argument('--min-empty-room-accuracy', type=float, default=DEFAULT_THRESHOLDS['min_empty_room_accuracy'])
    parser.add_argument('--min-leave-accuracy', type=float, default=DEFAULT_THRESHOLDS['min_leave_accuracy'])
    parser.add_argument('--max-spoil-rate', type=float, default=DEFAULT_THRESHOLDS['max_spoil_rate'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=0, help='torch CPU threads; 0 keeps the default')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-seconds', type=float, default=600)
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    thresholds = dict(min_empty_room_accuracy=args.min_empty_room_accuracy,
                      min_leave_accuracy=args.min_leave_accuracy, max_spoil_rate=args.max_spoil_rate)
    policy, _ = load_checkpoint(args.checkpoint, device=args.device)
    report = run_ladder(policy, seed=args.seed, count=args.count, nav_groups=args.nav_groups,
                        device=args.device, batch_size=args.batch_size, thresholds=thresholds,
                        guard=deadline(args.max_seconds))
    report.update(checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
                  finished_local=datetime.now().astimezone().isoformat())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'probe ladder (development gate, not a benchmark): {args.checkpoint}')
    print(format_table(report['checks'], report['informative']))
    print(f"overall: {'PASS' if report['passed'] else 'FAIL'}   wrote {args.out}")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
