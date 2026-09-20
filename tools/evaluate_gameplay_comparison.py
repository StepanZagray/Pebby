"""CPU-only paired neural checkpoint comparison using actual gameplay.

The first checkpoint is the baseline. Each arm plays one continuous shipped
seven-level session, then the same70 generated validation levels. These are
development panels, not untouched generalization tests or official scorecards.
No offline fidelity, stored route, oracle, fallback action or automatic promotion.
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
import tempfile
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

from pebby.agent import evaluate, gameplay_gate
from pebby.agent.neural_outcome_policy import weights_sha256
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.generate import build_level
from pebby.ls20.provenance import generated_context
from tools.evaluate_navigation_probe import load_policy
from tools.evaluate_reference_spatial_outcomes import guard as memory_guard, summary
from tools.evaluate_spatial_recovery import paired, validate_specs

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'pebby.gameplay-comparison.v1'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_report(path, report):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.gameplay-report-',
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.replace(path)  # Only the report exclusively created by this run.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def checked_specs(path):
    specs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    validate_specs(specs)
    if len(specs) != 70 or Counter(spec['difficulty'] for spec in specs) != Counter({tier: 10 for tier in range(1, 8)}):
        raise ValueError('require exactly70 generated validation levels, ten per tier1..7')
    for spec in specs:
        context = generated_context(spec)
        if spec.get('geometry_split', 'validation') != 'validation':
            raise ValueError('generated bank must use validation geometry')
        for key in ('context_index', 'training_context_index'):
            if key in spec and (type(spec[key]) is not int or spec[key] != context):
                raise ValueError('stored generated context disagrees with versioned context')
    return specs


class CheckedPolicy:
    """Preserve each family's public history API; validate every neural output."""
    def __init__(self, policy, guard):
        self.policy, self.guard = policy, guard

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def eval(self):
        if hasattr(self.policy, 'eval'):
            self.policy.eval()
        return self

    def __call__(self, *args, **kwargs):
        self.guard()
        scores = self.policy(*args, **kwargs)
        if (not isinstance(scores, torch.Tensor) or scores.ndim != 2 or scores.shape[1] != 4
                or not bool(torch.isfinite(scores).all())):
            raise ValueError('policy must return finite [B,4] neural movement logits')
        self.guard()
        return scores


class GuardedEnvironment:
    def __init__(self, env, guard):
        self.env, self.guard = env, guard

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        self.guard()
        frame = self.env.reset()
        self.guard()
        return frame

    def perform(self, action):
        self.guard()
        observation = self.env.perform(action)
        self.guard()
        if observation.frame is None and not observation.finished:
            raise RuntimeError('nonterminal transition has no public observation')
        return observation


def generated_rollout(policy, spec, guard):
    guard()
    context = generated_context(spec)
    env = GuardedEnvironment(Ls20Scenario(build_level(spec), context), guard)
    result = evaluate.rollout(CheckedPolicy(policy, guard), env, 300, torch.device('cpu'),
                              oracle_length=None, on_stall='repeat', temperature=0.)
    guard()  # A deadline after WIN still propagates; no partial game becomes a record.
    result.update(seed=spec['seed'], difficulty=spec['difficulty'], context_index=context,
                  evaluation_finished=True)
    return result


def refresh(report):
    for arm in report['arms'].values():
        runs = arm['runs']
        arm['generated_summary'] = {**summary(runs),
                                    'unrecorded_levels': report['bank_levels'] - len(runs),
                                    'failures': sum(not run['completed'] for run in runs)}
        arm['per_tier'] = {str(tier): {
            **summary([run for run in runs if run['difficulty'] == tier]),
            'failure_seeds': [run['seed'] for run in runs if run['difficulty'] == tier and not run['completed']]}
            for tier in range(1, 8)}
    baseline = report['arms'].get(report['baseline'])
    if baseline is not None:
        for name, arm in report['arms'].items():
            if name == report['baseline']:
                continue
            report['paired'][name] = paired(baseline['runs'], arm['runs'])
            report['paired'][name]['per_tier'] = {
                str(tier): paired([run for run in baseline['runs'] if run['difficulty'] == tier],
                                  [run for run in arm['runs'] if run['difficulty'] == tier])
                for tier in range(1, 8)}
            report['gameplay_gate'][name] = gameplay_gate.assess_gameplay(baseline.get('sequential'),
                                                                         arm.get('sequential'))


def source_paths():
    from arcengine import base_game
    return [Path(__file__), Path(base_game.__file__), ROOT / 'third_party/ls20/ls20.py',
            *[ROOT / 'tools' / name for name in ('evaluate_navigation_probe.py', 'train_navigation_probe.py',
                                                'evaluate_spatial_recovery.py', 'evaluate_reference_spatial_outcomes.py')],
            *sorted((ROOT / 'pebby/agent').glob('*.py')), *sorted((ROOT / 'pebby/ls20').glob('*.py'))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', nargs=3, metavar=('NAME', 'PATH', 'SHA256'), required=True)
    parser.add_argument('--bank', type=Path, default=ROOT / 'artifacts/reference-grounding-repair-v1/validation70.jsonl')
    parser.add_argument('--bank-sha256')
    parser.add_argument('--report-out', type=Path, required=True)
    parser.add_argument('--sequential-only', action='store_true')
    parser.add_argument('--max-seconds', type=int, default=1800)
    args = parser.parse_args(argv)
    if not 1 <= args.max_seconds <= 1800:
        parser.error('max-seconds must be in1..1800')
    names = [name for name, _, _ in args.checkpoint]
    if len(set(names)) != len(names) or any(not name.strip() for name in names):
        parser.error('checkpoint names must be nonempty and distinct; baseline first')
    if not args.sequential_only and args.bank_sha256 is None:
        parser.error('--bank-sha256 is required unless --sequential-only')
    if args.report_out.exists() or args.report_out.is_symlink():
        raise FileExistsError(args.report_out)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU evaluation requires a fresh process without initialized CUDA')
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError('evaluation requires ownership of the process real-time alarm')
    torch.set_num_threads(1)
    started = time.monotonic()
    report = dict(format=FORMAT, status='running', pid=os.getpid(),
                  start_ticks=int(Path('/proc/self/stat').read_text().rpartition(')')[2].split()[19]),
                  started_local=datetime.now().astimezone().isoformat(),
                  baseline=names[0], arms={}, paired={}, gameplay_gate={}, bank_levels=0,
                  training=False, oracle_calls=0, stored_solutions_used=False, promoted=False,
                  sequential_only=args.sequential_only, max_seconds=args.max_seconds,
                  shipped_gameplay_used_for_development_comparison=True,
                  untouched_test_or_generalization_claim=False,
                  runtime=dict(device='cpu', threads=1, precision='float32', cuda_visible_devices=''),
                  generated_protocol=dict(on_stall='repeat', temperature=0., max_actions=300,
                                          native_lives=3, reset_extension=False),
                  limitations=['Local gameplay, not an official API scorecard.',
                               'Shipped and generated validation panels are reused development evidence.',
                               'Interrupted and unplayed games are unrecorded, never counted as losses.',
                               'A completed300-action capped episode is a recorded protocol failure.',
                               'Generated games stop at GAME_OVER; shipped sessions allow charged current-level RESET.',
                               'No automatic promotion; paired gains require further gameplay confirmation.'])

    def guard():
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError('gameplay comparison deadline reached')
        available = memory_guard()
        report['minimum_memavailable_bytes'] = min(report.get('minimum_memavailable_bytes', available), available)

    def timeout(*_):
        raise TimeoutError('gameplay comparison hard deadline reached')

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open('x') as stream:
        json.dump(report, stream, indent=2)
    previous_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    policy = None
    try:
        guard()
        bound = [(Path(path), sha) for _, path, sha in args.checkpoint]
        if not args.sequential_only:
            bound.append((args.bank, args.bank_sha256))
        for path, expected in bound:
            if not re.fullmatch('[0-9a-fA-F]{64}', expected) or digest(path) != expected.lower():
                raise ValueError(f'exact SHA256 mismatch: {path}')
        bindings = {str(path.resolve()): digest(path) for path in [*source_paths(), *[path for path, _ in bound]]}
        report['source_sha256'] = bindings
        specs = [] if args.sequential_only else checked_specs(args.bank)
        report.update(bank=None if args.sequential_only else str(args.bank.resolve()),
                      bank_sha256=None if args.sequential_only else args.bank_sha256.lower(),
                      bank_levels=len(specs), selected_seeds=[spec['seed'] for spec in specs])
        write_report(args.report_out, report)
        print(json.dumps(dict(event='started', pid=os.getpid(), report=str(args.report_out))), flush=True)
        for name, path, expected in args.checkpoint:
            guard()
            arm = dict(status='running', checkpoint=str(Path(path).resolve()), checkpoint_sha256=expected.lower(),
                       phase='loading', sequential=None, runs=[])
            report['arms'][name] = arm
            policy, metadata = load_policy(path, 'cpu')
            arm.update(checkpoint_format=metadata.get('format'), config=policy.config(),
                       initial_weights_sha256=weights_sha256(policy.state_dict()), phase='sequential')
            arm['sequential'] = gameplay_gate.evaluate_sequential(CheckedPolicy(policy, guard), 'cpu', guard, 300)
            refresh(report)
            write_report(args.report_out, report)
            print(json.dumps(dict(event='sequential', arm=name, levels_completed=arm['sequential']['levels_completed'])), flush=True)
            arm['phase'] = 'generated'
            for spec in specs:
                arm['active_seed'] = spec['seed']
                run = generated_rollout(policy, spec, guard)
                arm['runs'].append(run)
                arm.pop('active_seed')
                refresh(report)
                write_report(args.report_out, report)
                if len(arm['runs']) % 10 == 0:
                    print(json.dumps(dict(event='generated', arm=name, recorded=len(arm['runs']),
                                          completed=arm['generated_summary']['completed'])), flush=True)
            if weights_sha256(policy.state_dict()) != arm['initial_weights_sha256']:
                raise RuntimeError('policy weights changed during gameplay')
            arm.update(status='complete', phase='complete', weights_unchanged=True)
            del policy
            policy = None
            gc.collect()
        guard()
        if any(digest(path) != expected for path, expected in bindings.items()):
            raise RuntimeError('bound source, checkpoint or bank changed during evaluation')
        report.update(status='complete', source_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        for arm in report['arms'].values():
            if arm['status'] == 'running':
                arm['status'] = 'partial_failed'
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        del policy
        gc.collect()
        refresh(report)
        report.update(elapsed_seconds=time.monotonic() - started,
                      finished_local=datetime.now().astimezone().isoformat(),
                      cuda_initialized=torch.cuda.is_initialized(),
                      process_cleanup='No child processes; launcher must verify this exact PID exits.')
        write_report(args.report_out, report)
    return report


if __name__ == '__main__':
    main()
