"""Bounded 70-level generated gameplay screen for the spatial outcome policy.

This is isolated generated-level evaluation, with one native three-life episode
per level. It is not a sequential official game or an official benchmark score.
"""

import argparse
from collections import Counter
from datetime import datetime, timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, report):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def guard():
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    if available < 7 * 2**30:
        raise MemoryError('7 GiB abort threshold protects the required 6 GiB host reserve')
    return available


def summary(runs):
    return dict(levels=len(runs), completed=sum(r['completed'] for r in runs),
                actions=sum(r['actions'] for r in runs), stalls=sum(r['stalls'] for r in runs),
                goals_cleared=sum(r['goals_cleared'] for r in runs), goals_total=sum(r['goals_total'] for r in runs),
                lives_lost=sum(3 - r['lives_left'] for r in runs), endings=dict(Counter(r['ending'] for r in runs)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--bank', type=Path, default=ROOT / 'artifacts/reference-grounding-repair-v1/validation70.jsonl')
    parser.add_argument('--bank-sha256', required=True)
    parser.add_argument('--report-out', type=Path, required=True)
    parser.add_argument('--direct-weight', type=float, default=0.)
    parser.add_argument('--planner-weight', type=float, default=1.)
    parser.add_argument('--max-actions', type=int, default=300)
    parser.add_argument('--max-seconds', type=int, default=600)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    if min(args.max_actions, args.max_seconds) <= 0:
        parser.error('positive time and action bounds required')
    if any(not math.isfinite(w) or w < 0 for w in (args.direct_weight, args.planner_weight)) or args.direct_weight + args.planner_weight <= 0:
        parser.error('score weights must be finite, nonnegative and not both zero')
    for name in ('checkpoint_sha256', 'bank_sha256'):
        if not re.fullmatch('[0-9a-fA-F]{64}', getattr(args, name)):
            parser.error(f'--{name.replace("_", "-")} requires an exact SHA256')
    if args.report_out.exists():
        raise FileExistsError(args.report_out)
    if sha(args.checkpoint) != args.checkpoint_sha256.lower() or sha(args.bank) != args.bank_sha256.lower():
        raise ValueError('checkpoint or bank hash differs from the required exact hash')
    specs = [json.loads(line) for line in args.bank.read_text().splitlines() if line.strip()]
    if (len(specs) != 70 or len({s['seed'] for s in specs}) != 70
            or Counter(s.get('difficulty') for s in specs) != Counter({tier: 10 for tier in range(1, 8)})
            or any(s.get('source') != 'generated_only' for s in specs)):
        raise ValueError('require 70 distinct generated-only levels, ten per difficulty 1..7')
    sources = [Path(__file__).resolve(), args.checkpoint.resolve(), args.bank.resolve(),
               *[ROOT / 'pebby/agent' / name for name in ('spatial_outcome_planner.py', 'spatial_outcome_policy.py',
                  'neural_outcome_policy.py', 'neural_outcome_planner.py', 'world_model.py', 'world_runtime.py',
                  'glyph_model.py', 'world_readout.py', 'evaluate.py', 'history.py', 'model.py', 'looped.py')],
               *[ROOT / 'pebby/ls20' / name for name in ('env.py', 'generate.py', 'bank.py', 'names.py', 'provenance.py')],
               ROOT / 'third_party/ls20/ls20.py']
    bindings = {str(path): sha(path) for path in sources}
    started = time.monotonic()
    local_start = datetime.now().astimezone()
    report = dict(status='running', pid=os.getpid(),
        start_ticks=int(Path('/proc/self/stat').read_text().rpartition(')')[2].split()[19]),
        started_local=local_start.isoformat(), deadline_local=(local_start + timedelta(seconds=args.max_seconds)).isoformat(),
        checkpoint_sha256=args.checkpoint_sha256.lower(), bank_sha256=args.bank_sha256.lower(),
        source_sha256=bindings, official_inputs_used=False, training=False, oracle_calls=0,
        protocol=dict(kind='isolated_generated_levels', max_actions=args.max_actions, on_stall='repeat',
                      temperature=0., native_lives=3, reset_extension=False),
        runtime=dict(device=args.device, precision='float32', execution='native_eager', temporal_backend='auto',
                     matmul_tf32=False, cudnn_tf32=True, history=8),
        decision=dict(planner_weight=args.planner_weight, direct_weight=args.direct_weight, old_ranker_weight=0.,
                      planner_horizon=1, planner_refinement_loops=1, learned_voluntary_reset=False),
        limitations=['Isolated generated levels; not a sequential official game or official score.',
                     'History clears at native life loss; persistent game memory is not implemented.'], runs=[])
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open('x') as handle:
        json.dump(report, handle, indent=2)
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError('bounded generated gameplay deadline reached')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    model = None
    try:
        report['minimum_memavailable_bytes'] = guard()
        if args.device == 'cuda':
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                    check=True, capture_output=True, text=True, timeout=10)
            foreign = [int(p) for p in result.stdout.splitlines() if p.strip() and int(p) != os.getpid()]
            if foreign:
                raise RuntimeError(f'foreign CUDA compute processes active: {foreign}')
        import torch
        from pebby.agent import evaluate
        from pebby.agent.spatial_outcome_policy import load_checkpoint
        from pebby.ls20.env import Ls20Scenario
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('highest')
        model, metadata = load_checkpoint(args.checkpoint, args.device, direct_weight=args.direct_weight,
                                          planner_weight=args.planner_weight)
        if model.config()['history'] != 8 or metadata.get('official_training_inputs') is not False:
            raise ValueError('generated-only H8 spatial checkpoint required')
        report['checkpoint_format'] = metadata['format']
        report['checkpoint_config'] = model.config()
        levels, optima, loaded = evaluate.bank_levels(args.bank)
        if loaded != specs:
            raise ValueError('bank changed while loading')
        with torch.inference_mode():
            for index, (level, optimal, spec) in enumerate(zip(levels, optima, specs)):
                report['minimum_memavailable_bytes'] = min(report['minimum_memavailable_bytes'], guard())
                run = evaluate.rollout(model, Ls20Scenario(level, int(spec.get('training_context_index', 0))),
                                       args.max_actions, torch.device(args.device), optimal, on_stall='repeat', temperature=0.)
                run.update(run=index, seed=spec['seed'], difficulty=spec['difficulty'])
                report['runs'].append(run)
                if (index + 1) % 10 == 0 or index + 1 == len(specs):
                    report['summary'] = summary(report['runs'])
                    report['elapsed_seconds'] = time.monotonic() - started
                    write(args.report_out, report)
                    print(json.dumps(dict(pid=os.getpid(), event='progress', **report['summary'])), flush=True)
        report['per_tier'] = {str(t): summary([r for r in report['runs'] if r['difficulty'] == t]) for t in range(1, 8)}
        if any(sha(path) != expected for path, expected in bindings.items()):
            raise ValueError('source or checkpoint changed during gameplay')
        report.update(status='complete', sources_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        del model
        gc.collect()
        if 'torch' in locals() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        report.update(elapsed_seconds=time.monotonic() - started, finished_local=datetime.now().astimezone().isoformat(),
                      process_cleanup='No persistent children; policy references released; launcher verifies PID exit.')
        write(args.report_out, report)
    return report


if __name__ == '__main__':
    main()
