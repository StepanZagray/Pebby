"""Evaluate a pinned navigation experiment, or retained spatial baseline.

Generated bank evaluation and the sequential shipped game have separate reports.
Confirmation is available only here, never in the training loop. A confirmation
panel becomes exposed after use; a reused panel is development evidence.
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time

import torch

from tools.train_navigation_probe import (Budget, digest, evaluate_roots, evaluate_rollouts,
                                         source_hashes, write_json)


def load_policy(path, device):
    from pebby.agent import navigation_probe, spatial_outcome_policy
    envelope = torch.load(path, map_location='cpu', weights_only=True)
    load = (spatial_outcome_policy.load_checkpoint if envelope.get('format') == spatial_outcome_policy.FORMAT
            else navigation_probe.load_checkpoint)
    return load(path, device=device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--bank-sha256')
    parser.add_argument('--split', choices=('train', 'development', 'confirmation'), default='development')
    parser.add_argument('--sequential', action='store_true')
    parser.add_argument('--per-level-cap', type=int, default=300)
    parser.add_argument('--rollout-cap', type=int, default=40)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-seconds', type=float, default=180)
    args = parser.parse_args(argv)
    if args.device == 'cpu':
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU evaluation isolation requires a fresh process without initialized CUDA')
        os.environ['CUDA_VISIBLE_DEVICES']=''
    if any(not math.isfinite(getattr(args,k)) or getattr(args,k)<=0
           for k in ('per_level_cap','rollout_cap','threads','batch_size','max_seconds')):
        parser.error('budgets and batch sizes must be positive')
    if args.sequential and (args.bank or args.bank_sha256):
        parser.error('use separate evaluations for the generated bank and shipped sequential game')
    if not args.sequential and (not args.bank or not args.bank_sha256):
        parser.error('generated evaluation requires --bank and --bank-sha256')
    if digest(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('checkpoint SHA256 differs')
    if args.bank and digest(args.bank) != args.bank_sha256:
        raise ValueError('bank SHA256 differs')
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    guard = Budget(args.max_seconds)
    report = dict(format='pebby.navigation-evaluation.v1', status='running', pid=os.getpid(),
                  started_local=datetime.now().astimezone().isoformat(),
                  checkpoint=str(args.checkpoint), checkpoint_sha256=args.checkpoint_sha256,
                  bank_sha256=args.bank_sha256, split=None if args.sequential else args.split,
                  source_sha256=source_hashes(), training=False,
                  confirmation_exposure=False,
                  limits=['Local diagnostic, not an official API scorecard.',
                          'Shipped levels are a repeatedly inspected target, not untouched generalization evidence.',
                          'No learned voluntary RESET, persistent memory or multistep search in these policies.'])
    write_json(args.out, report)
    print(json.dumps(dict(event='started',pid=os.getpid(),out=str(args.out))),flush=True)
    try:
        guard()
        torch.set_num_threads(args.threads)
        policy, metadata = load_policy(args.checkpoint,args.device)
        report['model_metadata'] = {k:v for k,v in metadata.items()
                                    if k not in ('encoder_weights','planner_weights','weights')}
        if args.sequential:
            from pebby.agent.competition import CompetitionSession, FourMovementDecision, run_competition
            class BoundedDecision(FourMovementDecision):
                def decide(self, context):
                    guard()
                    return super().decide(context)
            session = CompetitionSession()
            report['sequential'] = run_competition(BoundedDecision(policy,args.device),session,
                                                  per_level_caps=[args.per_level_cap]*session.level_count)
        else:
            from pebby.agent.navigation_diagnostics import collect_examples, public_frame_overlap
            bank = json.loads(args.bank.read_text())
            if bank.get('format') != 'pebby.navigation-bank.v1':
                raise ValueError('unsupported navigation bank format')
            overlap = public_frame_overlap(bank['splits'])
            if any(overlap.values()):
                raise ValueError('navigation bank has cross-split public-frame overlap')
            cases = bank['splits'][args.split]
            report['confirmation_exposure'] = args.split == 'confirmation'
            # Record exposure before scoring so a failed partial run cannot
            # accidentally be treated as an untouched confirmation panel.
            write_json(args.out,report)
            arrays = collect_examples(cases,history=policy.config().get('history',8),guard=guard)
            report['roots'] = evaluate_roots(policy,arrays,cases,args.device,args.batch_size,guard)
            report['rollouts'] = evaluate_rollouts(policy,cases,args.device,args.rollout_cap,guard)
            report['confirmation_exposure'] = args.split == 'confirmation'
        if (digest(args.checkpoint) != args.checkpoint_sha256
                or (args.bank and digest(args.bank) != args.bank_sha256)
                or any(digest(p)!=h for p,h in report['source_sha256'].items())):
            raise RuntimeError('model, bank or source changed during evaluation')
        report.update(status='complete',source_unchanged=True,checkpoint_unchanged=True)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report.update(finished_local=datetime.now().astimezone().isoformat(),
                      elapsed_seconds=time.monotonic()-guard.started,
                      cuda_initialized=torch.cuda.is_initialized())
        write_json(args.out,report)
    print(json.dumps(dict(status=report['status'],out=str(args.out))),flush=True)
    return report


if __name__ == '__main__':
    main()
