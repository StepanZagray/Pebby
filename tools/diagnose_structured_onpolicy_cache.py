"""Frozen head diagnosis on generated TRAIN visited states and expert anchors.

Actual successors are privileged diagnostics; imagined successors use only
public current fields and action IDs. This tool never optimizes parameters.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.model import load_checkpoint
from tools.build_structured_onpolicy_field_cache import FORMAT, check_hashes
from tools.train_structured_policy import check_policy_encoder, policy_terms
from tools.train_structured_transition import atomic_json, digest


def summarize(logits, masks, seeds, unsafe):
    terms = policy_terms(torch.from_numpy(logits), torch.from_numpy(masks).long())
    choices = logits.argmax(-1)
    correct = terms['correct'].numpy()
    unique = np.unique(seeds)
    return {'states': len(seeds), 'levels': len(unique),
            'optimal_choices': int(correct.sum()), 'optimal_rate': float(correct.mean()),
            'level_mean_optimal_rate': float(np.mean([correct[seeds == s].mean() for s in unique])),
            'mean_ce': float(terms['ce'].mean()),
            'chosen_unsafe': int(unsafe[np.arange(len(seeds)), choices].sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'checkpoint', 'report'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--seconds', type=int, default=180)
    args = p.parse_args()
    if args.report.exists() or not 1 <= args.seconds <= 600:
        p.error('new report and deadline1..600 required')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    print('PID', os.getpid(), flush=True)
    def expired(*_):
        raise TimeoutError('cache diagnostic deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.seconds)
    report = {'status': 'running', 'pid': os.getpid(), 'source': 'generated_only', 'split': 'train',
              'official_inputs_used': False, 'optimization_performed': False,
              'limits': ['TRAIN diagnostic, not validation or gameplay completion.',
                         'Actual-successor scoring is privileged and cannot be deployed.',
                         'Cached imagined fields include FP16 quantization. Unsafe means life loss or no current-life route.']}
    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        atomic_json(args.report, report)
    persist()
    try:
        path = args.cache / 'manifest.json'
        manifest = json.loads(path.read_text())
        if (manifest.get('format') != FORMAT or manifest.get('status') != 'complete'
                or manifest.get('split') != 'train' or manifest.get('source') != 'generated_only'):
            raise ValueError('completed generated TRAIN on-policy cache required')
        hashes = dict(manifest['source_hashes'])
        hashes.update({str(path): digest(path), str(args.checkpoint): digest(args.checkpoint),
                       __file__: digest(__file__), 'tools/train_structured_policy.py': digest('tools/train_structured_policy.py')})
        check_hashes(hashes)
        data = {}
        for name, info in manifest['arrays'].items():
            file = args.cache / (name + '.npy')
            hashes[str(file)] = info['sha256']
            if digest(file) != info['sha256']:
                raise ValueError('cache array hash mismatch: ' + name)
            value = np.load(file, mmap_mode='r', allow_pickle=False)
            if list(value.shape) != info['shape'] or str(value.dtype) != info['dtype']:
                raise ValueError('cache array shape/dtype mismatch: ' + name)
            data[name] = value
        policy, info = load_checkpoint(args.checkpoint, args.device)
        policy.float().eval().requires_grad_(False)
        check_policy_encoder(manifest['field_encoder'], policy, True)
        if manifest['actor_sources'] != policy.sources:
            raise ValueError('cached dynamics or encoder differs from public policy')
        report.update(checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
                      parameters=policy.parameter_count(), cache=str(args.cache), source_hashes=hashes)
        scores = {}
        with torch.inference_mode():
            for view, key in [('actual', 'next_fields'), ('imagined', 'imagined_fields')]:
                outputs = []
                for first in range(0, len(data['seeds']), 16):
                    fields = torch.as_tensor(np.array(data[key][first:first + 16]), device=args.device).float()
                    outputs.append(policy.readout(fields).float().cpu().numpy())
                scores[view] = np.concatenate(outputs)
        unsafe = (data['distances'] < 0) | data['lost_life']
        report['groups'] = {}
        for name, flag in [('on_policy', data['on_policy']), ('expert', ~data['on_policy'])]:
            if not flag.any():
                continue
            report['groups'][name] = {view: summarize(logits[flag], data['optimal'][flag], data['seeds'][flag], unsafe[flag])
                                      for view, logits in scores.items()}
            report['groups'][name]['action_disagreement'] = int((scores['actual'][flag].argmax(-1) != scores['imagined'][flag].argmax(-1)).sum())
        check_hashes(hashes)
        report.update(status='complete', source_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        signal.alarm(0)
        persist()


if __name__ == '__main__':
    main()
