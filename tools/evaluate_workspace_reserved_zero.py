"""Generated gameplay ablation of the predicted field's reserved channels.

Keep all weights and public H8 inference unchanged. Immediately before the
action readout, set channels 85:96 of each imagined successor to zero, matching
the actual encoder contract. No actual future or teacher enters inference.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from pebby.agent.evaluate import rollout
from pebby.agent.history import for_policy
from pebby.agent.model import load_checkpoint
from pebby.agent.structured_factored_policy import state_digest
from pebby.agent.world_data import history_arrays
from pebby.ls20.env import Ls20Scenario
from tools.evaluate_structured_workspace_gameplay import BANK, BANK_SHA, checked_bank, paired, summarize, verify_sources
from tools.train_structured_transition import atomic_json, digest


def zero_reserved(fields):
    if fields.ndim != 4 or tuple(fields.shape[1:]) != (4, 148, 96):
        raise ValueError('expected [B,4,148,96] imagined successors')
    result = fields.clone()
    result[..., 85:96] = 0
    return result


class ReservedZero:
    def __init__(self):
        self.calls = 0
        self.nonzero_values = 0
        self.max_abs = 0.

    def __call__(self, module, args):
        if len(args) != 1:
            raise ValueError('expected the public successor-only readout call')
        fields = args[0]
        reserved = fields[..., 85:96]
        self.calls += 1
        self.nonzero_values += int(torch.count_nonzero(reserved))
        self.max_abs = max(self.max_abs, float(reserved.abs().max()))
        return (zero_reserved(fields),)


def checked_baselines(path, specs):
    path = Path(path)
    raw = path.read_bytes()
    report = json.loads(raw)
    expected = dict(status='complete', sources_unchanged=True, source='generated_only',
                    official_inputs_used=False, training_performed=False, device='cuda',
                    precision='FP32; TF32 off', max_actions=200, protocol='strict',
                    on_stall='repeat', temperature=0., primary_depth=2, bank_sha256=BANK_SHA)
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError('completed identical generated CUDA protocol required')
    entries = {'onpolicy': report['arms']['onpolicy'], 'warmstart': report['reused_warmstart']}
    sources = dict(report['sources'])
    verify_sources(sources)
    for name, entry in entries.items():
        checkpoint = entry['checkpoint']
        if digest(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('baseline checkpoint changed')
        runs = entry['runs']
        if len(runs) != len(specs) or len(runs) != 100:
            raise ValueError('complete fixed monitor required')
        for run, spec in zip(runs, specs):
            if (run['seed'] != spec['seed'] or run['context'] != spec['training_context_index']
                    or run['optimal'] != spec['context_optimal_actions']
                    or run['on_stall'] != 'repeat' or run['temperature'] != 0):
                raise ValueError('baseline level/context/protocol mismatch')
        sources[checkpoint['path']] = checkpoint['sha256']
    sha = hashlib.sha256(raw).hexdigest()
    if digest(path) != sha:
        raise ValueError('baseline report changed while checking')
    sources[str(path)] = sha
    return entries, sources


@torch.inference_mode()
def public_wiring_check(policy, frame, device):
    history = for_policy(policy, frame, device)
    original = history.scores().clone()
    frames, valid, actions = history_arrays([frame], [-1], 8)
    current = policy.encoder(torch.from_numpy(frames[None]).long().to(device),
                             torch.from_numpy(valid[None]).bool().to(device),
                             torch.from_numpy(actions[None]).long().to(device))
    successors = policy.successor_fields(current)
    expected = policy.readout(zero_reserved(successors))[0]
    hook = ReservedZero()
    handle = policy.readout.register_forward_pre_hook(hook)
    try:
        actual = history.scores()
    finally:
        handle.remove()
    if not torch.equal(actual, expected) or hook.calls != 1:
        raise ValueError('public ablation does not equal explicit reserved-channel replacement')
    if not torch.equal(history.scores(), original):
        raise ValueError('hook removal changed original public scores')
    return {'public_equals_manual': True, 'original_restored': True,
            'first_state_choice_changed': int(actual.argmax()) != int(original.argmax())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', type=Path, default=Path('artifacts/structured-onpolicy-gameplay100.json'))
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=600)
    args = parser.parse_args()
    if args.report.exists() or not 1 <= args.seconds <= 600:
        parser.error('new report and deadline1..600 required')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    print('PID', os.getpid(), flush=True)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('ablation deadline')))
    signal.alarm(args.seconds)
    report = dict(status='validating', pid=os.getpid(), source='generated_only', split='validation',
                  official_inputs_used=False, training_performed=False, device='cuda', precision='FP32; TF32 off',
                  max_actions=200, protocol='strict', on_stall='repeat', temperature=0., primary_depth=2,
                  intervention='Set only imagined successor channels85:96 tozero immediately before action readout',
                  bank=str(BANK), bank_sha256=BANK_SHA, arms={},
                  limits=['Reused generated monitor and frozen baseline reports; not independent final testing.',
                          'Selected-first-error sensitivity motivated this test but does not predict whole-game gains.',
                          'All weights remain fixed; reserved channels may have acquired useful learned signals.'])
    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        atomic_json(args.report, report)
    policy = handle = None
    persist()
    try:
        levels, optima, specs = checked_bank()
        baselines, sources = checked_baselines(args.baseline_report, specs)
        sources[__file__] = digest(__file__)
        report['sources'] = sources
        for arm, baseline in baselines.items():
            entry = {'status': 'running', 'checkpoint': baseline['checkpoint'], 'runs': [],
                     'baseline_summary': summarize(baseline['runs'])}
            report['arms'][arm] = entry
            policy, info = load_checkpoint(baseline['checkpoint']['path'], 'cuda')
            policy.float().eval().requires_grad_(False)
            if policy.config().get('loops') != 2 or info.get('readout_config', {}).get('memory_mode') != 'evolving':
                raise ValueError('evolving depth2 workspace required')
            state_sha = state_digest(policy.state_dict())
            entry['wiring_check'] = public_wiring_check(policy, Ls20Scenario(levels[0], specs[0]['training_context_index']).reset(), 'cuda')
            hook = ReservedZero()
            handle = policy.readout.register_forward_pre_hook(hook)
            for index, (level, optimum, spec) in enumerate(zip(levels, optima, specs)):
                entry['incomplete_level'] = spec['seed']
                result = rollout(policy, Ls20Scenario(level, spec['training_context_index']),
                                 200, 'cuda', optimum, 'repeat', temperature=0.)
                result.update(seed=spec['seed'], difficulty=spec['difficulty'], context=spec['training_context_index'],
                              losses=3-result['lives_left'])
                entry['runs'].append(result)
                entry.pop('incomplete_level')
                entry['summary'] = summarize(entry['runs'])
                report['status'] = 'running'
                persist()
                if (index + 1) % 10 == 0:
                    print(json.dumps({'arm': arm, **entry['summary']['all']}), flush=True)
            handle.remove()
            handle = None
            if hook.calls != sum(run['actions'] for run in entry['runs']):
                raise ValueError('readout intervention/action census mismatch')
            if state_digest(policy.state_dict()) != state_sha:
                raise ValueError('frozen weights changed')
            entry.update(status='complete', weights_unchanged=True, intervention_calls=hook.calls,
                         reserved_nonzero_values=hook.nonzero_values, reserved_max_abs=hook.max_abs,
                         paired_vs_original=paired(entry['runs'], baseline['runs']))
            verify_sources(sources)
            persist()
            del policy
            policy = None
            gc.collect()
            torch.cuda.empty_cache()
        verify_sources(sources)
        report.update(status='complete', sources_unchanged=True)
    except BaseException as error:
        report.update(status='failed_partial', error=str(error))
        raise
    finally:
        if handle is not None:
            handle.remove()
        policy = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        signal.alarm(0)
        persist()


if __name__ == '__main__':
    main()
