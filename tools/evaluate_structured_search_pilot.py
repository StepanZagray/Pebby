"""Fixed generated-only actor/depth1/depth4 gameplay pilot, no optimization."""
import argparse
import gc
import json
import os
import signal
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch

from pebby.agent.evaluate import bank_levels, completion_rate
from pebby.agent.history import for_policy
from pebby.agent.model import load_checkpoint
from pebby.ls20.env import Ls20Scenario
from pebby.ls20 import names
from tools.train_structured_policy import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('refusing output overwrite')
    if not torch.cuda.is_available():
        parser.error('this bounded pilot requires CUDA')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('300s pilot deadline')))
    signal.alarm(300)
    started = time.monotonic()
    print('PID', os.getpid(), flush=True)
    levels, optima, specs = bank_levels(args.bank)
    if len(levels) != 5 or sorted(int(s['difficulty']) for s in specs) != [1, 2, 3, 4, 5]:
        raise ValueError('exactly five difficulty-balanced generated levels required')
    contexts = [int(s.get('training_context_index', 0)) for s in specs]
    report = {'status': 'running', 'pid': os.getpid(), 'bank': str(args.bank),
              'bank_sha256': digest(args.bank), 'official_inputs_used': False,
              'training_performed': False, 'device': 'cuda', 'max_actions': 200,
              'protocol': 'strict', 'on_stall': 'repeat', 'temperature': 0,
              'precision': 'FP32, TF32 disabled', 'arms': {},
              'runner_sha256': digest(__file__),
              'seeds': [s['seed'] for s in specs], 'contexts': contexts,
              'limitations': ['Five-level diagnostic, not an estimate of all-level reliability.',
                             'Search uses learned completion-before-next-loss surrogate; no terminal-failure calibration targets.']}

    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, args.report)

    checkpoints = {
        'actor': 'checkpoints/ls20-structured-policy-paired-local-h4-600.pt',
        'depth1': 'checkpoints/ls20-structured-search-ranking-d1.pt',
        'depth4': 'checkpoints/ls20-structured-search-ranking-d4.pt',
    }
    try:
        for arm, checkpoint in checkpoints.items():
            report['active_arm'] = arm
            persist()
            load_started = time.monotonic()
            policy, _ = load_checkpoint(checkpoint, 'cuda')
            result = {'checkpoint': checkpoint, 'checkpoint_sha256': digest(checkpoint),
                      'load_seconds': time.monotonic() - load_started,
                      'parameters': policy.parameter_count()}
            env = Ls20Scenario(levels[0], contexts[0])
            frame = env.reset()
            # Initial frame only. No action, oracle, label, or actual future enters
            # this throughput gate; each repetition has the same public history.
            history = for_policy(policy, frame, 'cuda')
            durations = []
            for repeat in range(6):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                history.scores()
                torch.cuda.synchronize()
                if repeat:
                    durations.append(time.perf_counter() - tick)
            result['initial_decision_seconds'] = durations
            result['initial_decision_median_seconds'] = statistics.median(durations)
            arm_budget = 180 if arm == 'depth4' else 60
            result['worst_case_projected_decision_seconds'] = max(durations) * 1000
            if result['worst_case_projected_decision_seconds'] > arm_budget:
                raise RuntimeError(f'{arm} projected1000-decision cost exceeds{arm_budget}s budget')
            pending = {}
            decisions = []
            if hasattr(policy, 'search_fields'):
                original_search = policy.search_fields
                original_outcomes = policy._outcomes

                def traced_outcomes(events):
                    outcomes = original_outcomes(events)
                    if 'first_outcomes' not in pending:
                        pending['first_outcomes'] = outcomes.detach().cpu().tolist()
                    return outcomes

                def traced_search(fields):
                    pending.clear()
                    searched = original_search(fields)
                    pending['search'] = searched['results'][0]
                    pending['transitions'] = searched['transition_counts'][0]
                    for root in pending['search']['roots']:
                        if not (-1e-6 <= root['score'] <= .99 + 1e-6 and
                                -1e-6 <= root['reward'] and -1e-6 <= root['bootstrap'] and
                                0 <= root['survival'] <= 1):
                            raise ValueError('search expected-return bounds violated')
                    return searched

                policy._outcomes = traced_outcomes
                policy.search_fields = traced_search

            inference_started = [None]

            def before_inference(module, inputs):
                inference_started[0] = time.perf_counter()

            def after_inference(module, inputs, output):
                torch.cuda.synchronize()
                pending['inference_seconds'] = time.perf_counter() - inference_started[0]

            pre_hook = policy.register_forward_pre_hook(before_inference)
            post_hook = policy.register_forward_hook(after_inference)

            def traced_scenario(level, context):
                scenario = Ls20Scenario(level, context)
                level_position = next(i for i, item in enumerate(levels) if item is level)
                original_perform = scenario.perform
                original_reset = scenario.reset
                last_frame = [None]

                def reset():
                    last_frame[0] = original_reset()
                    return last_frame[0]

                def perform(action):
                    # Everything below is post-selection diagnostics. The policy
                    # receives only the existing evaluator's public H8 interface.
                    old_lives = scenario.lives()
                    observation = original_perform(action)
                    action_index = names.ACTION_IDS.index(action)
                    record = {'level': level_position, 'seed': specs[level_position]['seed'],
                              'action_index': action_index, 'lost_life': scenario.lives() < old_lives,
                              'won': bool(observation.won),
                              'frame_changed': observation.frame != last_frame[0], **pending}
                    if 'search' in record and record['search']['action'] != action_index:
                        raise ValueError('executed action disagrees with public search')
                    decisions.append(record)
                    last_frame[0] = observation.frame
                    pending.clear()
                    return observation

                scenario.reset, scenario.perform = reset, perform
                return scenario

            playing = time.monotonic()
            result['status'] = 'playing'
            result['decisions'] = decisions
            report['arms'][arm] = result
            signal.alarm(max(1, min(arm_budget, int(300 - (time.monotonic() - started)))))
            with patch('pebby.agent.evaluate.Ls20Scenario', traced_scenario):
                result['evaluation'] = completion_rate(policy, levels, 200, 'cuda', optima, 'repeat', contexts)
            signal.alarm(max(1, int(300 - (time.monotonic() - started))))
            pre_hook.remove()
            post_hook.remove()
            result['gameplay_seconds'] = time.monotonic() - playing
            result['decisions'] = decisions
            result['life_losses'] = sum(d['lost_life'] for d in decisions)
            result['status'] = 'complete'
            if len(decisions) != sum(r['actions'] for r in result['evaluation']['runs']):
                raise ValueError('decision trace count mismatch')
            report['arms'][arm] = result
            persist()
            print(arm, 'completed', result['evaluation']['completed'], '/5',
                  'seconds', result['gameplay_seconds'], flush=True)
            del policy, history, env
            gc.collect()
            torch.cuda.empty_cache()
        if digest(args.bank) != report['bank_sha256']:
            raise ValueError('pilot bank changed')
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        persist()


if __name__ == '__main__':
    main()
