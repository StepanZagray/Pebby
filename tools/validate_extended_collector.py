"""Gate extended generated pilots through the unchanged mixed world-data collector.

Every collector expansion is checked against the complete contextual Oracle's
logical transition, including all four real engine branches. These checks cover
collected states and actions, not every reachable state in each level.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from unittest.mock import patch

import numpy as np

from pebby.agent import world_data
from pebby.ls20 import names
from pebby.ls20.extended_curriculum import ContractMismatch, VERSION, NAMESPACE
from pebby.ls20.plan import simulate


def checked_expansion(original, counts, env, oracle, before, seed, step):
    if oracle.truncated or not oracle.solvable:
        raise ContractMismatch(f'seed {seed}: collector used incomplete Oracle')
    state = oracle.state_of(env)
    result = original(env, oracle, before, seed, step)
    targets, branches, observations, mask = result
    for action, (branch, observation) in enumerate(zip(branches, observations)):
        predicted, outcome = simulate(oracle.layout, state, action, oracle.refills)
        if outcome == 'won':
            agrees = observation.won and branch.lives() == env.lives()
        elif outcome == 'died':
            agrees = (not observation.won and branch.lives() == env.lives() - 1
                      and (observation.finished or oracle.state_of(branch) == oracle.start))
        else:
            agrees = (not observation.finished and branch.lives() == env.lives()
                      and oracle.state_of(branch) == predicted)
        if not agrees:
            raise ContractMismatch(f'seed {seed} collected step {step} action {action}: '
                                   f'{outcome}, predicted {predicted}, actual {oracle.state_of(branch)}')
        if observation.finished and int(targets['next_optimal'][action]) != 0:
            raise ContractMismatch(f'seed {seed}: terminal successor has policy label')
        counts['branches'] += 1
        counts['outcome_' + outcome] += 1
    counts['expansions'] += 1
    return result


def collect_checked(specs):
    counts = Counter()
    original = world_data._expand
    def checked(*args, **kwargs):
        return checked_expansion(original, counts, *args, **kwargs)
    with patch.object(world_data, '_expand', side_effect=checked):
        arrays = world_data.build(specs, workers=1, history=8, samples=16, epsilon=.15,
                                  coverage='mixed_failure', search_limit=600000)
    expected_seeds = {int(spec['seed']) for spec in specs}
    if set(map(int, arrays['seeds'])) != expected_seeds:
        raise ContractMismatch('collector omitted requested generated levels')
    if len(arrays['meta']['levels']) != len(specs) or any('excluded' in row for row in arrays['meta']['levels']):
        raise ContractMismatch('collector produced excluded or missing level proofs')
    if not all(row.get('context_engine_verified') and not row.get('search_truncated')
               and row.get('context_index') == row['seed'] % 7 for row in arrays['meta']['levels']):
        raise ContractMismatch('incomplete or wrong-context collector proof')
    if not np.all(arrays['context_index'] == arrays['seeds'] % 7):
        raise ContractMismatch('wrong per-row context')
    if np.any(arrays['won'] & ~arrays['terminal']):
        raise ContractMismatch('nonterminal winning successor')
    win_seeds = set(map(int, arrays['seeds'][arrays['won'].any(axis=1)]))
    if win_seeds != expected_seeds:
        raise ContractMismatch('missing real winning successor coverage')
    if np.any(arrays['terminal'] & (arrays['next_optimal'] != 0)):
        raise ContractMismatch('nonzero terminal successor labels')
    if counts['branches'] != 4 * counts['expansions']:
        raise ContractMismatch('not all four branches were validated')
    arrays['meta']['extended_curriculum_version'] = VERSION
    arrays['meta']['generation_namespace'] = NAMESPACE
    arrays['meta']['collector_branch_verification'] = dict(counts)
    return arrays, dict(counts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=Path('data/ls20-extended-pilot-train.jsonl'))
    parser.add_argument('--validation', type=Path, default=Path('data/ls20-extended-pilot-validation.jsonl'))
    parser.add_argument('--out-dir', type=Path, default=Path('data'))
    parser.add_argument('--report', type=Path, default=Path('artifacts/world-extended-collector-pilot.json'))
    args = parser.parse_args(argv)
    started = time.monotonic()
    report = {'status': 'running', 'pid': os.getpid(), 'workers': 1, 'device': 'cpu',
              'history': 8, 'samples_per_level': 16, 'epsilon': .15, 'coverage': 'mixed_failure',
              'official_inputs_used': False, 'splits': {},
              'coverage_limit': 'All four actions from every collector-expanded state; not exhaustive reachable-state enumeration.'}
    def persist():
        report['elapsed_seconds'] = time.monotonic() - started
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temp = args.report.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(args.report)
    print('PID', os.getpid(), flush=True)
    persist()
    try:
        split_seeds = []
        for split, path in (('train', args.train), ('validation', args.validation)):
            output = args.out_dir / f'ls20-extended-pilot-{split}.npz'
            if output.exists():
                raise ValueError(f'refusing to overwrite {output}')
            specs = [json.loads(line) for line in path.read_text().splitlines()]
            if any(s.get('extended_curriculum_version') != VERSION or
                   s.get('generation_namespace') != NAMESPACE for s in specs):
                raise ValueError('source is not an extended curriculum bank')
            split_seeds.append({s['seed'] for s in specs})
            if len(split_seeds) > 1 and split_seeds[0] & split_seeds[1]:
                raise ValueError('train/validation seed overlap')
            arrays, counts = collect_checked(specs)
            report['splits'][split] = {'levels': len(specs), 'rows': len(arrays['seeds']),
                                     'all_levels_win_covered': arrays['meta']['win_covered_levels'] == len(specs),
                                     'valid_successor_labels': int((arrays['next_optimal'] != 0).sum()),
                                     'branch_verification': counts,
                                     'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                     'output': str(output)}
            arrays['meta']['source_bank'] = report['splits'][split]['source']
            arrays['meta']['source_bank_sha256'] = report['splits'][split]['source_sha256']
            temp = output.with_suffix('.tmp.npz')
            world_data.save(temp, arrays)
            temp.replace(output)
            report['splits'][split]['output_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
            print(split, report['splits'][split], flush=True)
            persist()
        report['status'] = 'complete'
    except Exception as error:
        report['status'], report['error'] = 'failed', repr(error)
        raise
    finally:
        persist()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
