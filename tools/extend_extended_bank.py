"""Append larger generated lessons after a successful all-pilot collector gate.

Creates separate banks with byte-exact pilot prefixes. One CPU worker; all new
rows retain complete contextual Oracle, engine-WIN and sampled-transition proof.
This writes JSON only and never modifies existing banks or active training data.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from pebby.ls20 import extended_curriculum as extended


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def check_prefix(rows, split, quality_profile=None):
    last = extended.SPLIT_STARTS[split] - 1
    for index, row in enumerate(rows):
        seed = row['seed']
        floor = extended.budget_floor(row)
        if quality_profile is not None and row.get('quality_profile') != quality_profile:
            raise ValueError('pilot quality profile mismatch')
        if seed <= last or not extended.SPLIT_STARTS[split] <= seed < (1000000 if split == 'train' else 2000000):
            raise ValueError('pilot seed order or namespace mismatch')
        fingerprint, partition = extended.geometry_partition(row)
        if (partition != split or row.get('geometry_split') != partition
                or row.get('geometry_sha256') != fingerprint):
            raise ValueError('pilot geometry partition mismatch')
        if row['difficulty'] != index % 5 + 1:
            raise ValueError('pilot difficulty sequence mismatch')
        if (row.get('generation_namespace') != extended.NAMESPACE or row.get('extended_curriculum_version') != extended.VERSION
                or row.get('context_engine_verified') is not True or row.get('search_truncated') is not False
                or row.get('training_context_index') != seed % 7 or row.get('verification_lives') != 3
                or row.get('oracle_backend') != 'fast' or row.get('random_transitions_checked', 0) < 8
                or row.get('slack_moves', -1) < floor or row.get('minimum_slack_moves', -1) < floor
                or row.get('budget_floor') != floor or row.get('quality_profile') not in ('learning', 'challenge')
                or row.get('step_counter') != 42 or row.get('step_cost') not in (1, 2)
                or row.get('distractor_count', 0) < 1
                or row.get('gameplay_sha256') != extended.gameplay_hash(row)):
            raise ValueError(f'pilot seed {seed} lacks extended contextual proof')
        last = seed
    return last + 1


def check_gate(gate_path, prefixes):
    gate = json.loads(Path(gate_path).read_text())
    if gate.get('status') != 'complete' or gate.get('coverage') != 'mixed' or gate.get('history') != 8 or gate.get('epsilon') != .15 or gate.get('samples_per_level') != 16:
        raise ValueError('pilot collector gate is incomplete or used wrong protocol')
    for split, path in prefixes.items():
        proof = gate['splits'][split]
        if (proof.get('source_sha256') != digest(path) or not proof.get('all_levels_win_covered')
                or proof.get('levels') != len(read_rows(path)) or proof.get('rows', 0) < 1
                or proof['branch_verification'].get('branches') != 4 * proof['branch_verification'].get('expansions', 0)):
            raise ValueError(f'{split}: pilot does not match successful collector gate')
        if proof.get('output_sha256') != digest(proof['output']):
            raise ValueError(f'{split}: gated collector NPZ changed')
    return gate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality-profile', choices=('learning', 'challenge'), default='learning')
    parser.add_argument('--train-count', type=int, default=10000)
    parser.add_argument('--validation-count', type=int, default=2000)
    parser.add_argument('--train-prefix', type=Path, default=Path('data/ls20-extended-pilot-train.jsonl'))
    parser.add_argument('--validation-prefix', type=Path, default=Path('data/ls20-extended-pilot-validation.jsonl'))
    parser.add_argument('--gate-report', type=Path, default=Path('artifacts/world-extended-collector-pilot.json'))
    parser.add_argument('--out-dir', type=Path, default=Path('data/extended-bank-v2'))
    parser.add_argument('--report', type=Path, default=Path('artifacts/world-extended-bank-v2.json'))
    parser.add_argument('--attempts', type=int, default=16)
    parser.add_argument('--search-limit', type=int, default=600000)
    parser.add_argument('--progress-every', type=int, default=25)
    parser.add_argument('--existing-banks', nargs='+', default=['data/ls20-verified-train.jsonl',
                        'data/ls20-verified-validation.jsonl'])
    args = parser.parse_args(argv)
    if min(args.train_count, args.validation_count, args.attempts, args.progress_every) < 1:
        parser.error('counts, attempts and progress interval must be positive')
    if args.attempts != 16 or args.search_limit != 600000:
        parser.error('v2 extension preserves the reviewed pilot bounds: attempts16/search-limit600000')
    started = time.monotonic()
    prefixes = {'train': args.train_prefix, 'validation': args.validation_prefix}
    gate = check_gate(args.gate_report, prefixes)
    rows = {split: read_rows(path) for split, path in prefixes.items()}
    next_seeds = {split: check_prefix(value, split, args.quality_profile) for split, value in rows.items()}
    totals = {'train': args.train_count, 'validation': args.validation_count}
    if any(totals[split] < len(rows[split]) for split in totals):
        parser.error('requested counts must include the full pilot prefixes')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {split: args.out_dir / f'{split}.jsonl' for split in totals}
    if any(path.exists() for path in outputs.values()):
        raise ValueError('refusing to overwrite existing output banks; use a new output directory')
    hashes, occupied = set(), set()
    source_hashes = {}
    for path in args.existing_banks:
        source_hashes[path] = digest(path)
        for row in read_rows(path):
            hashes.add(extended.gameplay_hash(row))
            occupied.add(row['seed'])
    original_hash_count = len(hashes)
    for split, values in rows.items():
        for row in values:
            fingerprint = row['gameplay_sha256']
            if fingerprint in hashes or row['seed'] in occupied:
                raise ValueError('pilot gameplay/seed overlaps another source or split')
            hashes.add(fingerprint)
            occupied.add(row['seed'])
    for split, path in outputs.items():
        with path.open('xb') as stream:
            content = prefixes[split].read_bytes()
            if not content.endswith(b'\n'):
                raise ValueError('pilot prefix must end with newline')
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    prefix_counts = {split: len(value) for split, value in rows.items()}
    initial_count = sum(prefix_counts.values())
    attempted, drafts, failures = 0, 0, []
    reasons = Counter()
    report = {'status': 'running', 'pid': os.getpid(), 'workers': 1, 'device': 'cpu',
              'version': extended.VERSION, 'namespace': extended.NAMESPACE, 'quality_profile': args.quality_profile, 'requested_total': totals,
              'pilot_prefix_levels': {split: len(value) for split, value in rows.items()},
              'pilot_prefix_sha256': {split: digest(path) for split, path in prefixes.items()},
              'collector_gate': str(args.gate_report), 'collector_gate_sha256': digest(args.gate_report),
              'existing_banks': source_hashes, 'output_paths': {s: str(p) for s,p in outputs.items()},
              'official_gameplay_inputs_used': False, 'full_image_collection_started': False,
              'original_banks_modified': False, 'search_limit': args.search_limit, 'attempts_per_seed': args.attempts,
              'limits': ['JSON bank extension only; no new rows added to active training.',
                         'All accepted levels have complete contextual Oracle and real-engine WIN3lives proof.',
                         'Eight random transitions per level are sampled from winning-route states, not all reachable states.',
                         'Only pilot collected states were audited across all four actions; full-bank images have not been collected.',
                         'Challenge proposes three moving kinds/shared rail lengths2..6; learning keeps at most two length2 patrollers in compact6..7 high-tier rooms.',
                         'Learning d5 favors obstacles/corridors; d4/5 required attribute offsets are one or two steps. Complex challenge candidates can exhaust complete-search bounds.',
                         'New v2 splits hold out translation-normalized geometry, not topology families or historical-v1 geometry.',
                         'Counter42/cost1|2; learning reserve8, challenge reserve0; context0+launcher remains excluded.',
                         'Challenge proposes up to6 refills/8 launchers; installed and used counts are separate. Three-required-kind extras cannot be non-required kinds.'],
              'code_sha256': {p: digest(p) for p in (__file__, 'pebby/ls20/extended_curriculum.py',
                               'pebby/ls20/generate.py','pebby/ls20/generation_quality.py','pebby/ls20/layout.py','pebby/ls20/plan.py',
                               'pebby/ls20/rails.py','pebby/ls20/fastplan.py','pebby/ls20/_fastplan.c')}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def persist():
        elapsed = time.monotonic() - started
        accepted = sum(map(len, rows.values()))
        added = accepted - initial_count
        throughput = added / max(elapsed, .001)
        remaining = sum(totals.values()) - accepted
        eta = (datetime.now().astimezone() + timedelta(seconds=remaining / throughput)) if throughput > 0 else None
        report.update(elapsed_seconds=elapsed, new_attempted_seeds=attempted, new_attempted_drafts=drafts,
                      accepted_total=accepted, added_levels=added, new_seed_failures=failures,
                      new_draft_exclusions=dict(reasons), next_seeds=next_seeds,
                      new_levels_per_second=throughput, eta_local=eta.strftime('%Y-%m-%d %H:%M:%S %Z') if eta else None,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      summaries={split: extended.summarize(value) for split,value in rows.items()},
                      dedup={'existing_unique_gameplay': original_hash_count,
                             'new_unique_gameplay_including_pilot': len(hashes)-original_hash_count,
                             'overlap_existing_or_other_split': 0})
        temp = args.report.with_suffix('.tmp')
        temp.write_text(json.dumps(report,indent=2)+'\n')
        temp.replace(args.report)
    persist()
    print('PID',os.getpid(),'prefixes',initial_count,'target',sum(totals.values()),flush=True)
    try:
        for split in ('train','validation'):
            with outputs[split].open('ab') as stream:
                split_seed_attempts = 0
                while len(rows[split]) < totals[split]:
                    if split_seed_attempts >= (totals[split] - prefix_counts[split]) * 8:
                        raise RuntimeError(f'{split}: bounded seed attempts exhausted')
                    seed = next_seeds[split]
                    next_seeds[split] += 1
                    split_seed_attempts += 1
                    if seed in occupied:
                        raise ValueError(f'{split}: seed collision {seed}')
                    difficulty = len(rows[split]) % 5 + 1
                    attempted += 1
                    accepted, excluded = extended.generate_level(seed,difficulty,args.attempts,args.search_limit,args.quality_profile)
                    reasons.update(excluded)
                    drafts += sum(excluded.values()) + int(accepted is not None)
                    if accepted is None:
                        failures.append({'split':split,'seed':seed,'difficulty':difficulty,'reasons':excluded})
                        persist()
                        continue
                    fingerprint = accepted['gameplay_sha256']
                    if fingerprint in hashes:
                        reasons['duplicate_gameplay'] += 1
                        persist()
                        continue
                    hashes.add(fingerprint)
                    occupied.add(seed)
                    rows[split].append(accepted)
                    stream.write((json.dumps(accepted,separators=(',',':'))+'\n').encode())
                    stream.flush()
                    os.fsync(stream.fileno())
                    if len(rows[split]) % args.progress_every == 0:
                        persist()
                        print(split,len(rows[split]),'/',totals[split], 'new levels/s',round(report['new_levels_per_second'],2),
                              'ETA',report['eta_local'],flush=True)
        report['status']='complete'
        report['output_sha256']={split:digest(path) for split,path in outputs.items()}
        for split,path in outputs.items():
            if not path.read_bytes().startswith(prefixes[split].read_bytes()):
                raise ValueError('pilot byte-prefix changed during extension')
    except BaseException as error:
        report['status']='blocked_contract_mismatch' if isinstance(error,extended.ContractMismatch) else 'failed'
        report['error']=repr(error)
        raise
    finally:
        persist()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
