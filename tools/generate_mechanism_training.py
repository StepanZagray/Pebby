"""Bounded TRAIN-only mechanism pilot; independent of existing/held-out banks.

Uses procedural candidates and full contextual native-oracle/actual-engine WIN
verification, explicit attribute composition and long-route quotas.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from tools.generate_mechanism_pilot import MODES, VERSION, MIN_SLACK, MIN_ACTIONS, candidate, verify, canonical, digest, coverage, generate_one, LIMITATIONS
from pebby.ls20.generate import FORMAT


def geometry(spec, normalized=False):
    free=sorted({(x,y) for x in range(12) for y in range(12)}-{tuple(p) for p in spec['walls']})
    if normalized:
        x0=min(p[0] for p in free);y0=min(p[1] for p in free)
        free=[(x-x0,y-y0) for x,y in free]
    return hashlib.sha256(json.dumps(free).encode()).hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-mode',type=int,default=20)
    parser.add_argument('--start-seed',type=int,default=700000)
    parser.add_argument('--seconds',type=int,default=120)
    parser.add_argument('--limit',type=int,default=600000)
    parser.add_argument('--attempts',type=int,default=40)
    parser.add_argument('--out',type=Path,default=Path('data/ls20-mechanism-training-v2.jsonl'))
    parser.add_argument('--report',type=Path,default=Path('artifacts/world-mechanism-training-v2.json'))
    args=parser.parse_args(argv)
    if not 700000<=args.start_seed<900000 or not 1<=args.seconds<=900 or not 1<=args.limit<=600000 or args.per_mode<1 or args.attempts<1:
        parser.error('require training namespace [700000,900000), seconds<=900, limit<=600000 and positive counts')
    if args.out.exists() or args.report.exists():raise FileExistsError('refuse existing output/report')
    started=time.monotonic();pid=os.getpid();print(json.dumps({'pid':pid,'seconds':args.seconds,'cpu_workers':1}),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.report.parent.mkdir(parents=True,exist_ok=True)
    report=dict(format='pebby.mechanism-training-pilot.v2',status='running',split='train',source='generated_only',pid=pid,
                cpu_workers=1,official_inputs_used=False,heldout_rows_used_as_candidates=False,
                namespace=[700000,900000],limits={'seconds':args.seconds,'states':args.limit,'attempts_per_seed':args.attempts},
                target_per_mode=args.per_mode,accepted=[],rejections=[],source_hashes={},
                limitations=LIMITATIONS)
    code=[Path(__file__),Path('tools/generate_mechanism_pilot.py'),*(Path('pebby/ls20')/p for p in ('generate.py','generation_quality.py','plan.py','fastplan.py','_fastplan.c','layout.py','rails.py')),Path('pebby/agent/world_data.py')]
    report['code_hashes']={str(p):digest(p) for p in code}
    def persist():
        report['elapsed_seconds']=time.monotonic()-started
        temporary=args.report.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(args.report)
    def timeout(*_):raise TimeoutError('bounded generation deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(args.seconds)
    seeds=set();spec_hashes=set();rows=[];generation_start=None
    try:
        # Inventory only generated JSONL under data; do not follow file symlinks.
        for path in sorted(Path('data').rglob('*.jsonl')):
            if path==args.out or path.is_symlink():continue
            used=False;before=digest(path)
            with path.open() as stream:
                for line in stream:
                    if not line.strip():continue
                    spec=json.loads(line)
                    if spec.get('format')!=FORMAT:continue
                    seeds.add(int(spec['seed']));spec_hashes.add(canonical(spec));used=True
            if used:
                if digest(path)!=before:raise RuntimeError(f'inventory changed: {path}')
                report['source_hashes'][str(path)]=before
        report['existing_distinct_seeds']=len(seeds);report['existing_distinct_gameplay_hashes']=len(spec_hashes)
        seed=args.start_seed;generation_start=time.monotonic();persist()
        with args.out.open('x') as output:
            # Round-robin mode order preserves balanced partial output on deadline.
            for number in range(args.per_mode):
                for mode in MODES:
                    accepted, seed = generate_one(seed, mode, attempts=args.attempts, limit=args.limit,
                        seen_seeds=seeds, seen_specs=spec_hashes, namespace_end=900000,
                        record_rejection=report['rejections'].append)
                    accepted['curriculum_version']=f'mechanism-training-v{VERSION}'
                    accepted['split']='train';accepted['source']='generated_only'
                    rows.append(accepted);seeds.add(accepted['seed']);spec_hashes.add(canonical(accepted))
                    output.write(json.dumps(accepted,separators=(',',':'))+'\n');output.flush();os.fsync(output.fileno())
                    report['accepted'].append({k:accepted[k] for k in ('seed','pilot_mode','optimal_actions','reachable_states','oracle_backend','solution_mechanics')})
                    persist();print(json.dumps({'accepted':len(rows),'seed':accepted['seed'],'mode':mode,'elapsed':report['elapsed_seconds']}),flush=True)
        report['status']='complete'
    except TimeoutError as error:report.update(status='bounded_partial',error=str(error))
    except (RuntimeError,ValueError) as error:report.update(status='failed_closed',error=str(error))
    finally:
        signal.alarm(0)
        report['bank_sha256']=digest(args.out) if args.out.exists() else None
        report['accepted_count']=len(rows);report['actual_engine_wins_three_lives']=len(rows)
        report['by_mode']=dict(Counter(r['pilot_mode'] for r in rows))
        report['coverage']=coverage(rows)
        report['quality_floors']={'route_actions':MIN_ACTIONS,'long_route_actions':72,'learning_slack_moves':MIN_SLACK,'challenge_slack_moves':0}
        report['rejection_counts']=dict(Counter(r['reason'] for r in report['rejections']))
        report['distinct_geometry_absolute']=len({geometry(r) for r in rows})
        report['distinct_geometry_translation_normalized']=len({geometry(r,True) for r in rows})
        report['distinct_gameplay_hashes']=len({canonical(r) for r in rows})
        report['mechanic_use_totals']=dict(sum((Counter(r['solution_mechanics']) for r in rows),Counter()))
        report['generation_seconds']=time.monotonic()-generation_start if generation_start else None
        rate=len(rows)/report['generation_seconds'] if rows else 0
        report['accepted_per_second']=rate
        report['eta_2000_generation_seconds_linear']=2000/rate if rate else None
        report['eta_caveat']='Linear extrapolation excludes inventory and cannot guarantee later acceptance or geometric diversity.'
        report['code_hashes_unchanged']=all(digest(p)==h for p,h in report['code_hashes'].items())
        report['inventory_hashes_unchanged']=all(digest(p)==h for p,h in report['source_hashes'].items())
        if not report['code_hashes_unchanged'] or not report['inventory_hashes_unchanged']:report['status']='failed_closed'
        persist();print(json.dumps({k:report[k] for k in ('status','accepted_count','elapsed_seconds','distinct_geometry_absolute','distinct_geometry_translation_normalized','accepted_per_second')}),flush=True)
    return 0 if report['status']=='complete' else 1

if __name__=='__main__':raise SystemExit(main())
