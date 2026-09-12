"""Build a small K4 index from existing verified public histories; no rendering."""
import argparse
import json
import os
from pathlib import Path
import resource
import time
import torch
from pebby.agent.world_train import load_dataset,require_verified_data,require_winning_coverage
from pebby.agent.world_exploratory_sequences import build_sidecar,load_sidecar
from pebby.agent.on_policy_provenance import file_digest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,default=Path('data/world-array-cache'))
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args(argv);torch.set_num_threads(1);started=time.monotonic()
    print('PID',os.getpid(),flush=True)
    data=load_dataset(args.source,history=8,cache_dir=args.cache_dir)
    require_verified_data(data);require_winning_coverage(data)
    index=build_sidecar(data,args.source,args.out)
    checked=load_sidecar(args.out,args.source,data)
    by_seed={int(level['seed']):int(level['difficulty']) for level in data['meta']['levels']}
    eligible = set(map(int,data['seeds'][checked.anchor_row]))
    report={**index.meta,'status':'complete','pid':os.getpid(),'device':'cpu','torch_threads':1,
            'output':str(args.out),'output_sha256':file_digest(args.out),'reloaded_and_reverified':True,
            'eligible_levels_by_difficulty':{str(d):sum(by_seed[s]==d for s in eligible) for d in range(1,6)},
            'elapsed_seconds':time.monotonic()-started,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
