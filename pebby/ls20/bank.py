"""Produce and load banks of verified levels.

Generation uses bounded complete planning and real-engine replay. Build banks
once rather than generating difficult levels during training. Seeds start at
0 / 1,000,000 / 2,000,000 for train / validation / test. Opt into generator
version 4 for distinct three-way geometry holdout and added goal mechanics;
the default version 3 preserves historical train/validation reconstruction.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

from .generate import DIFFICULTIES, GENERATOR_VERSION, SUPPORTED_GENERATOR_VERSIONS, generate_level

SPLIT_SEEDS = {"train": 0, "validation": 1_000_000, "test": 2_000_000}


def _one(job):
    seed, difficulty, split, version = job
    try:
        return generate_level(seed, difficulty, split=split, generator_version=version)
    except RuntimeError:
        return None  # a seed that never drafted a completable level; just skip it


def build(count, split="train", difficulties=DIFFICULTIES, workers=None, chunksize=8,
          *, generator_version=GENERATOR_VERSION):
    """Generate `count` levels, cycling through `difficulties`."""
    if split not in SPLIT_SEEDS:
        raise ValueError(f"split must be one of {sorted(SPLIT_SEEDS)}")
    if type(generator_version) is not int or generator_version not in (3, 4):
        raise ValueError('new banks require generator_version=3 or 4')
    if split == 'test' and generator_version == 3:
        raise ValueError('legacy v3 has no test geometry partition; use generator_version=4 / --generator-version 4')
    if type(count) is not int or not 1 <= count <= 1_000_000:
        raise ValueError('bank count must be 1..1000000 without crossing a seed split')
    if not difficulties or any(type(d) is not int or d not in DIFFICULTIES for d in difficulties):
        raise ValueError('difficulties must be nonempty and drawn from 1..7')
    workers = 2 if workers is None else workers
    if not 1 <= workers <= 2:
        raise ValueError('seven-tier generation allows at most two workers; later tiers need large complete searches')
    base = SPLIT_SEEDS[split]
    jobs = [(base + i, difficulties[i % len(difficulties)], split, generator_version) for i in range(count)]
    with Pool(processes=workers) as pool:
        specs = pool.map(_one, jobs, chunksize=chunksize)
    return [spec for spec in specs if spec is not None]


def save(specs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for spec in specs:
            handle.write(json.dumps(spec, separators=(",", ":")) + "\n")
    return path


def load(path):
    """Read compatible level geometry; reading is not proof certification.

    Historical v2 geometry remains usable for audits and contextual re-planning.
    Its stored index-zero routes must not be trusted as training-context proof.
    """
    with Path(path).open() as handle:
        specs = [json.loads(line) for line in handle if line.strip()]
    stale = {spec.get("generator_version") for spec in specs} - set(SUPPORTED_GENERATOR_VERSIONS)
    if stale:
        raise ValueError(f"bank was built by generator version(s) {sorted(stale)}; "
                         f"this build is version {GENERATOR_VERSION}")
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=1000)
    parser.add_argument("--split", choices=sorted(SPLIT_SEEDS), default="train")
    parser.add_argument("--difficulties", type=int, nargs="+", default=list(DIFFICULTIES))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--generator-version", type=int, choices=(3, 4), default=GENERATOR_VERSION,
                        help='3 preserves historical generation; 4 adds goal relations and disjoint test geometry')
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.levels < 1:
        parser.error("--levels must be positive")
    if any(d not in DIFFICULTIES for d in args.difficulties):
        parser.error(f"difficulties must be drawn from {DIFFICULTIES}")

    started = time.perf_counter()
    specs = build(args.levels, args.split, tuple(args.difficulties), args.workers,
                  generator_version=args.generator_version)
    elapsed = time.perf_counter() - started
    save(specs, args.out)
    by_difficulty = {d: sum(s["difficulty"] == d for s in specs) for d in args.difficulties}
    optimal = sorted(s["optimal_actions"] for s in specs)
    print(f"{len(specs)}/{args.levels} levels in {elapsed:.1f}s "
          f"({elapsed / max(1, len(specs)):.2f}s each) -> {args.out}")
    print(f"by difficulty: {by_difficulty}")
    if optimal:
        print(f"optimal actions: min {optimal[0]} median {optimal[len(optimal) // 2]} "
              f"max {optimal[-1]}")


if __name__ == "__main__":
    main()
