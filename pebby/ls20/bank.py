"""Produce and load banks of verified levels.

Generation costs about half a second per level, almost all of it planning, so
training data wants a bank built once across cores rather than levels made on
demand. Seeds are split by range so a bank can never leak between splits:
training starts at 0, validation at 1,000,000, test at 2,000,000.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

from .generate import DIFFICULTIES, GENERATOR_VERSION, generate_level

SPLIT_SEEDS = {"train": 0, "validation": 1_000_000, "test": 2_000_000}


def _one(job):
    seed, difficulty = job
    try:
        return generate_level(seed, difficulty)
    except RuntimeError:
        return None  # a seed that never drafted a completable level; just skip it


def build(count, split="train", difficulties=DIFFICULTIES, workers=None, chunksize=8):
    """Generate `count` levels, cycling through `difficulties`."""
    if split not in SPLIT_SEEDS:
        raise ValueError(f"split must be one of {sorted(SPLIT_SEEDS)}")
    base = SPLIT_SEEDS[split]
    jobs = [(base + i, difficulties[i % len(difficulties)]) for i in range(count)]
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
    stale = {spec.get("generator_version") for spec in specs} - {2, GENERATOR_VERSION}
    if stale:
        raise ValueError(f"bank was built by generator version(s) {sorted(stale)}; "
                         f"this build is version {GENERATOR_VERSION}")
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=1000)
    parser.add_argument("--split", choices=sorted(SPLIT_SEEDS), default="train")
    parser.add_argument("--difficulties", type=int, nargs="+", default=list(DIFFICULTIES))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.levels < 1:
        parser.error("--levels must be positive")
    if any(d not in DIFFICULTIES for d in args.difficulties):
        parser.error(f"difficulties must be drawn from {DIFFICULTIES}")

    started = time.perf_counter()
    specs = build(args.levels, args.split, tuple(args.difficulties), args.workers)
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
