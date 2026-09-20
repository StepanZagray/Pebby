"""Produce and load banks of verified S5I5 levels.

    python -m pebby.games.s5i5.bank --levels N --seed S --difficulty D --split train --out path.jsonl

Seeds S, S+1, ... are tried in order until N specs are accepted (or the seed
budget is exhausted); every stored spec carries its verified solution.
"""

import argparse
import json
from pathlib import Path
import time

from .generate import DIFFICULTIES, FORMAT, generate
from .generation_quality import SPLITS

SEED_ATTEMPTS_PER_LEVEL = 20


def build(count, seed=0, difficulty=1, max_seeds=None, *, split):
    if count < 0:
        raise ValueError("count must be nonnegative")
    if max_seeds is None:
        max_seeds = SEED_ATTEMPTS_PER_LEVEL * count
    if max_seeds < 0:
        raise ValueError("max_seeds must be nonnegative")
    specs = []
    tried = 0
    while len(specs) < count and tried < max_seeds:
        spec = generate(seed + tried, difficulty, split=split)
        tried += 1
        if spec is not None:
            specs.append(spec)
    return specs, tried


def save(specs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for spec in specs:
            handle.write(json.dumps(spec, separators=(",", ":")) + "\n")
    return path


def load(path):
    with Path(path).open() as handle:
        specs = [json.loads(line) for line in handle if line.strip()]
    bad = {s.get("format") for s in specs} - {FORMAT}
    if bad:
        raise ValueError(f"unexpected spec format(s) {sorted(map(str, bad))}; this build reads {FORMAT}")
    return specs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, choices=DIFFICULTIES, default=1)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seeds", type=int, default=None,
                        help="bound candidate seeds (default: 20 per requested level)")
    args = parser.parse_args(argv)
    if args.levels < 1:
        parser.error("--levels must be positive")
    if args.max_seeds is not None and args.max_seeds < 0:
        parser.error("--max-seeds must be nonnegative")
    started = time.perf_counter()
    specs, tried = build(args.levels, args.seed, args.difficulty, args.max_seeds,
                         split=args.split)
    elapsed = time.perf_counter() - started
    save(specs, args.out)
    lengths = sorted(s["solution_length"] for s in specs)
    print(f"{len(specs)}/{args.levels} levels from {tried} seeds in {elapsed:.1f}s -> {args.out}")
    if lengths:
        print(f"solution length: min {lengths[0]} median {lengths[len(lengths) // 2]} max {lengths[-1]}")
    return 0 if len(specs) == args.levels else 1


if __name__ == "__main__":
    raise SystemExit(main())
