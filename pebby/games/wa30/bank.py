"""Produce banks of verified WA30 levels.

    python -m pebby.games.wa30.bank --levels N --seed S --difficulty D --out path.jsonl

Seeds S, S+1, ... are tried in order until N levels are accepted (or the
attempt cap is hit). Every stored spec carries its verified solution.
"""

import argparse
import json
from pathlib import Path
import time

from .generate import DIFFICULTIES, generate

ATTEMPT_FACTOR = 10  # give up after this many seeds per requested level


def build(count, seed, difficulty):
    """Accepted specs and the number of seeds tried."""
    specs, tried = [], 0
    while len(specs) < count and tried < count * ATTEMPT_FACTOR:
        spec = generate(seed + tried, difficulty)
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
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, choices=DIFFICULTIES, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.levels < 1:
        parser.error("--levels must be positive")
    started = time.perf_counter()
    specs, tried = build(args.levels, args.seed, args.difficulty)
    elapsed = time.perf_counter() - started
    save(specs, args.out)
    lengths = sorted(s["solution_length"] for s in specs)
    print(f"{len(specs)}/{args.levels} levels from {tried} seeds in {elapsed:.1f}s -> {args.out}")
    if lengths:
        print(f"solution length: min {lengths[0]} median {lengths[len(lengths) // 2]} max {lengths[-1]}")
    return 0 if len(specs) == args.levels else 1


if __name__ == "__main__":
    raise SystemExit(main())
