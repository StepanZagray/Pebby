"""Write a bank of verified FT09 level specs as JSON lines.

    python -m pebby.games.ft09.bank --levels N --seed S --difficulty D --out path.jsonl

Seeds S, S+1, ... are tried in order until N specs are accepted; each line is
one spec including its verified solution.
"""

import argparse
import json
from pathlib import Path
import sys
import time

from .generate import DIFFICULTIES, generate


def build(levels, seed, difficulty, max_seeds=None, *, split=None):
    if levels < 0 or (max_seeds is not None and max_seeds < 0):
        raise ValueError("levels and max_seeds must be nonnegative")
    specs = []
    geometry_identities = set()
    gameplay_identities = set()
    current = seed
    max_seeds = levels * 50 if max_seeds is None else max_seeds
    while len(specs) < levels and current < seed + max_seeds:
        spec = generate(current, difficulty, split=split)
        if (spec is not None
                and spec["geometry_sha256"] not in geometry_identities
                and spec["gameplay_sha256"] not in gameplay_identities):
            specs.append(spec)
            geometry_identities.add(spec["geometry_sha256"])
            gameplay_identities.add(spec["gameplay_sha256"])
        current += 1
    return specs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, choices=DIFFICULTIES, default=1)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.levels < 0:
        parser.error("--levels must be nonnegative")
    start = time.monotonic()
    specs = build(args.levels, args.seed, args.difficulty, split=args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for spec in specs:
            handle.write(json.dumps(spec, separators=(",", ":")) + "\n")
    elapsed = time.monotonic() - start
    print(f"wrote {len(specs)} specs to {args.out} in {elapsed:.1f}s", file=sys.stderr)
    return 0 if len(specs) == args.levels else 1


if __name__ == "__main__":
    sys.exit(main())
