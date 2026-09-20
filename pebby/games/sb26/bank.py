"""Build JSONL banks of engine-certified SB26 levels.

Usage::

    python -m pebby.games.sb26.bank --levels N --seed S --difficulty D --out levels.jsonl
"""

import argparse
import json
from pathlib import Path
import time

from .generate import DEFAULT_ATTEMPTS, DEFAULT_NODE_LIMIT, DIFFICULTIES, generate


def build(count, seed=0, difficulty=1, max_seeds=None, attempts=DEFAULT_ATTEMPTS, node_limit=DEFAULT_NODE_LIMIT):
    count = int(count)
    if count < 0:
        raise ValueError("count must be non-negative")
    max_seeds = 20 * max(1, count) if max_seeds is None else int(max_seeds)
    specs = []
    tried = 0
    while len(specs) < count and tried < max_seeds:
        spec = generate(seed + tried, difficulty, attempts=attempts, node_limit=node_limit)
        tried += 1
        if spec is not None:
            specs.append(spec)
    return specs, tried


def save(specs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(spec, separators=(",", ":")) + "\n")
    return path


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, choices=DIFFICULTIES, default=1)
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--node-limit", type=int, default=DEFAULT_NODE_LIMIT)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    specs, tried = build(
        args.levels,
        seed=args.seed,
        difficulty=args.difficulty,
        max_seeds=args.max_seeds,
        attempts=args.attempts,
        node_limit=args.node_limit,
    )
    save(specs, args.out)
    print(
        f"wrote {len(specs)} levels ({tried} seeds tried, difficulty {args.difficulty}) "
        f"to {args.out} in {time.monotonic() - started:.2f}s"
    )
    return 0 if len(specs) == args.levels else 1


if __name__ == "__main__":
    raise SystemExit(main())
