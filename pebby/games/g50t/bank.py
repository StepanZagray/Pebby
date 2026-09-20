"""Build JSONL banks of independently verified G50T levels."""

import argparse
import json
from pathlib import Path

from .generate import DEFAULT_ATTEMPTS, DIFFICULTIES, generate
from .plan import DEFAULT_NODE_LIMIT


ATTEMPT_FACTOR = 12


def build(count, seed=0, difficulty=1, max_seeds=None,
          attempts=DEFAULT_ATTEMPTS, node_limit=DEFAULT_NODE_LIMIT):
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    cap = count * ATTEMPT_FACTOR if max_seeds is None else int(max_seeds)
    if cap < 0:
        raise ValueError("max_seeds must be non-negative")
    specs = []
    tried = 0
    while len(specs) < count and tried < cap:
        spec = generate(
            seed + tried,
            difficulty,
            attempts=attempts,
            node_limit=node_limit,
        )
        tried += 1
        if spec is not None:
            specs.append(spec)
    return specs, tried


def save(specs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(spec, sort_keys=True) + "\n")
    return path


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, choices=DIFFICULTIES, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--node-limit", type=int, default=DEFAULT_NODE_LIMIT)
    args = parser.parse_args(argv)
    if args.levels < 1:
        parser.error("--levels must be positive")
    specs, tried = build(
        args.levels,
        args.seed,
        args.difficulty,
        max_seeds=args.max_seeds,
        attempts=args.attempts,
        node_limit=args.node_limit,
    )
    save(specs, args.out)
    print(f"Wrote {len(specs)}/{args.levels} verified levels from {tried} seeds to {args.out}")
    return 0 if len(specs) == args.levels else 1


if __name__ == "__main__":
    raise SystemExit(main())
