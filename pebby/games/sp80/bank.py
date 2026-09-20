"""Build JSONL banks of replay-verified SP80 levels.

    python -m pebby.games.sp80.bank --levels N --seed S --difficulty D --out path.jsonl
"""

import argparse
import json
from pathlib import Path
import time

from .generate import DIFFICULTIES, generate


ATTEMPT_FACTOR = 12


def build(count, seed=0, difficulty=1, max_attempts=None):
    if count < 0:
        raise ValueError("count must be non-negative")
    cap = count * ATTEMPT_FACTOR if max_attempts is None else int(max_attempts)
    if cap < 0:
        raise ValueError("max_attempts must be non-negative")
    specs = []
    tried = 0
    while len(specs) < count and tried < cap:
        spec = generate(seed + tried, difficulty)
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
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, choices=DIFFICULTIES, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.levels < 1:
        parser.error("--levels must be positive")
    started = time.perf_counter()
    specs, tried = build(args.levels, args.seed, args.difficulty)
    save(specs, args.out)
    elapsed = time.perf_counter() - started
    print(
        f"wrote {len(specs)}/{args.levels} verified levels from {tried} seeds "
        f"(difficulty {args.difficulty}) to {args.out} in {elapsed:.2f}s"
    )
    return 0 if len(specs) == args.levels else 1


if __name__ == "__main__":
    raise SystemExit(main())
