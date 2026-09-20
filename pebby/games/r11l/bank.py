"""Build JSONL banks of verified R11L levels.

    python -m pebby.games.r11l.bank --levels N --seed S --difficulty D \
        --split train --out path.jsonl
"""

import argparse
import json
from pathlib import Path
import time

from .generate import DIFFICULTIES, SPLITS, generate


def build(count, seed=0, difficulty=1, max_attempts=None, *, split="train"):
    if count < 0:
        raise ValueError("count must be non-negative")
    max_attempts = max(1, count * 20) if max_attempts is None else max_attempts
    specs = []
    tried = 0
    while len(specs) < count and tried < max_attempts:
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
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, choices=DIFFICULTIES, default=1)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-attempts", type=int)
    args = parser.parse_args(argv)
    started = time.monotonic()
    specs, tried = build(
        args.levels, args.seed, args.difficulty, args.max_attempts, split=args.split
    )
    save(specs, args.out)
    print(f"wrote {len(specs)} levels ({tried} seeds tried, difficulty {args.difficulty}) "
          f"to {args.out} in {time.monotonic() - started:.2f}s")
    if len(specs) != args.levels:
        raise SystemExit("candidate seed limit reached; partial bank retained")


if __name__ == "__main__":
    main()
