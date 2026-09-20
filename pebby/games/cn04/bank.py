"""Write JSONL banks of engine-certified CN04 levels.

    python -m pebby.games.cn04.bank --levels N --seed S --difficulty D --out path.jsonl
"""

import argparse
import json
from pathlib import Path
import time

from .generate import DIFFICULTIES, generate


ATTEMPT_FACTOR = 20


def build(count, seed, difficulty, *, split, max_seeds=None):
    if count < 0:
        raise ValueError("count must be nonnegative")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    cap = count * ATTEMPT_FACTOR if max_seeds is None else int(max_seeds)
    if cap < 0:
        raise ValueError("max_seeds must be nonnegative")
    specs = []
    tried = 0
    while len(specs) < count and tried < cap:
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
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, choices=DIFFICULTIES, default=1)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.levels < 0:
        parser.error("--levels must be nonnegative")
    started = time.perf_counter()
    specs, tried = build(args.levels, args.seed, args.difficulty, split=args.split)
    save(specs, args.out)
    elapsed = time.perf_counter() - started
    print(f"{len(specs)}/{args.levels} levels from {tried} seeds in {elapsed:.1f}s -> {args.out}")
    if specs:
        lengths = sorted(spec["solution_length"] for spec in specs)
        print(f"solution length: min {lengths[0]} median {lengths[len(lengths) // 2]} max {lengths[-1]}")
    return 0 if len(specs) == args.levels else 1


if __name__ == "__main__":
    raise SystemExit(main())
