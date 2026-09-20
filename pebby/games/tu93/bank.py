"""Build JSONL banks of verified TU93 levels.

    python -m pebby.games.tu93.bank --levels N --seed S --difficulty D --out path.jsonl
"""

import argparse
import json
from pathlib import Path
import time

from .generate import DIFFICULTIES, SPLITS, generate, generate_game


ATTEMPT_FACTOR = 10


def build(count, seed=0, difficulty=1, max_attempts=None, split="train"):
    if count < 0:
        raise ValueError("count must be non-negative")
    cap = count * ATTEMPT_FACTOR if max_attempts is None else int(max_attempts)
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
    with path.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(spec, separators=(",", ":")) + "\n")
    return path


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, choices=DIFFICULTIES, default=1)
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument(
        "--whole-game", action="store_true",
        help="write one complete increasing-difficulty nine-level game",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if args.whole_game:
        if args.levels is not None:
            parser.error("--levels cannot be combined with --whole-game")
        specs = generate_game(args.seed, split=args.split)
        tried = 1
        if specs is None:
            specs = []
        requested = len(DIFFICULTIES)
    else:
        if args.levels is None or args.levels < 1:
            parser.error("--levels must be positive unless --whole-game is used")
        requested = args.levels
        specs, tried = build(
            args.levels, args.seed, args.difficulty, split=args.split
        )
    save(specs, args.out)
    elapsed = time.perf_counter() - started
    print(
        f"wrote {len(specs)}/{requested} verified levels from {tried} seeds "
        f"(difficulty {args.difficulty}, split {args.split}) to {args.out} in {elapsed:.2f}s"
    )
    return 0 if len(specs) == requested else 1


if __name__ == "__main__":
    raise SystemExit(main())
