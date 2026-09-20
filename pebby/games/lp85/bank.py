"""Build JSONL banks of replay-verified LP85 levels.

    python -m pebby.games.lp85.bank --levels N --seed S --difficulty D --out path.jsonl
"""

import argparse
import json
from pathlib import Path
import time

from .generate import DEFAULT_ATTEMPTS, DIFFICULTIES, PROFILES, SPLITS, generate


ATTEMPT_FACTOR = 12


def build(count, seed=0, difficulty=1, split="train", max_attempts=None,
          generator_attempts=DEFAULT_ATTEMPTS, node_limit=None):
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    cap = count * ATTEMPT_FACTOR if max_attempts is None else int(max_attempts)
    if cap < 0:
        raise ValueError("max_attempts must be non-negative")
    if split not in SPLITS:
        raise ValueError("split must be train, validation or test")
    specs = []
    tried = 0
    while len(specs) < count and tried < cap:
        spec = generate(
            seed + tried,
            difficulty,
            attempts=generator_attempts,
            node_limit=node_limit,
            split=split,
        )
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, choices=DIFFICULTIES, default=1)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--generator-attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--node-limit", type=int)
    args = parser.parse_args(argv)
    if args.levels < 1:
        parser.error("--levels must be positive")
    expected_work = PROFILES[args.difficulty]["search_work"]
    if args.generator_attempts < 1 or (
        args.node_limit is not None and args.node_limit != expected_work
    ):
        parser.error(
            f"--generator-attempts must be positive and --node-limit, if set, must equal {expected_work}"
        )
    started = time.perf_counter()
    specs, tried = build(
        args.levels,
        args.seed,
        args.difficulty,
        args.split,
        max_attempts=args.max_attempts,
        generator_attempts=args.generator_attempts,
        node_limit=args.node_limit,
    )
    save(specs, args.out)
    elapsed = time.perf_counter() - started
    print(
        f"wrote {len(specs)}/{args.levels} verified levels from {tried} seeds "
        f"(difficulty {args.difficulty}, split {args.split}) to {args.out} in {elapsed:.2f}s"
    )
    return 0 if len(specs) == args.levels else 1


if __name__ == "__main__":
    raise SystemExit(main())
