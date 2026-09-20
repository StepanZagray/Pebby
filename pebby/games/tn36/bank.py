"""Write bounded, replay-proven TN36 generated levels as JSONL."""

import argparse
import json
from pathlib import Path

from .generate import DEFAULT_ATTEMPTS, DIFFICULTIES, generate
def build(levels, seed=0, difficulty=1, max_seeds=None,
          attempts=DEFAULT_ATTEMPTS, node_limit=None):
    if levels < 1:
        raise ValueError("levels must be positive")
    max_seeds = levels * 20 if max_seeds is None else max_seeds
    if max_seeds < levels:
        raise ValueError("max_seeds must be at least levels")
    specs = []
    tried = 0
    for candidate in range(seed, seed + max_seeds):
        tried += 1
        spec = generate(candidate, difficulty, attempts=attempts, node_limit=node_limit)
        if spec is not None:
            specs.append(spec)
        if len(specs) == levels:
            break
    return specs, tried


def save(specs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        for spec in specs:
            output.write(json.dumps(spec, sort_keys=True) + "\n")
    return path


def load(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, choices=DIFFICULTIES, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--node-limit", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        specs, tried = build(args.levels, args.seed, args.difficulty, args.max_seeds,
                             args.attempts, args.node_limit)
    except ValueError as exc:
        parser.error(str(exc))
    save(specs, args.out)
    print(f"Wrote {len(specs)}/{args.levels} verified levels to {args.out} after {tried} seeds")
    if len(specs) != args.levels:
        raise SystemExit("Candidate seed limit reached; partial bank retained.")
    return 0


if __name__ == "__main__":
    main()
