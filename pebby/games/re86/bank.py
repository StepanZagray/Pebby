"""Write bounded, independently replayed RE86 levels as JSONL."""

import argparse
import json
from pathlib import Path

from .generate import DEFAULT_ATTEMPTS, DIFFICULTIES, generate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, choices=DIFFICULTIES, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--node-limit", type=int, default=100_000)
    args = parser.parse_args(argv)
    if args.levels < 1:
        parser.error("--levels must be positive")
    max_seeds = args.max_seeds if args.max_seeds is not None else args.levels * 20
    if max_seeds < args.levels:
        parser.error("--max-seeds must be at least --levels")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    with args.out.open("x", encoding="utf-8") as output:
        for candidate in range(args.seed, args.seed + max_seeds):
            spec = generate(
                candidate,
                args.difficulty,
                attempts=args.attempts,
                node_limit=args.node_limit,
            )
            if spec is None:
                continue
            output.write(json.dumps(spec, sort_keys=True) + "\n")
            output.flush()
            accepted += 1
            if accepted == args.levels:
                break
    print(f"Wrote {accepted}/{args.levels} verified levels to {args.out}")
    if accepted != args.levels:
        raise SystemExit("Candidate seed limit reached; partial bank retained.")
    return 0


if __name__ == "__main__":
    main()
