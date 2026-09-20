"""Write split-qualified, intended-context verified TR87 levels as JSONL."""

import argparse
import json
from pathlib import Path

from .generate import DIFFICULTIES, generate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, choices=DIFFICULTIES, default=1)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seeds", type=int, default=None,
                        help="Bound candidate seeds (default: 20 times requested levels).")
    args = parser.parse_args(argv)
    if args.levels < 1:
        parser.error("--levels must be positive")
    max_seeds = args.max_seeds if args.max_seeds is not None else args.levels * 20
    if max_seeds < args.levels:
        parser.error("--max-seeds must be at least --levels")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    seen_geometry = set()
    seen_gameplay = set()
    # Exclusive creation protects existing banks; partial output remains usable
    # if the bounded search fails to fill the requested count.
    with args.out.open("x", encoding="utf-8") as output:
        for seed in range(args.seed, args.seed + max_seeds):
            spec = generate(seed, args.difficulty, split=args.split)
            if spec is None:
                continue
            if (spec["geometry_d4_sha256"] in seen_geometry
                    or spec["gameplay_sha256"] in seen_gameplay):
                continue
            seen_geometry.add(spec["geometry_d4_sha256"])
            seen_gameplay.add(spec["gameplay_sha256"])
            output.write(json.dumps(spec, sort_keys=True) + "\n")
            output.flush()
            accepted += 1
            if accepted == args.levels:
                break
    print(f"Wrote {accepted}/{args.levels} verified levels to {args.out}")
    if accepted != args.levels:
        raise SystemExit("Candidate seed limit reached; partial bank retained.")


if __name__ == "__main__":
    main()
