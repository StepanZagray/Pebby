"""Write bounded, independently proved LS20 generated levels as JSONL."""

import argparse
import json
from pathlib import Path

from .generate import DIFFICULTIES, generate, generate_game


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--difficulty", type=int, choices=DIFFICULTIES)
    mode.add_argument(
        "--whole-game", action="store_true",
        help="Write one complete ordered seven-tier spec list per JSONL row.",
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="train",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--max-seeds", type=int, default=None,
        help="Bound candidate seeds (default: 20 times requested levels).",
    )
    args = parser.parse_args(argv)
    difficulty = 1 if args.difficulty is None else args.difficulty
    if args.levels < 1:
        parser.error("--levels must be positive")
    max_seeds = args.max_seeds if args.max_seeds is not None else args.levels * 20
    if max_seeds < args.levels:
        parser.error("--max-seeds must be at least --levels")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    with args.out.open("x", encoding="utf-8") as output:
        for candidate in range(args.seed, args.seed + max_seeds):
            value = (
                generate_game(candidate, split=args.split)
                if args.whole_game
                else generate(candidate, difficulty, split=args.split)
            )
            if value is None:
                continue
            output.write(json.dumps(value, sort_keys=True) + "\n")
            output.flush()
            accepted += 1
            if accepted == args.levels:
                break
    unit = "whole games" if args.whole_game else "levels"
    print(f"Wrote {accepted}/{args.levels} verified {unit} to {args.out}")
    if accepted != args.levels:
        raise SystemExit("Candidate seed limit reached; partial bank retained.")
    return 0


if __name__ == "__main__":
    main()
