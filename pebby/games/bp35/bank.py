"""Write replay-verified BP35 generated specs as JSON Lines."""

import argparse
import json
from pathlib import Path

from .generate import generate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--difficulty", type=int, choices=tuple(range(1, 10)), required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.levels < 0:
        parser.error("--levels must be nonnegative")
    records = []
    for offset in range(args.levels):
        spec = generate(args.seed + offset, args.difficulty, split=args.split)
        if spec is None:
            raise SystemExit(f"generation failed for seed {args.seed + offset}")
        records.append(spec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
