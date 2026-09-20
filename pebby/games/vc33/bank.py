"""Write bounded, replay-certified VC33 level specs as JSONL."""

import argparse
import json
from pathlib import Path

from .generate import DIFFICULTIES, generate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, choices=DIFFICULTIES, default=1)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--node-limit", type=int, default=None)
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
    with args.out.open("x", encoding="utf-8") as output:
        for candidate in range(args.seed, args.seed + max_seeds):
            spec = generate(candidate, args.difficulty, split=args.split,
                            attempts=args.attempts, node_limit=args.node_limit)
            if spec is None:
                continue
            geometry = spec["geometry_d4_sha256"]
            gameplay = spec["gameplay_sha256"]
            if geometry in seen_geometry or gameplay in seen_gameplay:
                continue
            seen_geometry.add(geometry)
            seen_gameplay.add(gameplay)
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
