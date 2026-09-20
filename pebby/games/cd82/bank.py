"""Produce banks of verified CD82 levels.

    python -m pebby.games.cd82.bank --levels N --seed S --difficulty D --out path.jsonl

Seeds S, S+1, ... are tried in order until N specs are accepted (or the
attempt cap is hit); every stored spec carries its engine-verified solution.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from .generate import DIFFICULTIES, generate, tier1_split_capacities


def build(count, seed=0, difficulty=1, max_attempts=None, *, split=None):
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if split is not None and split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation or test")
    max_attempts = 20 * count if max_attempts is None else max_attempts
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    requested_split = "train" if split is None else split
    capacity = tier1_split_capacities()[requested_split] if difficulty == 1 else None
    target_count = min(count, capacity) if capacity is not None else count
    specs = []
    attempts = 0
    seen_gameplay = set()
    seen_geometry = set()
    rejections = Counter()
    while len(specs) < target_count and attempts < max_attempts:
        spec = generate(seed + attempts, difficulty, split=split)
        attempts += 1
        if spec is None:
            rejections["seed_generation_failed"] += 1
            continue
        if spec["gameplay_sha256"] in seen_gameplay:
            rejections["duplicate_gameplay"] += 1
            continue
        if spec["geometry_sha256"] in seen_geometry:
            rejections["duplicate_geometry"] += 1
            continue
        seen_gameplay.add(spec["gameplay_sha256"])
        seen_geometry.add(spec["geometry_sha256"])
        specs.append(spec)
    build.last_report = {
        "requested": count,
        "accepted": len(specs),
        "seed_attempts": attempts,
        "split": requested_split,
        "known_capacity": capacity,
        "capacity_limited": capacity is not None and count > capacity,
        "rejections": dict(sorted(rejections.items())),
    }
    return specs, attempts


build.last_report = None


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, choices=DIFFICULTIES, default=1)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    started = time.time()
    specs, attempts = build(args.levels, args.seed, args.difficulty, split=args.split)
    save(specs, args.out)
    print(f"wrote {len(specs)} levels ({attempts} seeds tried, difficulty {args.difficulty}) "
          f"to {args.out} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
