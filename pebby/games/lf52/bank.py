"""Build JSONL banks of verified LF52 levels.

Usage: ``python -m pebby.games.lf52.bank --levels N --seed S --difficulty D --out FILE``
"""

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time

from .generate import DIFFICULTIES, generate


ATTEMPT_FACTOR = 10
FAILED_SEED_DIAGNOSTIC_LIMIT = 8
BANK_DIAGNOSTICS_FORMAT = "pebby.lf52.bank-diagnostics.v1"


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _serialized_bank(specs):
    return "".join(json.dumps(spec, separators=(",", ":")) + "\n" for spec in specs).encode()


def _failure_record(requested_seed):
    generation = _json_copy(generate.last_diagnostics)
    rejected = generation.get("rejected", {}) if isinstance(generation, dict) else {}
    causes = sorted(key for key, count in rejected.items() if type(count) is int and count > 0)
    cause = causes[0] if len(causes) == 1 else "multiple" if causes else "unknown"
    details = generation.get("examples", []) if isinstance(generation, dict) else []
    return {
        "requested_seed": requested_seed,
        "cause": cause,
        "details": _json_copy(details),
        "generation_diagnostics": generation,
    }


def build(count, seed=0, difficulty=1, max_attempts=None, **generate_kwargs):
    if type(count) is not int or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    cap = count * ATTEMPT_FACTOR if max_attempts is None else max_attempts
    if type(cap) is not int or cap < 0:
        raise ValueError("max_attempts must be a nonnegative integer")
    specs = []
    seen_geometry = set()
    seen_gameplay = set()
    failed_requested_seeds = []
    tried = 0
    rejected = {"generation_failed": 0, "duplicate_geometry": 0, "duplicate_gameplay": 0}
    while len(specs) < count and tried < cap:
        requested_seed = seed + tried
        spec = generate(requested_seed, difficulty, **generate_kwargs)
        tried += 1
        if spec is None:
            rejected["generation_failed"] += 1
            if len(failed_requested_seeds) < FAILED_SEED_DIAGNOSTIC_LIMIT:
                # Copy now: the next call replaces generate.last_diagnostics.
                failed_requested_seeds.append(_failure_record(requested_seed))
            continue
        geometry = spec.get("geometry_d4_sha256")
        gameplay = spec.get("gameplay_sha256")
        if geometry in seen_geometry:
            rejected["duplicate_geometry"] += 1
            continue
        if gameplay in seen_gameplay:
            rejected["duplicate_gameplay"] += 1
            continue
        seen_geometry.add(geometry)
        seen_gameplay.add(gameplay)
        specs.append(spec)
    serialized_bank_sha256 = hashlib.sha256(_serialized_bank(specs)).hexdigest()
    build.last_diagnostics = {
        "requested": count,
        "accepted": len(specs),
        "tried": tried,
        "hard_cap": cap,
        "rejected": rejected,
        "complete": len(specs) == count,
        "failed_requested_seeds": failed_requested_seeds,
        "failed_requested_seeds_omitted": rejected["generation_failed"] - len(failed_requested_seeds),
        "serialized_bank_sha256": serialized_bank_sha256,
        "accepted_geometry_d4_sha256": [spec["geometry_d4_sha256"] for spec in specs],
        "accepted_gameplay_sha256": [spec["gameplay_sha256"] for spec in specs],
        "accepted_action_sequence_sha256": [spec["action_sequence_sha256"] for spec in specs],
    }
    return specs, tried


build.last_diagnostics = None


def save(specs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(spec, separators=(",", ":")) + "\n")
    return path


def save_diagnostics(specs, path, diagnostics=None):
    """Persist an opt-in report bound to the exact JSONL bank serialization."""
    specs = list(specs)
    if diagnostics is None:
        diagnostics = build.last_diagnostics
    if not isinstance(diagnostics, Mapping):
        raise ValueError("bank diagnostics must be a mapping from a completed build")
    try:
        diagnostics = _json_copy(diagnostics)
        serialized_hash = hashlib.sha256(_serialized_bank(specs)).hexdigest()
        expected = {
            "serialized_bank_sha256": serialized_hash,
            "accepted": len(specs),
            "accepted_geometry_d4_sha256": [spec["geometry_d4_sha256"] for spec in specs],
            "accepted_gameplay_sha256": [spec["gameplay_sha256"] for spec in specs],
            "accepted_action_sequence_sha256": [spec["action_sequence_sha256"] for spec in specs],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"bank diagnostics/specs are malformed: {exc}") from exc
    mismatched = [key for key, value in expected.items() if diagnostics.get(key) != value]
    if mismatched:
        raise ValueError(
            "bank diagnostics do not describe the supplied specs: "
            + ", ".join(mismatched)
        )
    report = {
        "format": BANK_DIAGNOSTICS_FORMAT,
        "bank_sha256": serialized_hash,
        "diagnostics": diagnostics,
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return path


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, choices=DIFFICULTIES, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diagnostics-out", type=Path)
    parser.add_argument("--attempts", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    args = parser.parse_args(argv)
    if args.levels < 1:
        parser.error("--levels must be positive")
    kwargs = {}
    if args.attempts is not None:
        kwargs["attempts"] = args.attempts
    if args.limit is not None:
        kwargs["limit"] = args.limit
    kwargs["split"] = args.split
    if args.diagnostics_out is not None and args.diagnostics_out.resolve() == args.out.resolve():
        parser.error("--diagnostics-out must differ from --out")
    started = time.monotonic()
    specs, tried = build(args.levels, args.seed, args.difficulty, **kwargs)
    save(specs, args.out)
    if args.diagnostics_out is not None:
        save_diagnostics(specs, args.diagnostics_out)
    elapsed = time.monotonic() - started
    print(
        f"wrote {len(specs)}/{args.levels} verified levels from {tried} seeds "
        f"(difficulty {args.difficulty}) to {args.out} in {elapsed:.2f}s; "
        f"diagnostics={json.dumps(build.last_diagnostics, sort_keys=True, separators=(',', ':'))}"
    )
    return 0 if len(specs) == args.levels else 1


if __name__ == "__main__":
    raise SystemExit(main())
