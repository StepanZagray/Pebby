#!/usr/bin/env python3
"""Collect and independently validate a small generated mechanism world pilot.

The collector itself is the unchanged ``pebby.agent.world_data`` implementation.
Every retained state is expanded through all four real engine actions and checked
against the complete contextual Oracle.  The mechanism source bank is training
only; no held-out or official level is read.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time
from unittest.mock import patch

import numpy as np

from pebby.agent import world_data
from pebby.ls20.generate import FORMAT as LEVEL_FORMAT, GENERATOR_VERSION
from tools.generate_mechanism_pilot import MODES
from tools.validate_extended_collector import checked_expansion

def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def canonical_spec_hash(spec: dict) -> str:
    fields = ("size", "walls", "start", "start_triple", "goals", "cyclers", "rails",
              "launchers", "refills", "step_counter", "step_cost", "fog")
    payload = json.dumps({key: spec.get(key, []) for key in fields}, sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def choose_specs(path: Path, per_mode: int) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_mode = {mode: [] for mode in MODES}
    seen = set()
    for spec in rows:
        seed = int(spec.get("seed", -1))
        if spec.get("format") != LEVEL_FORMAT or spec.get("generator_version") != GENERATOR_VERSION:
            raise ValueError(f"unsupported source row for seed {seed}")
        if spec.get("split") != "train" or spec.get("source") != "generated_only":
            raise ValueError(f"mechanism source row is not training/generated-only: {seed}")
        if spec.get("pilot_mode") not in by_mode:
            raise ValueError(f"unknown mechanism mode for seed {seed}: {spec.get('pilot_mode')}")
        if seed in seen:
            raise ValueError(f"duplicate mechanism source seed {seed}")
        seen.add(seed)
        if not (700_000 <= seed < 900_000):
            raise ValueError(f"mechanism source seed outside training namespace: {seed}")
        if seed % 7 == 0 and spec.get("launchers"):
            raise ValueError(f"context-zero launchers are outside the oracle contract: {seed}")
        if spec.get("context_engine_verified") is not True or spec.get("search_truncated") is not False:
            raise ValueError(f"source context proof incomplete for seed {seed}")
        if spec.get("training_context_index") != seed % 7:
            raise ValueError(f"source context mismatch for seed {seed}")
        proof = spec.get("proof")
        if not isinstance(proof, dict) or proof.get("context_engine_verified") is not True \
                or proof.get("context_index") != seed % 7:
            raise ValueError(f"nested source proof incomplete for seed {seed}")
        by_mode[spec["pilot_mode"]].append(spec)
    for mode in MODES:
        by_mode[mode].sort(key=lambda spec: int(spec["seed"]))
        if len(by_mode[mode]) < per_mode:
            raise ValueError(f"mode {mode} has only {len(by_mode[mode])} rows; need {per_mode}")
    selected = [spec for mode in MODES for spec in by_mode[mode][:per_mode]]
    if len({int(spec["seed"]) for spec in selected}) != len(selected):
        raise ValueError("selected mechanism seeds are not distinct")
    return selected, {mode: [int(spec["seed"]) for spec in by_mode[mode][:per_mode]] for mode in MODES}


def collect_checked(specs: list[dict], history: int, samples: int, epsilon: float,
                    search_limit: int):
    counts = Counter()
    original = world_data._expand

    def checked(*args, **kwargs):
        return checked_expansion(original, counts, *args, **kwargs)

    # workers=1 is deliberate: it keeps the checked expansion in this process
    # and bounds both memory and process cleanup for the pilot.
    with patch.object(world_data, "_expand", side_effect=checked):
        arrays = world_data.build(specs, workers=1, history=history, samples=samples,
                                  epsilon=epsilon, coverage="mixed_failure", search_limit=search_limit,
                                  progress=True)
    expected = {int(spec["seed"]) for spec in specs}
    actual = set(map(int, np.unique(arrays["seeds"])))
    if actual != expected:
        raise ValueError(f"collector seed set mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
    if len(arrays["meta"].get("levels", [])) != len(specs):
        raise ValueError("collector omitted a level proof")
    for proof in arrays["meta"]["levels"]:
        seed = int(proof["seed"])
        if proof.get("context_engine_verified") is not True or proof.get("search_truncated") is not False \
                or proof.get("context_index") != seed % 7:
            raise ValueError(f"collector proof invalid for seed {seed}: {proof}")
    if not np.array_equal(arrays["context_index"], arrays["seeds"] % 7):
        raise ValueError("collector row context mismatch")
    if np.any(arrays["won"] & ~arrays["terminal"]):
        raise ValueError("winning successor is not terminal")
    if np.any(arrays["terminal"] & (arrays["next_optimal"] != 0)):
        raise ValueError("terminal successor has nonzero optimal mask")
    win_seeds = set(map(int, arrays["seeds"][np.any(arrays["won"], axis=1)]))
    if win_seeds != expected:
        raise ValueError(f"missing actual winning successor coverage: {sorted(expected - win_seeds)}")
    if counts["branches"] != 4 * counts["expansions"]:
        raise ValueError(f"four-action branch validation incomplete: {counts}")
    return arrays, dict(counts)


def require_verified_data(arrays):
    meta = arrays.get("meta", {})
    if meta.get("source") != "generated_only" or meta.get("oracle_search") != "complete_only":
        raise ValueError("pilot requires generated-only, complete-oracle provenance")
    proofs = {int(row["seed"]): row for row in meta.get("levels", [])}
    for seed in np.unique(arrays["seeds"]):
        proof = proofs.get(int(seed), {})
        if proof.get("context_engine_verified") is not True or proof.get("search_truncated", False):
            raise ValueError(f"seed {seed} lacks an untruncated contextual engine proof")
        if proof.get("context_index") != int(seed) % 7:
            raise ValueError(f"seed {seed} was verified in the wrong context")


def require_winning_coverage(arrays):
    seeds = np.asarray(arrays["seeds"])
    won = np.asarray(arrays["won"], dtype=bool)
    terminal = np.asarray(arrays["terminal"], dtype=bool)
    if np.any(won & ~terminal):
        raise ValueError("won successors must be terminal")
    if np.any(terminal & (arrays["next_optimal"] != 0)):
        raise ValueError("terminal successors must have zero next_optimal")
    winning_seeds = set(map(int, seeds[np.any(won, axis=1)]))
    missing = sorted(set(map(int, np.unique(seeds))) - winning_seeds)
    if missing:
        raise ValueError(f"missing actual winning successor for seeds {missing}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("data/mechanism-bank-v2/train.jsonl"))
    parser.add_argument("--per-mode", type=int, default=5)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--samples-per-level", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=.15)
    parser.add_argument("--search-limit", type=int, default=600_000)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--generator-report", type=Path,
                        help="Optional completed generation report bound to this exact bank")
    parser.add_argument("--generator-proof-sha256",
                        help="Optional expected digest; requires --generator-report")
    parser.add_argument("--out", type=Path, default=Path("data/ls20-world-mechanism-pilot25.npz"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/world-mechanism-pilot25.json"))
    args = parser.parse_args(argv)
    if args.per_mode < 1 or args.history < 1 or args.samples_per_level < 1 \
            or args.search_limit < 1 or not 0 <= args.epsilon <= 1 or not 1 <= args.seconds <= 900:
        parser.error("counts/search-limit/seconds must be positive, seconds <=900, and epsilon in [0,1]")
    if args.out.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite mechanism pilot output/report")
    if args.generator_proof_sha256 and args.generator_report is None:
        parser.error('--generator-proof-sha256 requires the actual --generator-report')
    if args.generator_report is not None:
        generator_report = json.loads(args.generator_report.read_text())
        actual_digest = digest(args.generator_report)
        if args.generator_proof_sha256 and args.generator_proof_sha256 != actual_digest:
            raise ValueError('generator report digest mismatch')
        bank_digest = digest(args.bank)
        bound_hashes = {entry.get('sha256') for entry in generator_report.get('banks', {}).values()}
        bound_hashes.add(generator_report.get('bank_sha256'))
        if generator_report.get('status') != 'complete' or bank_digest not in bound_hashes:
            raise ValueError('generation report does not certify this complete bank')
        args.generator_proof_sha256 = actual_digest
    started = time.monotonic()
    report = {"status": "running", "format": "pebby.ls20-world-mechanism-pilot.v1",
              "pid": os.getpid(), "device": "cpu", "workers": 1, "cpu_threads": 1,
              "official_inputs_used": False, "heldout_inputs_used": False,
              "history": args.history, "samples_per_level": args.samples_per_level,
              "epsilon": args.epsilon, "coverage": "mixed_failure", "per_mode": args.per_mode,
              "seconds": args.seconds, "generator_proof_sha256": args.generator_proof_sha256,
              "splits": {}, "coverage_limit": "All four actions from every retained collector state; not exhaustive reachable-state enumeration."}

    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.report)

    print("PID", os.getpid(), flush=True)
    persist()
    def timeout(_signum, _frame):
        raise TimeoutError(f"mechanism collection exceeded {args.seconds}-second bound")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.seconds)
    try:
        specs, selected_by_mode = choose_specs(args.bank, args.per_mode)
        report["selected_by_mode"] = selected_by_mode
        report["selected_levels"] = [{"seed": int(spec["seed"]), "pilot_mode": spec["pilot_mode"],
                                       "difficulty": int(spec["difficulty"]),
                                       "goals": len(spec.get("goals", [])),
                                       "source_spec_sha256": canonical_spec_hash(spec)} for spec in specs]
        code_paths = [Path(__file__), Path("tools/validate_extended_collector.py"),
                      Path("tools/generate_mechanism_pilot.py"), Path("pebby/agent/world_data.py"),
                      Path("pebby/ls20/env.py"), Path("pebby/ls20/generate.py"),
                      Path("pebby/ls20/plan.py"), Path("pebby/ls20/layout.py"),
                      Path("pebby/ls20/rails.py"), Path("pebby/ls20/fastplan.py"),
                      Path("pebby/ls20/_fastplan.c")]
        code_hashes = {str(path): digest(path) for path in code_paths}
        source_sha = digest(args.bank)
        report["source"] = {"path": str(args.bank), "sha256": source_sha,
                            "selected_rows": len(specs), "selected_by_mode": selected_by_mode,
                            "generator_proof_sha256": args.generator_proof_sha256}
        report["code_hashes"] = code_hashes
        persist()
        arrays, branch_counts = collect_checked(specs, args.history, args.samples_per_level,
                                                args.epsilon, args.search_limit)
        # Keep this generated-only pilot compatible with the normal trainer's
        # proof guards, without importing unrelated extended-bank namespaces.
        arrays["meta"].update({
            "source": "generated_only", "oracle_search": "complete_only",
            "mechanism_source_bank": str(args.bank), "mechanism_source_bank_sha256": source_sha,
            "generator_proof_sha256": args.generator_proof_sha256,
            "mechanism_pilot_modes": list(MODES), "mechanism_selected_by_mode": selected_by_mode,
            "collector_branch_verification": branch_counts,
            "provenance_code_hashes": code_hashes,
            "coverage_limit": report["coverage_limit"],
        })
        require_verified_data(arrays)
        require_winning_coverage(arrays)
        if len(np.unique(arrays["seeds"])) != len(specs):
            raise ValueError("collector output does not represent every selected level")
        finite_distances = arrays["distances"][arrays["distances"] >= 0]
        winning_row_mask = np.any(arrays["won"], axis=1)
        winning_level_count = len(np.unique(arrays["seeds"][winning_row_mask]))
        row_goal_counts = {str(mode): int(sum(len(spec["goals"]) == count for spec in specs
                                             if spec["pilot_mode"] == mode))
                           for mode, count in (("three_goals", 3), ("four_goals", 4))}
        outcomes = Counter()
        for key in ("won", "terminal", "lost_life"):
            outcomes[key] = int(np.asarray(arrays[key], dtype=bool).sum())
        report["arrays"] = {"rows": int(len(arrays["seeds"])),
                            "levels": int(len(np.unique(arrays["seeds"]))),
                            "three_goal_levels": row_goal_counts["three_goals"],
                            "four_goal_levels": row_goal_counts["four_goals"],
                            "finite_distance_values": int(finite_distances.size),
                            "finite_distance_gt63": int(np.sum(finite_distances > 63)),
                            "finite_distance_min": int(finite_distances.min()) if finite_distances.size else None,
                            "finite_distance_max": int(finite_distances.max()) if finite_distances.size else None,
                            "winning_successor_rows": int(winning_row_mask.sum()),
                            "winning_successor_levels": int(winning_level_count),
                            "terminal_successor_labels_zero": bool(np.all(arrays["next_optimal"][arrays["terminal"]] == 0)),
                            "outcomes": dict(outcomes)}
        report["branch_verification"] = branch_counts
        report["rates"] = {
            "branch_check_rate": branch_counts["branches"] / max(1, 4 * branch_counts["expansions"]),
            "actual_winning_successor_level_rate": report["arrays"]["winning_successor_levels"] / len(specs),
            "terminal_next_optimal_zero_rate": 1.0 if report["arrays"]["terminal_successor_labels_zero"] else 0.0,
        }
        temporary = args.out.with_suffix(".tmp.npz")
        world_data.save(temporary, arrays)
        temporary.replace(args.out)
        report["output"] = {"path": str(args.out), "sha256": digest(args.out),
                            "meta_format": arrays["meta"]["format"]}
        report["status"] = "complete"
    except TimeoutError as error:
        report["status"] = "bounded_partial"
        report["error"] = repr(error)
        raise
    except Exception as error:
        report["status"] = "failed_closed"
        report["error"] = repr(error)
        raise
    finally:
        signal.alarm(0)
        report["source_unchanged"] = args.bank.exists() and digest(args.bank) == report.get("source", {}).get("sha256")
        report["code_unchanged"] = all(digest(Path(path)) == value for path, value in report.get("code_hashes", {}).items())
        if not report["source_unchanged"] or not report["code_unchanged"]:
            report["status"] = "failed_closed"
        persist()
        print(json.dumps({key: report.get(key) for key in ("status", "elapsed_seconds", "peak_rss_mib", "output")}, indent=2), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
