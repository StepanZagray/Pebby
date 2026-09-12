#!/usr/bin/env python3
"""Faster generated life-loss transition diagnostics.

Ordinary trajectory steps execute and verify only the sampled real action.  If
that action actually loses a life, the untouched pre-action state is passed to
the v3 four-action capture routine, so retained rows keep the exact v3 labels,
frames, history contract, and mechanics checks.  This file is intentionally
separate from the accepted v3 collector.
"""

from __future__ import annotations

from pebby.ls20.provenance import generated_context, validate_difficulty

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

import numpy as np

from pebby.agent import world_data
from pebby.ls20 import names
from pebby.ls20.plan import simulate

from tools import collect_structured_event_transitions as base


FORMAT = "pebby.ls20-structured-event-transitions-fast.v1"
HISTORY = base.HISTORY
ACTION_COUNT = base.ACTION_COUNT


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def _verify_actual_branch(env, oracle, branch, result, seed, step, action_index):
    """Verify one sampled real branch against clone/reference and ``simulate``."""
    if result.frame is None:
        raise base.TransitionMismatch(
            f"seed {seed} step {step} action {action_index}: no frame")
    reference = world_data.clone_env(env)
    reference_result = reference.perform(names.ACTION_IDS[action_index])
    if (not base._same_frame(result.frame, reference_result.frame)
            or not base._same_branch_state(branch, reference)):
        raise base.TransitionMismatch(
            f"seed {seed} step {step} action {action_index}: clone/reference mismatch")

    current_state = oracle.state_of(env)
    branch_state = oracle.state_of(branch)
    predicted, outcome = simulate(oracle.layout, current_state, action_index, oracle.refills)
    if outcome == "won":
        agrees = result.won and branch.lives() == env.lives()
    elif outcome == "died":
        agrees = (not result.won and branch.lives() == env.lives() - 1
                  and (result.finished or branch_state == oracle.start))
    else:
        agrees = (not result.finished and branch.lives() == env.lives()
                  and branch_state == predicted)
    if not agrees:
        raise base.TransitionMismatch(
            f"seed {seed} step {step} action {action_index}: {outcome} disagrees")
    return 1, 1


def collect_level_fast(spec: dict, split: str, history: int, max_actions: int,
                       search_limit: int, rng: np.random.Generator):
    env, oracle, proof = world_data.verified_context(
        spec, context_index=generated_context(spec), search_limit=search_limit)
    if env is None or oracle is None:
        return [], {**proof, "split": split, "shortfall": "context_verification_failed"}, 0, 0, 0, 0

    frames, actions = [env.render()], [-1]
    rows = []
    ordinary_checks = 0
    branch_checks = 0
    engine_branch_checks = 0
    oracle_unverified_branches = 0
    retained_oracle_checked_branches = 0
    retained_oracle_unverified_branches = 0
    actual_losses = 0
    episode_status = "action_bound"
    step = -1
    for step in range(max_actions):
        if env.state.value in ("WIN", "GAME_OVER"):
            episode_status = env.state.value.lower()
            break
        observed, valid, previous = world_data.history_arrays(frames, actions, history)
        selected = int(rng.integers(0, ACTION_COUNT))

        # The original env is untouched until the selected branch is adopted.
        # If this branch loses a life, capture_branches receives this same
        # pre-action env and performs the complete four-action proof.
        branch = world_data.clone_env(env)
        result = branch.perform(names.ACTION_IDS[selected])
        engine_check, mechanics_check = _verify_actual_branch(
            env, oracle, branch, result, int(spec["seed"]), step, selected)
        ordinary_checks += mechanics_check
        engine_branch_checks += engine_check

        selected_loss = int(branch.lives()) < int(env.lives())
        selected_terminal = bool(result.finished)
        selected_won = bool(result.won)
        if selected_loss:
            capture = base.capture_branches(env, oracle, int(spec["seed"]), step)
            branch_checks += int(capture.pop("branch_checks"))
            engine_branch_checks += int(capture.pop("engine_branch_checks"))
            oracle_unverified_branches += int(capture["oracle_unverified_branches"])
            retained_oracle_checked_branches += (
                ACTION_COUNT - int(capture["oracle_unverified_branches"]))
            retained_oracle_unverified_branches += int(capture["oracle_unverified_branches"])
            # The selected branch was already checked above.  Ensure the
            # retained v3-equivalent branch has exactly the same real result.
            captured_branch = capture["_branches"][selected]
            captured_result = capture["_results"][selected]
            if (not base._same_frame(result.frame, captured_result.frame)
                    or not base._same_branch_state(branch, captured_branch)):
                raise base.TransitionMismatch(
                    f"seed {spec['seed']} step {step}: selected branch changed during capture")
            rows.append(base._row_from_capture(env, capture, observed, valid, previous,
                                               spec, split, step, selected))
            actual_losses += 1
            next_env = captured_branch
            next_result = captured_result
        else:
            next_env = branch
            next_result = result

        env, result = next_env, next_result
        if selected_terminal:
            episode_status = "won" if selected_won else "game_over"
            break
        if selected_loss:
            frames, actions = [env.render()], [-1]
        else:
            frames.append(result.frame)
            actions.append(selected)
            frames, actions = frames[-history:], actions[-history:]

    level_report = {
        **proof,
        "split": split,
        "seed": int(spec["seed"]),
        "difficulty": int(spec.get("difficulty", -1)),
        "source_spec_sha256": base.canonical_spec_hash(spec),
        "actions": int(step + 1 if max_actions else 0),
        "actual_life_losses": actual_losses,
        "actual_terminal_failure": bool(episode_status == "game_over"),
        "episode_status": episode_status,
        "shortfall_life_losses": max(0, 3 - actual_losses),
        "ordinary_mechanics_checks": ordinary_checks,
        "retained_branch_checks": branch_checks,
        "branch_checks": ordinary_checks + branch_checks,
        "engine_branch_checks": engine_branch_checks,
        "oracle_unverified_branches": oracle_unverified_branches,
        "retained_oracle_checked_branches": retained_oracle_checked_branches,
        "retained_oracle_unverified_branches": retained_oracle_unverified_branches,
        "retained_pre_loss_rows": len(rows),
    }
    return (rows, level_report, ordinary_checks, branch_checks,
            engine_branch_checks, oracle_unverified_branches)


def _write_arrays(path: Path, arrays: dict) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    world_data.save(temporary, arrays)
    temporary.replace(path)
    return digest(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/ls20-mechanism-training-pilot100.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/ls20-mechanism-validation50.jsonl"))
    parser.add_argument("--levels-per-split", type=int, default=5)
    parser.add_argument("--history", type=int, default=HISTORY)
    parser.add_argument("--max-actions", type=int, default=300)
    parser.add_argument("--search-limit", type=int, default=600_000)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--out", type=Path,
                        default=Path("data/ls20-structured-event-transitions-pilot-fast.npz"))
    parser.add_argument("--report", type=Path,
                        default=Path("artifacts/world-structured-event-transitions-pilot-fast.json"))
    args = parser.parse_args(argv)
    if min(args.levels_per_split, args.history, args.max_actions, args.search_limit, args.seconds) < 1:
        parser.error("counts must be positive")
    if args.history != HISTORY:
        parser.error("this pilot's public-history contract requires history=8")
    if args.out.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite fast structured-event pilot outputs")

    started = time.monotonic()
    code_paths = [Path(__file__), Path(base.__file__), Path("pebby/agent/world_data.py"),
                  Path("pebby/ls20/env.py"), Path("pebby/ls20/generate.py"),
                  Path("pebby/ls20/layout.py"), Path("pebby/ls20/plan.py"),
                  Path("pebby/ls20/rails.py"), Path("pebby/ls20/fastplan.py"),
                  Path("pebby/ls20/_fastplan.c")]
    code_hashes = {str(path): digest(path) for path in code_paths}
    v3_report = Path("artifacts/world-structured-event-transitions-pilot-v3.json")
    report = {
        "status": "running", "format": FORMAT, "pid": os.getpid(), "device": "cpu",
        "cpu_threads": 1, "official_inputs_used": False, "history": HISTORY,
        "alternatives_per_state": ACTION_COUNT, "max_actions_per_level": args.max_actions,
        "seconds": args.seconds, "source_code_hashes": code_hashes, "splits": {},
        "protocol": "One selected real clone/reference/simulate check per ordinary step; complete v3 four-branch capture only after actual life loss.",
        "v3_baseline_report": str(v3_report),
        "v3_baseline_report_sha256": digest(v3_report) if v3_report.exists() else None,
    }
    all_rows, level_reports = [], []
    ordinary_checks = retained_branch_checks = engine_branch_checks = 0
    oracle_unverified_branches = retained_checked = retained_unverified = 0
    error = None

    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.report)

    def timeout(_signum, _frame):
        raise TimeoutError(f"fast collection exceeded {args.seconds} seconds")

    print(f"PID {os.getpid()}", flush=True)
    persist()
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.seconds)
    try:
        train_specs = base.load_verified_specs(args.train, "train", args.levels_per_split)
        val_specs = base.load_verified_specs(args.validation, "validation", args.levels_per_split)
        train_seeds = {int(spec["seed"]) for spec in train_specs}
        val_seeds = {int(spec["seed"]) for spec in val_specs}
        if train_seeds & val_seeds:
            raise ValueError(f"train/validation seed overlap: {sorted(train_seeds & val_seeds)}")
        report["sources"] = {
            "train": {"path": str(args.train), "sha256_before": digest(args.train), "selected": sorted(train_seeds)},
            "validation": {"path": str(args.validation), "sha256_before": digest(args.validation), "selected": sorted(val_seeds)},
        }
        persist()
        for split, specs in (("train", train_specs), ("validation", val_specs)):
            split_rng = np.random.default_rng(args.seed + (0 if split == "train" else 1))
            split_rows = 0
            for spec in specs:
                result = collect_level_fast(spec, split, args.history, args.max_actions,
                                            args.search_limit, split_rng)
                rows, level_report, ordinary, retained, engine, unverified = result
                all_rows.extend(rows)
                level_reports.append(level_report)
                split_rows += len(rows)
                ordinary_checks += ordinary
                retained_branch_checks += retained
                engine_branch_checks += engine
                oracle_unverified_branches += unverified
                retained_checked += int(level_report.get("retained_oracle_checked_branches", 0))
                retained_unverified += int(level_report.get("retained_oracle_unverified_branches", 0))
                report["splits"].setdefault(split, {"levels": 0, "rows": 0})
                report["splits"][split]["levels"] += 1
                report["splits"][split]["rows"] = split_rows
                persist()

        metadata = {
            "format": FORMAT, "source": "generated_only", "oracle_search": "complete_only",
            "history": HISTORY, "alternatives_per_state": ACTION_COUNT,
            "retained_pre_loss_only": True, "policy_targets_are_diagnostic": True,
            "official_inputs_used": False, "levels": level_reports,
            "source_paths": {"train": str(args.train), "validation": str(args.validation)},
            "source_sha256": {"train": digest(args.train), "validation": digest(args.validation)},
            "comparison": "non-meta arrays must equal data/ls20-structured-event-transitions-pilot-v3.npz exactly",
        }
        arrays = base.stack_rows(all_rows, metadata)
        if arrays is None:
            raise ValueError("no retained life-loss rows")
        report["output"] = {"path": str(args.out), "sha256": _write_arrays(args.out, arrays),
                             "rows": len(all_rows), "levels": len(level_reports)}
        report["levels"] = level_reports
        report["ordinary_mechanics_checks"] = ordinary_checks
        report["retained_branch_checks"] = retained_branch_checks
        report["branch_checks"] = ordinary_checks + retained_branch_checks
        report["engine_branch_checks"] = engine_branch_checks
        report["oracle_unverified_branches"] = oracle_unverified_branches
        report["retained_oracle_checked_branches"] = retained_checked
        report["retained_oracle_unverified_branches"] = retained_unverified
        report["actual_terminal_failures"] = sum(bool(x.get("actual_terminal_failure")) for x in level_reports)
        report["levels_with_three_losses"] = sum(int(x.get("shortfall_life_losses", 1)) == 0 for x in level_reports)
        for source in ("train", "validation"):
            info = report["sources"][source]
            info["sha256_after"] = digest(Path(info["path"]))
            if info["sha256_after"] != info["sha256_before"]:
                raise ValueError(f"{source} source changed during collection")
        if v3_report.exists():
            baseline = json.loads(v3_report.read_text())
            report["speedup_vs_v3"] = (baseline.get("elapsed_seconds", 0) /
                                        max(report["elapsed_seconds"], 1e-9))
            report["v3_baseline_elapsed_seconds"] = baseline.get("elapsed_seconds")
        report["status"] = "complete"
    except TimeoutError as exc:
        error = exc
        report["status"] = "bounded_partial"
        report["error"] = repr(exc)
    except Exception as exc:
        error = exc
        report["status"] = "failed_closed"
        report["error"] = repr(exc)
    finally:
        signal.alarm(0)
        report["source_unchanged"] = all(
            Path(info["path"]).exists() and digest(Path(info["path"])) == info["sha256_before"]
            for info in report.get("sources", {}).values())
        report["code_unchanged"] = all(digest(Path(path)) == value for path, value in code_hashes.items())
        if not report["source_unchanged"] or not report["code_unchanged"]:
            report["status"] = "failed_closed"
            report["error"] = "source or code changed during collection"
        persist()
        print(json.dumps({key: report.get(key) for key in
                          ("status", "elapsed_seconds", "output", "ordinary_mechanics_checks",
                           "retained_branch_checks", "retained_oracle_checked_branches", "speedup_vs_v3")},
                         indent=2), flush=True)
    return 1 if error is not None or report["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
