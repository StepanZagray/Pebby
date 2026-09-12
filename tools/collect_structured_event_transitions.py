#!/usr/bin/env python3
"""Collect real generated LS20 life-loss transitions for structured-model diagnostics.

This is deliberately separate from ``world_data``.  It follows one seeded
uniform-random episode per verified generated level and retains only the states
whose selected real action loses a life.  All four actions from each retained
state are executed by the vendored engine on isolated clones.  A state whose
complete Oracle distance is absent is still useful transition data, but it gets
``current_reachable=False`` and a zero current policy mask; no policy target is
invented for a doomed state.
"""

from __future__ import annotations

from pebby.ls20.provenance import generated_context, validate_difficulty

import argparse
import copy
from collections import Counter
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
from pebby.ls20.generate import FORMAT as LEVEL_FORMAT, GENERATOR_VERSION, build_level
from pebby.ls20.layout import extract
from pebby.ls20.plan import Oracle, simulate


FORMAT = "pebby.ls20-structured-event-transitions.v1"
HISTORY = 8
ACTION_COUNT = 4
EVENT_LIVE = 0
EVENT_RESET_LOSS = 1
EVENT_TERMINAL_LOSS = 2
EVENT_WIN = 3


class TransitionMismatch(RuntimeError):
    """The real engine disagreed with the complete logical transition."""


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


def _verify_spec(spec: dict, path: Path, split: str) -> None:
    seed = int(spec.get("seed", -1))
    if spec.get("format") != LEVEL_FORMAT or spec.get("generator_version") not in (2, GENERATOR_VERSION):
        raise ValueError(f"{path}: seed {seed} is not a compatible generated level")
    if split == "train" and not 0 <= seed < 1_000_000:
        raise ValueError(f"{path}: training seed {seed} is outside the [0, 1000000) namespace")
    if split == "validation" and not 1_000_000 <= seed < 2_000_000:
        raise ValueError(f"{path}: validation seed {seed} is outside the [1000000, 2000000) namespace")
    if spec.get("context_engine_verified") is not True or spec.get("search_truncated") is not False:
        raise ValueError(f"{path}: seed {seed} lacks a complete contextual engine proof")
    if spec.get("engine_verified") is not True:
        raise ValueError(f"{path}: seed {seed} lacks engine_verified=true")
    if spec.get("training_context_index") != generated_context(spec):
        raise ValueError(f"{path}: seed {seed} has the wrong training context")
    proof = spec.get("proof")
    if not isinstance(proof, dict) or proof.get("context_engine_verified") is not True \
            or proof.get("context_index") != generated_context(spec):
        raise ValueError(f"{path}: seed {seed} lacks a matching nested context proof")
    # The training pilot explicitly carries generated_only.  The older verified
    # validation bank omits that optional field, so its generated format/proof
    # are required above and the input path is recorded verbatim in the report.
    if split == "train" and spec.get("source") != "generated_only":
        raise ValueError(f"{path}: training seed {seed} is not marked generated_only")
    if spec.get("source") not in (None, "generated_only"):
        raise ValueError(f"{path}: seed {seed} has an unsupported source marker")


def load_verified_specs(path: Path, split: str, count: int) -> list[dict]:
    if count < 1:
        raise ValueError("level count must be positive")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected = []
    seen = set()
    for spec in rows:
        _verify_spec(spec, path, split)
        seed = int(spec["seed"])
        if seed in seen:
            raise ValueError(f"{path}: duplicate seed {seed}")
        seen.add(seed)
        selected.append(spec)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"{path}: only {len(selected)} verified rows; need {count}")
    return selected


def _event_code(lost_life: bool, terminal: bool, won: bool) -> int:
    if won:
        return EVENT_WIN
    if lost_life and terminal:
        return EVENT_TERMINAL_LOSS
    if lost_life:
        return EVENT_RESET_LOSS
    return EVENT_LIVE


def _same_frame(left, right) -> bool:
    return np.array_equal(np.asarray(left, dtype=np.uint8), np.asarray(right, dtype=np.uint8))


def _same_branch_state(left, right) -> bool:
    return (tuple(left.player_cell()) == tuple(right.player_cell())
            and tuple(left.triple()) == tuple(right.triple())
            and int(left.steps_left()) == int(right.steps_left())
            and int(left.lives()) == int(right.lives())
            and left.state.value == right.state.value
            and tuple(left.goals_solved()) == tuple(right.goals_solved()))


def _oracle_next(oracle, branch, result):
    """Return a diagnostic successor distance/reachability/policy mask."""
    state = oracle.state_of(branch)
    if result.won:
        # WIN is a terminal success at logical distance zero.  It has no
        # successor policy target, but must still shorten a distance-one state.
        return 0, False, 0, state
    if result.finished:
        return -1, False, 0, state
    distance = oracle.distance_for(state)
    if distance is None or distance <= 0:
        return -1 if distance is None else int(distance), False, 0, state
    return int(distance), True, int(world_data.successor_optimal_mask(oracle, state)), state


def capture_branches(env, oracle, seed: int, step: int) -> dict:
    """Execute every action on a real isolated engine branch.

    Reachable states receive the same Oracle branch check used by the normal
    collector.  Unreachable states are intentionally treated as transition-only
    diagnostics: current_reachable is false and optimal is zero.
    """
    current_state = oracle.state_of(env)
    current_distance = oracle.distance_for(current_state)
    current_reachable = current_distance is not None and int(current_distance) > 0
    if current_reachable:
        current_distance = int(current_distance)

    next_frames, terminal, won, lost_life, next_player = [], [], [], [], []
    next_triple, next_steps, next_lives = [], [], []
    next_distance, next_reachable, next_optimal, branch_events = [], [], [], []
    branches, results = [], []
    branch_checks = 0
    engine_branch_checks = 0
    oracle_unverified_branches = 0

    for action_index, action in enumerate(names.ACTION_IDS):
        branch = world_data.clone_env(env)
        result = branch.perform(action)
        if result.frame is None:
            raise TransitionMismatch(f"seed {seed} step {step} action {action_index}: no frame")

        # Independent deepcopy comparison protects this diagnostic from a clone
        # implementation mistake while still using only the real engine.
        reference = copy.deepcopy(env, {id(env.module): env.module})
        reference_result = reference.perform(action)
        if not _same_frame(result.frame, reference_result.frame) or not _same_branch_state(branch, reference):
            raise TransitionMismatch(
                f"seed {seed} step {step} action {action_index}: clone/reference mismatch")
        engine_branch_checks += 1

        branch_state = oracle.state_of(branch)
        branch_distance, branch_reachable, branch_mask, _ = _oracle_next(oracle, branch, result)
        # state_of returns a complete logical tuple even when the state is not
        # in the Oracle's reachable-distance table.  Validate mechanics for
        # every branch regardless of policy reachability; only policy masks are
        # withheld for doomed current states below.
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
            raise TransitionMismatch(
                f"seed {seed} step {step} action {action_index}: {outcome} disagrees")
        branch_checks += 1

        branches.append(branch)
        results.append(result)
        terminal_flag = bool(result.finished)
        won_flag = bool(result.won)
        loss_flag = int(branch.lives()) < int(env.lives())
        next_frames.append(np.asarray(result.frame, dtype=np.uint8))
        terminal.append(terminal_flag)
        won.append(won_flag)
        lost_life.append(loss_flag)
        next_player.append(branch.player_cell())
        next_triple.append(branch.triple())
        next_steps.append(branch.steps_left())
        next_lives.append(branch.lives())
        next_distance.append(branch_distance)
        next_reachable.append(branch_reachable)
        next_optimal.append(branch_mask)
        branch_events.append(_event_code(loss_flag, terminal_flag, won_flag))

    optimal = 0
    if current_reachable:
        optimal = sum(1 << index for index, distance in enumerate(next_distance)
                      if distance == current_distance - 1 and not lost_life[index])
        if optimal == 0:
            raise TransitionMismatch(
                f"seed {seed} step {step}: reachable state has no safe optimal action")
    else:
        current_distance = -1 if current_distance is None else int(current_distance)

    return {
        "current_reachable": bool(current_reachable),
        "current_distance": np.int16(current_distance),
        "optimal": np.uint8(optimal),
        "next_frames": np.stack(next_frames),
        "terminal": np.asarray(terminal, dtype=bool),
        "won": np.asarray(won, dtype=bool),
        "lost_life": np.asarray(lost_life, dtype=bool),
        "next_player_cell": np.asarray(next_player, dtype=np.int16),
        "next_triple": np.asarray(next_triple, dtype=np.int16),
        "next_steps": np.asarray(next_steps, dtype=np.int16),
        "next_lives": np.asarray(next_lives, dtype=np.int16),
        "next_distance": np.asarray(next_distance, dtype=np.int16),
        "next_reachable": np.asarray(next_reachable, dtype=bool),
        "next_optimal": np.asarray(next_optimal, dtype=np.uint8),
        "branch_events": np.asarray(branch_events, dtype=np.uint8),
        "_branches": branches,
        "_results": results,
        "branch_checks": branch_checks,
        "engine_branch_checks": engine_branch_checks,
        "oracle_unverified_branches": oracle_unverified_branches,
    }


def _row_from_capture(env, capture, observed, valid, previous, spec, split, step,
                      selected_action: int) -> dict:
    selected_loss = bool(capture["lost_life"][selected_action])
    selected_terminal = bool(capture["terminal"][selected_action])
    selected_won = bool(capture["won"][selected_action])
    return {
        "frames": observed,
        "history_valid": valid,
        "previous_actions": previous,
        "next_frames": capture["next_frames"],
        "terminal": capture["terminal"],
        "won": capture["won"],
        "lost_life": capture["lost_life"],
        "branch_events": capture["branch_events"],
        "retained_oracle_checked_branches": np.int8(
            ACTION_COUNT - int(capture["oracle_unverified_branches"])),
        "retained_oracle_unverified_branches": np.int8(
            int(capture["oracle_unverified_branches"])),
        "next_player_cell": capture["next_player_cell"],
        "next_triple": capture["next_triple"],
        "next_steps": capture["next_steps"],
        "next_lives": capture["next_lives"],
        "next_distance": capture["next_distance"],
        "next_reachable": capture["next_reachable"],
        "next_optimal": capture["next_optimal"],
        "player_cell": np.asarray(env.player_cell(), dtype=np.int16),
        "current_triple": np.asarray(env.triple(), dtype=np.int16),
        "current_steps": np.int16(env.steps_left()),
        "current_lives": np.int16(env.lives()),
        "current_reachable": np.bool_(capture["current_reachable"]),
        "current_distance": np.int16(capture["current_distance"]),
        "optimal": np.uint8(capture["optimal"]),
        "selected_action": np.int8(selected_action),
        "actual_lost_life": np.bool_(selected_loss),
        "actual_terminal": np.bool_(selected_terminal),
        "actual_won": np.bool_(selected_won),
        "actual_event": np.uint8(_event_code(selected_loss, selected_terminal, selected_won)),
        "seed": np.int64(int(spec["seed"])),
        "context_index": np.int8(generated_context(spec)),
        "split_id": np.int8(0 if split == "train" else 1),
        "episode_step": np.int16(step),
    }


def collect_level(spec: dict, split: str, history: int, max_actions: int,
                  search_limit: int, rng: np.random.Generator) -> tuple[list[dict], dict, int, int, int]:
    env, oracle, proof = world_data.verified_context(spec, context_index=generated_context(spec),
                                                     search_limit=search_limit)
    if env is None or oracle is None:
        return [], {**proof, "split": split, "shortfall": "context_verification_failed"}, 0, 0, 0

    frames, actions = [env.render()], [-1]
    rows = []
    branch_checks = 0
    engine_branch_checks = 0
    oracle_unverified_branches = 0
    retained_oracle_checked_branches = 0
    retained_oracle_unverified_branches = 0
    retained_oracle_checked_branches = 0
    retained_oracle_unverified_branches = 0
    actual_losses = 0
    episode_status = "action_bound"
    for step in range(max_actions):
        if env.state.value in ("WIN", "GAME_OVER"):
            episode_status = env.state.value.lower()
            break
        observed, valid, previous = world_data.history_arrays(frames, actions, history)
        capture = capture_branches(env, oracle, int(spec["seed"]), step)
        branch_checks += int(capture.pop("branch_checks"))
        engine_branch_checks += int(capture.pop("engine_branch_checks"))
        oracle_unverified_branches += int(capture["oracle_unverified_branches"])
        branches, results = capture.pop("_branches"), capture.pop("_results")
        selected = int(rng.integers(0, ACTION_COUNT))
        selected_loss = bool(capture["lost_life"][selected])
        selected_terminal = bool(capture["terminal"][selected])
        selected_won = bool(capture["won"][selected])
        if selected_loss:
            rows.append(_row_from_capture(env, capture, observed, valid, previous, spec,
                                          split, step, selected))
            actual_losses += 1
            retained_oracle_checked_branches += (
                ACTION_COUNT - int(capture["oracle_unverified_branches"]))
            retained_oracle_unverified_branches += int(capture["oracle_unverified_branches"])

        env = branches[selected]
        result = results[selected]
        if selected_terminal:
            episode_status = "won" if selected_won else "game_over"
            break
        if selected_loss:
            # The engine has already completed the reset animation. The next
            # causal history starts with that real post-reset public frame.
            frames, actions = [env.render()], [-1]
        else:
            frames.append(result.frame)
            actions.append(selected)
            frames, actions = frames[-history:], actions[-history:]

    shortfall = max(0, 3 - actual_losses)
    level_report = {
        **proof,
        "split": split,
        "seed": int(spec["seed"]),
        "difficulty": int(spec.get("difficulty", -1)),
        "source_spec_sha256": canonical_spec_hash(spec),
        "actions": int(step + 1 if max_actions else 0),
        "actual_life_losses": actual_losses,
        "actual_terminal_failure": bool(episode_status == "game_over"),
        "episode_status": episode_status,
        "shortfall_life_losses": shortfall,
        "branch_checks": branch_checks,
        "engine_branch_checks": engine_branch_checks,
        "oracle_unverified_branches": oracle_unverified_branches,
        "retained_oracle_checked_branches": retained_oracle_checked_branches,
        "retained_oracle_unverified_branches": retained_oracle_unverified_branches,
        "retained_pre_loss_rows": len(rows),
    }
    return rows, level_report, branch_checks, engine_branch_checks, oracle_unverified_branches


def stack_rows(rows: list[dict], metadata: dict) -> dict | None:
    if not rows:
        return None
    keys = [key for key in rows[0] if key not in ("meta",)]
    arrays = {key: np.stack([row[key] for row in rows]) for key in keys}
    metadata = {**metadata, "rows": len(rows), "seeds": sorted({int(x) for x in arrays["seed"]})}
    arrays["meta"] = metadata
    return arrays


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
    parser.add_argument("--out", type=Path, default=Path("data/ls20-structured-event-transitions-pilot.npz"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/world-structured-event-transitions-pilot.json"))
    args = parser.parse_args(argv)
    if min(args.levels_per_split, args.history, args.max_actions, args.search_limit, args.seconds) < 1:
        parser.error("levels/history/max-actions/search-limit/seconds must be positive")
    if args.history != HISTORY:
        parser.error("this pilot's public-history contract requires history=8")
    if args.out.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite structured-event pilot outputs")

    started = time.monotonic()
    code_paths = [Path(__file__), Path("pebby/agent/world_data.py"), Path("pebby/ls20/env.py"),
                  Path("pebby/ls20/generate.py"), Path("pebby/ls20/layout.py"),
                  Path("pebby/ls20/plan.py"), Path("pebby/ls20/rails.py"),
                  Path("pebby/ls20/fastplan.py"), Path("pebby/ls20/_fastplan.c")]
    code_hashes = {str(path): digest(path) for path in code_paths}
    report = {
        "status": "running", "format": FORMAT, "pid": os.getpid(), "device": "cpu",
        "cpu_threads": 1, "official_inputs_used": False, "history": HISTORY,
        "alternatives_per_state": ACTION_COUNT, "max_actions_per_level": args.max_actions,
        "seconds": args.seconds, "source_code_hashes": code_hashes, "splits": {},
        "retained_rows_contract": "Only selected real life-loss pre-action states; terminal postframes are branch next_frames and no post-terminal history is emitted.",
        "event_codes": {"live": EVENT_LIVE, "reset_loss": EVENT_RESET_LOSS,
                        "terminal_loss": EVENT_TERMINAL_LOSS, "win": EVENT_WIN},
    }
    all_rows, level_reports = [], []
    branch_checks = 0
    engine_branch_checks = 0
    oracle_unverified_branches = 0
    retained_oracle_checked_branches = 0
    retained_oracle_unverified_branches = 0

    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.report)

    def timeout(_signum, _frame):
        raise TimeoutError(f"collection exceeded {args.seconds} seconds")

    print(f"PID {os.getpid()}", flush=True)
    persist()
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.seconds)
    error = None
    try:
        train_specs = load_verified_specs(args.train, "train", args.levels_per_split)
        val_specs = load_verified_specs(args.validation, "validation", args.levels_per_split)
        train_seeds = {int(spec["seed"]) for spec in train_specs}
        val_seeds = {int(spec["seed"]) for spec in val_specs}
        if train_seeds & val_seeds:
            raise ValueError(f"train/validation seed overlap: {sorted(train_seeds & val_seeds)}")
        report["sources"] = {
            "train": {"path": str(args.train), "sha256_before": digest(args.train), "selected": sorted(train_seeds)},
            "validation": {"path": str(args.validation), "sha256_before": digest(args.validation), "selected": sorted(val_seeds)},
            "generator_format": LEVEL_FORMAT, "generator_version": GENERATOR_VERSION,
            "source_generator_versions": sorted({spec["generator_version"]
                                                  for spec in train_specs + val_specs}),
        }
        persist()
        for split, specs in (("train", train_specs), ("validation", val_specs)):
            split_rng = np.random.default_rng(args.seed + (0 if split == "train" else 1))
            split_rows = 0
            for spec in specs:
                rows, level_report, checks, engine_checks, unverified_checks = collect_level(
                    spec, split, args.history, args.max_actions, args.search_limit, split_rng)
                all_rows.extend(rows)
                level_reports.append(level_report)
                split_rows += len(rows)
                branch_checks += checks
                engine_branch_checks += engine_checks
                oracle_unverified_branches += unverified_checks
                retained_oracle_checked_branches += int(level_report.get("retained_oracle_checked_branches", 0))
                retained_oracle_unverified_branches += int(level_report.get("retained_oracle_unverified_branches", 0))
                report["splits"].setdefault(split, {"levels": 0, "rows": 0})
                report["splits"][split]["levels"] += 1
                report["splits"][split]["rows"] = split_rows
                persist()

        metadata = {
            "format": FORMAT, "source": "generated_only", "oracle_search": "complete_only",
            "history": HISTORY, "alternatives_per_state": ACTION_COUNT,
            "retained_pre_loss_only": True, "policy_targets_are_diagnostic": True,
            "official_inputs_used": False, "levels": level_reports,
            "source_paths": {split: str(args.__dict__[split]) for split in ("train", "validation")},
            "source_sha256": {split: digest(args.__dict__[split]) for split in ("train", "validation")},
        }
        arrays = stack_rows(all_rows, metadata)
        if arrays is not None:
            report["output"] = {"path": str(args.out), "sha256": _write_arrays(args.out, arrays),
                                 "rows": len(all_rows), "levels": len(level_reports)}
        report["levels"] = level_reports
        report["branch_checks"] = branch_checks
        report["engine_branch_checks"] = engine_branch_checks
        report["oracle_unverified_branches"] = oracle_unverified_branches
        report["retained_oracle_checked_branches"] = retained_oracle_checked_branches
        report["retained_oracle_unverified_branches"] = retained_oracle_unverified_branches
        report["actual_terminal_failures"] = sum(bool(x.get("actual_terminal_failure")) for x in level_reports)
        report["levels_with_three_losses"] = sum(int(x.get("shortfall_life_losses", 1)) == 0 for x in level_reports)
        for source in ("train", "validation"):
            report["sources"][source]["sha256_after"] = digest(Path(report["sources"][source]["path"]))
            if report["sources"][source]["sha256_after"] != report["sources"][source]["sha256_before"]:
                raise ValueError(f"{source} source changed during collection")
            report["sources"][source]["sha256"] = report["sources"][source]["sha256_after"]
        report["status"] = "complete"
    except TimeoutError as exc:
        error = exc
        report["status"] = "bounded_partial"
        report["error"] = repr(exc)
    except Exception as exc:  # fail closed, but preserve the report
        error = exc
        report["status"] = "failed_closed"
        report["error"] = repr(exc)
    finally:
        signal.alarm(0)
        report["source_unchanged"] = all(
            Path(info["path"]).exists()
            and digest(Path(info["path"])) == info.get("sha256_before")
            for info in report.get("sources", {}).values()
            if isinstance(info, dict) and "path" in info and "sha256_before" in info)
        report["code_unchanged"] = all(digest(Path(path)) == value for path, value in code_hashes.items())
        if not report["source_unchanged"] or not report["code_unchanged"]:
            report["status"] = "failed_closed"
            report["error"] = "source or code changed during collection"
        persist()
        print(json.dumps({key: report.get(key) for key in
                          ("status", "elapsed_seconds", "peak_rss_mib", "output",
                           "actual_terminal_failures", "levels_with_three_losses")}, indent=2), flush=True)
    return 1 if error is not None or report["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
