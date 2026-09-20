#!/usr/bin/env python3
"""Replay bounded generated games into a new, resumable click-region corpus.

One committed record is the recovery unit. A partial manifest contains only
committed records and explicitly declares incompleteness until every source
record is processed. No source file is modified or hardlinked.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import threading

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pebby import multigame as M
from pebby.agent.multigame_model import load_supervised_game
from pebby.agent.multigame_training import sha256_file
from pebby.multigame_dataset import _variant_from_record
from pebby.multigame_variants import transform_frame
from tools.audit_multigame_coverage import _path

FORMAT = "pebby-click-backfill-v1"
ARTIFACTS = ("public_npz", "teacher_npz", "generated_specs")


class _DeadlineExceeded(BaseException):
    # Probe helpers catch Exception for invalid candidate clicks. The deadline
    # must escape those handlers rather than silently becoming a rejected click.
    pass


def _bounded_call(fn, seconds):
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        raise ValueError("bounded backfill requires a Unix main-thread timer")
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise ValueError("backfill cannot replace an already-active alarm")
    previous = signal.getsignal(signal.SIGALRM)

    def timeout(*unused):
        raise _DeadlineExceeded()

    signal.signal(signal.SIGALRM, timeout)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        return fn()
    except _DeadlineExceeded:
        raise ValueError(f"backfill game exceeded {seconds:g}s; no record committed") from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _verify_state(game, public, variant, row):
    if not np.array_equal(variant.public_frame(game.render()), public["frames"][row]):
        raise ValueError(f"replay frame mismatch at state {row}")
    if not np.array_equal(M.legal_mask(game, variant), public["legal_action_mask"][row]):
        raise ValueError(f"replay legal mask mismatch at state {row}")
    progress = game.progress
    for name in ("state", "level_index", "levels_completed", "terminal", "won"):
        if getattr(progress, name) != public[name][row]:
            raise ValueError(f"replay {name} mismatch at state {row}")


def _backfill_record(record, root, *, probe_limit, max_steps, game_seconds, modules=None):
    paths = {name: _path(root, record[name]) for name in ARTIFACTS if record.get(name)}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, actual in hashes.items():
        declared = record.get("file_hashes", {}).get(name, record.get(f"{name}_sha256"))
        if declared and declared != actual:
            raise ValueError(f"source {name} hash mismatch")
    result = copy.deepcopy(record)
    result["backfill_source_hashes"] = hashes
    result["backfill_runtime_bounds"] = {"max_steps_per_game": max_steps, "game_seconds": game_seconds}
    steps = int(record.get("steps", 0))
    if not steps:
        return result, paths, None
    sequence = load_supervised_game(paths["public_npz"], paths["teacher_npz"])
    if len(sequence) != steps:
        raise ValueError("record steps disagree with source arrays")
    with np.load(paths["teacher_npz"], allow_pickle=False) as loaded:
        teacher = {name: loaded[name].copy() for name in loaded.files}
    result["route_source_counts"] = M.route_source_counts_from_teacher(teacher, steps)
    clicks = sequence.target_action_id == M.CLICK_ACTION
    # Existing labels are preserved exactly; no claim that their one-step
    # equivalence has been independently reverified by this command.
    if sequence.has_click_regions or not clicks.any():
        result["click_backfill"] = {"status": "already_stored" if sequence.has_click_regions else "no_clicks",
                                    "verified_transitions": 0, "new_click_regions": 0}
        return result, paths, None
    if steps > max_steps:
        raise ValueError(f"game has {steps} transitions above --max-steps-per-game {max_steps}")
    variant = _variant_from_record(record)

    def replay():
        package = modules or M.preflight([record["source_id"]])[0]
        specs = json.loads(paths["generated_specs"].read_text())
        game = M.MultiGameEnv.from_specs(
            package, specs, require_full_standard=bool(record.get("full_standard_required")),
        )
        game.reset()
        with np.load(paths["public_npz"], allow_pickle=False) as loaded:
            public = {name: loaded[name] for name in loaded.files}
        regions = np.zeros((steps, 64, 64), dtype=np.uint8)
        candidates, elapsed = 0, 0.0
        _verify_state(game, public, variant, 0)
        for row in range(steps):
            if clicks[row]:
                raw = M.Action(*variant.raw_action(
                    M.CLICK_ACTION, int(sequence.target_action_x[row]), int(sequence.target_action_y[row]),
                ))
                region = M.equivalent_click_region(game, raw, limit=probe_limit)
                regions[row] = transform_frame(region.mask, variant.spatial)
                candidates += region.candidates
                elapsed += region.seconds
            public_action = M.Action(int(sequence.executed_action_id[row]),
                                    int(sequence.executed_action_x[row]) if sequence.executed_action_id[row] == 6 else None,
                                    int(sequence.executed_action_y[row]) if sequence.executed_action_id[row] == 6 else None)
            before = game.progress.levels_completed
            game.perform(M.Action(*variant.raw_action(*public_action.as_tuple())))
            _verify_state(game, public, variant, row + 1)
            if bool(game.progress.levels_completed > before) != bool(public["level_boundary"][row]):
                raise ValueError(f"replay boundary mismatch at transition {row}")
        return regions, candidates, elapsed

    regions, candidates, elapsed = _bounded_call(replay, game_seconds)
    for name, path in paths.items():
        if sha256_file(path) != hashes[name]:
            raise ValueError(f"source {name} changed during backfill")
    teacher["click_region_mask"] = regions
    teacher["click_region_size"] = regions.sum(axis=(1, 2)).astype(np.int16)
    sizes = teacher["click_region_size"][clicks]
    result["click_region_probe_limit"] = probe_limit
    result["click_region_stats"] = {
        "click_targets": int(clicks.sum()), "labelled": int(clicks.sum()), "probe_failures": 0,
        "mean_size": float(sizes.mean()), "median_size": float(np.median(sizes)),
        "min_size": int(sizes.min()), "max_size": int(sizes.max()),
        "candidates_probed": candidates, "probe_seconds": elapsed,
        "probe_seconds_per_click": elapsed / len(sizes),
    }
    result["click_backfill"] = {"status": "backfilled", "verified_transitions": steps,
                                "new_click_regions": int(clicks.sum())}
    return result, paths, teacher


def backfill(manifest_path, output_root, *, max_games=1, probe_limit=64,
             max_steps_per_game=4096, game_seconds=60.0, modules_by_source=None):
    """Resume with the same source/probe rule; runtime bounds may be raised safely."""
    source_path, output = Path(manifest_path).resolve(), Path(output_root).resolve()
    if output.is_relative_to(source_path.parent) or source_path.parent.is_relative_to(output):
        raise ValueError("backfill output and source corpus must be separate directory trees")
    output.mkdir(parents=True, exist_ok=True)
    # Keep the lock inode in the output. Unlinking it after release would let a
    # waiter and a new process lock different inodes for the same destination.
    with (output / ".backfill.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another backfill writer holds this output directory") from None
        return _backfill_locked(
            source_path, output, max_games=max_games, probe_limit=probe_limit,
            max_steps_per_game=max_steps_per_game, game_seconds=game_seconds,
            modules_by_source=modules_by_source,
        )


def _backfill_locked(manifest_path, output_root, *, max_games, probe_limit,
                     max_steps_per_game, game_seconds, modules_by_source):
    if (min(max_games, probe_limit, max_steps_per_game) < 1
            or not math.isfinite(game_seconds) or not game_seconds > 0):
        raise ValueError("all backfill bounds must be positive")
    source_path, output = Path(manifest_path).resolve(), Path(output_root).resolve()
    root = source_path.parent
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("backfill output and source corpus must be separate directory trees")
    source = M.load_manifest(source_path)  # also rejects held-out family records
    contract = {"format": FORMAT, "source_manifest": str(source_path),
                "source_sha256": sha256_file(source_path), "probe_limit": probe_limit}
    journal = output / "backfill.json"
    if journal.exists():
        if json.loads(journal.read_text()) != contract:
            raise ValueError("source manifest or label options changed since backfill began")
    else:
        if output.exists() and any(path.name != ".backfill.lock" for path in output.iterdir()):
            raise ValueError("backfill needs an empty new output directory")
        output.mkdir(parents=True, exist_ok=True)
        M._atomic_json(journal, contract)
    # Each atomic record file is its own commit marker. Rebuild the manifest on
    # resume, recovering a crash between a record commit and manifest replacement.
    records = []
    processed = 0
    for index, original in enumerate(source["records"]):
        key = f"backfill-{index:06d}"
        record_path = output / "records" / f"{key}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record["backfill_source_record_index"] != index:
                raise ValueError("backfill record index mismatch")
            for name, digest in record["backfill_source_hashes"].items():
                if sha256_file(_path(root, original[name])) != digest:
                    raise ValueError(f"source {index}:{name} changed since commit")
            for name, digest in record["file_hashes"].items():
                if sha256_file(_path(output, record[name])) != digest:
                    raise ValueError(f"backfill output {index}:{name} changed since commit")
        elif processed < max_games:
            record, paths, teacher = _backfill_record(
                original, root, probe_limit=probe_limit, max_steps=max_steps_per_game,
                game_seconds=game_seconds,
                modules=None if modules_by_source is None else modules_by_source[original["source_id"]],
            )
            relative = {"public_npz": f"games/{key}.npz", "teacher_npz": f"teacher/{key}.npz",
                        "generated_specs": f"teacher/{key}.levels.json"}
            hashes = {}
            for name, path in paths.items():
                destination = output / relative[name]
                destination.parent.mkdir(parents=True, exist_ok=True)
                if name == "teacher_npz" and teacher is not None:
                    M._atomic_npz(destination, teacher)
                else:
                    temporary = destination.with_suffix(destination.suffix + ".tmp")
                    shutil.copyfile(path, temporary)
                    temporary.replace(destination)
                record[name] = relative[name]
                record.pop(f"{name}_sha256", None)
                hashes[name] = sha256_file(destination)
                if not (name == "teacher_npz" and teacher is not None):
                    if hashes[name] != record["backfill_source_hashes"][name]:
                        raise ValueError(f"source {name} changed while copying")
            if int(record.get("steps", 0)):
                load_supervised_game(output / record["public_npz"], output / record["teacher_npz"])
            record["file_hashes"] = hashes
            record["record"] = str(record_path.relative_to(output))
            record["backfill_source_record_index"] = index
            record["backfill_source_manifest_sha256"] = contract["source_sha256"]
            M._atomic_json(record_path, record)  # sole per-game commit point
            processed += 1
        else:
            break
        records.append(record)
    if not source["records"]:
        raise ValueError("source manifest has no records")
    result = copy.deepcopy(source)
    # Every output artifact has its fresh digest on its output record.
    result.pop("file_hashes", None)
    result["records"] = records
    complete = len(records) == len(source["records"])
    result["click_backfill"] = {**contract, "complete": complete,
                               "source_records": len(source["records"]), "committed_records": len(records),
                               "last_invocation_bounds": {"max_games": max_games,
                                                          "max_steps_per_game": max_steps_per_game,
                                                          "game_seconds": game_seconds},
                               "equivalence": "bounded one-step public successor; not exhaustive future equivalence"}
    result["preparation_complete"] = complete and source.get("preparation_complete", True)
    if not complete:
        result["is_full_experiment_collection"] = False
    M.update_manifest_summary(result)
    M.save_manifest(output / "manifest.json", result)
    return {"complete": len(records) == len(source["records"]), "new_records": processed,
            "committed_records": len(records), "manifest": str(output / "manifest.json")}


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-games", type=int, default=1)
    parser.add_argument("--probe-limit", type=int, default=64)
    parser.add_argument("--max-steps-per-game", type=int, default=4096)
    parser.add_argument("--game-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    try:
        result = backfill(args.manifest, args.output_root, max_games=args.max_games,
                          probe_limit=args.probe_limit, max_steps_per_game=args.max_steps_per_game,
                          game_seconds=args.game_seconds)
        print(json.dumps(result, indent=2))
        return 0 if result["complete"] else 1
    except (OSError, ValueError, KeyError, M.PreflightError) as exc:
        print(f"click backfill rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
