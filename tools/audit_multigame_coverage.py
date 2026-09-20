#!/usr/bin/env python3
"""Read-only NPZ coverage census and optional generated-corpus acceptance gates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pebby import multigame as M
from pebby.agent.multigame_model import load_supervised_game
from pebby.multigame_dataset import _variant_from_record


def _path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"artifact path escapes manifest directory: {relative}")
    return path


def _bucket() -> dict:
    return {
        "records": 0, "transitions": 0, "targets": 0, "missing_targets": 0,
        "click_targets": 0, "stored_region_clicks": 0, "exact_fallback_clicks": 0,
        "multi_pixel_region_clicks": 0, "singleton_region_clicks": 0,
        "undo_targets": 0, "executed_undo": 0,
        "target_actions_canonical": Counter(), "executed_actions_canonical": Counter(),
        "route_source_counts": Counter(),
    }


def _finish(bucket: dict) -> dict:
    result = {key: dict(sorted(value.items())) if isinstance(value, Counter) else value
              for key, value in bucket.items()}
    clicks = result["click_targets"]
    result["stored_region_fraction"] = result["stored_region_clicks"] / clicks if clicks else None
    return result


def audit_manifest(path: Path) -> dict:
    """Count actual aligned rows, including zeros for generated-but-unvisited levels."""
    path = path.resolve()
    manifest = M.load_manifest(path)
    root = path.parent
    totals = _bucket()
    families: dict[str, dict] = {}
    discrepancies = []

    def family(source_id):
        if source_id not in {source.source_id for source in M.TRAIN_SOURCES}:
            raise ValueError(f"coverage audit accepts generated training families only: {source_id}")
        return families.setdefault(source_id, {"totals": _bucket(), "levels": {}})

    for source_id in manifest.get("requested_source_ids", ()):
        entry = family(source_id)
        for level in range(len(manifest.get("curricula_by_source", {}).get(source_id, ()))):
            entry["levels"][str(level)] = _bucket()
    for record in manifest["records"]:
        entry = family(record["source_id"])
        for level in range(int(record.get("levels_generated", len(record.get("levels", ()))))):
            entry["levels"].setdefault(str(level), _bucket())
        if int(record.get("steps", 0)) == 0:
            totals["records"] += 1
            entry["totals"]["records"] += 1
            continue
        public_path = _path(root, record["public_npz"])
        teacher_path = _path(root, record["teacher_npz"])
        game = load_supervised_game(public_path, teacher_path)
        if len(game) != int(record["steps"]):
            raise ValueError(f"{record['public_npz']} steps disagree with NPZ")
        with np.load(public_path, allow_pickle=False) as public:
            levels = public["level_index"][:-1].copy()
        with np.load(teacher_path, allow_pickle=False) as teacher:
            routes = (teacher["route_source"].astype(str) if "route_source" in teacher
                      else np.full(len(game), "unknown"))
            actual_routes = M.route_source_counts_from_teacher(teacher, len(game))
        if record.get("route_source_counts") != actual_routes:
            discrepancies.append({"record": record.get("record", record["public_npz"]),
                                  "declared": record.get("route_source_counts"),
                                  "actual": actual_routes})
        inverse = np.asarray(_variant_from_record(record).control_public_to_raw)
        targets = np.full(len(game), -1, dtype=np.int64)
        targets[game.target_valid] = inverse[game.target_action_id[game.target_valid]]
        executed = inverse[game.executed_action_id]
        sizes = (game.target_click_region.sum(axis=(1, 2))
                 if game.target_click_region is not None else np.zeros(len(game), dtype=int))

        def count(bucket, rows):
            target = targets[rows]
            action = executed[rows]
            click = target == M.CLICK_ACTION
            stored = sizes[rows] > 0
            bucket["records"] += 1
            bucket["transitions"] += len(target)
            bucket["targets"] += int((target >= 0).sum())
            bucket["missing_targets"] += int((target < 0).sum())
            bucket["click_targets"] += int(click.sum())
            bucket["stored_region_clicks"] += int((click & stored).sum())
            bucket["exact_fallback_clicks"] += int((click & ~stored).sum())
            bucket["multi_pixel_region_clicks"] += int((click & (sizes[rows] > 1)).sum())
            bucket["singleton_region_clicks"] += int((click & (sizes[rows] == 1)).sum())
            bucket["undo_targets"] += int((target == 7).sum())
            bucket["executed_undo"] += int((action == 7).sum())
            bucket["target_actions_canonical"].update(str(int(v)) for v in target if v >= 0)
            bucket["executed_actions_canonical"].update(str(int(v)) for v in action)
            bucket["route_source_counts"].update(str(v) for v in routes[rows])

        count(totals, slice(None))
        count(entry["totals"], slice(None))
        for level in np.unique(levels):
            count(entry["levels"].setdefault(str(int(level)), _bucket()), levels == level)
    result = {
        "manifest": str(path), "scope": manifest.get("scope"), "totals": _finish(totals),
        "by_family": {
            source: {"totals": _finish(entry["totals"]),
                     "by_level_index": {level: _finish(bucket) for level, bucket in
                                        sorted(entry["levels"].items(), key=lambda item: int(item[0]))}}
            for source, entry in sorted(families.items())
        },
        "route_metadata_discrepancies": discrepancies,
        "limitations": [
            "Stored regions are bounded one-step public-successor equivalence, not exhaustive future equivalence.",
            "Canonical undo is action 7; RESET is deliberately excluded from the policy protocol.",
            "Counts measure supervision, not independent puzzle diversity or gameplay mastery.",
        ],
    }
    actual = result["totals"]["route_source_counts"]
    result["manifest_route_counts_match"] = manifest.get("route_source_counts") == actual
    return result


def coverage_failures(report, *, min_click_region_fraction=0.0,
                      min_targets_per_level=0, min_undo_targets_per_family=0,
                      min_learner_steps_per_family=0):
    failures = []
    for source, family in report["by_family"].items():
        learner = family["totals"]["route_source_counts"].get(M.LEARNER_ROUTE_SOURCE, 0)
        if learner < min_learner_steps_per_family:
            failures.append(f"{source}: learner transitions {learner} < {min_learner_steps_per_family}; "
                            "collect with --perturbation learner --learner-checkpoint PATH")
        fraction = family["totals"]["stored_region_fraction"]
        if fraction is not None and fraction < min_click_region_fraction:
            failures.append(f"{source}: stored click-region fraction {fraction:.4f} < {min_click_region_fraction}; "
                            "collect with --click-region-probe-limit 64 or backfill verified labels")
        if family["totals"]["undo_targets"] < min_undo_targets_per_family:
            failures.append(f"{source}: undo targets {family['totals']['undo_targets']} < "
                            f"{min_undo_targets_per_family}; collect recovery states where undo is useful")
        if min_targets_per_level and not family["by_level_index"]:
            failures.append(f"{source}: no declared or observed levels")
        for level, counts in family["by_level_index"].items():
            if counts["targets"] < min_targets_per_level:
                failures.append(f"{source} level {level}: {counts['targets']} targets < {min_targets_per_level}; "
                                "collect trajectories that actually reach this level")
    return failures


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-click-region-fraction", "--min-click-region-coverage",
                        type=float, default=0.0)
    parser.add_argument("--require-level-coverage", action="store_true",
                        help="require at least one supervised target on each declared level")
    parser.add_argument("--min-targets-per-level", type=int, default=0)
    parser.add_argument("--min-undo-targets-per-family", type=int, default=0)
    parser.add_argument("--min-learner-steps-per-family", type=int, default=0)
    args = parser.parse_args(argv)
    if not 0 <= args.min_click_region_fraction <= 1 or min(
        args.min_targets_per_level, args.min_undo_targets_per_family, args.min_learner_steps_per_family,
    ) < 0:
        parser.error("coverage thresholds must be nonnegative; click fraction must be in [0,1]")
    try:
        if args.output and args.output.resolve() == args.manifest.resolve():
            raise ValueError("output must not replace the input manifest")
        report = audit_manifest(args.manifest)
        report["coverage_failures"] = coverage_failures(
            report, min_click_region_fraction=args.min_click_region_fraction,
            min_targets_per_level=max(args.min_targets_per_level, int(args.require_level_coverage)),
            min_undo_targets_per_family=args.min_undo_targets_per_family,
            min_learner_steps_per_family=args.min_learner_steps_per_family,
        )
        text = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Do not overwrite any existing corpus artifact or report accidentally.
            with args.output.open("x") as output:
                output.write(text)
        else:
            print(text, end="")
        return 1 if report["coverage_failures"] else 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"coverage audit rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
