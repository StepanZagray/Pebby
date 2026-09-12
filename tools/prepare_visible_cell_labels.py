#!/usr/bin/env python3
"""Extract generated-only, fully visible initial-state cell labels.

This deliberately labels only cells whose complete 5x5 play patch is inside
the public fog aperture and outside the HUD.  Engine-derived roles remain in
the output for auditing, but ``label_mask`` is the only mask a decoder may
use for supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pebby.ls20 import generate, names
from pebby.ls20.env import Ls20Scenario


ROLE_BITS = {
    "wall": 1,
    "goal": 2,
    "cycler_shape": 4,
    "cycler_color": 8,
    "cycler_rotation": 16,
    "launcher": 32,
    "refill": 64,
    "player": 128,
}
ROLE_NAMES = tuple(ROLE_BITS)
BOARD_CELLS = names.GRID_COLS * names.GRID_ROWS


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_content_sha256(specs) -> str:
    digest = hashlib.sha256()
    for spec in sorted(specs, key=lambda item: int(item["seed"])):
        encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def load_selected(path: Path, per_difficulty: int, selection_seed: int):
    grouped = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            if line.strip():
                spec = json.loads(line)
                grouped[int(spec["difficulty"])].append(spec)
    rng = random.Random(selection_seed)
    selected = []
    for difficulty in sorted(grouped):
        candidates = sorted(grouped[difficulty], key=lambda item: int(item["seed"]))
        if len(candidates) < per_difficulty:
            raise ValueError(f"{path}: difficulty {difficulty} has only {len(candidates)} rows")
        chosen = rng.sample(candidates, per_difficulty)
        selected.extend(sorted(chosen, key=lambda item: int(item["seed"])))
    return selected


def cell_index(col: int, row: int) -> int:
    return row * names.GRID_COLS + col


def board_patch(frame: np.ndarray, col: int, row: int) -> np.ndarray:
    x, y = names.cell_to_pixel(col, row)
    return frame[y : y + names.CELL, x : x + names.CELL]


def public_masks(env: Ls20Scenario, frame: np.ndarray):
    player = env.game.gudziatsk
    # render_interface uses math.dist((pixel_row, pixel_col),
    # (player.y + 1.5, player.x + 1.5)) > 20.0 for fog.
    any_visible = np.zeros((names.GRID_ROWS, names.GRID_COLS), dtype=bool)
    fully_visible = np.zeros_like(any_visible)
    hud_overlap = np.zeros_like(any_visible)
    for row in range(names.GRID_ROWS):
        for col in range(names.GRID_COLS):
            x, y = names.cell_to_pixel(col, row)
            visible_pixels = []
            for py in range(y, y + names.CELL):
                for px in range(x, x + names.CELL):
                    visible_pixels.append(
                        math.dist((py, px), (player.y + 1.5, player.x + 1.5)) <= 20.0
                        if env.fog() else True
                    )
            any_visible[row, col] = any(visible_pixels)
            fully_visible[row, col] = all(visible_pixels)
            # Keep the historical conservative HUD boundary.  It excludes
            # every pixel touched by the fixed interface, including y=52..54
            # where the interface panel may cover a board cell.
            hud_overlap[row, col] = any(py >= 52 for py in range(y, y + names.CELL))
    return any_visible, fully_visible, hud_overlap


def logical_labels(spec, env: Ls20Scenario):
    roles = np.zeros((names.GRID_ROWS, names.GRID_COLS), dtype=np.uint16)
    attrs = np.full((names.GRID_ROWS, names.GRID_COLS, 3), -1, dtype=np.int8)
    solved = np.zeros((names.GRID_ROWS, names.GRID_COLS), dtype=np.int8)
    goal_cells = np.zeros((names.GRID_ROWS, names.GRID_COLS), dtype=np.uint8)

    def add(cell, bit):
        col, row = map(int, cell)
        if not (0 <= col < names.GRID_COLS and 0 <= row < names.GRID_ROWS):
            raise ValueError(f"cell outside board: {cell}")
        roles[row, col] |= bit

    for cell in spec["walls"]:
        add(cell, ROLE_BITS["wall"])
    for entry in spec.get("cyclers", []):
        add(entry["cell"], ROLE_BITS[f"cycler_{entry['kind']}"])
    for entry in spec.get("launchers", []):
        add(entry["cell"], ROLE_BITS["launcher"])
    for cell in spec.get("refills", []):
        add(cell, ROLE_BITS["refill"])
    add(spec["start"], ROLE_BITS["player"])
    for index, goal in enumerate(spec["goals"]):
        col, row = map(int, goal["cell"])
        add(goal["cell"], ROLE_BITS["goal"])
        attrs[row, col] = np.asarray(goal["triple"], dtype=np.int8)
        goal_cells[row, col] = 1
        solved[row, col] = int(env.goals_solved()[index])
    return roles, attrs, solved, goal_cells


def patch_hash(patch: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(patch, dtype=np.uint8).tobytes()).hexdigest()


def context_hash(frame: np.ndarray, col: int, row: int) -> str:
    x, y = names.cell_to_pixel(col, row)
    patch = frame[y - names.CELL : y + 2 * names.CELL,
                 x - names.CELL : x + 2 * names.CELL]
    return hashlib.sha256(np.ascontiguousarray(patch, dtype=np.uint8).tobytes()).hexdigest()


def support_patch(frame: np.ndarray, col: int, row: int, padding: int = 2) -> np.ndarray | None:
    x, y = names.cell_to_pixel(col, row)
    left, top = x - padding, y - padding
    right, bottom = x + names.CELL + padding, y + names.CELL + padding
    if left < 0 or top < 0 or right > frame.shape[1] or bottom > frame.shape[0]:
        return None
    return frame[top:bottom, left:right]


def support_is_public(record, col: int, row: int, padding: int = 2) -> bool:
    """Whether a 7x7/9x9 pixel support region is entirely public and non-HUD."""
    return support_rejection_reason(record, col, row, padding) is None


def support_rejection_reason(record, col: int, row: int, padding: int = 2) -> str | None:
    """Return the first reason a support region is not fully public."""
    frame = record["frame"]
    x, y = names.cell_to_pixel(col, row)
    left, top = x - padding, y - padding
    right, bottom = x + names.CELL + padding, y + names.CELL + padding
    if left < 0 or top < 0 or right > frame.shape[1] or bottom > frame.shape[0]:
        return "boundary"
    if bottom > 52:
        return "hud"
    if not record["spec"].get("fog", False):
        return None
    px, py = names.cell_to_pixel(*record["player_cell"])
    for iy in range(top, bottom):
        for ix in range(left, right):
            if math.dist((iy, ix), (py + 1.5, px + 1.5)) > 20.0:
                return "fog"
    return None


def collect_record(spec, split: str):
    context_index = int(spec.get("training_context_index", 0))
    level = generate.build_level(spec)
    env = Ls20Scenario(level, context_index=context_index)
    frame = np.asarray(env.render(), dtype=np.uint8)
    second = np.asarray(env.render(), dtype=np.uint8)
    if frame.shape != (names.FRAME_SIZE, names.FRAME_SIZE) or frame.min() < 0 or frame.max() > 15:
        raise ValueError(f"bad render for seed {spec['seed']}: {frame.shape} {frame.min()}..{frame.max()}")
    if not np.array_equal(frame, second):
        raise ValueError(f"nondeterministic render for seed {spec['seed']}")
    if tuple(env.player_cell()) != tuple(spec["start"]):
        raise ValueError(f"player alignment failed for seed {spec['seed']}")
    expected_goals = [tuple(goal["triple"]) for goal in spec["goals"]]
    if [tuple(x) for x in env.goal_triples()] != expected_goals:
        raise ValueError(f"goal alignment failed for seed {spec['seed']}")
    if bool(env.fog()) != bool(spec.get("fog", False)):
        raise ValueError(f"fog alignment failed for seed {spec['seed']}")
    roles, attrs, solved, goal_cells = logical_labels(spec, env)
    any_visible, fully_visible, hud_overlap = public_masks(env, frame)
    label_mask = fully_visible & ~hud_overlap
    support7_mask = np.zeros_like(label_mask)
    support9_mask = np.zeros_like(label_mask)
    for row in range(names.GRID_ROWS):
        for col in range(names.GRID_COLS):
            support7_mask[row, col] = support_is_public({
                "frame": frame, "spec": spec, "player_cell": tuple(map(int, spec["start"]))
            }, col, row, 1)
            support9_mask[row, col] = support_is_public({
                "frame": frame, "spec": spec, "player_cell": tuple(map(int, spec["start"]))
            }, col, row, 2)
    return {
        "split": split,
        "seed": int(spec["seed"]),
        "difficulty": int(spec["difficulty"]),
        "context_index": context_index,
        "spec": spec,
        "frame": frame,
        "roles": roles,
        "attrs": attrs,
        "solved": solved,
        "goal_cells": goal_cells,
        "any_visible": any_visible,
        "fully_visible": fully_visible,
        "hud_overlap": hud_overlap,
        "any_label_mask": any_visible & ~hud_overlap,
        "label_mask": label_mask,
        "support7_label_mask": support7_mask,
        "support9_label_mask": support9_mask,
        "player_cell": tuple(map(int, spec["start"])),
    }


def census(records, mask_key="label_mask"):
    goal_groups = defaultdict(list)
    role_groups = defaultdict(list)
    goal_cell_count = 0
    role_cell_count = 0
    labelable_cell_count = 0
    role_value_counts = Counter()
    for record in records:
        frame = record["frame"]
        mask = record[mask_key]
        for row in range(names.GRID_ROWS):
            for col in range(names.GRID_COLS):
                if not mask[row, col]:
                    continue
                labelable_cell_count += 1
                padding = {"support7_label_mask": 1, "support9_label_mask": 2}.get(mask_key)
                patch = support_patch(frame, col, row, padding) if padding is not None else board_patch(frame, col, row)
                if patch is None:
                    raise AssertionError(f"persisted {mask_key} includes an out-of-bounds support at {(col, row)}")
                info = {
                    "split": record["split"],
                    "seed": record["seed"],
                    "difficulty": record["difficulty"],
                    "cell": [col, row],
                    "patch": patch.tolist(),
                    "role_mask": int(record["roles"][row, col]),
                    "goal_triple": record["attrs"][row, col].tolist()
                    if record["goal_cells"][row, col] else None,
                }
                if record["goal_cells"][row, col]:
                    goal_cell_count += 1
                    goal_groups[patch_hash(patch)].append(info)
                # Include blank floor (role_mask=0), otherwise this census
                # cannot detect a public patch alias between a role and floor.
                role_cell_count += 1
                role_value_counts[int(record["roles"][row, col])] += 1
                role_groups[patch_hash(patch)].append(info)

    def conflict_details(groups, key):
        result = {}
        for digest, occurrences in groups.items():
            values = sorted({tuple(o[key]) if isinstance(o[key], list) else o[key] for o in occurrences})
            if len(values) <= 1:
                continue
            result[digest] = {
                "values": [list(value) if isinstance(value, tuple) else value for value in values],
                "occurrences": occurrences[:12],
            }
        return result

    goal_conflicts = conflict_details(goal_groups, "goal_triple")
    role_conflicts = conflict_details(role_groups, "role_mask")
    conflict_role_counts = Counter()
    for occurrences in role_groups.values():
        values = {int(occurrence["role_mask"]) for occurrence in occurrences}
        if len(values) > 1:
            for occurrence in occurrences:
                conflict_role_counts[int(occurrence["role_mask"])] += 1

    def neighborhood_resolution(groups, key):
        out = {}
        for digest, occurrences in groups.items():
            if len({tuple(o[key]) if isinstance(o[key], list) else o[key] for o in occurrences}) <= 1:
                continue
            # Recompute context hashes only for occurrences whose entire 3x3
            # neighborhood is fully visible and HUD-free.  A differing hash
            # means neighboring public pixels could disambiguate that pair;
            # identical hashes with differing labels remain irreducible here.
            ctx = defaultdict(list)
            for o in occurrences:
                record = next(r for r in records if r["split"] == o["split"] and r["seed"] == o["seed"])
                col, row = o["cell"]
                lo_col, hi_col = col - 1, col + 1
                lo_row, hi_row = row - 1, row + 1
                if lo_col < 0 or hi_col >= names.GRID_COLS or lo_row < 0 or hi_row >= names.GRID_ROWS:
                    continue
                neighborhood_mask = record[mask_key][lo_row : hi_row + 1, lo_col : hi_col + 1]
                if not bool(np.all(neighborhood_mask)):
                    continue
                ctx[context_hash(record["frame"], col, row)].append(o)
            ctx_conflicts = []
            for context_digest, items in ctx.items():
                vals = {tuple(i[key]) if isinstance(i[key], list) else i[key] for i in items}
                if len(vals) > 1:
                    ctx_conflicts.append({"hash": context_digest, "values": [list(v) if isinstance(v, tuple) else v for v in vals]})
            out[digest] = {
                "eligible_occurrences": sum(len(items) for items in ctx.values()),
                "context_groups": len(ctx),
                "context_conflict_groups": ctx_conflicts,
            }
        return out

    return {
        "mask_key": mask_key,
        "mask_definition": "selected mask = all 25 board-patch pixels inside fog aperture AND no patch pixel y>=52"
        if mask_key == "label_mask" else "selected mask = any board-patch pixel inside fog aperture AND no patch pixel y>=52",
        "labelable_cells": labelable_cell_count,
        "labelable_goal_patches": goal_cell_count,
        "labelable_role_patches_including_blank_floor": role_cell_count,
        "goal_patch_groups": len(goal_groups),
        "goal_patch_groups_with_multiple_triples": len(goal_conflicts),
        "goal_conflict_examples": goal_conflicts,
        "role_patch_groups": len(role_groups),
        "role_patch_groups_with_multiple_role_masks": len(role_conflicts),
        "role_value_counts": {str(key): value for key, value in sorted(role_value_counts.items())},
        "role_conflict_occurrences_by_mask": {str(key): value for key, value in sorted(conflict_role_counts.items())},
        "role_conflict_examples": role_conflicts,
        "neighbor_context_analysis": {
            "goal": neighborhood_resolution(goal_groups, "goal_triple"),
            "role": neighborhood_resolution(role_groups, "role_mask"),
        },
    }


def save_npz(records, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format=np.asarray(["pebby.visible-cell-labels.v4"]),
        split=np.asarray([r["split"] for r in records]),
        seeds=np.asarray([r["seed"] for r in records], dtype=np.int64),
        difficulty=np.asarray([r["difficulty"] for r in records], dtype=np.int8),
        context_index=np.asarray([r["context_index"] for r in records], dtype=np.int16),
        frames=np.stack([r["frame"] for r in records]),
        roles=np.stack([r["roles"].reshape(-1) for r in records]),
        goal_attrs=np.stack([r["attrs"].reshape(-1, 3) for r in records]),
        goal_solved=np.stack([r["solved"].reshape(-1) for r in records]),
        goal_cells=np.stack([r["goal_cells"].reshape(-1) for r in records]),
        any_visible=np.stack([r["any_visible"].reshape(-1) for r in records]),
        any_label_mask=np.stack([r["any_label_mask"].reshape(-1) for r in records]),
        fully_visible=np.stack([r["fully_visible"].reshape(-1) for r in records]),
        hud_overlap=np.stack([r["hud_overlap"].reshape(-1) for r in records]),
        label_mask=np.stack([r["label_mask"].reshape(-1) for r in records]),
        support7_label_mask=np.stack([r["support7_label_mask"].reshape(-1) for r in records]),
        support9_label_mask=np.stack([r["support9_label_mask"].reshape(-1) for r in records]),
        player_cell=np.asarray([r["player_cell"] for r in records], dtype=np.int8),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/ls20-verified-train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/ls20-verified-validation.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/world-visible-cell-label-feasibility.json"))
    parser.add_argument("--out", type=Path, default=Path("data/ls20-visible-cell-labels.npz"))
    parser.add_argument("--train-per-difficulty", type=int, default=50)
    parser.add_argument("--validation-per-difficulty", type=int, default=20)
    args = parser.parse_args()
    if args.train_per_difficulty <= 0 or args.validation_per_difficulty <= 0:
        raise ValueError("per-difficulty counts must be positive")

    split_specs = {
        "train": load_selected(args.train, args.train_per_difficulty, 20260912),
        "validation": load_selected(args.validation, args.validation_per_difficulty, 20260913),
    }
    records = []
    for split, specs in split_specs.items():
        records.extend(collect_record(spec, split) for spec in specs)
    save_npz(records, args.out)
    # Re-read the persisted mask arrays before deriving any coverage or alias
    # statistics.  This makes the report a check of the actual training masks,
    # rather than only of transient in-memory arrays.
    with np.load(args.out, allow_pickle=False) as persisted:
        for index, record in enumerate(records):
            for key in ("any_label_mask", "label_mask", "support7_label_mask", "support9_label_mask"):
                record[key] = persisted[key][index].reshape(names.GRID_ROWS, names.GRID_COLS).astype(bool, copy=True)

    def coverage_for_mask(subset, mask_key):
        goal_labelable = sum(int(np.sum(r["goal_cells"] & r[mask_key])) for r in subset)
        goal_count = sum(int(np.sum(r["goal_cells"])) for r in subset)
        return {
            "levels": len(subset),
            "difficulty_counts": dict(Counter(str(r["difficulty"]) for r in subset)),
            "fog_levels": sum(bool(r["spec"].get("fog", False)) for r in subset),
            "two_goal_levels": sum(len(r["spec"]["goals"]) > 1 for r in subset),
            "rail_levels": sum(bool(r["spec"].get("rails")) for r in subset),
            "launcher_levels": sum(bool(r["spec"].get("launchers")) for r in subset),
            "mean_labelable_cells": float(np.mean([np.sum(r[mask_key]) for r in subset])),
            "min_labelable_cells": int(min(np.sum(r[mask_key]) for r in subset)),
            "max_labelable_cells": int(max(np.sum(r[mask_key]) for r in subset)),
            "goal_labelable_count": goal_labelable,
            "goal_count": goal_count,
            "goal_solved_positive_cells": sum(int(np.sum(r["goal_cells"] & r[mask_key] & (r["solved"] > 0))) for r in subset),
            "overlap_cells": sum(int(np.sum((r["roles"] != 0) & ((r["roles"] & (r["roles"] - 1)) != 0))) for r in subset),
        }

    coverage = {}
    for split in ("train", "validation"):
        subset = [r for r in records if r["split"] == split]
        coverage[split] = {
            "full_mask": coverage_for_mask(subset, "label_mask"),
            "any_visible_mask": coverage_for_mask(subset, "any_label_mask"),
            "support7_mask": coverage_for_mask(subset, "support7_label_mask"),
            "support9_mask": coverage_for_mask(subset, "support9_label_mask"),
        }

    rejection_counts = {"support7": Counter(), "support9": Counter()}
    rejection_examples = {"support7": {}, "support9": {}}
    support7_not_full = 0
    support9_not_support7 = 0
    for record in records:
        if np.any(record["support7_label_mask"] & ~record["label_mask"]):
            support7_not_full += int(np.sum(record["support7_label_mask"] & ~record["label_mask"]))
        if np.any(record["support9_label_mask"] & ~record["support7_label_mask"]):
            support9_not_support7 += int(np.sum(record["support9_label_mask"] & ~record["support7_label_mask"]))
        for row in range(names.GRID_ROWS):
            for col in range(names.GRID_COLS):
                for key, padding in (("support7", 1), ("support9", 2)):
                    reason = support_rejection_reason(record, col, row, padding)
                    if reason is None:
                        continue
                    rejection_counts[key][reason] += 1
                    candidate = {
                            "split": record["split"], "seed": record["seed"],
                            "difficulty": record["difficulty"], "cell": [col, row],
                            "center_full_mask": bool(record["label_mask"][row, col]),
                            "fog": bool(record["spec"].get("fog", False)),
                    }
                    previous = rejection_examples[key].get(reason)
                    if previous is None or (candidate["center_full_mask"] and not previous["center_full_mask"]):
                        rejection_examples[key][reason] = candidate
    mask_checks = {
        "support7_subset_of_full_mask_violations": support7_not_full,
        "support9_subset_of_support7_violations": support9_not_support7,
        "rejection_counts_by_reason": {key: dict(sorted(value.items())) for key, value in rejection_counts.items()},
        "rejection_examples": rejection_examples,
    }

    train_selected = split_specs["train"]
    validation_selected = split_specs["validation"]
    train_seeds = {int(spec["seed"]) for spec in train_selected}
    validation_seeds = {int(spec["seed"]) for spec in validation_selected}
    if train_seeds & validation_seeds:
        raise AssertionError("train and validation selected seeds overlap")
    selection = {
        "train_levels": len(train_selected),
        "validation_levels": len(validation_selected),
        "train_per_difficulty": args.train_per_difficulty,
        "validation_per_difficulty": args.validation_per_difficulty,
        "selection_seeds": {"train": 20260912, "validation": 20260913},
        "algorithm": "For each split, group rows by difficulty, sort candidates by integer seed, use random.Random(selection_seed).sample(per_difficulty) independently in difficulty order, then sort each chosen group by seed.",
        "same_records_for_mask_comparison": True,
    }
    source_split_content_separation = {
        "train_source": str(args.train),
        "validation_source": str(args.validation),
        "train_source_sha256": sha256_path(args.train),
        "validation_source_sha256": sha256_path(args.validation),
        "selected_train_rows": len(train_selected),
        "selected_validation_rows": len(validation_selected),
        "selected_train_unique_seeds": len(train_seeds),
        "selected_validation_unique_seeds": len(validation_seeds),
        "train_validation_selected_seed_overlap": sorted(train_seeds & validation_seeds),
        "selected_train_content_sha256": selected_content_sha256(train_selected),
        "selected_validation_content_sha256": selected_content_sha256(validation_selected),
        "source_files_distinct": args.train.resolve() != args.validation.resolve(),
    }
    if not source_split_content_separation["source_files_distinct"]:
        raise AssertionError("train and validation source files must be distinct")

    source_paths = [args.train, args.validation, Path("pebby/ls20/generate.py"), Path("pebby/ls20/env.py"), Path("pebby/ls20/names.py"), Path("third_party/ls20/ls20.py")]
    v1_path = Path("artifacts/world-visible-cell-label-feasibility-v1.json")
    v1_comparison = None
    if v1_path.exists():
        with v1_path.open() as handle:
            v1 = json.load(handle)
        v1_census = v1.get("pixel_collision_census", {})
        v1_comparison = {
            "path": str(v1_path),
            "sha256": sha256_path(v1_path),
            "mask_definition": v1.get("schema", {}).get("label_mask"),
            "visible_goal_patches": v1_census.get("visible_goal_patches"),
            "goal_patch_groups_with_multiple_triples": v1_census.get("goal_patch_groups_with_multiple_triples"),
            "role_patch_groups_with_multiple_role_masks": v1_census.get("role_patch_groups_with_multiple_role_masks"),
            "note": "v1 counted any-visible cells, so its partial-fog conflicts are not evidence of aliases among fully public patches",
        }
    v2_path = Path("artifacts/world-visible-cell-label-feasibility-v2.json")
    v2_comparison = {"path": str(v2_path), "sha256": sha256_path(v2_path)} if v2_path.exists() else None
    full_census = census(records, "label_mask")
    any_census = census(records, "any_label_mask")
    report = {
        "status": "complete",
        "format": "pebby.visible-cell-label-feasibility.v4",
        "initial_state_only": True,
        "schema": {
            "roles": "uint16 per 12x12 cell; wall=1 goal=2 cycler_shape=4 cycler_color=8 cycler_rotation=16 launcher=32 refill=64 player=128; overlaps permitted",
            "goal_attrs": "int8[3] shape/color/rotation at goal cells, -1 elsewhere",
            "goal_solved": "int8 at goal cells; initial audit has no positive solved goals",
            "visibility": "any_visible and fully_visible are computed from the engine's exact 20-pixel fog radius; supervision requires all 25 pixels",
            "hud_overlap": "any board-patch pixel y>=52; these cells are excluded",
            "label_mask": "fully_visible AND NOT hud_overlap; support7_label_mask and support9_label_mask require exact fully public 7x7/9x9 pixel support and are subsets of label_mask",
            "engine_only_omitted": ["rail/patrol support masks are invisible sprites", "step/life hidden state", "unsolved hidden goal state"],
        },
        "sources": {str(path): sha256_path(path) for path in source_paths},
        "extractor_sha256": sha256_path(Path(__file__)),
        "output_npz": {"path": str(args.out), "sha256": sha256_path(args.out)},
        "selection": selection,
        "source_split_content_separation": source_split_content_separation,
        "coverage": coverage,
        "mask_checks": mask_checks,
        "pixel_collision_census": full_census,
        "same_sample_mask_comparison": {
            "full_mask": full_census,
            "any_visible_mask": any_census,
            "support7_mask": census(records, "support7_label_mask"),
            "support9_mask": census(records, "support9_label_mask"),
            "non_hud_rule": "Both masks exclude every cell whose 5x5 patch touches y>=52; only fog visibility differs.",
        },
        "v1_comparison": v1_comparison,
        "v2_comparison": v2_comparison,
        "interpretation": "The v1 report used a different level sample and an any-visible mask, so its collision differences cannot be attributed to masking alone. This v4 report records any-visible, fully-visible 5x5, and persisted fully-public 7x7/9x9 masks on the same selected records and identical non-HUD rule. Fully visible 5x5 role decoding still has blank-floor versus launcher aliases in this sample; the wider masks are the proposed decoder-label masks for those roles. These generated-only counts are diagnostic and do not establish general identifiability.",
        "limitations": [
            "Initial-state labels only; no solved-goal positives or successor/life-reset labels.",
            "Labels are generated from engine/spec state for diagnostics and must never be provided as inference inputs.",
            "A collision-free neighborhood does not prove generalization under all partial observability or dynamic sprites.",
            "No official layouts, routes, frames, or labels were read.",
        ],
    }
    # Keep report stable and human-readable; output hash is recorded before JSON write.
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(args.report), "npz": str(args.out), "records": len(records), "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
