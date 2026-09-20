"""Bounded, reproducible quality audit for the full eight-tier generator."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from statistics import mean, median
import time

import numpy as np

from . import names
from .env import Env
from .generate import DIFFICULTIES, build_level, generate
from .generation_quality import SPLITS
from .layout import extract


def _component_sizes(points):
    remaining = set(points)
    sizes = []
    while remaining:
        stack = [remaining.pop()]
        size = 0
        while stack:
            x, y = stack.pop()
            size += 1
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def obstacle_pixel_metrics(env):
    """Measure immobile collision geometry from native sprite pixels.

    Raw pixels deliberately include off-camera geometry, matching the historic
    reference table.  Rendered unions are clipped separately, so an off-camera
    official boundary cannot be mistaken for visible obstacle presentation.
    """
    layout = extract(env)
    rods = {rod.name: rod for rod in layout.rods}
    moving = {rod.name for rod in layout.rods if rod.controlled}
    while True:
        expanded = moving | {
            child
            for name in moving
            for child in (rods[name].children if name in rods else ())
        }
        if expanded == moving:
            break
        moving = expanded

    frame = np.asarray(env.render(), dtype=np.int16)
    raw_collision = 0
    raw_visible = 0
    background_collision = 0
    frame_points = set()
    arena_points = set()
    tagged_rods = list(env.level.get_sprites_by_tag(names.TAG_ROD))
    generated = any(sprite.name.startswith("rod") for sprite in tagged_rods)
    for sprite in tagged_rods:
        if generated:
            is_obstacle = sprite.name.startswith(("obstacle", "boundary"))
        else:
            is_obstacle = sprite.name not in moving
        if not is_obstacle:
            continue
        pixels = np.asarray(sprite.pixels, dtype=np.int16)
        collision = pixels != -1
        visible = collision & (pixels >= 0) & (pixels != names.BACKGROUND_COLOR)
        raw_collision += int(np.count_nonzero(collision))
        raw_visible += int(np.count_nonzero(visible))
        background_collision += int(np.count_nonzero(collision & ~visible))
        for local_y, local_x in np.argwhere(visible):
            x = int(sprite.x) + int(local_x)
            y = int(sprite.y) + int(local_y)
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                # A collision position is visibly represented if native render
                # leaves any non-background cue there, including an overlapping
                # moving rod of the same visible wall system.
                if frame[y, x] != names.BACKGROUND_COLOR:
                    frame_points.add((x, y))
                    if y < 41:
                        arena_points.add((x, y))
    components = _component_sizes(arena_points)
    return {
        "raw_collision_pixels": raw_collision,
        "raw_visible_pixels": raw_visible,
        "background_collision_pixels": background_collision,
        "visible_collision_fraction": (
            raw_visible / raw_collision if raw_collision else 1.0
        ),
        "rendered_frame_visible_union": len(frame_points),
        "rendered_arena_visible_union": len(arena_points),
        "rendered_arena_components": components,
    }


def _frame_metrics(env):
    frame = np.asarray(env.render(), dtype=np.int16)
    arena = frame[:41]
    panel = frame[41:63]
    occupied = np.argwhere(arena != names.BACKGROUND_COLOR)
    bbox = None if not len(occupied) else [
        int(occupied[:, 1].min()), int(occupied[:, 0].min()),
        int(occupied[:, 1].max()), int(occupied[:, 0].max()),
    ]
    return {
        "arena_nonbackground": int(np.count_nonzero(arena != names.BACKGROUND_COLOR)),
        "panel_nonbackground": int(np.count_nonzero(panel != names.BACKGROUND_COLOR)),
        "arena_bbox": bbox,
        "palette": sorted(int(value) for value in np.unique(frame)),
        "obstacles": obstacle_pixel_metrics(env),
    }


def _presentation(spec):
    env = Env([build_level(spec)])
    frame = np.asarray(env.render(), dtype=np.int16)
    colors = {int(row["color"]) for row in spec["rails"] + spec["buttons"]}
    extents = all(
        sprite.name.startswith("boundary") or (
            0 <= sprite.x and 0 <= sprite.y
            and sprite.x + sprite.width <= names.FRAME
            and sprite.y + sprite.height <= names.FRAME
        )
        for sprite in env.level.get_sprites()
    )
    return {
        "extents_in_frame": extents,
        "control_cues_visible": all(
            np.any(frame[:41] == color) and np.any(frame[41:63] == color)
            for color in colors
        ),
        "pin_target_color_visible": (
            np.count_nonzero(frame[:41] == names.PIN_COLOR) >= len(spec["pins"])
        ),
        **_frame_metrics(env),
    }


def audit(sample_seeds=8, split="train"):
    if type(sample_seeds) is not int or sample_seeds < 1:
        raise ValueError("sample_seeds must be a positive integer")
    if split not in SPLITS:
        raise ValueError("unknown split")
    official = Env()
    report = {
        "sample_seeds": sample_seeds,
        "split": split,
        "tiers": [],
    }
    for difficulty in DIFFICULTIES:
        started = time.perf_counter()
        rejection_reasons = Counter()

        def record(row):
            rejection_reasons[row["reason"]] += 1

        specs = []
        for seed in range(sample_seeds):
            spec = generate(seed, difficulty, split=split, record_rejection=record)
            if spec is not None:
                specs.append(spec)
        presentations = [_presentation(spec) for spec in specs]
        official.set_level(difficulty - 1)
        attempts = [int(spec["generation_attempt"]) for spec in specs]
        lengths = [int(spec["solution_length"]) for spec in specs]
        mechanics = Counter()
        for spec in specs:
            mechanics.update({key: int(value) for key, value in spec["solution_mechanics"].items()
                              if isinstance(value, int) and not isinstance(value, bool)})
        candidates = len(specs) + sum(rejection_reasons.values())
        report["tiers"].append({
            "difficulty": difficulty,
            "accepted": len(specs),
            "requested": sample_seeds,
            "candidate_acceptance_rate": (len(specs) / candidates if candidates else 0.0),
            "seconds": round(time.perf_counter() - started, 3),
            "attempts": {
                "min": min(attempts) if attempts else None,
                "median": median(attempts) if attempts else None,
                "max": max(attempts) if attempts else None,
                "mean": round(mean(attempts), 3) if attempts else None,
            },
            "solution_lengths": {
                "min": min(lengths) if lengths else None,
                "median": median(lengths) if lengths else None,
                "max": max(lengths) if lengths else None,
            },
            "unique_geometry_d4": len({spec["geometry_d4_sha256"] for spec in specs}),
            "unique_gameplay": len({spec["gameplay_sha256"] for spec in specs}),
            "unique_solution_semantics": len({
                spec["solution_semantic_sha256"] for spec in specs
            }),
            "rejections": dict(sorted(rejection_reasons.items())),
            "mechanic_action_totals": dict(sorted(mechanics.items())),
            "presentation_all_pass": all(
                row["extents_in_frame"] and row["control_cues_visible"]
                and row["pin_target_color_visible"] for row in presentations
            ),
            "generated_arena_nonbackground": {
                "min": min(row["arena_nonbackground"] for row in presentations),
                "median": median(row["arena_nonbackground"] for row in presentations),
                "max": max(row["arena_nonbackground"] for row in presentations),
            },
            "generated_obstacle_pixels": {
                key: {
                    "min": min(row["obstacles"][key] for row in presentations),
                    "median": median(row["obstacles"][key] for row in presentations),
                    "max": max(row["obstacles"][key] for row in presentations),
                }
                for key in (
                    "raw_collision_pixels", "raw_visible_pixels",
                    "background_collision_pixels", "rendered_frame_visible_union",
                    "rendered_arena_visible_union",
                )
            },
            "official_frame": _frame_metrics(official),
        })
    report["all_accepted"] = all(row["accepted"] == sample_seeds for row in report["tiers"])
    report["all_geometry_unique"] = all(
        row["unique_geometry_d4"] == sample_seeds for row in report["tiers"]
    )
    report["all_solution_semantics_unique"] = all(
        row["unique_solution_semantics"] == sample_seeds for row in report["tiers"]
    )
    report["all_presentation_pass"] = all(
        row["presentation_all_pass"] for row in report["tiers"]
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--split", choices=SPLITS, default="train")
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.seeds, args.split), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
