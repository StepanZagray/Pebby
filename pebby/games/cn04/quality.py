"""Bounded family-local quality audit for generated CN04 rows."""

from collections import Counter, defaultdict
import statistics
import time

from .generate import DIFFICULTIES, FULL_STANDARD_CONTRACT, SPLITS, generate, validate_full_standard


REJECTION_REASONS = (
    "invalid_geometry",
    "profile_structure",
    "reference_action_length",
    "geometry_split",
    "constructive_witness",
    "official_copy",
    "context_index",
    "early_completion",
    "transition_mismatch",
    "native_replay",
    "teacher_search",
    "teacher_search_truncated",
    "profile_proof",
    "constraint_evidence",
    "search_truncated",
    "proven_unsolvable",
)


def audit_generated(*, seeds=(0,), splits=SPLITS, attempts=120):
    """Generate a modest stratified sample and return a JSON-ready report."""
    requested = accepted = 0
    failures = []
    rejection_counts = Counter({reason: 0 for reason in REJECTION_REASONS})
    rows = []
    timing = defaultdict(list)
    for split in splits:
        for difficulty in DIFFICULTIES:
            for seed in seeds:
                requested += 1
                started = time.perf_counter()
                row = generate(seed, difficulty, split=split, attempts=attempts)
                elapsed = time.perf_counter() - started
                timing[difficulty].append(elapsed)
                if row is None:
                    failures.append({
                        "seed": seed,
                        "difficulty": difficulty,
                        "split": split,
                        "rejections": dict(generate.last_rejections),
                    })
                    rejection_counts.update(generate.last_rejections)
                    continue
                errors = validate_full_standard(
                    row, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
                )
                if errors:
                    failures.append({
                        "seed": seed,
                        "difficulty": difficulty,
                        "split": split,
                        "validation_errors": errors,
                    })
                    continue
                accepted += 1
                rejection_counts.update(row["generation_exclusions"])
                rows.append(row)
    by_tier = {}
    for difficulty in DIFFICULTIES:
        tier_rows = [row for row in rows if row["difficulty"] == difficulty]
        lengths = [row["solution_length"] for row in tier_rows]
        densities = [row["visual_density"] for row in tier_rows]
        by_tier[str(difficulty)] = {
            "accepted": len(tier_rows),
            "mean_seconds": statistics.fmean(timing[difficulty]) if timing[difficulty] else 0.0,
            "witness_actions": {
                "min": min(lengths) if lengths else None,
                "median": statistics.median(lengths) if lengths else None,
                "max": max(lengths) if lengths else None,
            },
            "visual_density": {
                "min": min(densities) if densities else None,
                "max": max(densities) if densities else None,
            },
            "mechanic_exercise": {
                "stack_action_rows": sum(
                    row["solution_mechanics"]["stack_cycles_action"] > 0 for row in tier_rows
                ),
                "stack_click_rows": sum(
                    row["solution_mechanics"]["stack_cycles_click"] > 0 for row in tier_rows
                ),
                "bounce_reversal_rows": sum(
                    row["solution_mechanics"]["bounce_reversals"] > 0 for row in tier_rows
                ),
            },
            "relation_geometry_distinct": len({
                row["solution_constraints"]["relation_geometry_sha256"]
                for row in tier_rows
            }),
            "relation_topology_distinct": len({
                row["solution_constraints"]["topology_sha256"]
                for row in tier_rows
            }),
            "winning_alternate_assignments_distinct": len({
                tuple(row["solution_constraints"]["winning_alternates"])
                for row in tier_rows
            }),
            "teacher_work": {
                "min": min((row["reachable_states"] for row in tier_rows), default=None),
                "max": max((row["reachable_states"] for row in tier_rows), default=None),
                "bounded_assignment_caps": sum(
                    row["proof"]["bounded_assignment_caps"] for row in tier_rows
                ),
            },
        }
    geometry_by_split = {
        split: {row["geometry_d4_sha256"] for row in rows if row["split"] == split}
        for split in splits
    }
    gameplay_by_split = {
        split: {row["gameplay_sha256"] for row in rows if row["split"] == split}
        for split in splits
    }
    overlaps = {}
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1:]:
            overlaps[f"{left}:{right}"] = {
                "geometry_d4": len(geometry_by_split[left] & geometry_by_split[right]),
                "gameplay": len(gameplay_by_split[left] & gameplay_by_split[right]),
            }
    return {
        "requested": requested,
        "accepted": accepted,
        "accept_rate": accepted / requested if requested else 0.0,
        "failures": failures,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "by_tier": by_tier,
        "cross_split_overlap": overlaps,
        "distinct_geometry_d4": len({row["geometry_d4_sha256"] for row in rows}),
        "distinct_gameplay": len({row["gameplay_sha256"] for row in rows}),
        "calibration_caveat": (
            "Each tolerance is centered on one shipped level; this bounded sample "
            "is an implementation audit, not a population confidence estimate."
        ),
    }
