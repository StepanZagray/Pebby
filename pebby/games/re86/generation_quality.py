"""Bounded all-tier RE86 quality and diversity audit helpers."""

from collections import Counter
import statistics
import time

from .generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    generate,
    solution_semantic_identity,
    validate_full_standard,
)


def semantic_route_signature(spec):
    """Return the private route hash, separate from public puzzle identity."""
    return solution_semantic_identity(spec)


def audit_quality(*, seeds_per_tier=8, split="train", attempts=180,
                  seed_offset=0):
    """Generate/validate a modest sample and report honest distributions."""
    if type(seeds_per_tier) is not int or seeds_per_tier < 1:
        raise ValueError("seeds_per_tier must be a positive integer")
    started = time.monotonic()
    accepted = []
    failures = []
    rejections = Counter()
    for difficulty in DIFFICULTIES:
        for ordinal in range(seeds_per_tier):
            seed = seed_offset + difficulty * 100_000 + ordinal
            stats = {}
            spec = generate(
                seed, difficulty, attempts=attempts, stats=stats, split=split
            )
            rejections.update({key: value for key, value in stats.items()
                               if key != "accepted"})
            if spec is None:
                failures.append({"difficulty": difficulty, "seed": seed,
                                 "reason": "bounded_generation_exhausted"})
                continue
            errors = validate_full_standard(
                spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
            )
            if errors:
                failures.append({"difficulty": difficulty, "seed": seed,
                                 "reason": "validation", "errors": errors})
                continue
            accepted.append(spec)

    by_tier = {}
    for difficulty in DIFFICULTIES:
        rows = [spec for spec in accepted if spec["difficulty"] == difficulty]
        lengths = [spec["solution_length"] for spec in rows]
        densities = [spec["structural_metrics"]["visual_nonbackground"]
                     for spec in rows]
        route_signatures = {semantic_route_signature(spec) for spec in rows}
        by_tier[str(difficulty)] = {
            "requested": seeds_per_tier,
            "accepted": len(rows),
            "acceptance_rate": len(rows) / seeds_per_tier,
            "solution_length": _distribution(lengths),
            "visual_nonbackground": _distribution(densities),
            "distinct_geometry": len({spec["geometry_d4_sha256"] for spec in rows}),
            "distinct_gameplay": len({spec["gameplay_sha256"] for spec in rows}),
            "distinct_semantic_routes": len(route_signatures),
            "mechanic_event_totals": dict(sorted(_mechanic_totals(rows).items())),
        }
    return {
        "requested": len(DIFFICULTIES) * seeds_per_tier,
        "accepted": len(accepted),
        "acceptance_rate": len(accepted) / (len(DIFFICULTIES) * seeds_per_tier),
        "elapsed_seconds": time.monotonic() - started,
        "split": split,
        "seeds_per_tier": seeds_per_tier,
        "per_tier": by_tier,
        "rejections": dict(sorted(rejections.items())),
        "failures": failures,
    }


def _distribution(values):
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def _mechanic_totals(rows):
    totals = Counter()
    for spec in rows:
        for key, value in spec["solution_mechanics"].items():
            if type(value) is int and value > 0:
                totals[key] += value
    return totals
