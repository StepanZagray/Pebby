"""Canonical CD82 identities for novelty checks and disjoint data splits."""

import hashlib
import json
import time
from collections import Counter


SPLITS = ("train", "validation", "test")
IDENTITY_VERSION = "cd82-semantic-color-d4-v2"


def _compared_values(target):
    if len(target) != 10 or any(len(row) != 10 for row in target):
        raise ValueError("target must be 10x10")
    return [int(target[row][col]) for row in range(10) for col in range(10)
            if row != col and row + col != 9]


def _normalise(values):
    labels = {}
    result = []
    for value in values:
        if value not in labels:
            labels[value] = len(labels)
        result.append(f"colour:{labels[value]}")
    return result


def _legacy_geometry_normalise(values):
    """The v1 all-colour normalization, retained only for stable split buckets."""
    labels = {}
    result = []
    for value in values:
        if value not in labels:
            labels[value] = len(labels)
        result.append(labels[value])
    return result


def _d4(target):
    grid = tuple(tuple(int(value) for value in row) for row in target)
    variants = []
    current = grid
    for _ in range(4):
        variants.append(current)
        variants.append(tuple(tuple(reversed(row)) for row in current))
        current = tuple(tuple(row) for row in zip(*current[::-1]))
    return variants


def _digest(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _digest_in_bucket(payload, bucket):
    """Hash a v2 identity while retaining its established modulo-three split."""
    nonce = 0
    while True:
        digest = _digest({**payload, "partition_nonce": nonce})
        if int(digest, 16) % len(SPLITS) == bucket:
            return digest
        nonce += 1


def canonical_identities(spec):
    """Return ``(gameplay, geometry)`` SHA-256 identities.

    Both identities ignore diagonal pixels, palette ordering, private
    metadata, and numeric color labels while preserving the compared-cell
    color partition, indicator presence, and palette cardinality. Geometry
    additionally canonicalizes D4 transforms. Thus gameplay, split, and
    official-copy checks use one executable puzzle equivalence.
    """
    target = spec["target"]
    common = {"indicator": bool(spec["indicator"]),
              "palette_count": len(spec["palette"])}
    gameplay = _digest({"identity_version": IDENTITY_VERSION, **common,
                        "cells": _normalise(_compared_values(target))})
    geometry_cells = min(
        tuple(_normalise(_compared_values(variant)))
        for variant in _d4(target)
    )
    legacy_geometry_cells = min(
        tuple(_legacy_geometry_normalise(_compared_values(variant)))
        for variant in _d4(target)
    )
    legacy_digest = _digest({**common, "cells": legacy_geometry_cells})
    legacy_bucket = int(legacy_digest, 16) % len(SPLITS)
    geometry = _digest_in_bucket(
        {"identity_version": IDENTITY_VERSION, **common, "cells": geometry_cells},
        legacy_bucket,
    )
    return gameplay, geometry


def raw_geometry_identity(spec):
    """Orientation-preserving semantic target identity."""
    return _digest({
        "identity_version": IDENTITY_VERSION,
        "indicator": bool(spec["indicator"]),
        "palette_count": len(spec["palette"]),
        "cells": _normalise(_compared_values(spec["target"])),
    })


def geometry_split(geometry_sha256):
    """Stable three-way bucket for a canonical geometry digest."""
    if not isinstance(geometry_sha256, str) or len(geometry_sha256) != 64:
        raise ValueError("geometry identity must be a SHA-256 hex digest")
    try:
        bucket = int(geometry_sha256, 16) % len(SPLITS)
    except ValueError as error:
        raise ValueError("geometry identity must be a SHA-256 hex digest") from error
    return SPLITS[bucket]


def audit_quality(seeds=(0,), splits=("train",), *, attempts=120, limit=200_000):
    """Run a bounded all-tier generation audit and return JSON-safe evidence."""
    from .generate import DIFFICULTIES, generate_report
    from .reference_profiles import profile_errors

    if not seeds or not splits:
        raise ValueError("audit requires at least one seed and split")
    if any(split not in SPLITS for split in splits):
        raise ValueError("audit split must be train, validation or test")
    started = time.monotonic()
    tiers = {difficulty: {
        "requested": 0, "accepted": 0, "seconds": 0.0, "attempts": 0,
        "rejections": Counter(), "optimal_actions": [], "search_expanded": [],
        "solution_mechanics": Counter(), "failures": [],
    } for difficulty in DIFFICULTIES}
    identities = {split: {key: [] for key in
                          ("gameplay_sha256", "geometry_sha256", "geometry_d4_sha256")}
                  for split in splits}
    for split in splits:
        for seed in seeds:
            for difficulty in DIFFICULTIES:
                tier = tiers[difficulty]
                tier["requested"] += 1
                tick = time.monotonic()
                report = generate_report(seed, difficulty, attempts, limit, split=split)
                tier["seconds"] += time.monotonic() - tick
                tier["attempts"] += report["attempts"]
                tier["rejections"].update(report["rejections"])
                spec = report["spec"]
                if spec is None:
                    tier["failures"].append({"seed": seed, "split": split,
                                             "reason": "attempts_exhausted"})
                    continue
                errors = profile_errors(spec)
                if errors:
                    tier["failures"].append({"seed": seed, "split": split,
                                             "reason": "profile_validation", "errors": errors})
                    continue
                tier["accepted"] += 1
                tier["optimal_actions"].append(spec["optimal_actions"])
                tier["search_expanded"].append(spec["search_expanded"])
                tier["solution_mechanics"].update({
                    key: value for key, value in spec["solution_mechanics"].items()
                    if isinstance(value, int)
                })
                for key in identities[split]:
                    identities[split][key].append(spec[key])
    cross_split_overlap = {}
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1:]:
            for key in identities[left]:
                cross_split_overlap[f"{left}:{right}:{key}"] = len(
                    set(identities[left][key]) & set(identities[right][key]))
    serialised_tiers = {}
    for difficulty, tier in tiers.items():
        tier["seconds"] = round(tier["seconds"], 6)
        tier["accept_rate"] = tier["accepted"] / tier["requested"]
        tier["rejections"] = dict(sorted(tier["rejections"].items()))
        tier["solution_mechanics"] = dict(sorted(tier["solution_mechanics"].items()))
        serialised_tiers[str(difficulty)] = tier
    requested = sum(tier["requested"] for tier in tiers.values())
    accepted = sum(tier["accepted"] for tier in tiers.values())
    return {
        "audit_version": "cd82-bounded-quality-v2",
        "identity_version": IDENTITY_VERSION,
        "seeds": list(seeds),
        "splits": list(splits),
        "attempt_cap": attempts,
        "search_limit_request": limit,
        "requested": requested,
        "accepted": accepted,
        "accept_rate": accepted / requested,
        "seconds": round(time.monotonic() - started, 6),
        "tiers": serialised_tiers,
        "identity_counts": {
            split: {key: len(set(values)) for key, values in fields.items()}
            for split, fields in identities.items()
        },
        "cross_split_overlap": cross_split_overlap,
        "failures": sum((tier["failures"] for tier in tiers.values()), []),
    }
