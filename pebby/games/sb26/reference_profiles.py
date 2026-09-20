"""Measured contracts for SB26's eight shipped levels.

There is one official level per tier.  The numeric ranges below are therefore
explicit engineering tolerances around scarce references, not population
confidence intervals.  Source rows are ``third_party/arc3_games/sb26.py``
lines 361-679; traversal semantics are lines 889-1072.
"""

DIFFICULTIES = tuple(range(1, 9))


REFERENCE_PROFILES = {
    1: dict(frame_arities=(4,), connector_count=0, fixed_regular=0, fixed_links=0, movable_regular=4,
            movable_links=0, goals=4, distinct_regular_colours=4,
            reference_actions=9, reference_density=0.1143,
            required_use=dict(link_traversals=0, maximum_depth=1)),
    2: dict(frame_arities=(4, 4), connector_count=1, fixed_regular=0, fixed_links=1, movable_regular=7,
            movable_links=0, goals=7, distinct_regular_colours=7,
            reference_actions=15, reference_density=0.1934,
            required_use=dict(link_traversals=1, distinct_frames_entered=2, maximum_depth=2)),
    3: dict(frame_arities=(5, 2, 2), connector_count=0, fixed_regular=0, fixed_links=2, movable_regular=7,
            movable_links=0, goals=7, distinct_regular_colours=7,
            reference_actions=15, reference_density=0.2021,
            required_use=dict(link_traversals=2, distinct_frames_entered=3, maximum_depth=2)),
    4: dict(frame_arities=(5, 3), connector_count=0, fixed_regular=1, fixed_links=0, movable_regular=6,
            movable_links=1, goals=7, distinct_regular_colours=7,
            reference_actions=15, reference_density=0.1904,
            required_use=dict(link_traversals=1, fixed_regular_visits=1,
                              movable_link_visits=1, maximum_depth=2)),
    5: dict(frame_arities=(5, 3), connector_count=0, fixed_regular=0, fixed_links=0, movable_regular=6,
            movable_links=2, goals=9, distinct_regular_colours=5,
            reference_actions=17, reference_density=0.2178,
            required_use=dict(link_traversals=2, distinct_link_tiles=2,
                              repeated_frame_entries=1, maximum_depth=2)),
    6: dict(frame_arities=(3, 3, 3, 3), connector_count=0, fixed_regular=3, fixed_links=0, movable_regular=6,
            movable_links=3, goals=9, distinct_regular_colours=6,
            reference_actions=19, reference_density=0.2568,
            required_use=dict(link_traversals=3, distinct_frames_entered=4,
                              fixed_regular_visits=3, movable_link_visits=3, maximum_depth=2)),
    7: dict(frame_arities=(3, 3, 3), connector_count=0, fixed_regular=1, fixed_links=0, movable_regular=6,
            movable_links=2, goals=7, distinct_regular_colours=4,
            reference_actions=17, reference_density=0.2031,
            required_use=dict(link_traversals=2, distinct_frames_entered=3,
                              fixed_regular_visits=1, movable_link_visits=2, maximum_depth=3)),
    8: dict(frame_arities=(4, 4), connector_count=0, fixed_regular=0, fixed_links=0, movable_regular=6,
            movable_links=2, goals=12, distinct_regular_colours=6,
            reference_actions=17, reference_density=0.2502,
            required_use=dict(link_traversals=3, distinct_link_tiles=2,
                              repeated_frame_entries=2, cycle_reentries=1, maximum_depth=3)),
}


def profile_errors(difficulty, metrics, mechanics=None):
    """Return fail-closed structural, visual, and exercised-use mismatches."""
    profile = REFERENCE_PROFILES[difficulty]
    errors = []
    fields = (
        "frame_arities",
        "connector_count",
        "fixed_regular",
        "fixed_links",
        "movable_regular",
        "movable_links",
        "goals",
        "distinct_regular_colours",
    )
    for field in fields:
        actual = metrics.get(field)
        expected = profile[field]
        if field == "frame_arities" and actual is not None:
            actual = tuple(actual)
        if actual != expected:
            errors.append(f"{field}={actual!r}, expected {expected!r}")
    density = metrics.get("visual_density")
    if density is None or abs(float(density) - profile["reference_density"]) > 0.006:
        errors.append(
            f"visual_density={density!r} outside reference {profile['reference_density']:.4f} +/- 0.006"
        )
    if tuple(metrics.get("visible_bbox", ())) != (0, 0, 63, 60):
        errors.append(f"visible_bbox={metrics.get('visible_bbox')!r}, expected (0, 0, 63, 60)")
    if metrics.get("initial_energy") != 64:
        errors.append(f"initial_energy={metrics.get('initial_energy')!r}, expected 64")
    if mechanics is not None:
        if not mechanics.get("won"):
            errors.append("winning traversal certificate is absent")
        for field, floor in profile["required_use"].items():
            actual = mechanics.get(field)
            if actual != floor:
                errors.append(f"mechanic {field}={actual!r}, expected reference-calibrated value {floor}")
    return errors
