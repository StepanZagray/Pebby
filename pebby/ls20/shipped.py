"""Verified facts about the seven shipped LS20 levels.

Planning levels 6 and 7 exhaustively costs minutes and gigabytes (21.7M states
and 6.8 GiB for level 7), so nothing at request time should re-derive them.
These values are cached here instead. Regenerate with

    PYTHONPATH=. uv run python tools/shipped_optima.py

Each optimum was found by `plan.Oracle` and then replayed in the real game,
which reported the level completed. Human baselines are upstream's own
`baseline_actions` from `third_party/ls20/metadata.json`.
"""

LEVEL_COUNT = 7

# Fewest actions that clear each level, index 0..6.
OPTIMAL_ACTIONS = (13, 45, 39, 43, 44, 72, 53)

# Published human medians, for scale. Pebby's planner beats every one.
HUMAN_BASELINE = (22, 123, 73, 84, 96, 192, 186)

# Search limit each level actually needs. Levels 6 and 7 are why callers must
# not use the Oracle default (600,000) on the shipped set: it truncates
# silently, leaving `solvable` False and `solution()` None without raising.
SEARCH_LIMIT = (600_000, 600_000, 600_000, 600_000, 1_000_000, 13_000_000, 22_000_000)

# Rough peak resident memory of that search, in GiB. Level 7 needs most of a
# small machine; never run two at once.
SEARCH_PEAK_GIB = (0.1, 0.1, 0.1, 0.2, 0.5, 4.3, 6.8)

CHEAP_LEVELS = tuple(i for i, limit in enumerate(SEARCH_LIMIT) if limit <= 1_000_000)


def optimal(index):
    return OPTIMAL_ACTIONS[index]


def search_limit(index):
    """The limit to pass to `Oracle` for this level. Do not rely on the default."""
    return SEARCH_LIMIT[index]
