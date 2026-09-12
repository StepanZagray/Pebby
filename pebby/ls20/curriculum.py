"""Small generated-only mechanic lessons, exhaustively verified before release.

This extends the v2 spec with ``curriculum_version`` and optional ``rails``.
No shipped layouts, routes or observations inform the drafts. Rooms, corridors,
rail masks and triples are sampled here. The shared builder supplies sprite
prototypes and HUD chrome. These bounded lessons cover mechanics, not the full
distribution or difficulty of the official evaluation.

Verification uses the training context, seed modulo seven. Saved solutions and
complete-search distances are certified under that context's budget rules.
"""

import argparse
import random
from pathlib import Path

from . import names
from .env import Ls20Env, Ls20Scenario
from .generate import FORMAT, GENERATOR_VERSION, RESERVED, _connected, build_level
from .layout import extract
from .plan import Oracle, simulate

CURRICULUM_VERSION = 2
DIFFICULTIES = (1, 2, 3, 4, 5)
DEFAULT_LIMIT = 120_000
DEFAULT_ATTEMPTS = 24
KINDS = ("shape", "color", "rotation")
SIZES = (names.SHAPE_COUNT, names.COLOR_COUNT, names.ROTATION_COUNT)


def _draft(rng, difficulty):
    # A bounded connected region keeps exhaustive search cheap even with a
    # cyclic clock. Translation, aspect, obstacles and doors vary independently.
    width, height = rng.choice(((4, 4), (5, 4), (4, 5), (6, 3)))
    if difficulty == 4:
        width, height = 5, 4
    left, top = rng.randint(2, 10 - width), rng.randint(1, 9 - height)
    free = {(left + x, top + y) for x in range(width) for y in range(height)}
    topology = rng.choice(("room", "obstacles", "door"))
    if topology == "door":
        door = rng.randrange(height)
        free -= {(left + width // 2, top + y) for y in range(height) if y != door}
    elif topology == "obstacles":
        for cell in rng.sample(sorted(free), k=rng.randint(1, 3)):
            remaining = free - {cell}
            if len(_connected(remaining, min(remaining))) == len(remaining):
                free = remaining
    free -= RESERVED

    count = 1 if difficulty in (1, 3, 4) else (3 if difficulty == 5 and rng.randrange(3) == 0 else 2)
    kinds = rng.sample(KINDS, count)
    rail_mode = "short" if difficulty == 3 else "ring" if difficulty == 4 else "none"
    if difficulty == 5 and count == 2:
        rail_mode = rng.choice(("none", "short"))
    rails, rail_cells, cyclers = [], set(), []
    if rail_mode != "none":
        options = []
        for x, y in sorted(free):
            if rail_mode == "ring":
                cells = {(x + dx, y + dy) for dx in range(3) for dy in range(3)
                         if dx in (0, 2) or dy in (0, 2)}
                if cells <= free:
                    options.append(cells)
            else:
                for dx, dy in ((1, 0), (0, 1)):
                    for length in (2, 3):
                        cells = {(x + dx * i, y + dy * i) for i in range(length)}
                        if cells <= free:
                            options.append(cells)
        if not options:
            return None
        rail_cells = rng.choice(options)
        rails = [{"cells": sorted(rail_cells)}]
        cyclers.append({"cell": rng.choice(sorted(rail_cells)), "kind": kinds[0]})

    spots = sorted(free - rail_cells)
    rng.shuffle(spots)
    goal_count = 2 if difficulty == 5 else 1
    refill_count = rng.randrange(2) if difficulty >= 3 else 0
    if len(spots) < 1 + goal_count + count + refill_count:
        return None
    start = spots.pop()
    for kind in kinds[bool(rails):]:
        cyclers.append({"cell": spots.pop(), "kind": kind})
    refills = [spots.pop() for _ in range(refill_count)]
    start_triple = [rng.randrange(size) for size in SIZES]
    goals = []
    for _ in range(goal_count):
        triple = list(start_triple)
        for kind in kinds:
            i = KINDS.index(kind)
            triple[i] = (triple[i] + rng.randrange(1, SIZES[i])) % SIZES[i]
        if goals and triple == goals[0]["triple"]:
            i = KINDS.index(kinds[-1])
            triple[i] = (triple[i] + 1) % SIZES[i]
        goals.append({"cell": spots.pop(), "triple": triple})

    # A wall behind each pad keeps its second bounding-box trigger inaccessible.
    # Reserve the entire rail walk, not just the initial cycler cell.
    launchers = []
    if difficulty >= 2 and rng.randrange(2):
        options = []
        goal_cells = {goal["cell"] for goal in goals}
        for cell in spots:
            for dx, dy in names.ACTION_DELTAS:
                if (cell[0] - dx, cell[1] - dy) in free:
                    continue
                probe, distance = cell, 0
                while True:
                    probe = (probe[0] + dx, probe[1] + dy)
                    if probe not in free or probe in goal_cells:
                        break
                    distance += 1
                if distance >= 2:
                    options.append({"cell": cell, "delta": (dx, dy)})
        if options:
            launchers.append(rng.choice(options))

    cost = rng.choice((1, 2)) if difficulty >= 3 else 1
    return {
        "format": FORMAT, "generator_version": GENERATOR_VERSION,
        "curriculum_version": CURRICULUM_VERSION, "difficulty": difficulty,
        "size": names.FRAME_SIZE, "topology": topology, "rail_mode": rail_mode,
        "walls": sorted({(x, y) for x in range(names.GRID_COLS)
                         for y in range(names.GRID_ROWS)} - free),
        "start": start, "start_triple": start_triple, "goals": goals,
        "cyclers": cyclers, "rails": rails, "launchers": launchers,
        # The real HUD has only 42 budget pixels; exceeding it crashes rendering.
        "refills": sorted(refills), "step_counter": 42, "step_cost": cost,
        "fog": difficulty == 4 or (difficulty == 5 and bool(rng.randrange(2))),
    }


def _verify(spec, search_limit, min_slack):
    context_index = spec["seed"] % 7
    if context_index == 0 and spec.get("launchers"):
        return None
    env = Ls20Scenario(build_level(spec), context_index)
    try:
        layout = extract(env)
    except ValueError:
        return None
    # extract checks tile overlap across every tick; also reserve launchers,
    # whose trigger cells are intentionally stricter here than Layout requires.
    walked = set().union(*layout.moving_cyclers)
    if walked & {cell for pad in layout.launchers for cell in pad["triggers"]}:
        return None
    oracle = Oracle(layout, limit=search_limit)
    if oracle.truncated or not oracle.solvable:
        return None
    solution = oracle.solution(seed=spec["seed"])
    if not solution or len(solution) < 3:
        return None
    state = oracle.start
    used = {"moving_cycler": False, "launcher": False, "refill": False}
    for action in solution:
        before = state
        state, outcome = simulate(layout, state, names.ACTION_IDS.index(action), oracle.refills)
        used["moving_cycler"] |= before[1:4] != state[1:4] and (
            state[0] in layout.moving_cyclers[state[7]])
        used["launcher"] |= outcome == "launched"
        used["refill"] |= state[5] != before[5]
    slack = state[6] // layout.step_cost
    if slack < min_slack or (layout.patrollers and not used["moving_cycler"]):
        return None
    replay = Ls20Scenario(build_level(spec), context_index)
    for action in solution:
        observation = replay.perform(action)
    if not observation.won or replay.levels_completed != 1 or replay.lives() != 3:
        return None
    return {**spec, "optimal_actions": oracle.optimal_actions, "solution": solution,
            "slack_moves": slack, "reachable_states": oracle._reachable,
            "search_truncated": False, "search_limit": search_limit,
            "verification_level_index": context_index,
            "verification_match_hint": layout.match_hint,
            "context_index": context_index, "training_context_index": context_index,
            "context_solution": solution,
            "context_optimal_actions": oracle.optimal_actions,
            "context_engine_verified": True, "engine_win": True,
            "replay_lives": replay.lives(), "levels_completed": replay.levels_completed,
            "solution_mechanics": used, "engine_verified": True}


def generate_level(seed, difficulty=1, attempts=DEFAULT_ATTEMPTS, min_slack=3,
                   search_limit=DEFAULT_LIMIT):
    """Deterministic complete-search lesson, or an explicit bounded failure."""
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if attempts < 1 or search_limit < 1 or min_slack < 0:
        raise ValueError("attempts/search_limit must be positive and min_slack nonnegative")
    rng = random.Random(f"ls20-curriculum:{CURRICULUM_VERSION}:{seed}:{difficulty}")
    for attempt in range(attempts):
        spec = _draft(rng, difficulty)
        if spec is None:
            continue
        spec.update(seed=seed, generation_attempt=attempt + 1)
        verified = _verify(spec, search_limit, min_slack)
        if verified is not None:
            return verified
    raise RuntimeError(f"no fully verified curriculum level for seed={seed} "
                       f"difficulty={difficulty} within {attempts} attempts "
                       f"and {search_limit} states per search")


def main():
    from .bank import SPLIT_SEEDS, save

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=25)
    parser.add_argument("--split", choices=tuple(SPLIT_SEEDS), default="train")
    parser.add_argument("--difficulties", nargs="+", type=int, choices=DIFFICULTIES,
                        default=list(DIFFICULTIES))
    parser.add_argument("--seed", type=int, default=0, help="Offset within the split's seed range")
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--search-limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--min-slack", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.levels < 1 or args.seed < 0 or args.seed + args.levels > 1_000_000:
        parser.error("levels and seed must stay inside the selected million-seed split")
    if args.attempts < 1 or args.search_limit < 1 or args.min_slack < 0:
        parser.error("attempts/search-limit must be positive; min-slack must be nonnegative")
    specs = []
    # Intentionally one CPU worker. Failure stops the bank; no silent seed skip.
    for i in range(args.levels):
        spec = generate_level(SPLIT_SEEDS[args.split] + args.seed + i,
                              args.difficulties[i % len(args.difficulties)],
                              args.attempts, args.min_slack, args.search_limit)
        specs.append(spec)
        print(f"{i + 1}/{args.levels}: d{spec['difficulty']} seed={spec['seed']} "
              f"states={spec['reachable_states']} actions={spec['optimal_actions']}", flush=True)
    save(specs, args.out)
    print(f"Saved {len(specs)} fully verified curriculum levels to {args.out}")


if __name__ == "__main__":
    main()
