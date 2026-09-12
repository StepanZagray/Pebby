"""Seeded generator for LS20 levels that are provably completable.

A draft level is only returned after two independent checks agree:

1. the exact planner (`plan.Oracle`) finds an action sequence that clears every
   goal without ever overdrawing the step budget, and
2. that sequence is replayed in the *real* upstream game, which must report the
   level completed.

So "completable according to the rules of the level" is proved by the rules'
own implementation, not asserted by the generator.

Levels are assembled from upstream's own sprite prototypes, so a generated
level is the same kind of object as a shipped one and renders identically.
"""

import random

from . import names
from .env import Ls20Env, Ls20Scenario, upstream
from .layout import extract
from .plan import Oracle, advance as plan_advance

GENERATOR_VERSION = 3  # context-specific complete proofs; bounded hard drafts
FORMAT = "pebby.ls20.level.v1"

# Sprite prototypes, by the role each plays.
WALL = "ihdgageizm"
PLAYER = "sfqyzhzkij"
GOAL_PAD = "rjlbuycveu"
GOAL_ICON = "kvynsvxbpi"
GOAL_RING = "vjotnebuqo"
GOAL_HINT = "hoswmpiqkw"
GOAL_PLATE = "nszegiawib"
REFILL = "npxgalaybz"
CYCLER = {"shape": "mkjdaccuuf", "color": "soyhouuebz", "rotation": "rhsxkxzdjz"}
# A pad's name suffix is its launch direction, and it mounts one pixel back
# against its wall, so the sprite offset is the negated delta (ls20.py:1576-1598).
LAUNCHER = {(0, -1): "lujfinsby_t", (0, 1): "kapcaakvb_b",
            (-1, 0): "tihiodtoj_l", (1, 0): "yjgargdic_r"}

# The HUD chrome paints a 12x12 box over the bottom-left of the frame and wins
# the layer-0 tie, so the one lattice cell it would cover is kept solid.
RESERVED = {(1, 10)}

LEGACY_DIFFICULTY = {
    1: {"density": .10, "attributes": 1, "goals": 1, "refills": 0, "cost": 1, "distractors": 0, "launchers": 0},
    2: {"density": .16, "attributes": 2, "goals": 1, "refills": 0, "cost": 1, "distractors": 1, "launchers": 0},
    3: {"density": .22, "attributes": 3, "goals": 1, "refills": 0, "cost": 1, "distractors": 1, "launchers": 1},
    4: {"density": .22, "attributes": 3, "goals": 1, "refills": 2, "cost": 2, "distractors": 2, "launchers": 1},
    5: {"density": .18, "attributes": 3, "goals": 2, "refills": 2, "cost": 1, "distractors": 2, "launchers": 2},
}
LEGACY_DIFFICULTIES = tuple(sorted(LEGACY_DIFFICULTY))
from .reference_profiles import DIFFICULTIES, PROFILES as DIFFICULTY

_ATTRIBUTE_SIZES = {"shape": names.SHAPE_COUNT, "color": names.COLOR_COUNT,
                    "rotation": names.ROTATION_COUNT}


def _interior():
    return [(col, row) for col in range(1, names.GRID_COLS - 1)
            for row in range(1, names.GRID_ROWS - 1) if (col, row) not in RESERVED]


def _connected(free, start):
    seen, stack = {start}, [start]
    while stack:
        col, row = stack.pop()
        for dx, dy in names.ACTION_DELTAS:
            cell = (col + dx, row + dy)
            if cell in free and cell not in seen:
                seen.add(cell)
                stack.append(cell)
    return seen


def _draft(rng, difficulty):
    """One candidate level spec. May well be unsolvable; the caller checks."""
    settings = LEGACY_DIFFICULTY[difficulty]
    cells = _interior()
    free = {cell for cell in cells if rng.random() >= settings["density"]}
    if not free:
        return None
    region = _connected(free, rng.choice(sorted(free)))
    # Bound the spatial factor of the exhaustive state space, especially when
    # goals and consumable refills multiply it. Keep connected irregular maps.
    if difficulty >= 3:
        target = rng.randint(24, 32 if difficulty < 5 else 28)
        bounded = {rng.choice(sorted(region))}
        frontier = set()
        while len(bounded) < min(target, len(region)):
            for x, y in bounded:
                frontier.update((x + dx, y + dy) for dx, dy in names.ACTION_DELTAS
                                if (x + dx, y + dy) in region)
            frontier -= bounded
            bounded.add(rng.choice(sorted(frontier)))
        region = bounded
    needed = 2 + settings["goals"] + settings["refills"] + 3 + settings["distractors"]
    if len(region) < max(needed, 24):
        return None

    spots = sorted(region)
    rng.shuffle(spots)
    start = spots.pop()

    start_triple = (rng.randrange(names.SHAPE_COUNT), rng.randrange(names.COLOR_COUNT),
                    rng.randrange(names.ROTATION_COUNT))
    order = ["shape", "color", "rotation"]
    rng.shuffle(order)
    count = settings["attributes"]
    if difficulty >= 3:
        count = rng.choice((2, 3))
    changing = order[:count]
    goal_triple = list(start_triple)
    index_of = {"shape": 0, "color": 1, "rotation": 2}
    for attribute in changing:
        size = _ATTRIBUTE_SIZES[attribute]
        goal_triple[index_of[attribute]] = (
            start_triple[index_of[attribute]] + rng.randrange(1, size)) % size

    # One cycler per attribute that has to change, plus decoys that must be
    # avoided, so the policy cannot succeed by touching every cycler it sees.
    kinds = list(changing)
    non_required = [kind for kind in order if kind not in changing]
    if non_required:
        for _ in range(settings["distractors"]):
            kinds.append(rng.choice(non_required))
    cyclers, refills, goals = {}, [], []
    for kind in kinds:
        if not spots:
            return None
        cyclers[spots.pop()] = kind
    for _ in range(settings["refills"]):
        if not spots:
            return None
        refills.append(spots.pop())
    for _ in range(settings["goals"]):
        if not spots:
            return None
        goals.append(spots.pop())

    # With two goals both pads demand the same triple; upstream level 6 uses two
    # different ones, but one shared triple keeps the ordering problem honest
    # without needing a second full attribute plan.
    # Launcher pads mount against a wall and fling the player to the last free
    # cell before the next wall or goal pad. Requiring a wall directly behind the
    # pad keeps it to a single trigger cell, which is how upstream's read.
    blocked_ahead = set(goals)
    launchers = []
    for _ in range(rng.randint(0, settings["launchers"])):
        options = []
        for cell in spots:
            for delta in names.ACTION_DELTAS:
                behind = (cell[0] - delta[0], cell[1] - delta[1])
                if behind in region:
                    continue
                probe, reach = cell, 0
                while True:
                    probe = (probe[0] + delta[0], probe[1] + delta[1])
                    if probe not in region or probe in blocked_ahead:
                        break
                    reach += 1
                if reach >= 2:
                    options.append((cell, delta))
        if not options:
            break
        cell, delta = options[rng.randrange(len(options))]
        spots.remove(cell)
        launchers.append({"cell": cell, "delta": list(delta)})

    walls = [cell for cell in cells if cell not in region]
    walls += [(col, 0) for col in range(names.GRID_COLS)]
    walls += [(col, names.GRID_ROWS - 1) for col in range(names.GRID_COLS)]
    walls += [(0, row) for row in range(names.GRID_ROWS)]
    walls += [(names.GRID_COLS - 1, row) for row in range(names.GRID_ROWS)]
    walls += sorted(RESERVED)

    return {"format": FORMAT, "generator_version": GENERATOR_VERSION,
            "difficulty": difficulty, "size": names.FRAME_SIZE,
            "walls": sorted(set(walls)), "start": start, "start_triple": list(start_triple),
            "goals": [{"cell": cell, "triple": list(goal_triple)} for cell in goals],
            "cyclers": [{"cell": cell, "kind": kind} for cell, kind in sorted(cyclers.items())],
            "launchers": launchers,
            "refills": sorted(refills), "step_counter": 42, "step_cost": settings["cost"],
            "fog": False}


def _level_data(spec):
    triples = [goal["triple"] for goal in spec["goals"]]
    many = len(spec["goals"]) > 1
    shapes = [triple[0] for triple in triples]
    colors = [names.COLORS[triple[1]] for triple in triples]
    rotations = [names.ROTATIONS[triple[2]] for triple in triples]
    start_shape, start_color, start_rotation = spec["start_triple"]
    return {
        "StepCounter": spec["step_counter"],
        names.KEY_GOAL_SHAPE: shapes if many else shapes[0],
        names.KEY_GOAL_COLOR: colors if many else colors[0],
        names.KEY_GOAL_ROTATION: rotations if many else rotations[0],
        names.KEY_START_SHAPE: start_shape,
        names.KEY_START_COLOR: names.COLORS[start_color],
        names.KEY_START_ROTATION: names.ROTATIONS[start_rotation],
        names.KEY_FOG: spec["fog"],
        names.KEY_STEPS_DECREMENT: spec["step_cost"],
    }


def _chrome():
    """The fixed HUD furniture, cloned straight out of upstream's level 1."""
    module = upstream()
    gameplay = {names.TAG_WALL, names.TAG_GOAL_PAD, names.TAG_GOAL_ICON, names.TAG_GOAL_RING,
                names.TAG_GOAL_HINT_FRAME, names.TAG_STEP_REFILL, names.TAG_LAUNCHER,
                names.TAG_PATROL_RAIL, names.TAG_PLAYER, *names.CYCLER_TAGS}
    keep = []
    for sprite in module.levels[0].get_sprites():
        if set(sprite.tags or ()) & gameplay or sprite.name == GOAL_PLATE:
            continue
        keep.append(sprite.clone())
    return keep


def build_level(spec):
    """Turn a spec into a real ARCEngine Level made of upstream's own sprites."""
    from arcengine import Level, Sprite
    prototypes = upstream().sprites
    sprites = list(_chrome())

    def place(name, cell, offset=(0, 0), rotation=None):
        x, y = names.cell_to_pixel(*cell)
        sprite = prototypes[name].clone().set_position(x + offset[0], y + offset[1])
        if rotation is not None:
            sprite.set_rotation(rotation)
        sprites.append(sprite)
        return sprite

    for cell in spec["walls"]:
        place(WALL, tuple(cell))
    for entry in spec["cyclers"]:
        place(CYCLER[entry["kind"]], tuple(entry["cell"]))
    # Rails are invisible masks, not artwork copied from a shipped layout.
    # Full cell squares support upstream's pixel-perfect overlap pairing; -1
    # outside the requested cells keeps a ring's interior non-walkable.
    for index, entry in enumerate(spec.get("rails", ())):
        cells = [tuple(cell) for cell in entry["cells"]]
        left, top = min(c[0] for c in cells), min(c[1] for c in cells)
        width = (max(c[0] for c in cells) - left + 1) * names.CELL
        height = (max(c[1] for c in cells) - top + 1) * names.CELL
        pixels = [[-1] * width for _ in range(height)]
        for col, row in cells:
            for dy in range(names.CELL):
                for dx in range(names.CELL):
                    pixels[(row - top) * names.CELL + dy][(col - left) * names.CELL + dx] = 0
        x, y = names.cell_to_pixel(left, top)
        sprites.append(Sprite(pixels=pixels, name=f"generated_rail_{index}", x=x, y=y,
                              visible=False, tags=[names.TAG_PATROL_RAIL]))
    for cell in spec["refills"]:
        place(REFILL, tuple(cell), offset=(1, 1))
    for entry in spec.get("launchers", ()):
        delta = tuple(entry["delta"])
        place(LAUNCHER[delta], tuple(entry["cell"]), offset=(-delta[0], -delta[1]))
    for goal in spec["goals"]:
        cell = tuple(goal["cell"])
        place(GOAL_PLATE, cell, offset=(-2, -2), rotation=180)
        place(GOAL_RING, cell, offset=(-1, -1))
        place(GOAL_HINT, cell, offset=(-1, -1))
        place(GOAL_PAD, cell)
        place(GOAL_ICON, cell, offset=(1, 1))
    place(PLAYER, tuple(spec["start"]))
    return Level(sprites=sprites, grid_size=(names.FRAME_SIZE, names.FRAME_SIZE),
                 data=_level_data(spec))


def build_levels(specs):
    return [build_level(spec) for spec in specs]


def _verify(spec, min_slack, search_limit=600_000):
    """Plan the draft, then make the real game confirm the plan wins.

    Also requires budget headroom. LS20's step budget is tight, and a level with
    no slack is unlearnable: one wrong move makes it permanently unwinnable, so
    a policy would have to be flawless to score at all. `min_slack` is the
    number of spare moves that must remain after an optimal run.

    Returns the annotated spec, or None if the draft fails any check.
    """
    context_index = spec["seed"] % 7
    # Pending launcher hints are outside the oracle state. Re-roll the draft;
    # removing its launchers would silently change the requested mechanic.
    if context_index == 0 and spec.get("launchers"):
        return None
    env = Ls20Scenario(build_level(spec), context_index)
    try:
        layout = extract(env)
    except ValueError:
        return None
    oracle = Oracle(layout, limit=search_limit)
    if oracle.truncated or not oracle.solvable:
        return None
    solution = oracle.solution(seed=spec["seed"])
    if solution is None or len(solution) < 6:
        return None  # reject the trivial ones; they teach nothing
    state = oracle.start
    for action in solution:
        state = plan_advance(layout, state, names.ACTION_IDS.index(action), oracle.refills)
    slack = state[6] // layout.step_cost
    if slack < min_slack:
        return None
    replay = Ls20Scenario(build_level(spec), context_index)
    for action in solution:
        observation = replay.perform(action)
    if not observation.won or replay.levels_completed != 1 or replay.lives() != 3:
        return None
    spec["optimal_actions"] = oracle.optimal_actions
    spec["slack_moves"] = slack
    spec["solution"] = solution
    spec["reachable_states"] = oracle._reachable
    spec["search_truncated"] = oracle.truncated
    spec.update(search_limit=search_limit, engine_verified=True,
                context_index=context_index, training_context_index=context_index,
                context_solution=solution,
                context_optimal_actions=oracle.optimal_actions,
                context_engine_verified=True, verification_level_index=context_index,
                verification_match_hint=layout.match_hint, engine_win=True,
                replay_lives=replay.lives(), levels_completed=replay.levels_completed)
    return spec


def generate_legacy_level(seed, difficulty=1, attempts=400, min_slack=8):
    """A completable level for `seed`, or raise if none was found.

    Deterministic: the same (seed, difficulty) always gives the same level.
    """
    if difficulty not in LEGACY_DIFFICULTY:
        raise ValueError(f"legacy difficulty must be one of {LEGACY_DIFFICULTIES}")
    rng = random.Random((seed, difficulty).__hash__() ^ seed)
    for _ in range(attempts):
        spec = _draft(rng, difficulty)
        if spec is None:
            continue
        spec["seed"] = seed
        verified = _verify(spec, min_slack)
        if verified is not None:
            return verified
    raise RuntimeError(f"no completable level for seed={seed} difficulty={difficulty}")


def generate_level(seed, difficulty=1, attempts=400, min_slack=None, search_limit=None, **kwargs):
    """Generate a seven-tier reference-profile lesson with its actual tier context.

    Historical five-tier fixtures remain available through generate_legacy_level;
    their seed-context and difficulty semantics are deliberately unchanged.
    """
    from .reference_generator import generate_level as reference_level
    return reference_level(seed, difficulty, attempts=attempts, min_slack=min_slack,
                           search_limit=search_limit, **kwargs)


def generate_levels(seeds, difficulty=1, **kwargs):
    return [generate_level(seed, difficulty, **kwargs) for seed in seeds]
