"""Stateless LS20 operations: replay, oracle advice and learned-policy inference.

Nothing here keeps a game between calls. The caller owns the level and the list
of actions taken so far; every operation rebuilds the game and replays that list
(0.3 ms per action for a move that lands, about 1.8 ms for one that is blocked
and animates a rejection), so two clients can never see each other's state and a
crashed client loses nothing. The only thing cached is the oracle's search
result, which is a pure function of the level and therefore not state.

torch is never imported at module scope. The server has to boot and serve the UI
on a machine with no CUDA, no torch and no checkpoint, so nothing imports torch
until a checkpoint that actually exists is about to be loaded: with no
checkpoint on disk, the import never happens at all.
"""

from pebby.ls20.provenance import difficulty_version, generated_context

import json
import threading
from functools import lru_cache
from pathlib import Path

from pebby.ls20 import generate, names, shipped
from pebby.ls20.env import Ls20Env, Ls20Scenario
from pebby.ls20.plan import Unplannable, oracle_for

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "ls20-policy.pt"

RULESET = generate.FORMAT
GAME = "ls20"
SHIPPED_LEVELS = shipped.LEVEL_COUNT

# The planner's own default (600,000 reachable states) is sized for generated
# levels and truncates *silently* on the big shipped ones, leaving `solvable`
# False with no exception, so nothing here relies on it. This is the ceiling one
# request is allowed to allocate: 1,000,000 states is what shipped level 5 needs
# and costs about half a gigabyte. Levels 6 and 7 need 13M and 22M states and
# 4.3 and 6.8 GiB, which no request on an unauthenticated loopback server should
# be able to ask for, so they are answered from `pebby.ls20.shipped`'s cached,
# replay-verified numbers instead of being searched at all.
ORACLE_STATE_LIMIT = 1_000_000

# A replay is the only unbounded work a request can ask for, so it is bounded.
# The number is set from the measured worst case, not the average: 2048 actions
# that all walk into a wall take ~3.7 s, because a blocked move still renders a
# rejection animation, while 2048 that land take ~0.6 s. It still exceeds any
# honest game, which cannot spend more than 42 moves a life, three lives, across
# seven levels. 4096 was measured at ~7.5 s, which is too long to hand a client.
MAX_ACTIONS = 2048
MAX_SEED = 0xFFFFFFFF

# The ARC-AGI-3 palette, verbatim from the official toolkit: arcprize/ARC-AGI,
# `arc_agi/rendering.py` lines 20-38, `COLOR_MAP` ("Color mapping for frame
# values (0-15)"). The trailing FF alpha of the upstream RGBA strings is dropped,
# exactly as upstream's own `hex_to_rgb` does. The same 16 values in the same
# order also appear in arcprize/ARC-AGI-3-Agents `agents/templates/multimodal.py`
# and `agents/templates/reasoning_agent.py`. This is NOT the 10-colour ARC-AGI-1
# palette; the indices do not agree with it and substituting it renders LS20 in
# the wrong colours at almost every index.
PALETTE = (
    "#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333", "#000000",
    "#E53AA3", "#FF7BCC", "#F93C31", "#1E93FF", "#88D8F1", "#FFDC00",
    "#FF851B", "#921231", "#4FCC30", "#A356D6",
)
PALETTE_NAMES = (
    "white", "off-white", "light neutral", "neutral", "off-black", "black",
    "magenta", "light magenta", "red", "blue", "light blue", "yellow",
    "orange", "maroon", "green", "purple",
)

# Keys a generated spec may carry, in the order they are echoed back. The
# generator adds provenance fields the client returns untouched; anything else is
# a client inventing structure, which is rejected rather than ignored.
SPEC_FIELDS = ("format", "generator_version", "seed", "difficulty", "difficulty_version", "reference_profile", "size",
               "walls", "start", "start_triple", "goals", "cyclers", "launchers",
               "refills", "step_counter", "step_cost", "fog",
               "optimal_actions", "slack_moves", "solution", "reachable_states",
               "search_truncated", "search_limit", "context_index", "training_context_index",
               "verification_level_index", "verification_match_hint", "context_optimal_actions",
               "context_solution", "context_engine_verified", "engine_verified", "engine_win",
               "levels_completed", "replay_lives", "rails", "reference_level",
               "reference_optimal_actions", "reference_calibration", "quality_version", "quality_profile",
               "curriculum_version", "topology", "rail_mode", "free_cells", "corridor_fraction",
               "bbox_width", "bbox_height", "changed_kinds", "generation_attempt", "split", "source",
               "verification_lives", "oracle_backend", "proof", "minimum_slack_moves", "budget_floor",
               "geometry_sha256", "geometry_d4_sha256", "geometry_split", "geometry_version",
               "solution_mechanics", "patroller_count", "tick_period", "changing_attributes",
               "distractor_count", "nonrequired_distractor_count", "non_required_distractor_count",
               "gameplay_sha256", "generation_exclusions", "mechanics_version")
SPEC_KEYS = frozenset(SPEC_FIELDS)
# The subset that decides what the game does. Everything else is provenance, so
# it is kept out of the oracle cache key and cannot be used to thrash the cache.
GAMEPLAY_FIELDS = ("walls", "start", "start_triple", "goals", "cyclers", "launchers",
                   "refills", "step_counter", "step_cost", "fog", "rails")
# `launchers` arrived with generator version 2, so a version-1 spec that a client
# still holds stays loadable; validation fills it in as empty.
REQUIRED_SPEC_KEYS = frozenset(field for field in ("format",) + GAMEPLAY_FIELDS
                               if field not in ("launchers", "rails"))
TRIPLE_SIZES = (names.SHAPE_COUNT, names.COLOR_COUNT, names.ROTATION_COUNT)
LAUNCHER_DELTAS = [list(delta) for delta in names.ACTION_DELTAS]


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    """JSON has no integer type, and `True` is an int in Python. Neither is one here."""
    return type(value) is int


def _cell(value, what):
    # Tuples arrive from the generator, lists from JSON; both mean the same pair.
    _check(isinstance(value, (list, tuple)) and len(value) == 2 and all(_integer(v) for v in value),
           f"{what} must be a [column, row] pair of integers.")
    col, row = value
    _check(0 <= col < names.GRID_COLS and 0 <= row < names.GRID_ROWS,
           f"{what} must be inside the {names.GRID_COLS}x{names.GRID_ROWS} lattice.")
    return (col, row)


def _triple(value, what):
    _check(isinstance(value, (list, tuple)) and len(value) == 3 and all(_integer(v) for v in value),
           f"{what} must be a [shape, colour, rotation] triple of integers.")
    for index, size in enumerate(TRIPLE_SIZES):
        _check(0 <= value[index] < size,
               f"{what} entry {index} must be between 0 and {size - 1}.")
    return list(value)


def _list(value, what, limit):
    _check(isinstance(value, (list, tuple)), f"{what} must be a list.")
    _check(len(value) <= limit, f"{what} must hold at most {limit} entries.")
    return value


def validate_actions(value):
    """The client's whole action history, as ids 1..4."""
    _check(isinstance(value, list), "actions must be a list of action ids.")
    _check(len(value) <= MAX_ACTIONS, f"actions must hold at most {MAX_ACTIONS} ids.")
    for action in value:
        _check(_integer(action) and action in names.ACTION_IDS,
               f"Each action must be one of {list(names.ACTION_IDS)}.")
    return list(value)


def validate_level(value):
    """A shipped-level reference or a full generated spec, both untrusted.

    Returns the level in canonical form: known fields only, in a fixed order,
    cells as lists. Unknown fields are refused rather than ignored, so a
    generator that starts emitting something new fails here loudly instead of
    quietly dropping a field that changes how the level plays.
    """
    _check(isinstance(value, dict), "level must be a JSON object.")
    if "shipped" in value:
        _check(set(value) == {"shipped"}, 'A shipped level is exactly {"shipped": index}.')
        index = value["shipped"]
        _check(_integer(index) and 0 <= index < SHIPPED_LEVELS,
               f"shipped must be an integer between 0 and {SHIPPED_LEVELS - 1}.")
        return {"shipped": index}

    unknown = sorted(set(value) - SPEC_KEYS)
    _check(not unknown, f"Unknown level fields: {', '.join(unknown)}.")
    missing = sorted(REQUIRED_SPEC_KEYS - set(value))
    _check(not missing, f"Level is missing: {', '.join(missing)}.")
    _check(value["format"] == RULESET, f"level format must be {RULESET}.")

    # Both become sets in the layout reader, so order and repetition cannot
    # change the game. Collapsing them here means one level has exactly one
    # canonical form, and therefore exactly one entry in the plan cache.
    walls = sorted({_cell(cell, "Each wall")
                    for cell in _list(value["walls"], "walls", names.GRID_COLS * names.GRID_ROWS)})
    refills = sorted({_cell(cell, "Each refill") for cell in _list(value["refills"], "refills", 16)})
    goals, cyclers = [], []
    for goal in _list(value["goals"], "goals", 8):
        _check(isinstance(goal, dict) and {"cell", "triple"} <= set(goal)
               and set(goal) <= {"cell", "triple", "vanishing_ring"},
               'Each goal needs cell and triple, with an optional boolean vanishing_ring.')
        canonical = {"cell": list(_cell(goal["cell"], "Each goal cell")),
                     "triple": _triple(goal["triple"], "Each goal triple")}
        if "vanishing_ring" in goal:
            _check(type(goal["vanishing_ring"]) is bool, "vanishing_ring must be true or false.")
            _check(not goal["vanishing_ring"] or value.get("generator_version") == 4,
                   "Vanishing goal rings require generator_version 4.")
            # Explicit false is equivalent to the legacy plain ring.
            if goal["vanishing_ring"]:
                canonical["vanishing_ring"] = True
        goals.append(canonical)
    _check(goals, "A level needs at least one goal.")
    for cycler in _list(value["cyclers"], "cyclers", 32):
        _check(isinstance(cycler, dict) and set(cycler) == {"cell", "kind"},
               'Each cycler must be {"cell": [c, r], "kind": "shape"|"color"|"rotation"}.')
        # `x in dict` raises for an unhashable x, and an uncaught TypeError is a
        # dropped connection rather than a 400, so the type comes first.
        _check(isinstance(cycler["kind"], str) and cycler["kind"] in generate.CYCLER,
               'Each cycler kind must be "shape", "color" or "rotation".')
        cyclers.append({"cell": list(_cell(cycler["cell"], "Each cycler cell")),
                        "kind": cycler["kind"]})

    launchers = []
    for launcher in _list(value.get("launchers", []), "launchers", 8):
        _check(isinstance(launcher, dict) and set(launcher) == {"cell", "delta"},
               'Each launcher must be {"cell": [c, r], "delta": [dx, dy]}.')
        delta = launcher["delta"]
        # `[True, 0] == [1, 0]` and `[1.0, 0] == [1, 0]` are both true in Python,
        # so membership alone would let a non-integer delta through and give the
        # same level two different canonical forms.
        _check(isinstance(delta, (list, tuple)) and len(delta) == 2 and all(_integer(v) for v in delta)
               and list(delta) in LAUNCHER_DELTAS,
               f"Each launcher delta must be one of {LAUNCHER_DELTAS}.")
        launchers.append({"cell": list(_cell(launcher["cell"], "Each launcher cell")),
                          "delta": list(delta)})

    rails = []
    for rail in _list(value.get("rails", []), "rails", 32):
        _check(isinstance(rail, dict) and set(rail) == {"cells"},
               'Each rail must be {"cells": [[column, row], ...]}.')
        cells = sorted({_cell(cell, "Each rail cell")
                        for cell in _list(rail["cells"], "rail cells", names.GRID_COLS * names.GRID_ROWS)})
        _check(len(cells) >= 2, "A rail needs at least two distinct cells.")
        rails.append({"cells": [list(cell) for cell in cells]})
    rails.sort(key=lambda rail: rail["cells"])

    start = _cell(value["start"], "start")
    _check(_integer(value["step_counter"]) and 0 < value["step_counter"] <= 1000,
           "step_counter must be an integer between 1 and 1000.")
    _check(_integer(value["step_cost"]) and 0 < value["step_cost"] <= 10,
           "step_cost must be an integer between 1 and 10.")
    _check(type(value["fog"]) is bool, "fog must be true or false.")
    context = value.get('training_context_index', 0)
    try:
        if difficulty_version(value):
            _check(context == generated_context(value),
                   'calibrated training_context_index must equal difficulty - 1.')
    except ValueError as error:
        _check(False, str(error))
    _check(_integer(context) and 0 <= context < SHIPPED_LEVELS,
           'training_context_index must be an integer between 0 and 6.')
    for field in ('context_index', 'verification_level_index'):
        if field in value:
            _check(_integer(value[field]) and value[field] == context,
                   f'{field} must agree with training_context_index.')

    # The layout reader refuses levels that stack two interacting tiles on one
    # cell, because the effect order would then depend on sprite list order.
    # Rejecting that here keeps every accepted level plannable and describable.
    wall_cells = {tuple(cell) for cell in walls}
    specials = [tuple(cell) for cell in refills] + [tuple(g["cell"]) for g in goals] \
        + [tuple(c["cell"]) for c in cyclers] + [tuple(l["cell"]) for l in launchers]
    _check(len(specials) == len(set(specials)), "Two interacting tiles share one cell.")
    _check(not (set(specials) & wall_cells), "An interacting tile sits inside a wall.")
    _check(start not in wall_cells, "start sits inside a wall.")

    walked = [tuple(cell) for rail in rails for cell in rail["cells"]]
    _check(len(walked) == len(set(walked)), "Two rails share a cell.")
    _check(not (set(walked) & wall_cells), "A rail sits inside a wall.")
    fixed_specials = set(refills) | {tuple(g["cell"]) for g in goals} | {tuple(l["cell"]) for l in launchers}
    _check(not (set(walked) & fixed_specials), "A rail overlaps a fixed interacting tile.")
    for rail in rails:
        cells = {tuple(cell) for cell in rail["cells"]}
        _check(sum(tuple(cycler["cell"]) in cells for cycler in cyclers) == 1,
               "Each rail needs exactly one cycler.")

    level = {key: value[key] for key in SPEC_FIELDS if key in value}
    level.update({"walls": [list(cell) for cell in walls], "refills": [list(cell) for cell in refills],
                  "goals": goals, "cyclers": cyclers, "launchers": launchers, "rails": rails, "start": list(start),
                  "start_triple": _triple(value["start_triple"], "start_triple")})
    return level


def cache_key(level):
    """Canonical text for the parts of a level that change how it plays.

    Only ever called on the output of `validate_level`, which has already put the
    order-free parts in one fixed order; the parts left in the client's order —
    goal pads, whose index decides which required triple they carry — are ones
    where a different order is a different game.
    """
    if "shipped" in level:
        return json.dumps(level, separators=(",", ":"))
    gameplay = {key: level.get(key, []) if key == "rails" else level[key] for key in GAMEPLAY_FIELDS}
    gameplay['training_context_index'] = level.get('training_context_index', 0)
    return json.dumps(gameplay,
                      sort_keys=True, separators=(",", ":"))


def build_env(level):
    """A fresh game positioned on `level`, having performed no action."""
    if "shipped" in level:
        env = Ls20Env()
        if level["shipped"]:
            env.set_level(level["shipped"])
        return env
    context = level.get('training_context_index', 0)
    if context == 0:
        return Ls20Env(generate.build_levels([dict(level)]))
    return Ls20Scenario(generate.build_level(dict(level)), context)


def replay(level, actions, history=None):
    """Rebuild the game and apply every action. Returns (env, frames, frame).

    `frames` is what the final action rendered, which is what an ARC-AGI-3 agent
    receives; `frame` is the last frame still available. A terminal action
    returns no frames upstream, so the previous one is carried forward rather
    than handing the client a null board.
    """
    env = build_env(level)
    frame = env.render()
    if history is not None:
        history.observe(frame, reset=True)
    frames = [frame]
    for action in actions:
        old_lives, old_level = env.lives(), env.level_index
        observation = env.perform(action)
        if observation.frames:
            frames = observation.frames
            frame = observation.frame
            if history is not None:
                history.observe(frame, names.ACTION_IDS.index(action),
                                reset=env.lives() < old_lives or env.level_index != old_level)
    return env, frames, frame


def status_of(env):
    """Everything the UI needs to describe the game without decoding HUD pixels."""
    state = env.state
    return {
        "state": state.name,
        "level_index": env.level_index,
        "level_count": env.level_count,
        "levels_completed": env.levels_completed,
        "steps_left": env.steps_left(),
        "step_cost": env.step_cost(),
        "lives": env.lives(),
        "triple": list(env.triple()),
        "goal_triples": [list(triple) for triple in env.goal_triples()],
        "goals_solved": [bool(solved) for solved in env.goals_solved()],
        "player_cell": list(env.player_cell()),
        "fog": env.fog(),
        "finished": state.name in ("WIN", "GAME_OVER"),
        "won": state.name == "WIN",
    }


def oracle_levels():
    """Shipped levels the oracle can answer step by step inside a request.

    Since rail support landed the planner understands all seven, so "does it
    refuse this level" no longer separates anything; what separates them is
    whether the search fits in a request. `shipped.CHEAP_LEVELS` is that,
    measured: levels 1-5 need at most a million states, levels 6 and 7 need 13
    and 22 million. This is not a claim about solvability either way.
    """
    return tuple(index for index in shipped.CHEAP_LEVELS if 0 <= index < SHIPPED_LEVELS)


def known_optimum(level, level_index):
    """What is already known about this level's best solution, without searching.

    Shipped levels come from `pebby.ls20.shipped`, where every optimum was found
    by the planner and then replayed in the real game, which reported the level
    completed. Generated levels carry their own, proved the same way at
    generation time.
    """
    if "shipped" in level:
        return {"optimal": shipped.optimal(level_index),
                "human_baseline": shipped.HUMAN_BASELINE[level_index]}
    return {"optimal": level.get("optimal_actions"), "human_baseline": None}


_oracle_lock = threading.Lock()


def planned(level, level_index):
    """The plan for one level, computed at most once.

    Serialised on purpose. The search is CPU bound, so two clients asking for the
    same plan at the same time would each compute it and make each other slower;
    behind the lock the second one waits and then finds it in the cache. The UI
    asks for a plan on every move, so this is the common case, not a rare race.
    """
    with _oracle_lock:
        return _cached_oracle(cache_key(level), level_index)


@lru_cache(maxsize=8)
def _cached_oracle(canonical, level_index):
    """Plan one level. Pure in (level, level index), so caching is not state.

    Small cache on purpose: each entry holds a distance map over every reachable
    game state, which is hundreds of thousands of entries for a hard level.
    """
    key = json.loads(canonical)
    is_shipped = "shipped" in key
    env = build_env(key if is_shipped else {**key, "format": RULESET})
    if env.level_index != level_index:
        env.set_level(level_index)
    # Explicit, never the planner's default. For a shipped level the measured
    # requirement is used when it fits the ceiling, so level 5 gets the million
    # states it needs instead of silently truncating at 600,000.
    limit = min(shipped.search_limit(level_index), ORACLE_STATE_LIMIT) if is_shipped \
        else ORACLE_STATE_LIMIT
    return oracle_for(env, limit=limit)


def oracle_advice(level, actions):
    """The optimal next action for the replayed position, plus what is already known.

    Four distinguishable answers, because "no" means four different things here:
    the game is over; the level is too big to search inside a request; the
    planner searched and its search was cut off; or it searched, understood the
    position, and this particular position can no longer be finished. None of
    them is a claim that a level is unsolvable, which nobody has established for
    levels 6 and 7.
    """
    env, _, _ = replay(level, actions)
    index = env.level_index
    known = known_optimum(level, index)
    refused = {"action": None, "available": False, "remaining": None, **known}

    if status_of(env)["finished"]:
        return {**refused, "reason": "The game is over."}
    if "shipped" in level and index not in oracle_levels():
        states = shipped.search_limit(index)
        return {**refused, "reason":
                f"Level {index + 1} needs about {states // 1_000_000} million planner states and "
                f"{shipped.SEARCH_PEAK_GIB[index]} GiB, past the {ORACLE_STATE_LIMIT:,}-state limit a "
                f"request may use, so a search here would hit that limit rather than finish. Its "
                f"optimal solution is a cached {known['optimal']} actions, but the oracle cannot "
                f"step through it."}
    try:
        oracle = planned(level, index)
    except Unplannable as error:
        return {**refused, "reason": f"Level {index + 1} is not plannable: {error}"}
    except ValueError as error:
        return {**refused, "reason": f"Level {index + 1} cannot be read: {error}"}

    state = oracle.state_of(env)
    action = oracle.action_at(env)
    if action is None and not oracle.solvable:
        # Searched, and still nothing. A truncated search says so, because
        # "found nothing" and "stopped looking" are not the same statement.
        limit = " Its search hit the state limit." if getattr(oracle, "truncated", False) else ""
        return {**refused, "reason": f"The planner found no plan for level {index + 1}.{limit}"}
    return {"action": action, "available": True, "remaining": oracle.distance_for(state), **known,
            "reason": None if action is not None
                      else "This position can no longer be completed; undo or reset."}


class AgentPolicy:
    """The learned policy, loaded on first use and never at import time.

    Every failure mode here — no checkpoint, no torch, a checkpoint from a model
    revision that no longer exists — has to come back as an ordinary answer the
    UI can render, because the UI is the thing people use to discover that no
    agent is trained yet.
    """

    def __init__(self, checkpoint=None):
        self.path = Path(checkpoint) if checkpoint else None
        self.explicit = checkpoint is not None
        self.loaded = False
        self.parameters = None
        self.metadata = {}
        self.reason = "The agent has not been loaded yet."
        self._model = None
        self._attempted = False

    def ensure(self):
        """Try once. A second call after a failure returns the same explanation."""
        if self._attempted:
            return self.loaded
        self._attempted = True
        path = self.path or DEFAULT_CHECKPOINT
        if not path.exists():
            self.reason = (f"No agent checkpoint at {path}. Train one with "
                           "`uv run python -m pebby.agent.train`, then restart with --checkpoint.")
            return False
        try:
            import torch  # noqa: PLC0415 - lazy on purpose

            from pebby.agent.model import load_checkpoint  # noqa: PLC0415
            torch.set_num_threads(1)  # One request must not seize every core.
            model, checkpoint = load_checkpoint(path)
        except Exception as error:  # noqa: BLE001 - any import or load failure must degrade
            self.reason = f"Could not load {path}: {type(error).__name__}: {error}"
            return False
        self._model = model
        self.loaded = True
        self.reason = None
        self.metadata = {key: value for key, value in checkpoint.items()
                         if key not in {"weights", "encoder_weights", "planner_weights", "perceptor_weights"}}
        self.parameters = self.metadata.get("parameters")
        if self.parameters is None and hasattr(model, "parameter_count"):
            self.parameters = model.parameter_count()
        if self.parameters is None:
            self.parameters = sum(parameter.numel() for parameter in model.parameters())
        return True

    def act(self, frame, history=None):
        """Greedy action id and the four move probabilities for one 64x64 frame."""
        import torch  # noqa: PLC0415 - lazy on purpose
        from pebby.agent.model import frames_to_tensor  # noqa: PLC0415

        with torch.inference_mode():
            output = history.scores() if history is not None else self._model(frames_to_tensor(frame))
        # The policy has carried a value head in some revisions and not in
        # others; only the logits were ever the policy.
        logits = output[0] if isinstance(output, tuple) else output
        if logits.dim() == 2:
            logits = logits[0]
        probabilities = logits.softmax(-1).tolist()
        action = names.ACTION_IDS[int(max(range(len(probabilities)), key=probabilities.__getitem__))]
        return action, probabilities

    def describe(self):
        return {"loaded": self.loaded, "parameters": self.parameters,
                "checkpoint": str(self.path) if self.explicit else None,
                "reason": self.reason}


class Engine:
    """Every operation the UI and the HostAI bridge can ask for."""

    def __init__(self, checkpoint=None, *, banks=None):
        from pebby.level_banks import LevelBanks
        self.agent = AgentPolicy(checkpoint)
        self.banks = banks if banks is not None else LevelBanks()

    def info(self):
        # Attempt the load here rather than reporting "not loaded" for a
        # checkpoint nobody has exercised yet. With no checkpoint on disk this
        # still costs nothing and still does not import torch.
        self.agent.ensure()
        return {
            "game": GAME,
            "ruleset": RULESET,
            "generator_version": generate.GENERATOR_VERSION,
            "actions": [{"id": id, "name": name}
                        for id, name in zip(names.ACTION_IDS, names.ACTION_NAMES)],
            "palette": list(PALETTE),
            "palette_names": list(PALETTE_NAMES),
            "palette_source": "arcprize/ARC-AGI arc_agi/rendering.py COLOR_MAP",
            "shipped_levels": SHIPPED_LEVELS,
            "oracle_levels": list(oracle_levels()),
            "difficulties": list(generate.DIFFICULTIES),
            "max_actions": MAX_ACTIONS,
            "max_seed": MAX_SEED,
            "grid": {"cols": names.GRID_COLS, "rows": names.GRID_ROWS, "cell": names.CELL,
                     "x_origin": names.X_ORIGIN, "y_origin": names.Y_ORIGIN,
                     "frame_size": names.FRAME_SIZE},
            "triple": {"shapes": names.SHAPE_COUNT, "colors": list(names.COLORS),
                       "rotations": list(names.ROTATIONS)},
            "agent": self.agent.describe(),
        }

    def boot(self, limit=50):
        """Everything the viewer needs to paint its first screen, in one trip.

        The viewer used to assemble this from four dependent requests — info,
        then banks, then the first page of rows, then that row's level — each
        waiting on the one before it. The work here is identical; only the
        waiting goes away. Every part stays individually reachable, so nothing
        below depends on a client having asked for the composite first.
        """
        catalogue = self.banks.catalogue()
        result = {"info": self.info(), "banks": catalogue["banks"], "page": None, "level": None}
        if not catalogue["banks"]:
            return result
        # The viewer opens on the first bank's training split, so that is what
        # arrives prefetched; every other view is a normal bank_levels call.
        page = self.banks.levels(catalogue["banks"][0]["id"], "train", None, 0, limit)
        result["page"] = page
        if page["levels"]:
            result["level"] = self.bank_level(page["bank"], page["levels"][0]["id"])
        return result

    def generate(self, seed, difficulty):
        _check(_integer(seed) and 0 <= seed <= MAX_SEED,
               f"seed must be an integer between 0 and {MAX_SEED}.")
        _check(_integer(difficulty) and difficulty in generate.DIFFICULTY,
               f"difficulty must be one of {list(generate.DIFFICULTIES)}.")
        try:
            level = generate.generate_level(seed, difficulty)
        except RuntimeError as error:
            raise ValueError(str(error)) from error
        level = validate_level(level)
        env, _, frame = replay(level, [])
        return {"level": level, "frame": frame, "status": status_of(env)}

    def bank_level(self, bank, row_id):
        row, metadata = self.banks.level(bank, row_id)
        level = validate_level(row)
        env, _, frame = replay(level, [])
        return {"level": level, "frame": frame, "status": status_of(env), **metadata}

    def shipped(self, index):
        level = validate_level({"shipped": index})
        env, _, frame = replay(level, [])
        return {"level": level, "frame": frame, "status": status_of(env)}

    def play(self, level, actions):
        env, frames, frame = replay(level, actions)
        return {"frames": frames, "frame": frame, "status": status_of(env)}

    def oracle(self, level, actions):
        return oracle_advice(level, actions)

    def act(self, level, actions):
        if not self.agent.ensure():
            return {"action": None, "probabilities": None, "loaded": False,
                    "reason": self.agent.reason}
        try:
            history = None
            if self.agent._model.config().get("architecture") == "world":
                from pebby.agent.history import PolicyHistory
                history = PolicyHistory(self.agent._model)
            env, _, frame = replay(level, actions, history)
            action, probabilities = self.agent.act(frame, history)
        except Exception as error:  # noqa: BLE001 - a broken policy must not 500
            return {"action": None, "probabilities": None, "loaded": True,
                    "reason": f"The policy failed on this frame: {type(error).__name__}: {error}"}
        return {"action": action, "probabilities": probabilities, "loaded": True,
                "reason": None, "status": status_of(env)}

    def dispatch(self, request):
        """One JSON object in, one JSON object out. Both HTTP routes land here."""
        _check(isinstance(request, dict), "Request must be a JSON object.")
        op = request.get("op")
        _check(isinstance(op, str), 'Request must contain a string "op".')
        fields = set(request)
        if op == "boot" and fields <= {"op", "limit"}:
            return self.boot(request.get("limit", 50))
        if op == "banks" and fields == {"op"}:
            return self.banks.catalogue()
        if op == "bank_levels" and fields <= {"op", "bank", "split", "difficulty", "offset", "limit"}:
            _check("bank" in request, "bank is required.")
            return self.banks.levels(request["bank"], request.get("split", "train"),
                                    request.get("difficulty"), request.get("offset", 0), request.get("limit", 50))
        if op == "bank_level" and fields == {"op", "bank", "id"}:
            return self.bank_level(request["bank"], request["id"])
        if op == "info" and fields == {"op"}:
            return self.info()
        if op == "generate" and fields <= {"op", "seed", "difficulty"}:
            return self.generate(request.get("seed", 0), request.get("difficulty", 1))
        if op == "shipped" and fields == {"op", "index"}:
            return self.shipped(request["index"])
        if op in ("play", "oracle", "agent") and fields <= {"op", "level", "actions"}:
            _check("level" in request, "level is required.")
            level = validate_level(request["level"])
            actions = validate_actions(request.get("actions", []))
            if op == "play":
                return self.play(level, actions)
            if op == "oracle":
                return self.oracle(level, actions)
            return self.act(level, actions)
        raise ValueError("Unknown operation or unexpected fields.")
