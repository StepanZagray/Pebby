"""Bounded LS20 v4 generation across the seven calibrated reference tiers.

The reference generator's historical seed ranges encode train/validation/test
splits. Shared collection supplies arbitrary 63-bit seeds, so this adapter maps
them deterministically into the explicitly requested native split and records
both seeds.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
from numbers import Integral

from pebby.ls20.generate import build_level as _build_level
from pebby.ls20.reference_generator_v2 import (
    GEOMETRY_VERSION,
    MECHANICS_VERSION,
    generate_level as _generate_v4,
    geometry_partition as _geometry_partition,
    seed_split as _native_seed_split,
)
from pebby.ls20.extended_curriculum import gameplay_hash as _gameplay_hash
from pebby.ls20.generation_quality import (
    geometry_d4_hash as _geometry_d4_hash,
    route_budget_slack as _route_budget_slack,
)
from pebby.ls20.reference_generator import patroller_contacts as _patroller_contacts
from pebby.ls20.reference_profiles import (
    DIFFICULTIES,
    DIFFICULTY_VERSION,
    PROFILES,
    profile_errors,
)


GENERATOR_VERSION = 4
# Raised from 50: over game seeds 0-5 the tier-2 child seeds succeeded 2/6 at
# 50 native draft attempts and 6/6 at 200 (~0.6-8 s each). Tier-3 children
# needed attempts 247, 263, 15, 64, 9 and 45 (~3 s per 200 rejected drafts),
# so a whole seven-tier game still relies on the shared collector's outer
# reseeding rather than this default alone.
DEFAULT_ATTEMPTS = 200
DEFAULT_NODE_LIMIT = 32_000_000
_TRAIN_SEED_COUNT = 6_000_000
_SPLIT_SEED_COUNTS = {"train": _TRAIN_SEED_COUNT, "validation": 1_000_000, "test": 1_000_000}

FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    # Accepted after the native bank audit plus shared split, enrichment,
    # validator, build_game, and real-engine whole-game replay checks.
    "status": "ready",
    "source_id": "ls20-9607627b",
    "mechanics_inventory_version": MECHANICS_VERSION,
    "quality_profile_version": DIFFICULTY_VERSION,
    "curriculum": [
        {
            "difficulty": difficulty,
            "context_index": difficulty - 1,
            "search_work": PROFILES[difficulty]["search_limit"],
        }
        for difficulty in DIFFICULTIES
    ],
    "evidence": {
        "official_tier_characterization": "official-seven-tier-characterization:5167ac2b",
        "solution_mechanics": "ls20-reference-profile-use-matrix-v1",
        "native_budget": "ls20-route-budget-slack-v1",
        "context_engine_replay": "ls20-adapter-native-replay:f00495649a5607b3",
        "novelty_split": GEOMETRY_VERSION,
        "bounded_rejections": "v4-generation-report:95253ed1",
        "official_copy_exclusion": "ls20-official-d4-geometry-exclusion-v1",
    },
    "caveats": [
        "launcher/refill participation uses tier floors rather than every installed device",
        "D4 identity does not hold out graph-isomorphic topology families",
        "the v4 bank replay audit reused stored proofs and did not rerun optimality searches",
    ],
}


def effective_seed(seed, split="train"):
    """Map a nonnegative shared seed into one declared native v4 split range."""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    original = int(seed)
    if split not in _SPLIT_SEED_COUNTS:
        raise ValueError("split must be train, validation, or test")
    # Native v4 interprets every seed >= 8,000,000 as validation. Shared
    # collection produces 63-bit seeds, so passing them through would silently
    # mislabel almost every training episode. Compress into all six million
    # native train seeds: [0, 1m) plus [3m, 8m).
    if split == "train":
        slot = original % _TRAIN_SEED_COUNT
        mapped = slot if slot < 1_000_000 else slot + 2_000_000
    elif split == "validation":
        mapped = 1_000_000 + original % _SPLIT_SEED_COUNTS[split]
    else:
        mapped = 2_000_000 + original % _SPLIT_SEED_COUNTS[split]
    return mapped, split


def build_level(spec):
    """Build a real ARCEngine level with the existing audited LS20 builder."""
    return _build_level(spec)


_official_geometry_cache = None


def _official_geometry_hashes():
    """Canonical D4 geometry hashes of the seven shipped LS20 levels.

    The shipped levels live in the same 12x12 spec space as generated rows and
    their free-cell counts fall inside the calibrated tier bands, so a redrawn
    copy would pass the structural profile. Only the wall geometry is needed:
    ``gameplay_sha256`` includes the raw walls, so any gameplay-level copy of a
    shipped level necessarily also matches its D4 geometry hash, and this
    single check dominates a separate gameplay comparison.
    """
    global _official_geometry_cache
    if _official_geometry_cache is None:
        from pebby.ls20.layout import extract

        from .env import Env, official_levels

        env, hashes = Env(), set()
        for index in range(len(official_levels())):
            env.set_level(index)
            hashes.add(_geometry_d4_hash({"walls": sorted(extract(env).walls)}))
        _official_geometry_cache = frozenset(hashes)
    return _official_geometry_cache


def _game_level_seed(game_seed, level_index, difficulty):
    if isinstance(game_seed, bool) or not isinstance(game_seed, Integral) or game_seed < 0:
        raise ValueError("game seed must be a nonnegative integer")
    material = (
        f"ls20-9607627b:{int(game_seed)}:{int(level_index)}:{int(difficulty)}"
    ).encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big") & (
        (1 << 63) - 1
    )


def generate(
    seed,
    difficulty,
    attempts=DEFAULT_ATTEMPTS,
    node_limit=DEFAULT_NODE_LIMIT,
    *,
    split=None,
):
    """Return a JSON-ready, exact-planned and engine-replayed level or ``None``."""
    if (
        isinstance(difficulty, bool)
        or not isinstance(difficulty, Integral)
        or difficulty not in DIFFICULTIES
    ):
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if isinstance(attempts, bool) or not isinstance(attempts, Integral) or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    if isinstance(node_limit, bool) or not isinstance(node_limit, Integral) or node_limit < 1:
        raise ValueError("node_limit must be a positive integer")

    difficulty = int(difficulty)
    attempts = int(attempts)
    node_limit = int(node_limit)
    requested_seed = int(seed)
    requested_split = "train" if split is None else split
    mapped_seed, effective_split = effective_seed(seed, requested_split)
    tier_search_limit = PROFILES[int(difficulty)]["search_limit"]
    try:
        spec = _generate_v4(
            mapped_seed,
            difficulty,
            attempts=attempts,
            search_limit=min(node_limit, tier_search_limit),
            split=effective_split,
        )
    except RuntimeError as exc:
        # Exhausting the explicitly bounded draft attempts is a normal rejected
        # seed for the shared outer generation loop. Engine/planner contract
        # failures use a different exception and deliberately remain visible.
        if str(exc).startswith("no v4 reference level"):
            return None
        raise

    # The native v4 loop is deterministic per (seed, difficulty) and returns its
    # first accepted draft, so a shipped-geometry collision cannot be redrawn
    # here without editing the frozen generator: reject the seed as a normal
    # bounded rejection in every split and let the outer loop reseed.
    if _geometry_d4_hash(spec) in _official_geometry_hashes():
        return None

    native_solution = [int(action) for action in spec["solution"]]
    action_triples = [[action, None, None] for action in native_solution]
    spec.update(
        requested_seed=requested_seed,
        original_seed=requested_seed,
        effective_seed=mapped_seed,
        effective_split=effective_split,
        native_difficulty=difficulty,
        official_copy=False,
        native_solution=native_solution,
        solution=action_triples,
        context_solution=action_triples,
        solution_length=len(action_triples),
        coverage={
            "difficulty_profile": difficulty,
            "generated_only": True,
            "full_reference_profile": True,
            "goal_count": len(spec["goals"]),
            "launcher_count": len(spec.get("launchers", ())),
            "refill_count": len(spec.get("refills", ())),
            "moving_rail_count": len(spec.get("rails", ())),
        },
        limitations=(
            "all seven calibrated tiers and shared split/whole-game adapter checks are accepted; "
            "launcher/refill floors and D4-only novelty remain declared limits; finite split ranges make the "
            "63-bit seed map many-to-one, so dataset audits must deduplicate effective seed "
            "and geometry/gameplay identities"
        ),
        seed_mapping=(
            "requested_seed modulo 6,000,000 selects across native v4 train ranges "
            "[0,1,000,000) and [3,000,000,8,000,000)"
        ),
    )
    return spec


def generate_game(
    seed,
    *,
    split="train",
    difficulties=None,
    attempts=DEFAULT_ATTEMPTS,
    node_limit=DEFAULT_NODE_LIMIT,
):
    """Generate an ordered set of specs with stable per-tier child seeds.

    Omitting ``difficulties`` is the full seven-level mode. A reduced sequence
    can be useful for inspecting individual tiers, but :func:`build_game`
    deliberately rejects it because LS20's proof context uses native indices.
    """
    selected = DIFFICULTIES if difficulties is None else tuple(difficulties)
    if not selected:
        raise ValueError("a generated game needs at least one difficulty")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in selected
    ):
        raise ValueError("game difficulties must be integers")
    selected = tuple(int(value) for value in selected)
    if any(value not in DIFFICULTIES for value in selected):
        raise ValueError(f"game difficulties must be drawn from {DIFFICULTIES}")
    if selected != tuple(sorted(set(selected))):
        raise ValueError("game difficulties must be strictly increasing and distinct")
    specs = []
    for level_index, difficulty in enumerate(selected):
        child_seed = _game_level_seed(seed, level_index, difficulty)
        spec = generate(
            child_seed,
            difficulty,
            attempts=attempts,
            node_limit=node_limit,
            split=split,
        )
        if spec is None:
            return None
        spec.update(game_seed=int(seed), game_level_index=level_index)
        specs.append(spec)
    return specs


def build_game(specs):
    """Build the complete native seven-level sequence without index shifting."""
    if (
        not isinstance(specs, Sequence)
        or isinstance(specs, (str, bytes))
        or len(specs) != len(DIFFICULTIES)
    ):
        raise ValueError(
            f"LS20 build_game requires exactly {len(DIFFICULTIES)} ordered specs"
        )
    split = None
    levels = []
    for level_index, (spec, difficulty) in enumerate(zip(specs, DIFFICULTIES)):
        if not isinstance(spec, Mapping):
            raise ValueError(f"spec {level_index} must be an object")
        if spec.get("difficulty") != difficulty:
            raise ValueError("game specs must use difficulties 1..7 in order")
        for key in ("context_index", "training_context_index", "verification_level_index"):
            if spec.get(key) != level_index:
                raise ValueError(f"spec {level_index} has a shifted native {key}")
        if split is None:
            split = spec.get("split")
            if split not in _SPLIT_SEED_COUNTS:
                raise ValueError("game specs must declare a native split")
        elif spec.get("split") != split:
            raise ValueError("game specs must all use the same split")
        entry = FULL_STANDARD_CONTRACT["curriculum"][level_index]
        errors = validate_full_standard(spec, entry)
        if errors:
            raise ValueError(
                f"spec {level_index} does not satisfy the full contract: "
                + "; ".join(errors)
            )
        levels.append(_build_level(spec))
    return levels


def _recompute_route_certificate(spec, native_solution, context_index):
    """Replay a stored route and recompute its cheap mechanic certificate."""
    from pebby.ls20 import names
    from pebby.ls20.env import Ls20Scenario
    from pebby.ls20.layout import extract
    from pebby.ls20.plan import simulate

    env = Ls20Scenario(_build_level(spec), context_index)
    layout = extract(env)
    if not layout.exact:
        raise ValueError("generated layout is not exactly represented by the planner")
    refills = tuple(sorted(layout.refills))
    state = (
        layout.start_cell,
        *layout.start_triple,
        0,
        0,
        layout.max_steps,
        0,
    )
    minimum_slack = state[6] // layout.step_cost
    used = Counter()
    moving_contacts = set()
    launcher_contacts = set()
    result = None
    final_outcome = None
    for action in native_solution:
        before = state
        action_index = names.ACTION_IDS.index(action)
        state, outcome = simulate(layout, state, action_index, refills)
        final_outcome = outcome
        minimum_slack = min(
            minimum_slack,
            _route_budget_slack(
                layout, before, state, action=action_index, outcome=outcome,
            ),
        )
        moving_contacts.update(
            _patroller_contacts(layout, before, state, action_index, outcome)
        )
        dx, dy = names.ACTION_DELTAS[action_index]
        target = (before[0][0] + dx, before[0][1] + dy)
        if outcome == "launched":
            entry = target if layout.free(target) else before[0]
            for number, pad in enumerate(layout.launchers):
                if entry in pad["triggers"] and pad["distance"] > 0:
                    launcher_contacts.add(number)
                    break
        used[outcome] += 1
        used["refills_consumed"] += (state[5] ^ before[5]).bit_count()
        used["goals_cleared"] += (state[4] ^ before[4]).bit_count()
        result = env.perform(action)
    if (
        final_outcome != "won"
        or result is None
        or not result.won
        or env.lives() != 3
        or env.levels_completed != 1
    ):
        raise ValueError("stored route does not produce a one-level WIN with three lives")
    used.update(
        used_patroller_count=len(moving_contacts),
        used_launcher_count=len(launcher_contacts),
    )
    mechanics = dict(used)
    mechanics.update(
        won=True,
        moving_cycler=bool(moving_contacts),
        launcher=bool(launcher_contacts),
        refill=bool(used["refills_consumed"]),
    )
    return {
        "solution_mechanics": mechanics,
        "minimum_slack_moves": minimum_slack,
        "slack_moves": state[6] // layout.step_cost,
    }


def validate_full_standard(spec, curriculum_entry):
    """Validate one adapter row against its complete calibrated tier certificate."""
    errors = []
    if not isinstance(spec, Mapping):
        return ["generated spec must be an object"]
    if not isinstance(curriculum_entry, dict):
        return ["curriculum entry must be an object"]
    difficulty = curriculum_entry.get("difficulty")
    if isinstance(difficulty, bool) or not isinstance(difficulty, Integral) or difficulty not in DIFFICULTIES:
        return ["curriculum difficulty must be in 1..7"]
    difficulty = int(difficulty)
    expected_context = difficulty - 1
    expected_search = PROFILES[difficulty]["search_limit"]
    if curriculum_entry.get("context_index") != expected_context:
        errors.append("curriculum context does not match tier-1")
    if curriculum_entry.get("search_work") != expected_search:
        errors.append("curriculum search work differs from calibrated tier cap")
    try:
        errors.extend(profile_errors(spec))
    except (AttributeError, KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"profile validation failed on malformed spec: {exc}")
    expected_versions = {
        "generator_version": GENERATOR_VERSION,
        "quality_version": 3,
        "mechanics_version": MECHANICS_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "difficulty_version": DIFFICULTY_VERSION,
    }
    for key, expected in expected_versions.items():
        if spec.get(key) != expected:
            errors.append(f"{key} differs from the accepted full-standard version")
    if type(spec.get("difficulty")) is not int or spec.get("difficulty") != difficulty:
        errors.append("spec difficulty differs from curriculum")
    for key in ("context_index", "training_context_index", "verification_level_index"):
        if spec.get(key) != expected_context:
            errors.append(f"{key} differs from the calibrated context")
    for key, expected in (
        ("context_engine_verified", True),
        ("engine_verified", True),
        ("engine_win", True),
        ("search_truncated", False),
        ("verification_lives", 3),
        ("replay_lives", 3),
        ("levels_completed", 1),
        ("search_limit", expected_search),
        ("source", "generated_only"),
    ):
        if spec.get(key) != expected:
            errors.append(f"{key} is missing or inconsistent")
    if spec.get("official_copy") is not False:
        errors.append("official_copy must be recorded as False by the adapter's shipped-level check")
    native_solution = spec.get("native_solution")
    solution = spec.get("solution")
    expected_solution = (
        [[action, None, None] for action in native_solution]
        if isinstance(native_solution, list)
        and all(type(action) is int and action in (1, 2, 3, 4) for action in native_solution)
        else None
    )
    optimal = spec.get("optimal_actions")
    if expected_solution is None or not isinstance(solution, list):
        errors.append("native and shared solutions must be lists")
    elif solution != expected_solution or spec.get("context_solution") != expected_solution:
        errors.append("shared action triples do not exactly mirror the native route")
    elif not (
        optimal == spec.get("context_optimal_actions")
        == len(native_solution) == len(solution)
    ):
        errors.append("optimal action counts do not match stored solutions")
    proof = spec.get("proof")
    if not isinstance(proof, dict):
        errors.append("nested proof is missing")
    else:
        mirrors = {
            "seed": spec.get("effective_seed"),
            "difficulty": difficulty,
            "difficulty_version": DIFFICULTY_VERSION,
            "context_index": expected_context,
            "context_engine_verified": True,
            "search_truncated": False,
            "optimal_actions": optimal,
            "context_optimal_actions": optimal,
            "engine_win": True,
            "replay_lives": 3,
            "levels_completed": 1,
            "search_limit": expected_search,
            "reachable_states": spec.get("reachable_states"),
            "oracle_backend": spec.get("oracle_backend"),
            "generator_version": GENERATOR_VERSION,
            "mechanics_version": MECHANICS_VERSION,
            "split": spec.get("split"),
            "geometry_version": GEOMETRY_VERSION,
            "geometry_split": spec.get("geometry_split"),
        }
        for key, expected in mirrors.items():
            if proof.get(key) != expected:
                errors.append(f"proof.{key} does not mirror the top-level certificate")
    for key in ("geometry_sha256", "geometry_d4_sha256", "gameplay_sha256"):
        value = spec.get(key)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"{key} must be a SHA-256 identity")
    split = spec.get("split")
    requested_seed = spec.get("requested_seed")
    effective = spec.get("effective_seed")
    if split not in _SPLIT_SEED_COUNTS or spec.get("effective_split") != split:
        errors.append("effective split differs from the generated spec split")
    if (
        isinstance(requested_seed, bool)
        or not isinstance(requested_seed, Integral)
        or requested_seed < 0
        or isinstance(effective, bool)
        or not isinstance(effective, Integral)
        or effective < 0
    ):
        errors.append("requested/effective seeds must be nonnegative integers")
    elif split in _SPLIT_SEED_COUNTS:
        mapped, _ = effective_seed(requested_seed, split)
        if int(effective) != mapped or spec.get("seed") != mapped:
            errors.append("effective seed does not match the requested seed/split mapping")
        elif _native_seed_split(mapped) != split:
            errors.append("effective seed is outside the native split namespace")
    try:
        geometry, geometry_split = _geometry_partition(spec)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"geometry identity could not be recomputed: {exc}")
    else:
        if (
            spec.get("geometry_sha256") != geometry
            or spec.get("geometry_d4_sha256") != geometry
            or spec.get("geometry_split") != geometry_split
            or split != geometry_split
        ):
            errors.append("stored geometry identity/partition differs from recomputation")
        if geometry in _official_geometry_hashes():
            errors.append("official_copy: canonical D4 geometry matches a shipped LS20 level")
    try:
        gameplay = _gameplay_hash(spec)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"gameplay identity could not be recomputed: {exc}")
    else:
        if spec.get("gameplay_sha256") != gameplay:
            errors.append("stored gameplay identity differs from recomputation")
    exclusions = spec.get("generation_exclusions")
    if not isinstance(exclusions, dict) or any(
        not isinstance(key, str)
        or isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 0
        for key, value in (exclusions.items() if isinstance(exclusions, dict) else ())
    ):
        errors.append("bounded generation rejection counts are missing or malformed")
    if expected_solution is not None:
        try:
            route = _recompute_route_certificate(spec, native_solution, expected_context)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"stored route could not be replayed: {type(exc).__name__}: {exc}")
        else:
            for key, actual in route.items():
                if spec.get(key) != actual:
                    errors.append(f"{key} differs from recomputed route evidence")
    return errors


def replays_to_completion(spec, *, context_index=None):
    """Replay a stored solution at the context in which its proof was made."""
    from .env import Env

    index = spec.get("verification_level_index", spec["difficulty"] - 1)
    if context_index is not None:
        index = context_index
    level = build_level(spec)
    env = Env([level.clone() for _ in range(int(index) + 1)])
    env.set_level(int(index))
    observation = None
    for action in spec["solution"]:
        action_id = action[0] if isinstance(action, (list, tuple)) else action
        observation = env.perform(int(action_id), None, None)
    return bool(observation and observation.won and env.levels_completed == 1)
