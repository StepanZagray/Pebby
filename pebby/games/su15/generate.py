"""Full nine-tier, reference-calibrated SU15 procedural generation.

Drafts contain fresh fruit, pursuer, target, and interaction assignments.  The
official levels contribute only aggregate per-tier measurements.  Acceptance
requires a settled-action constructive witness and a replay in the actual
native level context; no shortest-route claim is made.
"""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import random

from arcengine import GameState, Level, Sprite

from . import names
from .env import Env, UPSTREAM, upstream
from .quality import (
    GAMEPLAY_VERSION,
    GEOMETRY_VERSION,
    MECHANICS_VERSION,
    PROFILE_VERSION,
    REFERENCE_PROFILES,
    gameplay_identity,
    geometry_partition,
    profile_errors,
    raw_geometry_identity,
    structural_metrics,
)


SOURCE_SHA256 = "a5f91f7c963d6ca6447dae0ab21342b48a3f511601c40dfa8e972bdc59b4651e"
if hashlib.sha256(UPSTREAM.read_bytes()).hexdigest() != SOURCE_SHA256:
    raise RuntimeError("vendored SU15 source bytes differ from the calibrated source")

SOURCE_ID = "su15-1944f8ab"
FORMAT = "pebby.su15.full-level.v3"
GENERATOR_VERSION = 3
PROOF_VERSION = "su15-native-proof-v2"
WORK_VERSION = "su15-native-transition-work-v2"
DIFFICULTIES = tuple(range(1, 10))
SPLITS = ("train", "validation", "test")
MAX_ATTEMPTS = 72
DEFAULT_LIMIT = 50_000
MAX_COLLECTION = {"fruits": 32, "enemies": 16, "targets": 8, "requirements": 8}
MAX_SOLUTION_ACTIONS = 256
MAX_SEED = (1 << 63) - 1

FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "status": "ready",
    "source_id": SOURCE_ID,
    "mechanics_inventory_version": MECHANICS_VERSION,
    "quality_profile_version": PROFILE_VERSION,
    "curriculum": tuple(
        {"difficulty": difficulty, "context_index": difficulty - 1,
         "search_work": REFERENCE_PROFILES[difficulty]["search_work"]}
        for difficulty in DIFFICULTIES
    ),
    "evidence": {
        "official_tier_characterization": "su15.md#official-reference-characterization",
        "solution_mechanics": "solution_mechanics and witnessed zones recomputed from native action boundaries",
        "native_budget": "su15.reference_profiles exact shipped steps 32/48",
        "context_engine_replay": "proof.context_engine_verified and validator native replay",
        "novelty_split": "width-aware typed reflection identity, official semantic exclusion, and three-way partition",
        "bounded_rejections": "generate.last_report reason counters",
        "independent_closure": "docs/generator-evidence/external-astra-su15/su15-closure-detailed.md",
        "final_sequence_closure": "docs/generator-evidence/external-su15-sequence-fix/su15-detailed.md and root focused rerun: 2 passed in 8.92s",
        "root_primary_collector": "pre-final-validation hashes: train/validation/test all-nine WIN in 100/98/103 actions and 10.084 seconds; the final sequence patch changes no grammar, RNG, route, or rendering behavior",
        "root_native_frame_review": "nine generated native frames viewed by root before the final validation-only patch; that patch changes no rendering behavior",
        "root_acceptance": "su15-root-metadata-readiness-authorization-2026-09-19",
    },
    "caveats": (
        "Each tier has one shipped reference; tolerances are explicit engineering bands, not population estimates.",
        "Constructive witnesses are replay-certified but are not claimed shortest.",
        "Root readiness accepts the bounded evidence above; the full 72-attempt audit and official-teacher search were not repeated after the final validation-only patch.",
        "Tier 1 is necessarily a small tutorial class; its diversity is spatial rather than mechanic-compositional.",
        "The procedural grammar is finite and no graph-isomorphism or minimum-hardness claim is made.",
        "Tier 9 witnesses two of three installed target zones; installed zones and witnessed use are distinct, and no all-enemy indispensability claim is made.",
        "Planner work counts planner-controlled native transitions and excludes admission replay, validator replay, and full-game replay.",
    ),
}


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _position(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be [x, y]")
    return _integer(value[0], f"{label}[0]"), _integer(value[1], f"{label}[1]")


def _native_requirements(spec):
    values = []
    for index, requirement in enumerate(spec["requirements"]):
        kind = requirement.get("kind")
        tier = _integer(requirement.get("tier"), f"requirements[{index}].tier")
        count = _integer(requirement.get("count"), f"requirements[{index}].count")
        if count < 1:
            raise ValueError("requirement counts must be positive")
        if kind == "fruit" and 0 <= tier <= 8:
            key = tier
        elif kind == "enemy" and tier in names.ENEMY_REQUIREMENT_KEYS:
            key = names.ENEMY_REQUIREMENT_KEYS[tier]
        else:
            raise ValueError("unknown requirement kind/tier")
        values.append([key, count])
    return values[0] if len(values) == 1 else values


def _legend_sprite(module, pixels_source, x, y, name):
    prototype = module.sprites[pixels_source]
    return Sprite(
        pixels=prototype.pixels.copy(), name=name, visible=True,
        collidable=False, tags=[], layer=5,
    ).set_position(x, y)


def build_level(spec):
    """Rebuild one real native level from a JSON-compatible full spec."""
    if not isinstance(spec, dict) or spec.get("format") != FORMAT:
        raise ValueError(f"expected format {FORMAT!r}")
    difficulty = _integer(spec.get("difficulty"), "difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if _integer(spec.get("context_index"), "context_index") != difficulty - 1:
        raise ValueError("context_index must equal difficulty - 1")
    steps = _integer(spec.get("steps"), "steps")
    if not 1 <= steps <= 128:
        raise ValueError("steps must be in 1..128")

    module = upstream()
    sprites = [module.sprites[names.SPRITE_BACKGROUND].clone()]
    for index, value in enumerate(spec.get("fruits", ())):
        tier = _integer(value.get("tier"), f"fruits[{index}].tier")
        if tier not in range(9):
            raise ValueError("fruit tiers must be in 0..8")
        x, y = _position(value.get("position"), f"fruits[{index}].position")
        prototype = module.sprites[str(tier)]
        height, width = map(int, prototype.pixels.shape)
        if not (0 <= x <= 64 - width and names.PLAY_MIN_Y <= y <= 63 - height):
            raise ValueError("fruit does not fit the playable frame")
        sprites.append(prototype.clone().set_position(x, y))
    for index, value in enumerate(spec.get("enemies", ())):
        kind = _integer(value.get("kind"), f"enemies[{index}].kind")
        if kind not in names.ENEMY_SPRITES:
            raise ValueError("enemy kinds must be in 1..3")
        x, y = _position(value.get("position"), f"enemies[{index}].position")
        prototype = module.sprites[names.ENEMY_SPRITES[kind]]
        height, width = map(int, prototype.pixels.shape)
        if not (0 <= x <= 64 - width and names.PLAY_MIN_Y <= y <= 63 - height):
            raise ValueError("enemy does not fit the playable frame")
        sprites.append(prototype.clone().set_position(x, y))
    for index, value in enumerate(spec.get("targets", ())):
        x, y = _position(value.get("position"), f"targets[{index}].position")
        if not (0 <= x <= 55 and names.PLAY_MIN_Y <= y <= 54):
            raise ValueError("target does not fit the playable frame")
        sprites.append(module.sprites[names.SPRITE_TARGET].clone().set_position(x, y))
    if not spec.get("fruits") or not spec.get("targets"):
        raise ValueError("SU15 levels require fruit and target zones")

    # The colored fruit progression is an instructional rule cue in every
    # shipped post-tutorial board, not decorative art.  Late pursuer-building
    # tiers also show the native class progression.  Tag-free copies cannot
    # enter live object lists or solve the puzzle privately.
    if difficulty >= 2:
        sprites.append(_legend_sprite(
            module, names.SPRITE_FRUIT_PROGRESSION, 1, 1,
            "pebby_su15_fruit_progression",
        ).set_scale(2))
    if difficulty >= 8:
        sprites.append(_legend_sprite(
            module, names.SPRITE_ENEMY_PROGRESSION, 0, 5,
            "pebby_su15_enemy_progression",
        ))

    # Exact required types remain visible at the top.
    legend_x = 30
    for index, requirement in enumerate(spec["requirements"]):
        kind, tier, count = requirement["kind"], int(requirement["tier"]), int(requirement["count"])
        source = str(tier) if kind == "fruit" else names.ENEMY_SPRITES[tier]
        for occurrence in range(count):
            sprites.append(_legend_sprite(
                module, source, legend_x, 2,
                f"pebby_su15_legend_{index}_{occurrence}",
            ))
            legend_x += int(module.sprites[source].width) + 2
    if difficulty == 1:
        fruit_x, fruit_y = map(int, spec["fruits"][0]["position"])
        sprites.append(module.sprites[names.SPRITE_TUTORIAL_ARROW].clone()
                       .set_position(min(60, fruit_x + 5), max(10, fruit_y - 6)))
    else:
        # Su15.on_set_level indexes one tutorial-tagged sprite while the
        # temporary game initially enters logical index zero.
        sprites.append(module.sprites[names.SPRITE_TUTORIAL].clone().set_position(500, 500))

    descriptor = copy.deepcopy(spec)
    descriptor["kind"] = names.GENERATED_KIND
    return Level(
        sprites=sprites,
        grid_size=(64, 64),
        data={names.KEY_REQUIREMENTS: _native_requirements(spec), names.KEY_STEPS: steps,
              names.KEY_GENERATED: descriptor},
        name=f"generated-su15-full-d{difficulty}-s{spec.get('seed', 0)}",
    )


def _cluster(tier, count, anchor, pair_gap, group_gap, vertical_gap):
    x, y = anchor
    templates = {
        1: [(0, 0)], 2: [(0, 0), (pair_gap, 0)],
        3: [(0, 0), (pair_gap, 0), (group_gap, vertical_gap)],
        4: [(0, 0), (pair_gap, 0), (group_gap, 0), (group_gap + pair_gap, 0)],
        6: [(0, 0), (pair_gap, 0), (group_gap, 0), (group_gap + pair_gap, 0),
            (0, vertical_gap), (pair_gap, vertical_gap)],
        8: [(0, 0), (pair_gap, 0), (group_gap, 0), (group_gap + pair_gap, 0),
            (0, vertical_gap), (pair_gap, vertical_gap),
            (group_gap, vertical_gap), (group_gap + pair_gap, vertical_gap)],
    }
    return [{"tier": tier, "position": [x + dx, y + dy]}
            for dx, dy in templates[count]]


def _mirror_values(values):
    result = []
    for value in values:
        changed = copy.deepcopy(value)
        width = 9 if "tier" not in value and "kind" not in value else (
            value.get("tier", 0) + 1 if "tier" in value else 5)
        changed["position"][0] = 64 - width - int(value["position"][0])
        result.append(changed)
    return result


def _draft(rng, difficulty):
    profile = REFERENCE_PROFILES[difficulty]
    pair_gap = rng.choice((2, 3, 4))
    # Pair clusters must be more than two selection radii apart.  The engine
    # promotes an arbitrary connected group only one tier, so packing four
    # equal fruits together would silently destroy half their merge value.
    group_gap = rng.choice((20, 22, 24))
    vertical_gap = rng.choice((11, 13, 15))
    jitter_x, jitter_y = rng.randint(-2, 2), rng.randint(-2, 2)
    relative_x, relative_y = rng.choice((-2, 0, 2)), rng.choice((-2, 0, 2))

    if difficulty == 1:
        fruits = [{"tier": 2, "position": [5 + jitter_x, 54 + jitter_y]}]
    elif difficulty in (2, 4):
        fruits = _cluster(0, 8, (5 + jitter_x, 39 + jitter_y), pair_gap, group_gap, vertical_gap)
    elif difficulty == 3:
        fruits = (_cluster(0, 6, (4 + jitter_x, 37 + jitter_y), pair_gap, group_gap, vertical_gap)
                  + _cluster(1, 3, (35 + jitter_x, 16 + jitter_y), pair_gap + 1, 10, 11))
    elif difficulty == 5:
        fruits = (_cluster(0, 4, (33 + jitter_x, 56 + jitter_y), pair_gap, group_gap, vertical_gap)
                  + _cluster(1, 4, (5 + jitter_x, 37 + jitter_y), pair_gap + 1, group_gap, vertical_gap))
    elif difficulty == 6:
        fruits = [{"tier": 5, "position": [31 + jitter_x, 31 + jitter_y]}]
    elif difficulty == 7:
        fruits = ([{"tier": 1, "position": [40 + jitter_x, 44 + jitter_y]},
                   {"tier": 1, "position": [40 + pair_gap + jitter_x, 44 + jitter_y]},
                   {"tier": 1, "position": [40 + jitter_x, 44 + vertical_gap + jitter_y]},
                   {"tier": 1, "position": [40 + pair_gap + jitter_x,
                                               44 + vertical_gap + jitter_y]}]
                  + [{"tier": 5, "position": [31 + jitter_x, 29 + jitter_y]}])
    elif difficulty == 8:
        fruits = (_cluster(3, 2, (49 + jitter_x, 47 + jitter_y), pair_gap + 3, group_gap, vertical_gap)
                  + [{"tier": 5, "position": [31 + jitter_x, 29 + jitter_y]}])
    else:
        fruits = (_cluster(1, 2, (6 + jitter_x, 48 + jitter_y), pair_gap + 2, group_gap, vertical_gap)
                  + [{"tier": 5, "position": [31 + jitter_x, 29 + jitter_y]}])

    enemies = []
    if difficulty == 4:
        enemies = [{"kind": 1, "position": [55, 12 + rng.randint(0, 3)]}]
    elif difficulty == 5:
        enemies = [{"kind": 1, "position": [55, 12]}, {"kind": 1, "position": [46, 14]}]
    elif difficulty == 6:
        enemies = [{"kind": 1, "position": [14 + jitter_x + relative_x,
                                               max(10, 12 + jitter_y + relative_y)]}]
    elif difficulty == 7:
        enemies = [{"kind": 1, "position": [14 + jitter_x, 32 + jitter_y]},
                   {"kind": 1, "position": [56, 11]}]
    elif difficulty == 8:
        enemies = [{"kind": 1, "position": [14 + jitter_x, 32 + jitter_y]},
                   {"kind": 1, "position": [6, 50]}, {"kind": 1, "position": [9, 50]}]
    elif difficulty == 9:
        upper_y = max(10, 12 + jitter_y + relative_y)
        enemies = [{"kind": 1, "position": [6 + jitter_x, upper_y]},
                   {"kind": 1, "position": [9 + jitter_x, upper_y]},
                   {"kind": 1, "position": [48 + jitter_x, 34 + jitter_y]},
                   {"kind": 1, "position": [51 + jitter_x, 34 + jitter_y]}]

    target_templates = {
        1: [(46, 12)], 2: [(46, 12)], 3: [(3, 12), (49, 12)],
        4: [(46, 12)], 5: [(27, 12)], 6: [(5, 13), (49, 50)],
        7: [(50, 29), (40, 43)],
        8: [(50, 29), (49, 47), (4, 48), (4, 12)],
        9: [(17, 28), (27, 29), (5, 45)],
    }
    targets = [{"position": [x, y]} for x, y in target_templates[difficulty]]
    if difficulty == 1:
        targets[0]["position"][0] += relative_x
        targets[0]["position"][1] += relative_y
    elif difficulty == 6:
        for target in targets:
            target["position"][0] += relative_x // 2
            target["position"][1] += relative_y // 2
    requirements = [{"kind": kind, "tier": tier, "count": count}
                    for kind, tier, count in profile["requirements"]]
    mirrored = bool(rng.randrange(2))
    if mirrored:
        fruits, enemies, targets = map(_mirror_values, (fruits, enemies, targets))
    spec = {
        "format": FORMAT, "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION,
        "quality_profile_version": PROFILE_VERSION,
        "geometry_version": GEOMETRY_VERSION, "gameplay_version": GAMEPLAY_VERSION,
        "proof_version": PROOF_VERSION, "work_version": WORK_VERSION,
        "difficulty": difficulty, "context_index": difficulty - 1,
        "reference_level": difficulty, "fruits": fruits, "enemies": enemies,
        "targets": targets, "requirements": requirements, "steps": profile["steps"],
        "source": "generated_only",
        "reference_calibration": "aggregate measurements only; no official geometry or route copied",
        "interaction_plan": {
            "pair_gap": pair_gap, "group_gap": group_gap, "vertical_gap": vertical_gap,
            "mirrored": mirrored,
            "fruit_merge_tree": difficulty in (2, 3, 4, 5, 7, 8, 9),
            "pursuer_degradation": difficulty in (6, 7, 8, 9),
            "pursuer_merge_tree": difficulty in (8, 9),
        },
    }
    metrics = structural_metrics(spec)
    spec.update(bbox_width=metrics["bbox_width"], bbox_height=metrics["bbox_height"])
    return spec


def _context_env(spec):
    context = int(spec["context_index"])
    level = build_level(spec)
    env = Env([level.clone() for _ in range(context + 1)])
    env.set_level(context)
    return env


def _initial_non_background_pixels(spec):
    frame = _context_env(spec).render()
    return sum(value not in (-1, 3, 4, 5) for row in frame for value in row)


_OFFICIAL_GAMEPLAY_IDENTITIES = None


def _official_gameplay_identities():
    """Canonical semantic starts, excluding every header/tutorial sprite."""
    global _OFFICIAL_GAMEPLAY_IDENTITIES
    if _OFFICIAL_GAMEPLAY_IDENTITIES is not None:
        return _OFFICIAL_GAMEPLAY_IDENTITIES
    env = Env()
    env.reset()
    reverse = {value: key for key, value in names.ENEMY_REQUIREMENT_KEYS.items()}
    identities = set()
    for index in range(len(DIFFICULTIES)):
        env.set_level(index)
        raw = env.game.dsqlbvwaj
        rows = raw if isinstance(raw[0], (list, tuple)) else [raw]
        requirements = []
        for key, count in rows:
            if key in reverse:
                requirements.append({"kind": "enemy", "tier": reverse[key], "count": int(count)})
            else:
                requirements.append({"kind": "fruit", "tier": int(key), "count": int(count)})
        semantic = {
            "fruits": [
                {"tier": int(env.game.kqywaxhmsb[s]), "position": [int(s.x), int(s.y)]}
                for s in env.fruits()
            ],
            "enemies": [
                {"kind": int(env.game.dfqhmningy(env.game.kcuphgwar[s])),
                 "position": [int(s.x), int(s.y)]}
                for s in env.enemies()
            ],
            "targets": [{"position": [int(s.x), int(s.y)]} for s in env.targets()],
            "requirements": requirements,
            "steps": env.native_steps_left,
        }
        identities.add(gameplay_identity(semantic))
    _OFFICIAL_GAMEPLAY_IDENTITIES = frozenset(identities)
    return _OFFICIAL_GAMEPLAY_IDENTITIES


def _typed_state(env):
    fruits = tuple(sorted((int(env.game.kqywaxhmsb[s]), int(s.x), int(s.y)) for s in env.fruits()))
    enemies = tuple(sorted((int(env.game.dfqhmningy(env.game.kcuphgwar[s])), int(s.x), int(s.y))
                           for s in env.enemies()))
    return fruits, enemies


def mechanic_trace(spec, actions):
    """Recompute event use and first completion from native transitions."""
    env = _context_env(spec)
    start_score = env.levels_completed
    trace = Counter()
    for key in ("fruit_merges", "fruit_degrades", "enemy_merges",
                "pursuer_motion_actions", "undo_actions",
                "unequal_collision_penalties"):
        trace[key] = 0
    first_win = None
    minimum_steps = env.native_steps_left
    witnessed_target_zones = 0
    original_complete = env.game.cbdhpcilgb
    required_types = {(kind, tier) for kind, tier, _count in
                      REFERENCE_PROFILES[int(spec["difficulty"])]["requirements"]}

    def traced_complete():
        nonlocal witnessed_target_zones
        completed = original_complete()
        if not completed:
            return False
        live = []
        for sprite in env.fruits():
            typed = ("fruit", int(env.game.kqywaxhmsb[sprite]))
            if typed in required_types:
                live.append(sprite)
        for sprite in env.enemies():
            typed = ("enemy", int(env.game.dfqhmningy(env.game.kcuphgwar[sprite])))
            if typed in required_types:
                live.append(sprite)
        witnessed_target_zones = max(witnessed_target_zones, sum(
            any(env.game.jnieciwfsv(*env.game.jdeyppambj(sprite), target)
                for sprite in live)
            for target in env.targets()
        ))
        return True

    env.game.cbdhpcilgb = traced_complete
    for index, action in enumerate(actions):
        if not isinstance(action, (list, tuple)) or len(action) != 3:
            raise ValueError(f"malformed action {index}")
        action_id, x, y = action
        if type(action_id) is not int or action_id not in env.available_actions:
            raise ValueError(f"invalid action id at {index}")
        before_fruits, before_enemies = _typed_state(env)
        before_steps = env.native_steps_left
        observation = env.perform(action_id, x, y)
        after_fruits, after_enemies = _typed_state(env)
        after_steps = env.native_steps_left
        before_fruit_mass = sum(1 << tier for tier, _, _ in before_fruits)
        after_fruit_mass = sum(1 << tier for tier, _, _ in after_fruits)
        before_enemy_mass = sum(1 << (kind - 1) for kind, _, _ in before_enemies)
        after_enemy_mass = sum(1 << (kind - 1) for kind, _, _ in after_enemies)
        if len(after_fruits) < len(before_fruits) and after_fruit_mass == before_fruit_mass:
            trace["fruit_merges"] += len(before_fruits) - len(after_fruits)
        if after_fruit_mass < before_fruit_mass:
            trace["fruit_degrades"] += 1
        if len(after_enemies) < len(before_enemies) and after_enemy_mass == before_enemy_mass:
            trace["enemy_merges"] += len(before_enemies) - len(after_enemies)
        if before_enemies != after_enemies:
            trace["pursuer_motion_actions"] += 1
        if action_id == names.ACTION_UNDO:
            trace["undo_actions"] += 1
        if before_steps - after_steps > 1:
            trace["unequal_collision_penalties"] += 1
        minimum_steps = min(minimum_steps, after_steps)
        if env.levels_completed > start_score or observation.state == GameState.WIN:
            first_win = index
            break
        if observation.state == GameState.GAME_OVER:
            break
    result = dict(trace)
    result.update(won=first_win is not None,
                  first_win_action=-1 if first_win is None else first_win,
                  installed_target_zones=len(spec["targets"]),
                  witnessed_target_zones=witnessed_target_zones,
                  minimum_native_steps=minimum_steps,
                  final_native_steps=env.native_steps_left)
    return result, env


def _verify(spec, limit):
    from .plan import search
    if profile_errors(spec, require_proof=False):
        return None, "profile_structure"
    semantic_identity = gameplay_identity(spec)
    if semantic_identity in _official_gameplay_identities():
        return None, "official_gameplay_copy"
    result = search(_context_env(spec), limit=limit)
    if result.actions is None:
        return None, "search_cutoff" if result.truncated else "constructive_no_witness"
    actions = [list(action) for action in result.actions]
    mechanics, _ = mechanic_trace(spec, actions)
    if not mechanics["won"] or mechanics["first_win_action"] != len(actions) - 1:
        return None, "native_first_win_replay"
    row = dict(spec)
    row.update(solution=actions, context_solution=copy.deepcopy(actions),
               solution_length=len(actions), solution_mechanics=mechanics,
               initial_non_background_pixels=_initial_non_background_pixels(spec),
               engine_verified=True, context_engine_verified=True,
               native_budget=spec["steps"], search_limit=limit,
               search_work_used=result.work, search_truncated=False,
               proof={"context_index": spec["context_index"],
                      "proof_version": PROOF_VERSION, "work_version": WORK_VERSION,
                      "context_engine_verified": True, "search_truncated": False,
                      "engine_verified": True, "search_limit": limit,
                      "search_work_used": result.work,
                      "witness_kind": "constructive-native-replay-not-claimed-optimal",
                      "first_win_action": mechanics["first_win_action"],
                      "levels_completed": 1})
    if profile_errors(row):
        return None, "profile_proof"
    geometry_hash, partition = geometry_partition(row)
    row.update(geometry_sha256=geometry_hash, geometry_d4_sha256=geometry_hash,
               raw_geometry_sha256=raw_geometry_identity(row), geometry_split=partition,
               gameplay_sha256=semantic_identity)
    return row, None


_BASE_SPEC_FIELDS = {
    "format", "generator_version", "mechanics_version", "quality_profile_version",
    "geometry_version", "gameplay_version", "proof_version", "work_version",
    "difficulty", "context_index", "reference_level", "fruits", "enemies",
    "targets", "requirements", "steps", "source", "reference_calibration",
    "interaction_plan", "bbox_width", "bbox_height", "seed", "generation_attempt",
    "split", "geometry_sha256", "geometry_d4_sha256", "raw_geometry_sha256",
    "geometry_split", "gameplay_sha256", "solution", "context_solution",
    "solution_length", "solution_mechanics", "initial_non_background_pixels",
    "engine_verified", "context_engine_verified", "native_budget", "search_limit",
    "search_work_used", "search_truncated", "proof", "generation_rejections",
}
_SEQUENCE_FIELDS = {
    "parent_game_seed", "game_position", "child_seed", "sequence_kind",
    "game_sequence_sha256",
}
_SMOKE_SEQUENCE_FIELDS = _SEQUENCE_FIELDS - {"game_sequence_sha256"}
_MECHANIC_FIELDS = {
    "fruit_merges", "fruit_degrades", "enemy_merges", "pursuer_motion_actions",
    "undo_actions", "unequal_collision_penalties", "won", "first_win_action",
    "installed_target_zones", "witnessed_target_zones", "minimum_native_steps",
    "final_native_steps",
}
_PROOF_FIELDS = {
    "context_index", "proof_version", "work_version", "context_engine_verified",
    "engine_verified", "search_truncated", "search_limit", "search_work_used",
    "witness_kind", "first_win_action", "levels_completed",
}


def _strict_int(value, low=None, high=None):
    return (type(value) is int and (low is None or value >= low)
            and (high is None or value <= high))


def _hex_digest(value):
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _parse_actions(value, label, errors):
    parsed = []
    if type(value) is not list or not 1 <= len(value) <= MAX_SOLUTION_ACTIONS:
        errors.append(f"{label} must be a nonempty capped list")
        return parsed
    for index, action in enumerate(value):
        if type(action) is not list or len(action) != 3:
            errors.append(f"{label}[{index}] must be a three-item JSON list")
            continue
        action_id, x, y = action
        if not _strict_int(action_id) or action_id not in names.AVAILABLE_ACTIONS:
            errors.append(f"{label}[{index}] has an invalid action id")
        elif action_id == names.ACTION_CLICK:
            if not (_strict_int(x, 0, 63) and _strict_int(y, 0, 63)):
                errors.append(f"{label}[{index}] has invalid click coordinates")
            else:
                parsed.append((action_id, x, y))
        elif x is not None or y is not None:
            errors.append(f"{label}[{index}] undo coordinates must be null")
        else:
            parsed.append((action_id, x, y))
    return parsed


def _schema_errors(spec, curriculum_entry):
    """Validate every public shape before native construction or arithmetic."""
    errors = []
    if type(spec) is not dict or type(curriculum_entry) is not dict:
        return ["spec and curriculum entry must be plain mappings"]
    if set(curriculum_entry) != {"difficulty", "context_index", "search_work"}:
        errors.append("curriculum entry has an incomplete or unexpected schema")
        return errors
    difficulty = spec.get("difficulty")
    if not _strict_int(difficulty, 1, len(DIFFICULTIES)):
        return ["difficulty must be an integer in 1..9"]
    canonical_entry = FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
    if any(not _strict_int(curriculum_entry.get(field)) for field in canonical_entry):
        errors.append("curriculum fields must be integers, not bools or floats")
    elif curriculum_entry != canonical_entry:
        errors.append("curriculum entry does not match the canonical tier")

    fields = set(spec)
    if not _BASE_SPEC_FIELDS <= fields:
        errors.append(f"missing required fields: {sorted(_BASE_SPEC_FIELDS - fields)}")
    if fields - (_BASE_SPEC_FIELDS | _SEQUENCE_FIELDS):
        errors.append(f"unexpected fields: {sorted(fields - (_BASE_SPEC_FIELDS | _SEQUENCE_FIELDS))}")

    fixed = {
        "format": FORMAT, "generator_version": GENERATOR_VERSION,
        "mechanics_version": MECHANICS_VERSION, "quality_profile_version": PROFILE_VERSION,
        "geometry_version": GEOMETRY_VERSION, "gameplay_version": GAMEPLAY_VERSION,
        "proof_version": PROOF_VERSION, "work_version": WORK_VERSION,
        "context_index": difficulty - 1, "reference_level": difficulty,
        "source": "generated_only",
        "reference_calibration": "aggregate measurements only; no official geometry or route copied",
    }
    for field, expected in fixed.items():
        value = spec.get(field)
        if type(value) is not type(expected) or value != expected:
            errors.append(f"{field} mismatch")
    if not _strict_int(spec.get("steps"), 1, 128):
        errors.append("steps must be an integer in 1..128")
    if not _strict_int(spec.get("seed"), 0, MAX_SEED):
        errors.append("seed is outside the supported integer domain")
    if not _strict_int(spec.get("generation_attempt"), 1, MAX_ATTEMPTS):
        errors.append("generation_attempt is outside the bounded attempt domain")
    split = spec.get("split")
    if type(split) is not str or split not in SPLITS:
        errors.append("split is invalid")
    if type(spec.get("geometry_split")) is not str or spec.get("geometry_split") != split:
        errors.append("split/geometry partition mirror mismatch")
    for field in ("geometry_sha256", "geometry_d4_sha256", "raw_geometry_sha256",
                  "gameplay_sha256"):
        if not _hex_digest(spec.get(field)):
            errors.append(f"{field} must be a lowercase SHA-256 digest")
    if spec.get("geometry_sha256") != spec.get("geometry_d4_sha256"):
        errors.append("geometry digest mirrors disagree")
    if not _strict_int(spec.get("bbox_width"), 1, 64):
        errors.append("bbox_width is invalid")
    if not _strict_int(spec.get("bbox_height"), 1, 64):
        errors.append("bbox_height is invalid")
    if not _strict_int(spec.get("initial_non_background_pixels"), 0, 4096):
        errors.append("initial_non_background_pixels is invalid")

    collection_shapes = {
        "fruits": {"tier", "position"},
        "enemies": {"kind", "position"},
        "targets": {"position"},
        "requirements": {"kind", "tier", "count"},
    }
    for field, expected_keys in collection_shapes.items():
        values = spec.get(field)
        minimum = 0 if field == "enemies" else 1
        if type(values) is not list or not minimum <= len(values) <= MAX_COLLECTION[field]:
            errors.append(f"{field} must be a capped list with at least {minimum} entries")
            continue
        for index, value in enumerate(values):
            if type(value) is not dict or set(value) != expected_keys:
                errors.append(f"{field}[{index}] has an invalid object schema")
                continue
            if field == "requirements":
                kind, tier, count = value["kind"], value["tier"], value["count"]
                if type(kind) is not str or kind not in ("fruit", "enemy"):
                    errors.append(f"requirements[{index}].kind is invalid")
                if not _strict_int(tier) or not ((kind == "fruit" and 0 <= tier <= 8)
                                                  or (kind == "enemy" and 1 <= tier <= 3)):
                    errors.append(f"requirements[{index}].tier is invalid")
                if not _strict_int(count, 1, 8):
                    errors.append(f"requirements[{index}].count is invalid")
                continue
            type_field = "tier" if field == "fruits" else "kind" if field == "enemies" else None
            if type_field is not None:
                low, high = (0, 8) if type_field == "tier" else (1, 3)
                if not _strict_int(value[type_field], low, high):
                    errors.append(f"{field}[{index}].{type_field} is invalid")
                    continue
            position = value["position"]
            if (type(position) is not list or len(position) != 2
                    or not all(_strict_int(coordinate, 0, 63) for coordinate in position)):
                errors.append(f"{field}[{index}].position is invalid")
                continue
            x, y = position
            width = ((value.get("tier", 8) + 1) if field == "fruits"
                     else 5 if field == "enemies" else 9)
            height = width if field != "enemies" else 4
            if x + width > 64 or y < names.PLAY_MIN_Y or y + height > 64:
                errors.append(f"{field}[{index}] does not fit the playable frame")
    requirements = spec.get("requirements")
    if type(requirements) is list and all(
            type(value) is dict and set(value) == collection_shapes["requirements"]
            and value.get("kind") in ("fruit", "enemy")
            and _strict_int(value.get("tier")) and _strict_int(value.get("count"), 1, 8)
            for value in requirements):
        legend_width = sum(
            ((value["tier"] + 1) if value["kind"] == "fruit" else 5) * value["count"]
            + 2 * value["count"]
            for value in requirements
        )
        if legend_width > 34:
            errors.append("requirement legend exceeds its capped header extent")

    plan = spec.get("interaction_plan")
    plan_fields = {"pair_gap", "group_gap", "vertical_gap", "mirrored",
                   "fruit_merge_tree", "pursuer_degradation", "pursuer_merge_tree"}
    if type(plan) is not dict or set(plan) != plan_fields:
        errors.append("interaction_plan has an invalid schema")
    else:
        if not _strict_int(plan["pair_gap"], 2, 4):
            errors.append("interaction_plan.pair_gap is invalid")
        if not _strict_int(plan["group_gap"], 20, 24):
            errors.append("interaction_plan.group_gap is invalid")
        if not _strict_int(plan["vertical_gap"], 11, 15):
            errors.append("interaction_plan.vertical_gap is invalid")
        if type(plan["mirrored"]) is not bool:
            errors.append("interaction_plan.mirrored must be boolean")
        expected_flags = {
            "fruit_merge_tree": difficulty in (2, 3, 4, 5, 7, 8, 9),
            "pursuer_degradation": difficulty in (6, 7, 8, 9),
            "pursuer_merge_tree": difficulty in (8, 9),
        }
        for field, expected in expected_flags.items():
            if type(plan[field]) is not bool or plan[field] is not expected:
                errors.append(f"interaction_plan.{field} mismatch")

    solution = _parse_actions(spec.get("solution"), "solution", errors)
    context_solution = _parse_actions(spec.get("context_solution"), "context_solution", errors)
    if spec.get("context_solution") != spec.get("solution") or context_solution != solution:
        errors.append("context_solution must exactly mirror solution")
    if not _strict_int(spec.get("solution_length"), 1, MAX_SOLUTION_ACTIONS):
        errors.append("solution_length is invalid")
    elif spec["solution_length"] != len(solution):
        errors.append("solution_length does not match the parsed route")

    mechanics = spec.get("solution_mechanics")
    if type(mechanics) is not dict or set(mechanics) != _MECHANIC_FIELDS:
        errors.append("solution_mechanics has an invalid complete schema")
    else:
        for field in _MECHANIC_FIELDS - {"won", "first_win_action"}:
            if not _strict_int(mechanics[field], 0, MAX_SOLUTION_ACTIONS):
                errors.append(f"solution_mechanics.{field} is invalid")
        if mechanics["won"] is not True:
            errors.append("solution_mechanics.won must be true")
        if not _strict_int(mechanics["first_win_action"], 0, MAX_SOLUTION_ACTIONS - 1):
            errors.append("solution_mechanics.first_win_action is invalid")

    if spec.get("engine_verified") is not True or spec.get("context_engine_verified") is not True:
        errors.append("engine verification flags must be true")
    if spec.get("search_truncated") is not False:
        errors.append("search_truncated must be false")
    if not _strict_int(spec.get("native_budget"), 1, 128) or spec.get("native_budget") != spec.get("steps"):
        errors.append("native_budget must exactly mirror steps")
    approved_cap = canonical_entry["search_work"]
    if not _strict_int(spec.get("search_limit"), 1, approved_cap):
        errors.append("search_limit is outside the approved cap")
    if (not _strict_int(spec.get("search_work_used"), 1, approved_cap)
            or (_strict_int(spec.get("search_limit"), 1, approved_cap)
                and spec.get("search_work_used") > spec.get("search_limit"))):
        errors.append("search_work_used is outside the declared cap")

    proof = spec.get("proof")
    if type(proof) is not dict or not _PROOF_FIELDS <= set(proof) or set(proof) - (_PROOF_FIELDS | {"full_game_replay"}):
        errors.append("proof has an invalid complete schema")
    else:
        proof_fixed = {
            "context_index": difficulty - 1, "proof_version": PROOF_VERSION,
            "work_version": WORK_VERSION, "context_engine_verified": True,
            "engine_verified": True, "search_truncated": False,
            "search_limit": spec.get("search_limit"),
            "search_work_used": spec.get("search_work_used"),
            "witness_kind": "constructive-native-replay-not-claimed-optimal",
            "first_win_action": (spec.get("solution_length") - 1
                                 if _strict_int(spec.get("solution_length")) else None),
            "levels_completed": 1,
        }
        for field, expected in proof_fixed.items():
            if type(proof.get(field)) is not type(expected) or proof.get(field) != expected:
                errors.append(f"proof.{field} mismatch")
        replay_row = proof.get("full_game_replay")
        if replay_row is not None:
            if (type(replay_row) is not dict
                    or set(replay_row) != {"context_index", "levels_completed", "state"}
                    or not _strict_int(replay_row.get("context_index"), 0, 8)
                    or replay_row.get("context_index") != difficulty - 1
                    or not _strict_int(replay_row.get("levels_completed"), 1, 9)
                    or replay_row.get("levels_completed") != difficulty
                    or type(replay_row.get("state")) is not str
                    or replay_row.get("state") not in {"NOT_FINISHED", "WIN"}):
                errors.append("proof.full_game_replay is invalid")

    rejections = spec.get("generation_rejections")
    if type(rejections) is not dict:
        errors.append("generation_rejections must be a mapping")
    elif (any(type(reason) is not str or not reason or not _strict_int(count, 1, MAX_ATTEMPTS)
              for reason, count in rejections.items())
          or (_strict_int(spec.get("generation_attempt"), 1, MAX_ATTEMPTS)
              and sum(rejections.values()) != spec["generation_attempt"] - 1)):
        errors.append("generation_rejections does not mirror prior bounded attempts")

    sequence_fields = fields & _SEQUENCE_FIELDS
    proof_has_replay = type(proof) is dict and "full_game_replay" in proof
    if sequence_fields or proof_has_replay:
        parent_seed = spec.get("parent_game_seed")
        game_position = spec.get("game_position")
        child_seed = spec.get("child_seed")
        sequence_kind = spec.get("sequence_kind")
        if not _strict_int(parent_seed, 0, MAX_SEED):
            errors.append("parent_game_seed is invalid")
        if not _strict_int(game_position, 0, 8):
            errors.append("game_position is invalid")
        if not _strict_int(child_seed, 0, MAX_SEED) or child_seed != spec.get("seed"):
            errors.append("child_seed must exactly mirror seed")
        if type(sequence_kind) is not str or sequence_kind not in (
                "full-official-context", "explicit-smoke-subset"):
            errors.append("sequence_kind is invalid")

        if sequence_kind == "explicit-smoke-subset":
            if sequence_fields != _SMOKE_SEQUENCE_FIELDS or proof_has_replay:
                errors.append("smoke sequence enrichment must use exactly its reduced metadata")
        elif sequence_kind == "full-official-context":
            if sequence_fields != _SEQUENCE_FIELDS or not proof_has_replay:
                errors.append("full sequence enrichment and replay proof must be complete")
            if game_position != difficulty - 1:
                errors.append("full sequence position must match the official context")
            if not _hex_digest(spec.get("game_sequence_sha256")):
                errors.append("full sequence hash is missing or invalid")
            expected_replay = {
                "context_index": difficulty - 1,
                "levels_completed": difficulty,
                "state": "WIN" if difficulty == DIFFICULTIES[-1] else "NOT_FINISHED",
            }
            if type(proof) is dict and proof.get("full_game_replay") != expected_replay:
                errors.append("proof.full_game_replay does not match the tier sequence state")
        else:
            errors.append("sequence enrichment must be absent or use one complete supported mode")

        if (_strict_int(parent_seed, 0, MAX_SEED)
                and _strict_int(game_position, 0, 8)
                and _strict_int(child_seed, 0, MAX_SEED)
                and child_seed != _child_seed(parent_seed, game_position, difficulty)):
            errors.append("child_seed is not derived from parent, position, and difficulty")
    return errors


def validate_full_standard(spec, curriculum_entry):
    """Fail closed after a complete schema gate, then recompute all evidence."""
    errors = _schema_errors(spec, curriculum_entry)
    if errors:
        return errors
    difficulty = spec["difficulty"]
    split = spec["split"]
    proof = spec["proof"]
    parsed = [tuple(action) for action in spec["solution"]]
    recomputed = dict(spec)
    try:
        metrics = structural_metrics(spec)
        if spec["bbox_width"] != metrics["bbox_width"] or spec["bbox_height"] != metrics["bbox_height"]:
            errors.append("stored bounding box metrics mismatch")
        build_level(spec)
        measured_pixels = _initial_non_background_pixels(spec)
        if spec["initial_non_background_pixels"] != measured_pixels:
            errors.append("initial visual density mismatch")
        geometry_hash, partition = geometry_partition(spec)
        if spec["geometry_sha256"] != geometry_hash or spec["geometry_d4_sha256"] != geometry_hash:
            errors.append("canonical geometry identity mismatch")
        if spec["raw_geometry_sha256"] != raw_geometry_identity(spec):
            errors.append("raw geometry identity mismatch")
        if partition != split:
            errors.append("canonical geometry hashes to another split")
        semantic_identity = gameplay_identity(spec)
        if spec["gameplay_sha256"] != semantic_identity:
            errors.append("gameplay identity mismatch")
        if semantic_identity in _official_gameplay_identities():
            errors.append("official gameplay-equivalent start is forbidden")
        mechanics, _ = mechanic_trace(spec, parsed)
        recomputed["solution_mechanics"] = mechanics
        if mechanics != spec["solution_mechanics"]:
            errors.append("stored mechanic trace mismatch")
        if not mechanics["won"] or mechanics["first_win_action"] != len(parsed) - 1:
            errors.append("route does not first-win on its final action")
        if proof["first_win_action"] != mechanics["first_win_action"]:
            errors.append("proof first-win index differs from native replay")
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        errors.append(f"native evidence rebuild failed: {exc}")
    errors.extend(profile_errors(recomputed))
    return errors


def generate(seed, difficulty=1, attempts=MAX_ATTEMPTS, limit=None, *, split="train",
             max_attempts=None, search_limit=None, record_rejection=None):
    """Generate one deterministic, split-bound, replay-certified tier."""
    seed, difficulty = _integer(seed, "seed"), _integer(difficulty, "difficulty")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in 0..{MAX_SEED}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if max_attempts is not None:
        attempts = max_attempts
    if search_limit is not None:
        limit = search_limit
    attempts = _integer(attempts, "attempts")
    limit = REFERENCE_PROFILES[difficulty]["search_work"] if limit is None else _integer(limit, "limit")
    if not 1 <= attempts <= MAX_ATTEMPTS:
        raise ValueError(f"attempts must be in 1..{MAX_ATTEMPTS}")
    if not 0 < limit <= REFERENCE_PROFILES[difficulty]["search_work"]:
        raise ValueError("limit must be positive and no greater than the approved tier cap")
    rng, rejected = random.Random(f"{MECHANICS_VERSION}:{seed}:{difficulty}:{split}"), Counter()
    for attempt in range(1, attempts + 1):
        spec = _draft(rng, difficulty)
        geometry_hash, partition = geometry_partition(spec)
        if partition != split:
            reason = "geometry_split"
        else:
            spec.update(seed=seed, generation_attempt=attempt, split=split,
                        geometry_sha256=geometry_hash, geometry_d4_sha256=geometry_hash,
                        raw_geometry_sha256=raw_geometry_identity(spec), geometry_split=partition)
            accepted, reason = _verify(spec, limit)
            if accepted is not None:
                accepted["generation_rejections"] = dict(rejected)
                validation = validate_full_standard(
                    accepted, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1])
                if not validation:
                    generate.last_report = {"accepted": True, "seed": seed,
                                            "difficulty": difficulty, "split": split,
                                            "attempt": attempt, "rejections": dict(rejected),
                                            "search_work": accepted["search_work_used"]}
                    return accepted
                reason = "contract: " + "; ".join(validation)
        rejected[reason] += 1
        if record_rejection is not None:
            record_rejection({"seed": seed, "difficulty": difficulty, "split": split,
                              "attempt": attempt, "reason": reason})
    generate.last_report = {"accepted": False, "seed": seed, "difficulty": difficulty,
                            "split": split, "attempts": attempts, "rejections": dict(rejected)}
    return None


generate.last_report = None


def _child_seed(seed, ordinal, difficulty):
    payload = f"{SOURCE_ID}\0{seed}\0{ordinal}\0{difficulty}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _difficulty_sequence(difficulties):
    sequence = DIFFICULTIES if difficulties is None else tuple(difficulties)
    if not sequence or any(type(value) is not int or value not in DIFFICULTIES for value in sequence):
        raise ValueError("difficulties must be a nonempty sequence drawn from 1..9")
    if tuple(sorted(set(sequence))) != sequence:
        raise ValueError("difficulties must be unique and strictly increasing")
    return sequence


def _replay_full_game(specs):
    env = Env([build_level(spec) for spec in specs])
    env.reset()
    rows, observation = [], None
    for index, spec in enumerate(specs):
        if env.level_index != index or env.levels_completed != index:
            raise ValueError(f"native game did not enter tier {index + 1}")
        start_score = env.levels_completed
        for action_index, action in enumerate(spec["solution"]):
            observation = env.perform(*action)
            if env.levels_completed > start_score:
                if action_index != len(spec["solution"]) - 1:
                    raise ValueError(f"tier {index + 1} completes before the stored suffix ends")
                break
        if env.levels_completed != index + 1:
            raise ValueError(f"native game failed tier {index + 1}")
        rows.append({"context_index": index, "levels_completed": env.levels_completed,
                     "state": observation.state.name})
    if observation is None or observation.state != GameState.WIN:
        raise ValueError("complete nine-tier episode did not reach WIN")
    return rows


def _game_sequence_sha256(specs):
    """Bind a full game to its ordered, independently recomputed gameplay identities."""
    identities = [gameplay_identity(spec) for spec in specs]
    return hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()


def generate_game(seed, *, split="train", difficulties=None, attempts=MAX_ATTEMPTS, limit=None):
    """Generate the exact nine-tier game, or an explicit increasing smoke subset."""
    parent_seed, sequence = _integer(seed, "seed"), _difficulty_sequence(difficulties)
    if not 0 <= parent_seed <= MAX_SEED:
        raise ValueError(f"seed must be in 0..{MAX_SEED}")
    specs, reports = [], []
    for ordinal, difficulty in enumerate(sequence):
        child_seed = _child_seed(parent_seed, ordinal, difficulty)
        spec = generate(child_seed, difficulty, attempts=attempts, limit=limit, split=split)
        reports.append(copy.deepcopy(generate.last_report))
        if spec is None:
            generate_game.last_report = {"accepted": False, "parent_game_seed": parent_seed,
                                         "failed_position": ordinal,
                                         "failed_difficulty": difficulty,
                                         "tier_reports": reports}
            return None
        specs.append(dict(spec, parent_game_seed=parent_seed, game_position=ordinal,
                          child_seed=child_seed))
    if sequence != DIFFICULTIES:
        for spec in specs:
            spec["sequence_kind"] = "explicit-smoke-subset"
        generate_game.last_report = {"accepted": True, "smoke": True, "tier_reports": reports}
        return specs
    try:
        replay_rows = _replay_full_game(specs)
    except ValueError as exc:
        generate_game.last_report = {"accepted": False, "reason": str(exc), "tier_reports": reports}
        return None
    sequence_hash = _game_sequence_sha256(specs)
    for spec, replay_row in zip(specs, replay_rows):
        spec["sequence_kind"] = "full-official-context"
        spec["game_sequence_sha256"] = sequence_hash
        spec["proof"]["full_game_replay"] = replay_row
    generate_game.last_report = {"accepted": True, "parent_game_seed": parent_seed,
                                 "split": split, "tier_reports": reports}
    return specs


generate_game.last_report = None


def build_game(specs):
    """Validate and replay exactly nine independently generated tier specs."""
    if not isinstance(specs, (list, tuple)) or len(specs) != len(DIFFICULTIES):
        raise ValueError("full SU15 games require exactly nine specs")
    specs = list(specs)
    if tuple(spec.get("difficulty") for spec in specs) != DIFFICULTIES:
        raise ValueError("full SU15 game difficulties must be exactly 1..9")
    splits = {spec.get("split") for spec in specs}
    if len(splits) != 1 or next(iter(splits)) not in SPLITS:
        raise ValueError("all tiers must use one valid split")
    for index, (spec, entry) in enumerate(zip(specs, FULL_STANDARD_CONTRACT["curriculum"])):
        errors = validate_full_standard(spec, entry)
        if errors:
            raise ValueError(f"tier {index + 1} failed validation: {'; '.join(errors)}")

    sequence_kinds = [spec.get("sequence_kind") for spec in specs]
    if any(kind is not None for kind in sequence_kinds):
        if any(kind != "full-official-context" for kind in sequence_kinds):
            raise ValueError("full games must be wholly standalone or wholly full-sequence enriched")
        parents = {spec["parent_game_seed"] for spec in specs}
        if len(parents) != 1:
            raise ValueError("full sequence rows must share one parent seed")
        parent_seed = next(iter(parents))
        for index, spec in enumerate(specs):
            if spec["game_position"] != index:
                raise ValueError(f"tier {index + 1} has the wrong ordered game position")
            expected_child = _child_seed(parent_seed, index, spec["difficulty"])
            if spec["child_seed"] != expected_child or spec["seed"] != expected_child:
                raise ValueError(f"tier {index + 1} has invalid derived child provenance")
        expected_hash = _game_sequence_sha256(specs)
        if any(spec["game_sequence_sha256"] != expected_hash for spec in specs):
            raise ValueError("full sequence rows do not match the recomputed ordered hash")

    replay_rows = _replay_full_game(specs)
    if all(kind == "full-official-context" for kind in sequence_kinds):
        supplied_rows = [spec["proof"]["full_game_replay"] for spec in specs]
        if supplied_rows != replay_rows:
            raise ValueError("supplied full-game replay proof differs from native sequential replay")
    return [build_level(spec) for spec in specs]
