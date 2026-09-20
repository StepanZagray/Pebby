"""Full six-tier, reference-calibrated procedural generator for TR87.

Every accepted row is constructed independently of official geometry/routes,
checked against its official-tier structural profile, and replayed through the
vendored game at the intended native level index. Stored witnesses are
constructive certificates; only proofs explicitly marked optimal make a
shortest-route claim.
"""

from collections.abc import Mapping
import hashlib
import random

from . import names
from .env import Env, official_levels, upstream
from .generation_quality import gameplay_identity, geometry_identity, geometry_partition
from .layout import extract, translate, translation_trace
from .plan import _route_offsets
from .reference_profiles import (
    DIFFICULTIES,
    DIFFICULTY_VERSION,
    MECHANICS_INVENTORY_VERSION,
    PROFILES,
    profile_errors,
    structural_metrics,
)


GENERATOR_VERSION = 3
FORMAT = "pebby.tr87.full-level.v3"
QUALITY_PROFILE_VERSION = "tr87-reference-quality-v3"
MAX_ATTEMPTS = 180
MODES = tuple(PROFILES[d]["mode"] for d in DIFFICULTIES)
DISPLAY_SIZE = 64
# The shipped levels leave a visible gutter. Generated content keeps at least
# two columns on either side so a complete 7x7 backing can never sit flush
# against the camera crop and read as clipped.
CONTENT_LEFT = 2
CONTENT_RIGHT = 62


FULL_STANDARD_CONTRACT = {
    "format": "pebby-full-generator-contract-v1",
    "status": "ready",
    "source_id": "tr87-cd924810",
    "mechanics_inventory_version": MECHANICS_INVENTORY_VERSION,
    "quality_profile_version": QUALITY_PROFILE_VERSION,
    "curriculum": [
        {"difficulty": d, "context_index": PROFILES[d]["context_index"],
         "search_work": PROFILES[d]["search_work"]}
        for d in DIFFICULTIES
    ],
    "evidence": {
        "official_tier_characterization": "docs/generator-evidence/tr87.md",
        "solution_mechanics": "spec.proof + spec.solution_mechanics",
        "native_budget": "spec.native_budget + context replay",
        "context_engine_replay": "spec.proof.context_engine_verified",
        "novelty_split": "spec.geometry_sha256/gameplay_sha256/geometry_split",
        "bounded_rejections": "spec.generation_exclusions",
    },
    "caveats": [
        "The six tiers are calibrated from one shipped reference level each.",
        "At tier 6 the upstream tree branch precedes and shadows the double-translation branch.",
        "Generated tiers 5 and 6 use constructive replay witnesses and make no shortest-route claim.",
    ],
}


def _symbols(family, rng, count):
    return [family + str(digit) for digit in rng.sample(range(1, 8), count)]


def _semantic_puzzle(rng, difficulty):
    """New rules and solved rows with the reference tier's grammar shape."""
    if difficulty == 1:
        heads, outputs = _symbols("A", rng, 6), _symbols("B", rng, 6)
        rules = [([a], [b]) for a, b in zip(heads, outputs)]
        order = rng.sample(range(6), 5)
        source = [heads[i] for i in order]
        target = [outputs[i] for i in order]
    elif difficulty == 2:
        heads = _symbols("B", rng, 6)
        arities = (1, 3, 2, 2, 3, 1)
        rules = [([head], ["C" + str(rng.randint(1, 7)) for _ in range(size)])
                 for head, size in zip(heads, arities)]
        order = (0, 4, 3, 5)
        source = [heads[i] for i in order]
        target = [symbol for i in order for symbol in rules[i][1]]
    elif difficulty == 3:
        heads = _symbols("C", rng, 6)
        arities = ((1, 1), (2, 2), (1, 2), (2, 1), (3, 1), (1, 1))
        rules = []
        for head, (left, right) in zip(heads, arities):
            lhs = [head] + [rng.choice(heads) for _ in range(left - 1)]
            rhs = ["A" + str(rng.randint(1, 7)) for _ in range(right)]
            rules.append((lhs, rhs))
        order = (0, 4, 2, 5, 1)
        source = [symbol for i in order for symbol in rules[i][0]]
        target = [symbol for i in order for symbol in rules[i][1]]
    elif difficulty == 4:
        outer = _symbols("A", rng, 4)
        middle = _symbols("B", rng, 4)
        outputs = _symbols("C", rng, 4)
        first = [([a], [b]) for a, b in zip(outer, middle)]
        second = [([b], [c]) for b, c in zip(middle, outputs)]
        rules = first + second
        order = list(range(4)) + [rng.randrange(4) for _ in range(3)]
        rng.shuffle(order)
        source = [outer[i] for i in order]
        target = [outputs[i] for i in order]
    elif difficulty == 5:
        heads = _symbols("A", rng, 4)
        tails = _symbols("A", rng, 4)
        arities = ((1, 1), (1, 2), (2, 1), (1, 1))
        rules = []
        for i, (left, right) in enumerate(arities):
            lhs = [heads[i]] + ([tails[i]] if left == 2 else [])
            rhs = ["B" + str(rng.randint(1, 7)) for _ in range(right)]
            rules.append((lhs, rhs))
        order = list(range(4))
        rng.shuffle(order)
        source = [symbol for i in order for symbol in rules[i][0]]
        target = [symbol for i in order for symbol in rules[i][1]]
    else:
        outer = _symbols("A", rng, 3)
        middle = _symbols("B", rng, 3)
        outputs = _symbols("C", rng, 3)
        # Tier 6's official winning grammar couples two identical children in
        # one tree expansion while retaining mixed-child expansions in the
        # same parse. Choose the repeated head/child procedurally, then ensure
        # the other two children occur in live mixed expansions so every
        # secondary rule participates in the winning native trace.
        repeated_head = rng.randrange(len(outer))
        repeated_child = rng.choice(middle)
        required_mixed_children = [child for child in middle if child != repeated_child]
        rng.shuffle(required_mixed_children)
        first = []
        for index, head in enumerate(outer):
            if index == repeated_head:
                children = [repeated_child, repeated_child]
            else:
                required = required_mixed_children.pop()
                companion = rng.choice([child for child in middle if child != required])
                children = [required, companion]
                rng.shuffle(children)
            first.append(([head], children))
        second = [([b], [c]) for b, c in zip(middle, outputs)]
        rules = [item for pair in zip(first, second) for item in pair]
        order = list(range(3))
        rng.shuffle(order)
        source = [outer[i] for i in order]
        mapping = dict(zip(middle, outputs))
        target = [mapping[symbol] for i in order for symbol in first[i][1]]
    return rules, source, target


def _rule_width(rule):
    return 7 * (len(rule[0]) + len(rule[1])) + 1


def _rule_ys(rng, difficulty):
    if difficulty <= 3:
        base, gap = rng.randint(4, 6), rng.randint(8, 9)
        return [base + i * gap for i in range(3)]
    if difficulty == 4:
        base = rng.randint(3, 5)
        return [base + i * 8 for i in range(4)]
    if difficulty == 5:
        base, gap = rng.randint(9, 11), rng.randint(11, 13)
        return [base, base + gap]
    base, gap = rng.randint(5, 7), rng.randint(11, 12)
    return [base + i * gap for i in range(3)]


def _place_rules(rng, rules, difficulty):
    """Pack exactly two rules per reference row with sampled new geometry."""
    rows = PROFILES[difficulty]["rule_rows"]
    for _ in range(64):
        ordered = list(rules)
        rng.shuffle(ordered)
        pairs = [ordered[i:i + 2] for i in range(0, len(ordered), 2)]
        if len(pairs) != rows:
            return None
        gaps = [rng.randint(7, 13) for _ in pairs]
        if all(_rule_width(a) + gap + _rule_width(b) <= CONTENT_RIGHT - CONTENT_LEFT - 2
               for (a, b), gap in zip(pairs, gaps)):
            break
    else:
        return None
    specs = []
    for y, pair, gap in zip(_rule_ys(rng, difficulty), pairs, gaps):
        first, second = pair
        total = _rule_width(first) + gap + _rule_width(second)
        # Backings begin one pixel before the tile anchor; the paired visual
        # extent ends one pixel after this packing width.
        x = rng.randint(CONTENT_LEFT + 1, CONTENT_RIGHT - total - 1)
        specs.append({"lhs": list(first[0]), "rhs": list(first[1]), "x": x, "y": y})
        x += _rule_width(first) + gap
        specs.append({"lhs": list(second[0]), "rhs": list(second[1]), "x": x, "y": y})
    return specs


def _draft(rng, difficulty):
    profile = PROFILES[difficulty]
    rules, source, target = _semantic_puzzle(rng, difficulty)
    # Shipped direct-translation levels do not store an already solved target;
    # their arbitrary same-family glyphs are then scrambled again on entry.
    # Sample that independent clean row so action difficulty is not collapsed
    # to the fixed engine scramble alone.
    if difficulty <= 4:
        target = [symbol[0] + str(rng.randint(1, 7)) for symbol in target]
    placed = _place_rules(rng, rules, difficulty)
    if placed is None:
        return None
    witness_rules = None
    pre_scramble_offsets = None
    if difficulty >= 5:
        witness_rules = [{"lhs": list(rule["lhs"]), "rhs": list(rule["rhs"])}
                         for rule in placed]
        pre_scramble_offsets = []
        for rule in placed:
            for side in ("lhs", "rhs"):
                offset = rng.randrange(names.SYMBOL_COUNT)
                pre_scramble_offsets.append(offset)
                rule[side] = [names.cycle(symbol, offset) for symbol in rule[side]]
    max_source_x = CONTENT_RIGHT - 6 - names.TILE_SPACING * (len(source) - 1)
    max_target_x = CONTENT_RIGHT - 6 - names.TILE_SPACING * (len(target) - 1)
    source_y = (41, 41, 41, 41, 44, 45)[difficulty - 1]
    target_y = (52, 52, 52, 52, 53, 54)[difficulty - 1]
    spec = {
        "format": FORMAT,
        "generator_version": GENERATOR_VERSION,
        "difficulty": difficulty,
        "difficulty_version": DIFFICULTY_VERSION,
        "quality_profile_version": QUALITY_PROFILE_VERSION,
        "reference_level": difficulty,
        "reference_actions": profile["reference_actions"],
        "reference_actions_optimal": profile["reference_actions_optimal"],
        "mode": profile["mode"],
        "mechanics": dict(profile["flags"]),
        "split_y": profile["split_y"],
        "rules": placed,
        "source": {"y": source_y, "x0": rng.randint(CONTENT_LEFT + 1, max_source_x), "symbols": source},
        "target": {"y": target_y, "x0": rng.randint(CONTENT_LEFT + 1, max_target_x), "symbols": target},
        "reference_calibration": "one shipped level per tier; explicit tolerance, not a confidence interval",
    }
    if witness_rules is not None:
        spec["witness_rules"] = witness_rules
        spec["pre_scramble_rule_offsets"] = pre_scramble_offsets
    spec.update(structural_metrics(spec))
    return None if profile_errors(spec, require_proof=False) else spec


def rule_tile_positions(rule):
    """[(symbol, x, y)] for one rule plus its connector-strip position."""
    tiles, x = [], rule["x"]
    for symbol in rule["lhs"]:
        tiles.append((symbol, x, rule["y"]))
        x += names.TILE_SPACING
    last_lhs_x = x - names.TILE_SPACING
    strip = (last_lhs_x + 2, rule["y"] + 2)
    x = last_lhs_x + 10
    for symbol in rule["rhs"]:
        tiles.append((symbol, x, rule["y"]))
        x += names.TILE_SPACING
    return tiles, strip


def build_level(spec):
    """Build a native ARCEngine level exclusively from upstream prototypes."""
    module = upstream()
    protos = module.sprites
    sprites = [protos[names.SPRITE_BACKGROUND].clone().set_position(0, spec["split_y"]).set_scale(32)]

    def add_tile(symbol, x, y):
        sprites.append(protos[names.SPRITE_BACKING + symbol[0]].clone().set_position(x - 1, y - 1))
        sprites.append(protos[names.tile_sprite_name(symbol)].clone().set_position(x, y))

    for rule in spec["rules"]:
        tiles, (strip_x, strip_y) = rule_tile_positions(rule)
        sprites.append(protos[names.SPRITE_STRIP].clone().set_position(strip_x, strip_y))
        for symbol, x, y in tiles:
            add_tile(symbol, x, y)
    for row in (spec["source"], spec["target"]):
        for i, symbol in enumerate(row["symbols"]):
            add_tile(symbol, row["x0"] + i * names.TILE_SPACING, row["y"])
    data = {key: True for key, enabled in spec["mechanics"].items() if enabled}
    if spec.get("witness_rules") is not None:
        data[names.KEY_TEACHER_RULES] = [[list(rule["lhs"]), list(rule["rhs"])]
                                         for rule in spec["witness_rules"]]
    return module.Level(sprites=sprites, grid_size=(64, 64), data=data)


def _render_extent_errors(spec):
    """Check complete native sprite rectangles, rather than anchor points."""
    errors = []
    for index, sprite in enumerate(build_level(spec).get_sprites()):
        if sprite.name == names.SPRITE_BACKGROUND:
            continue
        left, top = int(sprite.x), int(sprite.y)
        right = left + int(sprite.width)
        bottom = top + int(sprite.height)
        if (left < CONTENT_LEFT or right > CONTENT_RIGHT
                or top < 0 or bottom > DISPLAY_SIZE):
            errors.append(
                f"sprite {index} extent ({left},{top})-({right},{bottom}) "
                "is outside the generated render-safe bounds"
            )
    return errors


def env_for(spec, context_index=None):
    """Fresh environment with ``spec`` at its intended real engine index."""
    context = spec.get("difficulty", 1) - 1 if context_index is None else context_index
    if context not in range(len(names.BUDGET_BY_LEVEL_INDEX)):
        raise ValueError("context_index must be in 0..5")
    levels = [level.clone() for level in official_levels()[:context]] + [build_level(spec)]
    env = Env(levels)
    env.set_level(context)
    return env


def _uniform_delta(current, wanted):
    if len(current) != len(wanted):
        return None
    deltas = {(int(b[1]) - int(a[1])) % names.SYMBOL_COUNT for a, b in zip(current, wanted)
              if a[0] == b[0]}
    if len(deltas) != 1 or any(a[0] != b[0] for a, b in zip(current, wanted)):
        return None
    return deltas.pop()


def _constructive_actions(spec, layout, node_limit):
    if not layout.alter_rules:
        wanted = translate(layout)
        if wanted is None or len(wanted) != len(layout.target):
            return None, 0, False
        offsets = [_uniform_delta((have,), (goal,)) for have, goal in zip(layout.target, wanted)]
    else:
        current = [side for rule in layout.rules for side in rule]
        wanted = [tuple(rule[side]) for rule in spec["witness_rules"] for side in ("lhs", "rhs")]
        offsets = [_uniform_delta(have, goal) for have, goal in zip(current, wanted)]
    if any(delta is None for delta in offsets):
        return None, 0, False
    return _route_offsets(tuple(offsets), layout.cursor, node_limit=node_limit)


def _official_identity_sets():
    """Canonical clean official identities, derived without storing layouts."""
    geometry, gameplay = set(), set()
    for difficulty, level in enumerate(official_levels(), 1):
        split_y = level.get_sprites_by_name(names.SPRITE_BACKGROUND)[0].y
        tiles = level.get_sprites_by_tag(names.TAG_TILE)
        by_position = {(tile.x, tile.y): names.symbol(tile.name) for tile in tiles}
        rules = []
        for strip in sorted(level.get_sprites_by_name(names.SPRITE_STRIP), key=lambda s: (s.y, s.x)):
            y = strip.y - 2
            containing = [tile for tile in tiles
                          if tile.x <= strip.x < tile.x + tile.width
                          and tile.y <= strip.y < tile.y + tile.height]
            if not containing:
                raise ValueError("official connector overlaps no left-hand tile")
            last_x = max(tile.x for tile in containing)
            lhs = [(last_x, by_position[(last_x, y)])]
            x = last_x - names.TILE_SPACING
            while (x, y) in by_position:
                lhs.insert(0, (x, by_position[(x, y)]))
                x -= names.TILE_SPACING
            rhs, x = [], last_x + 10
            while (x, y) in by_position:
                rhs.append((x, by_position[(x, y)]))
                x += names.TILE_SPACING
            rules.append({"lhs": [s for _, s in lhs], "rhs": [s for _, s in rhs],
                          "x": lhs[0][0], "y": y})
        lower = sorted((tile for tile in tiles if tile.y > split_y), key=lambda s: (s.y, s.x))
        source_y = min(tile.y for tile in lower)
        source_tiles = [tile for tile in lower if tile.y == source_y]
        target_tiles = [tile for tile in lower if tile.y != source_y]
        profile = PROFILES[difficulty]
        clean = {
            "mode": profile["mode"], "mechanics": dict(profile["flags"]), "split_y": split_y,
            "rules": rules,
            "source": {"x0": source_tiles[0].x, "y": source_y,
                       "symbols": [names.symbol(tile.name) for tile in source_tiles]},
            "target": {"x0": target_tiles[0].x, "y": target_tiles[0].y,
                       "symbols": [names.symbol(tile.name) for tile in target_tiles]},
        }
        geometry.add(geometry_identity(clean))
        gameplay.add(gameplay_identity(clean))
    return geometry, gameplay


_OFFICIAL_GEOMETRY, _OFFICIAL_GAMEPLAY = _official_identity_sets()


def verify(spec, node_limit):
    """Return ``(accepted_spec, reason)`` after intended-context native replay."""
    structural_errors = profile_errors(spec, require_proof=False)
    if structural_errors:
        return None, "profile_structure:" + "|".join(sorted(structural_errors))
    extent_errors = _render_extent_errors(spec)
    if extent_errors:
        return None, "render_extent:" + "|".join(sorted(extent_errors))
    context = PROFILES[spec["difficulty"]]["context_index"]
    env = env_for(spec, context)
    layout = extract(env)
    initial = layout.to_dict()
    frame = env.render()
    density = sum(pixel not in (names.BACKGROUND_COLOR, names.PADDING_COLOR)
                  for row in frame for pixel in row)
    proposed, search_work, search_truncated = _constructive_actions(spec, layout, node_limit)
    if search_truncated:
        return None, "constructive_search_truncated"
    if not proposed:
        return None, "constructive_witness_unavailable"
    actual, touched = [], set()
    cycle_actions = select_actions = 0
    for action, x, y in proposed:
        if action in names.CYCLE_DELTA:
            touched.add(env.cursor())
            cycle_actions += 1
        else:
            select_actions += 1
        observation = env.perform(action)
        actual.append((action, x, y))
        if env.levels_completed:
            break
    if env.levels_completed != 1 or not observation.won:
        return None, "context_replay_failed"
    solved_layout = extract(env)
    trace = translation_trace(solved_layout)
    if (not trace["success"] or tuple(solved_layout.target[:len(trace["output"])]) != trace["output"]):
        return None, "translation_trace_failed"
    mechanics = _mechanics_evidence(solved_layout, trace, touched, cycle_actions, select_actions)
    accepted = dict(spec)
    accepted.update({
        "initial": initial,
        "solution": [list(step) for step in actual],
        "context_solution": [list(step) for step in actual],
        "solution_length": len(actual),
        "solution_optimal": spec["difficulty"] <= 4,
        "context_index": context,
        "training_context_index": context,
        "verification_level_index": context,
        "native_budget": layout.budget,
        "budget_remaining": layout.budget - len(actual),
        "context_engine_verified": True,
        "engine_verified": True,
        "engine_win": True,
        "levels_completed": 1,
        "search_work": search_work,
        "search_truncated": False,
        "visual_nonbackground_pixels": density,
        "solution_mechanics": mechanics,
        "proof": {
            "kind": "constructive-native-replay",
            "difficulty": spec["difficulty"],
            "context_index": context,
            "native_budget": layout.budget,
            "context_engine_verified": True,
            "engine_win": True,
            "solution_optimal": spec["difficulty"] <= 4,
            "search_truncated": False,
            "search_work": search_work,
            "split": spec["split"],
            "geometry_d4_sha256": spec["geometry_d4_sha256"],
            "gameplay_sha256": spec["gameplay_sha256"],
            "generator_version": GENERATOR_VERSION,
            "difficulty_version": DIFFICULTY_VERSION,
            "quality_profile_version": QUALITY_PROFILE_VERSION,
        },
    })
    errors = profile_errors(accepted)
    return (accepted, None) if not errors else (None, "profile:" + "|".join(sorted(errors)))


def generate(seed, difficulty, node_limit=None, attempts=MAX_ATTEMPTS, stats=None, *, split=None):
    """Generate a deterministic, split-qualified, fully certified level.

    ``node_limit`` bounds measured routing states and ``attempts`` bounds all
    redraws. Omitting ``split`` preserves the historical smoke API and selects
    ``train``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if type(attempts) is not int or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    if node_limit is not None and (type(node_limit) is not int or node_limit < 1):
        raise ValueError("node_limit must be a positive integer when supplied")
    route_limit = PROFILES[difficulty]["search_work"] if node_limit is None else node_limit
    split = "train" if split is None else split
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation, or test")
    rng = random.Random(f"{DIFFICULTY_VERSION}:{seed}:{difficulty}:{split}")
    exclusions = {}
    for attempt in range(1, attempts + 1):
        spec = _draft(rng, difficulty)
        if spec is None:
            _count(stats, exclusions, "invalid_draft")
            continue
        geometry_hash, partition = geometry_partition(spec)
        gameplay_hash = gameplay_identity(spec)
        if partition != split:
            _count(stats, exclusions, "geometry_split")
            continue
        if geometry_hash in _OFFICIAL_GEOMETRY or gameplay_hash in _OFFICIAL_GAMEPLAY:
            _count(stats, exclusions, "official_copy")
            continue
        spec.update({
            "seed": seed,
            "generation_attempt": attempt,
            "split": split,
            "geometry_split": partition,
            "geometry_sha256": geometry_hash,
            "geometry_d4_sha256": geometry_hash,
            "gameplay_sha256": gameplay_hash,
            "geometry_version": "d4-translation-normalized-v1",
            "gameplay_identity_version": "cyclic-symbol-normalized-v1",
            "official_copy": False,
            "generation_exclusions": dict(exclusions),
        })
        accepted, reason = verify(spec, route_limit)
        if accepted is not None:
            if stats is not None:
                stats["accepted"] = stats.get("accepted", 0) + 1
            accepted["generation_exclusions"] = dict(exclusions)
            return accepted
        _count(stats, exclusions, reason)
    return None


def _count(external, local, key):
    local[key] = local.get(key, 0) + 1
    if external is not None:
        external[key] = external.get(key, 0) + 1


def _child_seed(game_seed, ordinal, difficulty):
    payload = f"tr87-cd924810:{game_seed}:{ordinal}:{difficulty}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def generate_game(seed, *, split, difficulties=None, attempts=MAX_ATTEMPTS, node_limit=None,
                  stats=None):
    """Generate an ordered full episode (or an explicitly reduced smoke set)."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    tiers = DIFFICULTIES if difficulties is None else tuple(difficulties)
    if not tiers or any(type(d) is not int or d not in DIFFICULTIES for d in tiers):
        raise ValueError("difficulties must be a nonempty sequence drawn from DIFFICULTIES")
    if tuple(d for d in DIFFICULTIES if d in tiers) != tiers:
        raise ValueError("explicit difficulties must be unique and in increasing official order")
    specs = []
    for ordinal, difficulty in enumerate(tiers):
        child = _child_seed(seed, ordinal, difficulty)
        spec = generate(child, difficulty, node_limit=node_limit, attempts=attempts,
                        stats=stats, split=split)
        if spec is None:
            return None
        spec["game_seed"] = seed
        spec["game_ordinal"] = ordinal
        specs.append(spec)
    return specs


def build_game(specs):
    """Validate and build exactly one full six-context native episode."""
    if not isinstance(specs, (list, tuple)):
        raise ValueError("full-standard TR87 game specs must be a list or tuple")
    specs = list(specs)
    if len(specs) != len(DIFFICULTIES):
        raise ValueError("full-standard TR87 games require exactly six specs")
    if any(not isinstance(spec, Mapping) for spec in specs):
        raise ValueError("every game spec must be a mapping")
    splits = {spec.get("split") for spec in specs}
    if len(splits) != 1:
        raise ValueError("all game specs must use the same split")
    geometries, gameplays = set(), set()
    for index, (difficulty, spec, entry) in enumerate(
            zip(DIFFICULTIES, specs, FULL_STANDARD_CONTRACT["curriculum"])):
        if (spec.get("difficulty") != difficulty or spec.get("context_index") != index
                or spec.get("training_context_index") != index):
            raise ValueError("game specs must be ordered tiers 1..6 at contexts 0..5")
        errors = validate_full_standard(spec, entry)
        if errors:
            raise ValueError("invalid full-standard spec: " + "; ".join(errors))
        geometry, gameplay = spec["geometry_d4_sha256"], spec["gameplay_sha256"]
        if geometry in geometries or gameplay in gameplays:
            raise ValueError("duplicate geometry or gameplay identity in game")
        geometries.add(geometry)
        gameplays.add(gameplay)
    levels = [build_level(spec) for spec in specs]
    episode = Env(levels)
    episode.reset()
    observation = None
    for index, spec in enumerate(specs):
        if episode.level_index != index or episode.levels_completed != index:
            raise ValueError("native episode entered the wrong curriculum context")
        for action, x, y in spec["context_solution"]:
            if x is not None or y is not None:
                raise ValueError("TR87 episode witness must use null display coordinates")
            observation = episode.perform(action)
        if episode.levels_completed != index + 1:
            raise ValueError("native episode witness did not advance exactly one tier")
    if observation is None or not observation.won:
        raise ValueError("native six-tier episode did not reach WIN")
    return levels


def validate_full_standard(spec, curriculum_entry):
    """Errors proving whether ``spec`` satisfies a shared curriculum entry."""
    if not isinstance(spec, Mapping):
        return ["spec must be a mapping"]
    try:
        errors = list(profile_errors(spec))
    except Exception as error:  # malformed public data must fail closed
        errors = [f"malformed spec: {type(error).__name__}"]
    try:
        errors.extend(_render_extent_errors(spec))
    except Exception as error:
        errors.append(f"render extent validation failed: {type(error).__name__}")
    if not isinstance(curriculum_entry, Mapping):
        return errors + ["curriculum entry must be a mapping"]
    if curriculum_entry.get("difficulty") != spec.get("difficulty"):
        errors.append("curriculum difficulty mismatch")
    if curriculum_entry.get("context_index") != spec.get("training_context_index"):
        errors.append("curriculum context mismatch")
    allowed_work = curriculum_entry.get("search_work")
    if type(allowed_work) is not int or not 1 <= allowed_work <= 32_000_000:
        errors.append("curriculum search_work must be an integer in 1..32000000")
    proof = spec.get("proof", {})
    if not isinstance(proof, Mapping):
        errors.append("proof must be a mapping")
        proof = {}
    if proof.get("context_index") != curriculum_entry.get("context_index"):
        errors.append("proof context mismatch")
    if proof.get("native_budget") != spec.get("native_budget"):
        errors.append("proof native budget mismatch")
    if proof.get("context_engine_verified") is not True or proof.get("engine_win") is not True:
        errors.append("proof lacks a successful native replay")
    if proof.get("search_truncated") is not False:
        errors.append("proof is truncated or does not say")
    proof_work = proof.get("search_work")
    if type(proof_work) is not int or proof_work < 1:
        errors.append("proof has no bounded work measurement")
    if (type(proof_work) is not int
            or type(allowed_work) is int and proof_work > allowed_work):
        errors.append("proof exceeds curriculum search_work")
    for key in ("difficulty", "split", "geometry_d4_sha256", "gameplay_sha256",
                "generator_version", "difficulty_version", "quality_profile_version"):
        if proof.get(key) != spec.get(key):
            errors.append(f"proof {key} mismatch")
    try:
        identity, partition = geometry_partition(spec)
        if identity != spec.get("geometry_d4_sha256") or partition != spec.get("split"):
            errors.append("recomputed geometry identity/partition mismatch")
        gameplay = gameplay_identity(spec)
        if gameplay != spec.get("gameplay_sha256"):
            errors.append("recomputed gameplay identity mismatch")
        if identity in _OFFICIAL_GEOMETRY or gameplay in _OFFICIAL_GAMEPLAY:
            errors.append("recomputed identity matches an official level")
        expected_context = curriculum_entry.get("context_index")
        if type(expected_context) is int and expected_context in range(len(DIFFICULTIES)):
            errors.extend(_native_witness_errors(spec, expected_context))
    except Exception as error:  # native builders may reject malformed coordinates/symbols
        errors.append(f"native witness validation failed: {type(error).__name__}")
    return errors


def _native_witness_errors(spec, context):
    """Recompute bounded proof facts by replaying only the stored witness."""
    errors = []
    env = env_for(spec, context)
    layout = extract(env)
    initial_frame = env.render()
    density = sum(pixel not in (names.BACKGROUND_COLOR, names.PADDING_COLOR)
                  for row in initial_frame for pixel in row)
    if layout.to_dict() != spec.get("initial"):
        errors.append("stored initial state differs from native context")
    if layout.budget != spec.get("native_budget"):
        errors.append("stored native budget differs from replay")
    actions = spec.get("solution")
    if not isinstance(actions, list):
        return errors + ["solution must be a list"]
    route, measured_work, truncated = _constructive_actions(
        spec, layout, PROFILES[spec["difficulty"]]["search_work"]
    )
    if truncated or route is None:
        errors.append("certified constructive route cannot be recomputed within its bound")
    else:
        normalized_actions = [tuple(step) for step in actions]
        if route != normalized_actions:
            errors.append("stored witness differs from the recomputed constructive route")
        if (spec.get("search_work") != measured_work
                or spec.get("proof", {}).get("search_work") != measured_work):
            errors.append("stored search work differs from measured routing work")
    touched = set()
    cycle_actions = select_actions = 0
    observation = None
    for index, step in enumerate(actions):
        if (not isinstance(step, (list, tuple)) or len(step) != 3
                or step[0] not in names.ACTION_IDS or step[1] is not None or step[2] is not None):
            return errors + ["malformed native display action"]
        action = step[0]
        if action in names.CYCLE_DELTA:
            touched.add(env.cursor())
            cycle_actions += 1
        else:
            select_actions += 1
        observation = env.perform(action)
        if env.levels_completed and index != len(actions) - 1:
            errors.append("witness completes before its final stored action")
            break
    if observation is None or env.levels_completed != 1 or not observation.won:
        errors.append("stored witness does not win in its native context")
    solved_layout = extract(env)
    trace = translation_trace(solved_layout)
    if not trace["success"]:
        errors.append("native solved state has no successful translation trace")
    recomputed_mechanics = _mechanics_evidence(
        solved_layout, trace, touched, cycle_actions, select_actions)
    if recomputed_mechanics != spec.get("solution_mechanics"):
        errors.append("solution mechanic evidence differs from native replay")
    if density != spec.get("visual_nonbackground_pixels"):
        errors.append("visual density differs from native rendering")
    if layout.budget - len(actions) != spec.get("budget_remaining"):
        errors.append("remaining budget differs from native witness accounting")
    return errors


def _mechanics_evidence(layout, trace, touched, cycle_actions, select_actions):
    """Route participation plus the actual successful native grammar trace."""
    return {
        "cycle_actions": cycle_actions,
        "select_actions": select_actions,
        "edited_groups": len(touched),
        "edited_rule_groups": len(touched) if layout.alter_rules else 0,
        "edited_target_tiles": len(touched) if not layout.alter_rules else 0,
        "translation_depth": trace["translation_depth"],
        "translation_branch": trace["branch"],
        "translation_segments": len(trace["outer_rule_matches"]),
        "matched_rule_applications": (len(trace["outer_rule_matches"])
                                      + len(trace["secondary_rule_matches"])),
        "translated_symbols": len(trace["output"]),
        "double_compositions": trace["double_compositions"],
        "tree_expansions": trace["tree_expansions"],
        "tree_mixed_child_expansions": trace["tree_mixed_child_expansions"],
        "tree_repeated_child_expansions": trace["tree_repeated_child_expansions"],
        "double_branch_exercised": trace["branch"] == "double" and trace["double_compositions"] > 0,
        "tree_branch_exercised": trace["branch"] == "tree" and trace["tree_expansions"] > 0,
        "tree_shadows_double": trace["branch"] == "tree" and layout.double_translation,
    }
