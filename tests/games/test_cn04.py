import copy
import json
import subprocess
import sys
from collections import Counter

import pytest
from arcengine import Level, Sprite

from pebby.games.cn04 import names
import pebby.games.cn04.generate as cn_generate
import pebby.games.cn04.plan as cn_plan
from pebby.games.cn04.bank import build, load, save
from pebby.games.cn04.env import Env, completes_level, official_levels, replay
from pebby.games.cn04.generate import (
    DIFFICULTIES,
    FORMAT,
    FULL_STANDARD_CONTRACT,
    build_game,
    build_level,
    generate,
    generate_game,
    gameplay_hash,
    geometry_hashes,
    geometry_split,
    load_level,
    validate_full_standard,
)
from pebby.games.cn04.layout import extract
from pebby.games.cn04.plan import search, solve, trace, transition
from pebby.games.cn04.reference_profiles import (
    OFFICIAL_TIER6_NO_BOUNCE_SOLUTION,
    PROFILES,
    STACK_PIN_SPECTRA,
    WINNING_PIN_DEGREES,
    native_semantic_groups,
    official_characterization,
    profile_errors,
)


def env_for(specs):
    if isinstance(specs, dict):
        specs = [specs]
    return Env([build_level(spec) for spec in specs])


def assert_action_triples(actions, available=names.ACTION_IDS):
    assert actions is not None
    for action in actions:
        assert isinstance(action, (list, tuple)) and len(action) == 3
        assert action[0] in available
        if action[0] == names.ACTION_CLICK:
            assert all(isinstance(value, int) and 0 <= value < 64 for value in action[1:])
        else:
            assert tuple(action[1:]) == (None, None)


@pytest.fixture(scope="module")
def official_witnesses():
    """Full teacher coverage, computed in each intended native context."""
    witnesses = []
    for difficulty in DIFFICULTIES:
        env = Env()
        if difficulty > 1:
            env.set_level(difficulty - 1)
        result = search(env, limit=PROFILES[difficulty]["search_work"])
        assert result.solved, (difficulty, result.reason, result.expanded)
        assert not result.truncated and not result.unsupported
        assert_action_triples(result.actions, env.available_actions)
        assert completes_level([official_levels()[difficulty - 1]], result.actions)
        witnesses.append(result.actions)
    return witnesses


def test_all_six_official_tiers_are_solved_and_replay_as_one_native_episode(
        official_witnesses):
    assert len(official_levels()) == len(DIFFICULTIES) == 6
    env = Env()
    frame = env.reset()
    assert len(frame) == 64 and all(len(row) == 64 for row in frame)
    for index, actions in enumerate(official_witnesses):
        assert env.level_index == index
        before = env.levels_completed
        for action_index, action in enumerate(actions):
            observation = env.perform(*action)
            if action_index < len(actions) - 1:
                assert env.levels_completed == before
        assert env.levels_completed == before + 1
    assert observation.finished and observation.won
    assert env.levels_completed == 6


def test_official_characterization_is_six_scarce_reference_rows():
    rows = official_characterization()
    assert [row["difficulty"] for row in rows] == list(DIFFICULTIES)
    assert [row["context_index"] for row in rows] == list(range(6))
    assert [row["sprites"] for row in rows] == [2, 4, 3, 4, 8, 13]
    assert [tuple(row["stack_sizes"]) for row in rows] == [(), (), (), (), (5,), (6, 4)]
    assert [row["max_steps"] for row in rows] == [75, 100, 125, 125, 150, 200]
    assert [tuple(row["winning_pin_degrees"]) for row in rows] == [
        WINNING_PIN_DEGREES[difficulty] for difficulty in DIFFICULTIES
    ]
    assert "confidence" not in FULL_STANDARD_CONTRACT["evidence"]["official_tier_characterization"]


def test_setting_an_already_fresh_context_is_idempotent():
    env = Env()
    before = extract(env).start
    pristine = {sprite.name: env.original_pixels(sprite).tolist() for sprite in env.sprites()}
    env.set_level(0)
    assert extract(env).start == before
    assert {sprite.name: env.original_pixels(sprite).tolist()
            for sprite in env.sprites()} == pristine


@pytest.mark.parametrize("split", ("train", "validation", "test"))
def test_every_tier_generates_with_profiles_native_proofs_and_disjoint_split(split):
    rows = [generate(0, difficulty, split=split) for difficulty in DIFFICULTIES]
    assert all(rows)
    assert len({row["geometry_d4_sha256"] for row in rows}) == 6
    assert len({row["gameplay_sha256"] for row in rows}) == 6
    for difficulty, row in zip(DIFFICULTIES, rows):
        assert row["format"] == FORMAT
        assert row["split"] == row["geometry_split"] == split
        assert row["official_copy"] is False
        assert profile_errors(row) == []
        assert validate_full_standard(
            row, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
        ) == []
        assert completes_level([build_level(row)], [tuple(action) for action in row["solution"]])
        teacher = search(env_for(row), limit=PROFILES[difficulty]["search_work"])
        assert teacher.solved and teacher.actions == [tuple(action) for action in row["solution"]]
        assert row["solution_constraints"]["winning_pin_counts"] == list(
            WINNING_PIN_DEGREES[difficulty]
        )
        if difficulty == 5:
            assert row["solution_constraints"]["parallel_relation_counts"] == [3, 2, 2]
        assert row["proof"]["teacher_reason"] == teacher.reason
        assert row["proof"]["bounded_assignment_caps"] == teacher.assignment_caps
    assert rows[4]["solution_mechanics"]["distinct_stacks_cycled"] == 1
    assert rows[5]["solution_mechanics"]["distinct_stacks_cycled"] == 2
    assert rows[5]["solution_mechanics"]["bounce_reversals"] == 0


def test_split_is_geometry_derived_and_generation_is_deterministic():
    a = generate(17, 6, split="validation")
    b = generate(17, 6, split="validation")
    assert a == b
    private_change = copy.deepcopy(a)
    private_change["constructed_target"][0] = [19, 18, 270, 5]
    private_change["solution"] = list(reversed(private_change["solution"]))
    private_change["solution_mechanics"] = {"forged": "private certificate"}
    private_change["proof"] = {"forged": "private certificate"}
    assert geometry_hashes(private_change) == geometry_hashes(a)
    assert gameplay_hash(private_change) == gameplay_hash(a)
    assert geometry_split(geometry_hashes(private_change)[1]) == a["split"]

    # Authored group/alternate labels are ignored by native CN04.  A
    # consistent relabel must leave identities, partition, validation, and the
    # complete native episode unchanged.
    relabelled = copy.deepcopy(a)
    for piece in relabelled["pieces"]:
        piece["group"] = 100 + piece["group"] * 7
        piece["alternate"] = 500 - piece["alternate"] * 11
    assert geometry_hashes(relabelled) == geometry_hashes(a)
    assert gameplay_hash(relabelled) == gameplay_hash(a)
    assert validate_full_standard(
        relabelled, FULL_STANDARD_CONTRACT["curriculum"][5]
    ) == []
    original_env, relabelled_env = env_for(a), env_for(relabelled)
    assert original_env.reset() == relabelled_env.reset()
    for action in a["solution"]:
        original = original_env.perform(*action)
        relabelled_observation = relabelled_env.perform(*action)
        assert original.frame == relabelled_observation.frame
        assert original.levels_completed == relabelled_observation.levels_completed
    assert original.won and relabelled_observation.won

    contradictory = copy.deepcopy(a)
    contradictory["pieces"][-1]["group"] = contradictory["pieces"][0]["group"]
    assert any("contradictory" in error or "recomputation" in error
               for error in validate_full_standard(
                   contradictory, FULL_STANDARD_CONTRACT["curriculum"][5]
               ))

    tied_layers = copy.deepcopy(a)
    origin = (tied_layers["pieces"][0]["x"], tied_layers["pieces"][0]["y"])
    tied_stack = [piece for piece in tied_layers["pieces"]
                  if (piece["x"], piece["y"]) == origin]
    assert len(tied_stack) == 6
    for piece in tied_stack:
        piece["layer"] = 7
    semantic_names = [piece["name"] for _, piece
                      in native_semantic_groups(tied_layers)[0]]
    tied_env = env_for(tied_layers)
    native_names = [piece.name for piece in tied_env.stacks()[tied_env.sprites()[0]]]
    assert semantic_names == native_names == [piece["name"] for piece in tied_stack]

    reassigned = copy.deepcopy(a)
    reassigned["pieces"][0]["pixels"], reassigned["pieces"][1]["pixels"] = (
        reassigned["pieces"][1]["pixels"], reassigned["pieces"][0]["pixels"],
    )
    assert gameplay_hash(reassigned) != gameplay_hash(a)
    with pytest.raises(ValueError, match="split"):
        generate(0, 1, split="dev")
    with pytest.raises(TypeError):
        generate(0, 1)
    for bad_seed in (True, 1.0, "1"):
        with pytest.raises(ValueError, match="seed must be an integer"):
            generate(bad_seed, 1, split="train")
    for bad_difficulty in (True, 1.0, "1"):
        with pytest.raises(ValueError, match="difficulty must be an integer"):
            generate(0, bad_difficulty, split="train")
    for bad_attempts in (True, 1.0, 0, -1):
        with pytest.raises(ValueError, match="attempts must be a positive integer"):
            generate(0, 1, split="train", attempts=bad_attempts)


def test_validator_recomputes_route_geometry_mechanics_and_proof():
    row = generate(0, 6, split="test")
    contract = FULL_STANDARD_CONTRACT["curriculum"][5]
    assert validate_full_standard(row, contract) == []

    wrong_route = copy.deepcopy(row)
    wrong_route["solution"][0] = [names.ACTION_UP, None, None]
    assert validate_full_standard(wrong_route, contract)

    wrong_geometry = copy.deepcopy(row)
    wrong_geometry["pieces"][0]["x"] += 1
    assert any("identity" in error or "route" in error
               for error in validate_full_standard(wrong_geometry, contract))

    wrong_mechanics = copy.deepcopy(row)
    wrong_mechanics["solution_mechanics"]["stack_cycles_action"] = 0
    assert validate_full_standard(wrong_mechanics, contract)

    wrong_proof = copy.deepcopy(row)
    wrong_proof["proof"]["native_transition_match"] = False
    assert validate_full_standard(wrong_proof, contract)

    malformed = copy.deepcopy(row)
    malformed["pieces"] = [{"pixels": None}]
    assert validate_full_standard(malformed, contract)

    for bad_difficulty in (True, 1.0, "1", [1]):
        malformed = copy.deepcopy(row)
        malformed["difficulty"] = bad_difficulty
        assert validate_full_standard(malformed, contract)
    for bad_proof in (None, [], True):
        malformed = copy.deepcopy(row)
        malformed["proof"] = bad_proof
        assert "proof must be a mapping" in validate_full_standard(malformed, contract)
    assert validate_full_standard(row, None) == ["curriculum entry must be a mapping"]

    strict_mutations = (
        (("seed",), True),
        (("engine_verified",), False),
        (("search_limit",), 0),
        (("search_truncated",), True),
        (("reachable_states",), -1),
        (("verification_level_index",), False),
        (("proof", "generator_version"), -1),
        (("proof", "verification_level_index"), 0),
        (("proof", "engine_win"), "false"),
        (("proof", "levels_completed"), 6),
        (("proof", "search_limit"), 0),
        (("proof", "bounded_assignment_caps"), True),
        (("native_budget", "max_steps"), True),
    )
    for path, value in strict_mutations:
        malformed = copy.deepcopy(row)
        target = malformed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert validate_full_standard(malformed, contract), path


@pytest.fixture(scope="module")
def generated_tier_one_row():
    row = generate(0, 1, split="train")
    assert row is not None
    return row


@pytest.mark.parametrize(
    "mutation",
    (
        "float_pin",
        "oversized_pin",
        "boolean_pixel",
        "float_pixels",
        "unsupported_pixel_low",
        "unsupported_pixel_high",
        "ragged_pixels",
        "empty_pixel_row",
        "missing_pixels",
        "string_rotation",
        "non_quarter_rotation",
        "missing_rotation",
        "float_grid_size",
        "float_background",
        "float_max_steps",
        "integer_grey_masking",
        "float_generator_version",
        "missing_optimal_actions",
        "integer_won",
        "float_curriculum_difficulty",
        "boolean_curriculum_context",
        "float_curriculum_work",
    ),
)
def test_validator_rejects_malformed_primitive_schema(generated_tier_one_row, mutation):
    row = copy.deepcopy(generated_tier_one_row)
    contract = copy.deepcopy(FULL_STANDARD_CONTRACT["curriculum"][0])

    if mutation in {
        "float_pin", "oversized_pin", "boolean_pixel",
        "unsupported_pixel_low", "unsupported_pixel_high",
    }:
        replacement = {
            "float_pin": 8.9,
            "oversized_pin": 2 ** 100,
            "boolean_pixel": True,
            "unsupported_pixel_low": -3,
            "unsupported_pixel_high": 16,
        }[mutation]
        for pixels in row["pieces"][0]["pixels"]:
            for index, value in enumerate(pixels):
                if value == names.PIN_A:
                    pixels[index] = replacement
                    break
            else:
                continue
            break
    elif mutation == "float_pixels":
        row["pieces"][0]["pixels"] = [
            [float(value) for value in pixels]
            for pixels in row["pieces"][0]["pixels"]
        ]
    elif mutation == "ragged_pixels":
        row["pieces"][0]["pixels"][0].append(names.TRANSPARENT)
    elif mutation == "empty_pixel_row":
        row["pieces"][0]["pixels"][0] = []
    elif mutation == "missing_pixels":
        del row["pieces"][0]["pixels"]
    elif mutation == "string_rotation":
        row["pieces"][0]["rotation"] = str(row["pieces"][0]["rotation"])
    elif mutation == "non_quarter_rotation":
        row["pieces"][0]["rotation"] = 45
    elif mutation == "missing_rotation":
        del row["pieces"][0]["rotation"]
    elif mutation == "float_grid_size":
        row["grid_size"][0] = float(row["grid_size"][0])
    elif mutation == "float_background":
        row["background"] = float(row["background"])
    elif mutation == "float_max_steps":
        row["max_steps"] = float(row["max_steps"])
    elif mutation == "integer_grey_masking":
        row["grey_masking"] = int(row["grey_masking"])
    elif mutation == "float_generator_version":
        row["generator_version"] = float(row["generator_version"])
    elif mutation == "missing_optimal_actions":
        del row["proof"]["optimal_actions"]
    elif mutation == "integer_won":
        row["solution_mechanics"]["won"] = 1
    elif mutation == "float_curriculum_difficulty":
        contract["difficulty"] = 1.0
    elif mutation == "boolean_curriculum_context":
        contract["context_index"] = False
    elif mutation == "float_curriculum_work":
        contract["search_work"] = float(contract["search_work"])
    else:  # pragma: no cover - keeps additions to the parameter list explicit
        raise AssertionError(f"unknown mutation: {mutation}")

    assert validate_full_standard(row, contract), mutation


@pytest.mark.parametrize("seed", (0, 1))
def test_validator_accepts_bounded_ordinary_tier_one_rows(seed):
    row = generate(seed, 1, split="train")
    assert row is not None
    assert validate_full_standard(row, FULL_STANDARD_CONTRACT["curriculum"][0]) == []


def test_validator_accepts_native_unclickable_transparency(generated_tier_one_row):
    row = copy.deepcopy(generated_tier_one_row)
    for pixels in row["pieces"][0]["pixels"]:
        for index, value in enumerate(pixels):
            if value == names.TRANSPARENT:
                pixels[index] = names.TRANSPARENT_UNCLICKABLE
                assert validate_full_standard(
                    row, FULL_STANDARD_CONTRACT["curriculum"][0]
                ) == []
                return
    raise AssertionError("generated row has no transparent pixel")


def test_generated_stack_spectra_leave_equal_count_and_colour_ambiguity():
    for difficulty in (5, 6):
        row = generate(0, difficulty, split="train")
        groups = native_semantic_groups(row)
        spectra = []
        for group_index, group in enumerate(groups):
            if len(group) == 1:
                continue
            signatures = []
            for _, piece in group:
                pins = [
                    value for values in piece["pixels"] for value in values
                    if value in names.PIN_COLORS
                ]
                signatures.append((len(pins), tuple(sorted(Counter(pins).items()))))
            spectra.append(tuple(sorted(count for count, _ in signatures)))
            if difficulty == 6 and len(group) == 4:
                winning = row["solution_constraints"]["winning_alternates"][group_index]
                winner_signature = signatures[winning]
                assert sum(signature == winner_signature for signature in signatures) >= 2
        assert tuple(spectra) == STACK_PIN_SPECTRA[difficulty]


def test_generated_teacher_is_independent_of_quality_action_floor(monkeypatch):
    row = generate(0, 5, split="train")
    baseline = search(env_for(row), limit=PROFILES[5]["search_work"])
    monkeypatch.setitem(PROFILES[5], "witness_actions", (1_000, 1_001))
    repeated = search(env_for(row), limit=PROFILES[5]["search_work"])
    assert not hasattr(cn_plan, "_GENERATED_INITIAL_ACTION_RANGES")
    assert repeated.actions == baseline.actions
    assert repeated.expanded == baseline.expanded
    assert repeated.reason == baseline.reason


def test_json_round_trip_and_exact_native_stack_differential(tmp_path):
    row = generate(0, 6, split="test")
    path = save([row], tmp_path / "cn04.jsonl")
    restored = load(path)[0]
    assert restored == json.loads(json.dumps(row))
    level = load_level(restored)
    env = Env([level])
    env.reset()
    layout = extract(env)
    state = layout.start
    for index, action in enumerate(restored["solution"]):
        state, _, _ = transition(layout, state, tuple(action))
        observation = env.perform(*action)
        if index < len(restored["solution"]) - 1:
            assert extract(env).start == state
            assert not observation.finished
    assert observation.won
    assert trace(layout, restored["solution"]) == restored["solution_mechanics"]


def test_action5_clamps_a_new_alternate_but_click_cycle_does_not():
    small = Sprite(pixels=[[0, 0, 8]], name="small", x=17, y=5, layer=0,
                   visible=True, tags=[names.CLICK_TAG])
    wide = Sprite(pixels=[[0, 9, 9, 9, 8]], name="wide", x=17, y=5, layer=1,
                  visible=False, tags=[names.CLICK_TAG])
    level = Level(sprites=[small, wide], grid_size=(20, 20),
                  data={names.KEY_MAX_STEPS: 20, names.KEY_GREY_MASKING: True})

    click_env = Env([level])
    click_layout = extract(click_env)
    # Use the second zero body cell, not the planner's canonical first click,
    # to cover arbitrary native display coordinates with the same semantics.
    click = names.grid_to_display(18, 5)
    click_env.perform(names.ACTION_CLICK, *click)
    assert click_env.selected().name == "wide"
    assert click_env.selected().x == 17  # width 5 intentionally protrudes

    action_env = Env([level])
    action_env.perform(names.ACTION_ROTATE)
    assert action_env.selected().name == "wide"
    assert action_env.selected().x == 15  # ACTION5 alone runs clamp


def test_tier_six_bounce_is_optional_for_the_witness_but_live_recovery_works():
    official = official_characterization()[5]
    assert official["reference_no_bounce_actions"] == 54
    assert official["reference_no_bounce_reversals"] == 0
    assert official["reference_bounce_required"] is False

    official_env = Env()
    official_env.set_level(5)
    official_layout = extract(official_env)
    before = official_env.levels_completed
    for index, action in enumerate(OFFICIAL_TIER6_NO_BOUNCE_SOLUTION):
        observation = official_env.perform(*action)
        if index < len(OFFICIAL_TIER6_NO_BOUNCE_SOLUTION) - 1:
            assert official_env.levels_completed == before
    official_trace = trace(official_layout, OFFICIAL_TIER6_NO_BOUNCE_SOLUTION)
    assert observation.won and official_env.levels_completed == before + 1
    assert len(OFFICIAL_TIER6_NO_BOUNCE_SOLUTION) == 54
    assert official_trace["won"] and official_trace["bounce_reversals"] == 0

    row = generate(0, 6, split="train")
    assert row["solution_mechanics"]["bounce_reversals"] == 0

    env = env_for(row)
    env.reset()
    # The first generated group is a six-alternate stack starting at 0F.
    # Six ACTION5 inputs traverse to the endpoint and bounce back to 4B.
    for _ in range(6):
        env.perform(names.ACTION_ROTATE)
    live = extract(env)
    assert live.start[1][0] == 4
    assert live.start[3] is False

    result = search(env, limit=PROFILES[6]["search_work"])
    assert result.solved and not result.truncated and not result.unsupported
    clone = env.clone()
    before = clone.levels_completed
    for action in result.actions:
        observation = clone.perform(*action)
    assert observation.won and clone.levels_completed == before + 1


def _already_matched_level():
    pixels = [[-1] * 7 for _ in range(7)]
    pixels[3][3] = names.PIN_A
    first = Sprite(pixels=pixels, name="first", x=3, y=3,
                   tags=[names.CLICK_TAG])
    second = Sprite(pixels=[[9, names.PIN_A]], name="second", x=5, y=6,
                    tags=[names.CLICK_TAG])
    # Different initial top-lefts keep these as singleton groups; both pins are
    # at (6,6) because the second piece's pin sits at local x=1.
    return Level(sprites=[first, second], grid_size=(20, 20),
                 data={names.KEY_MAX_STEPS: 20})


def test_click_does_not_check_completion_but_action5_does():
    env = Env([_already_matched_level()])
    assert env.is_solved()
    click = names.grid_to_display(6, 6)
    observation = env.perform(names.ACTION_CLICK, *click)
    assert not observation.finished and env.levels_completed == 0
    env.perform(names.ACTION_CLICK, *click)  # reselect first singleton
    observation = env.perform(names.ACTION_ROTATE)
    assert observation.won and env.levels_completed == 1


def test_live_prefix_teacher_recovers_on_stacked_official_tier(official_witnesses):
    env = Env()
    env.set_level(5)
    prefix = official_witnesses[5][:12]
    for action in prefix:
        env.perform(*action)
    result = search(env, limit=PROFILES[6]["search_work"])
    assert result.solved and not result.truncated and not result.unsupported
    assert replay(env.clone(), result.actions)


def test_generated_stacked_live_prefix_uses_native_replayed_fast_teacher():
    row = generate(3, 6, split="validation")
    env = Env([build_level(row)])
    for action in (
        (names.ACTION_RIGHT, None, None),
        (names.ACTION_DOWN, None, None),
        (names.ACTION_ROTATE, None, None),
    ):
        env.perform(*action)
    result = search(env, limit=PROFILES[6]["search_work"])
    assert result.solved and not result.truncated and not result.unsupported
    assert result.reason.startswith(
        "floor-independent minimum-translation semantic assembly route won in native replay"
    )
    assert replay(env.clone(), result.actions)


def test_generate_game_builds_and_replays_exact_six_tier_sequence(monkeypatch):
    rows = generate_game(0, split="train")
    assert rows is not None and [row["difficulty"] for row in rows] == list(DIFFICULTIES)
    assert len({row["seed"] for row in rows}) == 6
    assert all(row["proof"]["full_game_replay"] for row in rows)
    assert len(build_game(rows)) == 6
    reduced = generate_game(0, split="train", difficulties=(2, 4))
    assert [row["difficulty"] for row in reduced] == [2, 4]
    with pytest.raises(ValueError, match="exactly six"):
        build_game(reduced)
    with pytest.raises(ValueError, match="ordered"):
        build_game(list(reversed(rows)))
    with pytest.raises(ValueError, match="mappings"):
        build_game([None] * len(DIFFICULTIES))
    for bad_seed in (True, 1.0, "1"):
        with pytest.raises(ValueError, match="seed must be an integer"):
            generate_game(bad_seed, split="train")

    # Direct construction must establish a fresh uninterrupted native replay;
    # it cannot rely only on the per-row validator's isolated certificates.
    broken = copy.deepcopy(rows)
    broken[1]["solution"] = []
    monkeypatch.setattr(cn_generate, "validate_full_standard", lambda *_: [])
    with pytest.raises(ValueError, match="failed tier 2"):
        build_game(broken)


def test_bank_builder_and_cli_require_explicit_split(tmp_path):
    specs, tried = build(3, seed=0, difficulty=1, split="test")
    assert len(specs) == 3 and tried >= 3
    assert all(spec["split"] == "test" for spec in specs)
    path = tmp_path / "bank.jsonl"
    completed = subprocess.run(
        [sys.executable, "-m", "pebby.games.cn04.bank", "--levels", "3",
         "--seed", "0", "--difficulty", "1", "--split", "test",
         "--out", str(path)],
        check=True, timeout=60, capture_output=True, text=True,
    )
    assert "3/3 levels" in completed.stdout
    assert len(load(path)) == 3


def test_pairing_requires_exactly_two_pins_of_the_same_colour():
    def pin_level(colours):
        sprites = []
        for index, colour in enumerate(colours):
            pixels = [[9] * (index + 1)]
            pixels[0][-1] = colour
            sprites.append(Sprite(pixels=pixels, name=f"pin{index}", x=6 - index,
                                  y=6, tags=[names.CLICK_TAG]))
        return Level(sprites=sprites, grid_size=(20, 20),
                     data={names.KEY_MAX_STEPS: 20})

    same = Env([pin_level([names.PIN_A, names.PIN_A])])
    mixed = Env([pin_level([names.PIN_A, names.PIN_B])])
    triple = Env([pin_level([names.PIN_A, names.PIN_A, names.PIN_A])])
    assert same.is_solved() and extract(same).complete(extract(same).start)
    assert not mixed.is_solved() and not extract(mixed).complete(extract(mixed).start)
    assert not triple.is_solved() and not extract(triple).complete(extract(triple).start)


def test_cut_off_search_is_explicitly_inconclusive():
    env = env_for(generate(0, 1, split="test"))
    result = search(env, limit=0)
    assert result.truncated and not result.unsupported and not result.solved
    assert solve(env, limit=0) is None
    assert solve.truncated and not solve.unsupported
