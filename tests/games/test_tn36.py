"""Full-mechanic TN36 teacher, generation, and contract tests."""

import copy
import json
from pathlib import Path
import tempfile

from arcengine import GameState
import pytest

from pebby.games.tn36 import names
from pebby.games.tn36.bank import build as build_bank, load as load_bank, save as save_bank
from pebby.games.tn36.env import Env, official_levels, replay
from pebby.games.tn36.generate import (
    DEFAULT_ATTEMPTS, DIFFICULTIES, FORMAT, FULL_STANDARD_CONTRACT, build_game,
    build_level, _context_env, _official_equivalence_ids, _trace, generate,
    generate_game, validate_full_standard,
)
from pebby.games.tn36.layout import extract
from pebby.games.tn36.plan import execute, execute_program, program_actions, search
from pebby.games.tn36.quality import (
    REFERENCE_PROFILES, REQUIRED_MECHANIC_EVENTS, identities,
    official_equivalence_identity, split_accepts, split_bucket,
)
from pebby.multigame import collect_generated_game, preflight


def assert_legal(actions):
    assert actions
    for action_id, x, y in actions:
        assert action_id == names.ACTION_CLICK
        assert isinstance(x, int) and isinstance(y, int)
        assert 0 <= x < 64 and 0 <= y < 64


def test_all_seven_official_levels_have_positive_native_teacher_replays():
    env = Env()
    assert env.level_count == len(official_levels()) == 7
    frame = env.reset()
    assert len(frame) == 64 and all(len(row) == 64 for row in frame)
    lengths = []
    for context in range(7):
        assert env.level_index == context
        result = search(env, limit=env.clicks_left, node_limit=400_000)
        assert result.solved and not result.truncated and not result.unsupported, result
        assert result.exact is (context < 5)  # later checkpoint routes are constructive
        assert_legal(result.actions)
        lengths.append(len(result.actions))
        assert replay(env, result.actions)
        assert env.levels_completed == context + 1
    assert lengths == [7, 9, 9, 12, 16, 18, 16]
    assert env.state == GameState.WIN


def test_official_reference_structure_and_native_budgets_are_pinned():
    expected = [
        (5, 2, 0, 0, 0, 0, 61),
        (4, 6, 2, 0, 0, 0, 61),
        (6, 6, 4, 2, 0, 0, 61),
        (6, 6, 4, 2, 0, 0, 61),
        (6, 6, 5, 2, 0, 0, 61),
        (6, 6, 4, 4, 3, 0, 122),
        (6, 6, 4, 5, 4, 2, 122),
    ]
    for context, row in enumerate(expected):
        env = Env()
        env.set_level(context)
        layout = extract(env)
        actual = (
            len(layout.program), layout.bit_widths[0], len(layout.presets),
            len(layout.walls), len(layout.platforms), len(layout.gates), env.clicks_left,
        )
        assert actual == row
        profile = REFERENCE_PROFILES[context + 1]
        assert (profile["slots"], profile["bit_width"], profile["selectors"],
                profile["walls"], profile["platforms"], profile["gates"],
                profile["native_budget"]) == row


def test_every_official_preset_tier_selects_a_locked_example_then_recovers():
    for context in range(1, 7):
        env = Env()
        env.set_level(context)
        layout = extract(env)
        assert layout.presets
        assert all(reset is False for reset in env.level.get_data("Reset"))
        x, y = layout.presets[-1].click
        env.perform(names.ACTION_CLICK, x, y)
        assert not env.controller.deredwcqze
        result = search(env, node_limit=400_000)
        assert result.solved and not result.truncated
        assert replay(env, result.actions)


def test_clone_is_isolated_and_live_prefix_replans_from_actual_bits():
    env = Env([official_levels()[0]])
    clone = env.clone()
    before = env.render()
    x, y = extract(clone).bit_clicks[1][0]
    clone.perform(names.ACTION_CLICK, x, y)
    assert clone.render() != before and env.render() == before
    result = search(clone, node_limit=50_000)
    assert result.solved and not result.truncated
    assert replay(clone, result.actions)


def test_bounds_and_cutoff_are_explicit_unknowns():
    capped = search(Env([official_levels()[0]]), node_limit=1)
    assert capped.truncated and capped.actions is None
    action_capped = search(Env([official_levels()[0]]), limit=1, node_limit=50_000)
    assert action_capped.truncated and action_capped.actions is None
    assert "unknown" in action_capped.reason
    for action in ((1, None, None), (6, None, 1), (6, -1, 0), (6, 0, 64)):
        try:
            Env([official_levels()[0]]).perform(*action)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid action {action}")


def test_every_generated_tier_validates_and_exercises_its_required_mechanics():
    expected = {
        1: {"translation"},
        2: {"translation", "preset_selection"},
        3: {"translation", "preset_selection", "collision_rollback"},
        4: {"translation", "preset_selection", "collision_rollback", "scale"},
        5: {"translation", "preset_selection", "scale", "rotation", "recolor"},
        6: {"translation", "preset_selection", "collision_rollback", "platform_checkpoint"},
        7: {"translation", "preset_selection", "platform_checkpoint", "gate_toggle"},
    }
    for difficulty in DIFFICULTIES:
        spec = generate(700 + difficulty, difficulty, split="train")
        assert spec is not None, generate.last_report
        entry = FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
        assert validate_full_standard(spec, entry) == []
        assert expected[difficulty] <= set(spec["mechanic_mechanics"]["events"])
        assert expected[difficulty] <= set(spec["solution_mechanics"]["events"])
        assert spec["solution_mechanics"]["first_win_action"] == len(spec["solution"]) - 1
        assert spec["mechanic_mechanics"]["first_win_action"] == len(spec["mechanic_solution"]) - 1
        assert replay(Env([build_level(spec)]), spec["solution"])
        if difficulty == 7:
            recovery_events = spec["recovery_mechanics"]["events"]
            assert recovery_events["gate_destroy"] >= 1
            assert recovery_events["failed_run_reset"] >= 1


def test_explicit_splits_derive_from_canonical_executable_gameplay():
    rows = {}
    for offset, split in enumerate(("train", "validation", "test")):
        spec = generate(800 + offset, 5, attempts=96, split=split)
        assert spec is not None, generate.last_report
        rows[split] = spec
        geometry, canonical, gameplay = identities(spec)
        assert spec["geometry_sha256"] == geometry
        assert spec["geometry_d4_sha256"] == canonical
        assert spec["gameplay_sha256"] == gameplay
        assert split_accepts(split, split_bucket(gameplay))
    assert len({row["gameplay_sha256"] for row in rows.values()}) == 3


def test_json_roundtrip_tamper_rejection_and_bank_compatibility():
    spec = generate(907, 7, split="train")
    assert spec is not None, generate.last_report
    stored = json.loads(json.dumps(spec))
    entry = FULL_STANDARD_CONTRACT["curriculum"][6]
    assert stored == spec and stored["format"] == FORMAT
    assert validate_full_standard(stored, entry) == []
    for mutation in ("target", "solution", "gameplay_sha256", "proof"):
        broken = copy.deepcopy(stored)
        if mutation == "target":
            broken[mutation][0] += 4
        elif mutation == "solution":
            broken[mutation][-1][1] -= 1
        elif mutation == "proof":
            broken[mutation]["context_engine_verified"] = False
        else:
            broken[mutation] = "0" * 64
        assert validate_full_standard(broken, entry), mutation

    broken_recovery = copy.deepcopy(stored)
    broken_recovery["proof"]["recovery"]["native_win"] = 1
    assert validate_full_standard(broken_recovery, entry)
    assert not ({"search_work", "search_expanded", "search_generated", "search_truncated"}
                & stored["proof"]["recovery"].keys())
    malformed_recovery = copy.deepcopy(stored)
    malformed_recovery["recovery_solution"] = None
    assert validate_full_standard(malformed_recovery, entry)
    forged_recovery_work = copy.deepcopy(stored)
    forged_recovery_work["proof"]["recovery"]["search_expanded"] = 1
    assert any("unsupported recovery-work claim" in error for error in
               validate_full_standard(forged_recovery_work, entry))
    float_recovery_count = copy.deepcopy(stored)
    float_recovery_count["proof"]["recovery"]["action_count"] = float(
        float_recovery_count["proof"]["recovery"]["action_count"])
    assert validate_full_standard(float_recovery_count, entry)

    with tempfile.TemporaryDirectory() as directory:
        specs, tried = build_bank(3, seed=30, difficulty=1)
        assert len(specs) == 3 and tried >= 3
        path = save_bank(specs, Path(directory) / "tn36.jsonl")
        assert load_bank(path) == specs


def test_generate_game_and_independently_composed_specs_replay_all_contexts():
    specs = generate_game(36, split="train", attempts=96)
    assert specs is not None, generate_game.last_report
    assert tuple(spec["difficulty"] for spec in specs) == DIFFICULTIES
    assert len(build_game(specs)) == 7

    independent = [generate(1_000 + d, d, split="train") for d in DIFFICULTIES]
    assert all(independent), [generate.last_report]
    assert len(build_game(independent)) == 7


def test_semantic_program_and_action_diversity_across_eight_seeds_per_tier():
    for difficulty in DIFFICULTIES:
        rows = [generate(2_000 + difficulty * 100 + seed, difficulty, split="train")
                for seed in range(8)]
        assert all(rows)
        program_signatures = {
            json.dumps(row["solution_mechanics"]["programs_executed"], separators=(",", ":"))
            for row in rows
        }
        action_signatures = {json.dumps(row["solution"], separators=(",", ":")) for row in rows}
        gameplay = {row["gameplay_sha256"] for row in rows}
        geometry = {row["geometry_d4_sha256"] for row in rows}
        assert len(program_signatures) >= (2 if difficulty == 7 else 3)
        assert len(action_signatures) >= (2 if difficulty == 7 else 6)
        assert len(gameplay) == 8
        # Semantic geometry may legitimately repeat when active programs differ.
        assert geometry


def test_shared_collector_uses_plan_search_entry_for_smoke_level():
    modules, = preflight(["tn36"])
    collected = collect_generated_game(
        modules, master_seed=36, game_index=0, difficulties=[1],
        outer_generation_attempts=3, generator_attempts=8,
    )
    assert collected.record["status"] == "won", collected.record["errors"]
    assert collected.record["levels_completed"] == 1


def test_public_teacher_proof_is_bound_to_search_and_mechanics_are_separate():
    for difficulty in DIFFICULTIES:
        spec = generate(700 + difficulty, difficulty, split="train")
        assert spec is not None, generate.last_report
        low, high = REFERENCE_PROFILES[difficulty]["actions"]
        assert low <= len(spec["solution"]) <= high
        assert spec["proof"]["action_count"] == len(spec["solution"])
        assert spec["proof"]["shortest_route_claimed"] is False
        assert spec["required_solution_events"] == list(REQUIRED_MECHANIC_EVENTS[difficulty])
        assert set(REQUIRED_MECHANIC_EVENTS[difficulty]) <= set(
            spec["mechanic_mechanics"]["events"]
        )
        independently_recomputed = search(
            _context_env(spec),
            limit=REFERENCE_PROFILES[difficulty]["native_budget"],
            node_limit=REFERENCE_PROFILES[difficulty]["search_work"],
            required_events=REQUIRED_MECHANIC_EVENTS[difficulty],
        )
        assert tuple(map(tuple, spec["solution"])) == independently_recomputed.actions
        assert spec["search_expanded"] == independently_recomputed.expanded
        for event in REQUIRED_MECHANIC_EVENTS[difficulty]:
            assert spec["solution_mechanics"]["events"].get(event, 0) >= 1, event


def _native_public_replay(spec):
    """Replay the public route on the native engine and record native facts.

    Returns the selector indices the engine reports after each selector click,
    one (program, native start, native end) row per program run, and whether
    the native level completed. Nothing here reads the stored certificates or
    the auxiliary construction witness.
    """
    env = Env([build_level(spec)])
    selected = []
    runs = []
    won = False
    for action_id, x, y in spec["solution"]:
        layout = extract(env)
        preset_clicks = [preset.click for preset in layout.presets]
        is_selector = (x, y) in preset_clicks
        is_run = layout.run_click == (x, y)
        program = tuple(layout.program)
        before = layout.current
        observation = env.perform(action_id, x, y)
        if env.levels_completed > 0 or observation.state == GameState.WIN:
            won = True
            if is_run:
                runs.append((program, before, tuple(spec["target"])))
            break
        if is_selector:
            selected.append(env.controller.jwmpcflifn)
        if is_run:
            runs.append((program, before, extract(env).current))
    return selected, runs, won


def test_every_tier_public_solution_exercises_required_events_natively():
    for difficulty in DIFFICULTIES:
        spec = generate(700 + difficulty, difficulty, split="train")
        assert spec is not None, generate.last_report
        required = REQUIRED_MECHANIC_EVENTS[difficulty]
        assert spec["required_solution_events"] == list(required)

        # Native replay of the public route with the event model cross-checked
        # against the settled engine after every run; this never consults the
        # auxiliary mechanic_solution witness.
        trace = _trace(spec, spec["solution"])
        for event in required:
            assert trace["events"].get(event, 0) >= 1, (difficulty, event, trace["events"])
        assert trace["wins"] == 1

        selected, runs, won = _native_public_replay(spec)
        assert won
        assert runs, "public route must execute at least one program run"
        if "preset_selection" in required:
            assert selected == trace["selector_indices"]
            assert len(selected) >= 1
            assert all(0 <= index < len(spec["preset_programs"]) for index in selected)
        else:
            assert selected == []
        if "collision_rollback" in required:
            # A wall must have rolled back at least one step in the native
            # engine: applying the same opcodes without obstacles lands
            # somewhere else than where the native actor actually ended.
            assert any(execute(before, program) != after for program, before, after in runs)
        start = tuple(spec["actor"])
        end = tuple(spec["target"])
        if "translation" in required:
            assert end[:2] != start[:2] or any(after[:2] != before[:2] for _, before, after in runs)
        if "scale" in required:
            assert end[3] != start[3] or any(after[3] != before[3] for _, before, after in runs)
        if "rotation" in required:
            assert end[2] != start[2] or any(after[2] != before[2] for _, before, after in runs)
        if "recolor" in required:
            assert end[4] != start[4] or any(after[4] != before[4] for _, before, after in runs)
        if "platform_checkpoint" in required:
            assert trace["events"].get("program_run", 0) >= 2


def test_constrained_search_adds_selector_and_rollback_without_changing_plain_search():
    env = Env()
    env.set_level(2)
    plain = search(env, limit=env.clicks_left, node_limit=400_000)
    constrained = search(
        env, limit=env.clicks_left, node_limit=400_000,
        required_events=REQUIRED_MECHANIC_EVENTS[3],
    )
    assert plain.solved and constrained.solved and not constrained.truncated
    assert len(plain.actions) == 9
    layout = extract(env)
    preset_clicks = [preset.click for preset in layout.presets]
    assert not any(action[1:] in preset_clicks for action in plain.actions)
    assert constrained.actions[0][1:] in preset_clicks
    assert len(constrained.actions) >= len(plain.actions) + 1
    # The official tier-3 min-click route avoids its walls; the constrained
    # route must natively bump one: unobstructed opcode application of the
    # executed program does not land on the target the engine reached.
    live = env.clone()
    programs = []
    for action in constrained.actions:
        current = extract(live)
        if current.run_click == action[1:]:
            programs.append((tuple(current.program), current.current))
        live.perform(*action)
    assert live.levels_completed == 1
    assert any(execute(before, program) != layout.target for program, before in programs)
    with pytest.raises(ValueError, match="unknown required events"):
        search(env, node_limit=1_000, required_events=("teleport",))


@pytest.mark.parametrize("split", ("train", "validation", "test"))
def test_generate_game_defaults_succeed_for_every_split(split):
    for seed in (1, 2, 3):
        specs = generate_game(seed, split=split)
        assert specs is not None, generate_game.last_report
        assert all(spec["generation_attempt"] <= DEFAULT_ATTEMPTS for spec in specs)
        assert all(spec["split"] == split for spec in specs)
        assert len(build_game(specs)) == 7


def test_identity_normalizes_presentation_but_keeps_program_and_gate_body_semantics():
    tutorial = generate(701, 1, split="train")
    assert tutorial is not None
    translated = copy.deepcopy(tutorial)
    translated["actor"][0] += 2
    translated["target"][0] += 2
    assert identities(translated)[0] != identities(tutorial)[0]
    assert identities(translated)[1:] == identities(tutorial)[1:]
    for color in (8, 9, 15):
        recolored = copy.deepcopy(tutorial)
        recolored["actor"][4] = recolored["target"][4] = color
        assert identities(recolored)[1:] == identities(tutorial)[1:]

    gated = generate(707, 7, split="train")
    assert gated is not None, generate.last_report
    changed_body = copy.deepcopy(gated)
    changed_body["gates"][0]["body_x"] -= 2
    assert identities(changed_body)[1:] != identities(gated)[1:]
    changed_program = copy.deepcopy(gated)
    changed_program["initial_program"][0] ^= 1
    assert identities(changed_program)[1] == identities(gated)[1]
    assert identities(changed_program)[2] != identities(gated)[2]


def test_translated_official_tutorial_is_excluded_by_executable_equivalence():
    spec = generate(701, 1, split="train")
    assert spec is not None
    env = Env()
    layout = extract(env)
    spec["actor"] = list(layout.initial)
    spec["target"] = list(layout.target)
    spec["actor"][0] += 1
    spec["target"][0] += 1
    spec["initial_program"] = list(layout.program)
    assert official_equivalence_identity(spec) in _official_equivalence_ids()
    geometry, canonical, gameplay = identities(spec)
    split = next(name for name in ("train", "validation", "test")
                 if split_accepts(name, split_bucket(gameplay)))
    spec.update(
        geometry_sha256=geometry, geometry_d4_sha256=canonical,
        gameplay_sha256=gameplay, split=split, geometry_split=split,
        split_partition_bucket=split_bucket(gameplay),
    )
    spec["proof"]["geometry_split"] = split
    errors = validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][0])
    assert any("duplicates an official tier" in error for error in errors)


def test_clone_preserves_selected_preset_checkpoint_reset_and_callback_ownership():
    env = Env()
    env.set_level(1)
    selector = extract(env).presets[-1].click
    env.perform(names.ACTION_CLICK, *selector)
    clone = env.clone()
    original_controller = env.controller
    cloned_controller = clone.controller
    assert cloned_controller.jwmpcflifn == original_controller.jwmpcflifn
    for original, copied in (
        (original_controller.mvqheosngn, cloned_controller.mvqheosngn),
        (original_controller.bzirenxmrg, cloned_controller.bzirenxmrg),
    ):
        assert copied.vupcwzjtxu.vkuvtkaerv == original.vupcwzjtxu.vkuvtkaerv
        assert copied.vupcwzjtxu.kviwnrvuri is original.vupcwzjtxu.kviwnrvuri
        assert (copied.fwrnsvyvrz, copied.bmhxacplut, copied.qixyeojolu,
                copied.fpofcohbab, copied.nzmblccilq) == (
            original.fwrnsvyvrz, original.bmhxacplut, original.qixyeojolu,
            original.fpofcohbab, original.nzmblccilq,
        )
    assert clone.render() == env.render()
    before = env.render()
    click = extract(clone).bit_clicks[0][0]
    clone.perform(names.ACTION_CLICK, *click)
    assert env.render() == before
    env.perform(names.ACTION_CLICK, *click)
    assert clone.render() == env.render()


def test_no_reset_runs_restart_from_saved_checkpoint_and_clone_preserves_extension_state():
    env = Env()
    env.set_level(1)
    env.controller.bzirenxmrg.vupcwzjtxu.iqqmvkctrb(False)
    layout = extract(env)
    assert layout.reset_after_run is False
    world = (layout.current, layout.initial, layout.actor_alive,
             tuple(gate.visible for gate in layout.gates))
    program = (2, 0, 0, 0)
    for run_index in range(2):
        live = extract(env)
        for action in program_actions(live, live.program, program):
            env.perform(*action)
        world, won, _ = execute_program(layout, world, program)
        assert not won
        native = extract(env)
        assert native.current == world[0]
        assert native.initial == world[1]
        if run_index == 0:
            clone = env.clone()
            cloned = extract(clone)
            assert cloned.current == native.current
            assert cloned.initial == native.initial
            assert cloned.reset_after_run is False
    assert extract(env).current[0] == layout.initial[0] + 4


def test_strict_schema_proof_caps_and_fixed_obligations_fail_closed():
    spec = generate(701, 1, split="train")
    assert spec is not None
    entry = FULL_STANDARD_CONTRACT["curriculum"][0]
    mutations = []
    mutations.append(lambda row: row["proof"].__setitem__("context_engine_verified", 1))
    mutations.append(lambda row: row.__setitem__("search_truncated", True))
    mutations.append(lambda row: row.__setitem__("search_limit", 1))
    mutations.append(lambda row: row["proof"].__setitem__(
        "context_engine_replay", {"levels_completed": 999}))
    mutations.append(lambda row: row.pop("seed"))
    mutations.append(lambda row: row.__setitem__("required_solution_events", []))
    mutations.append(lambda row: (
        row.__setitem__("search_expanded", 999_999_999),
        row["proof"].__setitem__("search_expanded", 999_999_999),
    ))
    mutations.append(lambda row: row.__setitem__("bit_width", 2.0))
    mutations.append(lambda row: row.__setitem__("native_budget", 61.0))
    mutations.append(lambda row: row.__setitem__("solution_mechanics", []))
    mutations.append(lambda row: row["proof"].__setitem__("program_runs", []))
    mutations.append(lambda row: row.__setitem__("generator_version", 4.0))
    mutations.append(lambda row: row["proof"]["context_engine_replay"].__setitem__(
        "levels_completed", True))
    mutations.append(lambda row: row.__setitem__("seed", 999))
    mutations.append(lambda row: row.__setitem__("generation_attempt", 10_001))
    mutations.append(lambda row: row["solution"].__setitem__(0, tuple(row["solution"][0])))
    for mutate in mutations:
        broken = copy.deepcopy(spec)
        mutate(broken)
        assert validate_full_standard(broken, entry)

    malformed_slots = copy.deepcopy(generate(702, 2, split="train"))
    malformed_slots["left_slot_count"] = 7
    with pytest.raises(ValueError, match="preset programs"):
        build_level(malformed_slots)
    malformed_primitive = copy.deepcopy(spec)
    malformed_primitive["walls"] = [{
        "sprite": [], "x": 33, "y": 4, "rotation": 0, "width": 1, "height": 1,
    }]
    with pytest.raises(ValueError, match="sprite must be a string"):
        build_level(malformed_primitive)


def test_selector_first_hit_access_and_duplicate_positions_fail_closed():
    # Tier 5 necessarily packs five nine-pixel selectors into the left panel;
    # partial overlap is valid because each ordered native object has a unique
    # first-hit pixel. Exact overlap makes the later selector unreachable.
    spec = generate(705, 5, split="train")
    assert spec is not None, generate.last_report
    entry = FULL_STANDARD_CONTRACT["curriculum"][4]
    assert validate_full_standard(spec, entry) == []
    duplicated = copy.deepcopy(spec)
    duplicated["selector_xs"][1] = duplicated["selector_xs"][0]
    with pytest.raises(ValueError, match="independently accessible"):
        build_level(duplicated)
    assert any("independently accessible" in error for error in
               validate_full_standard(duplicated, entry))


def test_tier_four_test_split_yield_regression_uses_unchanged_quality_band():
    seeds = (
        801886752782924456,
        3655502145692660390,
        8467496176440153150,
    )
    low, high = REFERENCE_PROFILES[4]["actions"]
    assert (low, high) == (8, 19)
    for seed in seeds:
        spec = generate(seed, 4, attempts=96, split="test")
        assert spec is not None, generate.last_report
        assert low <= len(spec["solution"]) <= high
        assert spec["generation_attempt"] <= 96
        assert validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][3]) == []


def test_bank_uses_tier_default_work_when_cap_is_omitted():
    for difficulty, seed in ((6, 706), (7, 707)):
        specs, tried = build_bank(1, seed=seed, difficulty=difficulty, max_seeds=1)
        assert len(specs) == 1 and tried == 1
    with pytest.raises(ValueError, match="cannot lower"):
        build_bank(1, seed=706, difficulty=6, max_seeds=1, node_limit=400_000)


def test_contract_is_ready_after_root_audit():
    assert DIFFICULTIES == tuple(range(1, len(official_levels()) + 1))
    assert FULL_STANDARD_CONTRACT["status"] == "ready"
    assert [row["context_index"] for row in FULL_STANDARD_CONTRACT["curriculum"]] == list(range(7))
