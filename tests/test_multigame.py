"""Shared multi-game registry, collection, and data-boundary tests."""

import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pebby import multigame as M
from tools import collect_multigame_games as collector


class _ToyEnv:
    """Small deterministic real-state stand-in for mixed rollout tests."""

    def __init__(self, levels):
        self.levels = list(levels)
        self.reset()

    def reset(self):
        self.state = "NOT_FINISHED"
        self.level_index = 0
        self.levels_completed = 0
        self.actions = 0

    @property
    def available_actions(self):
        return (1, 2)

    def render(self):
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[0, 0] = self.levels_completed
        frame[0, 1] = self.actions % 16
        return frame

    def perform(self, action_id, x=None, y=None):
        self.actions += 1
        if action_id == 1:
            self.levels_completed += 1
            if self.levels_completed == len(self.levels):
                self.state = "WIN"
            else:
                self.level_index = self.levels_completed

    def clone(self):
        cloned = object.__new__(type(self))
        cloned.levels = list(self.levels)
        cloned.state = self.state
        cloned.level_index = self.level_index
        cloned.levels_completed = self.levels_completed
        cloned.actions = self.actions
        return cloned


class _RenderFailsAfterAction(_ToyEnv):
    def render(self):
        if self.actions:
            raise RuntimeError("successor render failed")
        return super().render()


def _toy_modules(*, search=None, env_cls=_ToyEnv):
    def generate(seed, difficulty):
        return {
            "seed": seed,
            "difficulty": difficulty,
            "coverage": {"toy": True},
            "omitted_mechanics": ["none invented"],
            "generator_notes": "test declaration",
            "limitations": ["test only"],
        }

    if search is None:
        def search(env, limit):
            return SimpleNamespace(
                actions=[(1, None, None)], truncated=False, unsupported=False,
                exact=True, expanded=1, reason="toy exact plan",
            )

    return M.GameModules(
        source=M.source_for("cd82"),
        env=SimpleNamespace(Env=env_cls),
        generate=SimpleNamespace(generate=generate, build_level=lambda spec: dict(spec)),
        plan=SimpleNamespace(search=search),
        solver_adapter="work_limit",
    )


def _full_toy_modules(
    *, solution=((1, None, None),), solutions=None, search=None, env_cls=_ToyEnv,
):
    raw_solutions = (solution,) if solutions is None else tuple(solutions)
    stored_solutions = tuple(
        [list(action) for action in tier_solution]
        for tier_solution in raw_solutions
    )

    def generate(seed, difficulty, *, split):
        stored_solution = stored_solutions[difficulty - 1]
        return {
            "seed": seed,
            "difficulty": difficulty,
            "split": split,
            "solution": stored_solution,
            "solution_length": len(stored_solution),
        }

    if search is None:
        def search(env, limit):
            return SimpleNamespace(
                actions=[(1, None, None)], truncated=False, unsupported=False,
                exact=True, expanded=1, reason="toy live recovery",
            )

    source = M.source_for("cd82")
    curriculum = tuple(
        M.CurriculumEntry(difficulty, difficulty - 1, 10)
        for difficulty in range(1, len(stored_solutions) + 1)
    )
    contract = M.FullStandardContract(
        source.source_id,
        "ready",
        "fixture-mechanics",
        "fixture-quality",
        len(stored_solutions),
        curriculum,
        tuple((key, f"fixture-{key}") for key in M.FULL_STANDARD_EVIDENCE),
        ("synthetic certified-route fixture",),
    )
    generator = SimpleNamespace(
        DIFFICULTIES=tuple(range(1, len(stored_solutions) + 1)),
        generate=generate,
        generate_game=lambda seed, *, split, difficulties=None: [],
        build_level=lambda spec: dict(spec),
        build_game=lambda specs: [dict(spec) for spec in specs],
        validate_full_standard=lambda spec, entry: [],
    )
    return M.GameModules(
        source=source,
        env=SimpleNamespace(Env=env_cls),
        generate=generator,
        plan=SimpleNamespace(search=search),
        solver_adapter="work_limit",
        full_standard=contract,
    )


@pytest.fixture(scope="module")
def real_collections():
    results = {}
    for slug in ("cd82", "ft09", "tr87"):
        modules = M.preflight([slug])[0]
        results[slug] = M.collect_generated_game(
            modules,
            master_seed=17,
            game_index=0,
            difficulties=(1, 2),
            limits=M.SearchLimits(max_actions_per_level=256, max_search_work=2_000_000),
            outer_generation_attempts=3,
            generator_attempts=20,
        )
    return results


def test_registry_has_all_sources_and_never_collects_heldout():
    assert len(M.SOURCES) == 25
    assert len(M.TRAIN_SOURCES) == 24
    assert {source.source_id for source in M.TRAIN_SOURCES} == set(M.TRAIN_SOURCE_IDS)
    assert M.HELD_OUT_SOURCE_ID not in M.TRAIN_SOURCE_IDS
    assert M.source_for("m0r0").held_out
    with pytest.raises(ValueError, match="held-out"):
        M.resolve_collection_sources(["m0r0"])
    with pytest.raises(ValueError, match="held-out"):
        M.resolve_collection_sources([M.HELD_OUT_SOURCE_ID])


def test_custom_source_object_is_canonicalized_and_cannot_bypass_holdout():
    forged = M.Source("m0r0", M.HELD_OUT_SOURCE_ID, "forged", held_out=False)
    assert M.source_for(forged) is M.SOURCE_BY_SLUG["m0r0"]
    with pytest.raises(ValueError, match="held-out"):
        M.resolve_collection_sources([forged])
    mismatched = M.Source("cd82", M.HELD_OUT_SOURCE_ID, "bad")
    with pytest.raises(ValueError, match="does not match"):
        M.source_for(mismatched)
    with pytest.raises(ValueError, match="held-out"):
        M.new_manifest(
            sources=[forged], explicit_subset=True, seed=0, games_per_source=1,
            difficulties=(1,), limits=M.SearchLimits(),
        )


def test_default_preflight_requires_all_24_before_output(tmp_path, monkeypatch):
    def one_missing(values=M.SOURCES):
        sources = tuple(M.source_for(value) for value in values)
        return tuple(
            M.Availability(
                source_id=source.source_id,
                slug=source.slug,
                ready=index != 0,
                    missing=(f"{source.package}.generate",) if index == 0 else (),
                    solver_adapter=None if index == 0 else "work_limit",
                    full_standard_ready=index != 0,
                )
            for index, source in enumerate(sources)
        )

    monkeypatch.setattr(M, "inspect_availability", one_missing)
    out = tmp_path / "must-not-exist"
    with pytest.raises(M.PreflightError) as caught:
        collector.main(["--out-dir", str(out), "--quiet"])
    assert len(caught.value.availability) == 1
    assert "default 24-game full-standard preflight rejected" in str(caught.value)
    assert not out.exists()


def test_known_packages_report_deliberate_solver_limit_adapters():
    expected = {
        "cd82": "work_limit",
        "ft09": "work_limit",
        "tr87": "action_limit+node_limit",
        "s5i5": "action_limit+max_nodes",
        "wa30": "work_limit+action_budget",
    }
    for slug, adapter in expected.items():
        status = M.inspect_source(slug)
        assert status.ready, status
        assert status.solver_adapter == adapter


def test_full_contract_matches_actual_official_count_and_requires_two_generation_modes():
    source = M.source_for("cd82")
    env = SimpleNamespace(official_levels=lambda: [object(), object()])

    def one(seed, difficulty, *, split):
        return {"seed": seed, "difficulty": difficulty, "split": split}

    def whole(seed, *, split, difficulties=None):
        return []

    base = {
        "format": M.FULL_STANDARD_FORMAT,
        "status": "ready",
        "source_id": source.source_id,
        "mechanics_inventory_version": "fixture-mechanics",
        "quality_profile_version": "fixture-quality",
        "curriculum": [
            {"difficulty": 1, "context_index": 0, "search_work": 10},
            {"difficulty": 2, "context_index": 1, "search_work": 20},
        ],
        "evidence": {key: f"fixture-{key}" for key in M.FULL_STANDARD_EVIDENCE},
        "caveats": ["synthetic contract parser fixture"],
    }
    built = []

    def build_game(specs):
        built.append(tuple(specs))
        return list(specs)

    generator = SimpleNamespace(
        DIFFICULTIES=(1, 2), FULL_STANDARD_CONTRACT=base,
        generate=one, generate_game=whole,
        build_game=build_game,
        build_level=lambda spec: dict(spec),
        validate_full_standard=lambda spec, entry: [],
    )
    contract = M._full_standard_contract(source, env, generator)
    assert contract.official_level_count == 2
    assert contract.to_dict()["official_level_count"] == 2
    modules = M.GameModules(
        source, SimpleNamespace(Env=_ToyEnv), generator, SimpleNamespace(),
        "work_limit", full_standard=contract,
    )
    game = M.MultiGameEnv.from_specs(
        modules, [{"difficulty": 1}, {"difficulty": 2}], require_full_standard=True,
    )
    assert game.level_count == 2 and len(built) == 1

    generator.DIFFICULTIES = (1,)
    with pytest.raises(ValueError, match="official level count"):
        M._full_standard_contract(source, env, generator)
    generator.DIFFICULTIES = (True, 2)
    with pytest.raises(ValueError, match="official level count"):
        M._full_standard_contract(source, env, generator)
    generator.DIFFICULTIES = (1, 2)
    generator.FULL_STANDARD_CONTRACT = {
        **base,
        "curriculum": [
            {"difficulty": 1, "context_index": 0, "search_work": 10},
            {"difficulty": 2, "context_index": 0, "search_work": 20},
        ],
    }
    with pytest.raises(ValueError, match="contexts"):
        M._full_standard_contract(source, env, generator)

    generator.FULL_STANDARD_CONTRACT = base
    del generator.generate_game
    with pytest.raises(ValueError, match="generate_game"):
        M._full_standard_contract(source, env, generator)


def _assert_file_identity(identity):
    path = Path(identity["path"])
    if not path.is_absolute():
        path = M.ROOT / path
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == identity["sha256"]


@pytest.mark.parametrize(
    "slug,vendored_path",
    (
        ("ls20", "third_party/ls20/ls20.py"),
        ("bp35", "third_party/arc3_games/bp35.py"),
        ("ar25", "third_party/arc3_games/ar25.py"),
    ),
)
def test_production_preflight_binds_private_provenance_to_actual_bytes(slug, vendored_path):
    modules = M.preflight([slug])[0]
    provenance = modules.provenance
    assert provenance["format"] == M.PROVENANCE_FORMAT
    assert provenance["status"] == "available"
    assert provenance["source_id"] == modules.source.source_id
    assert provenance["vendored_game"]["path"] == vendored_path
    _assert_file_identity(provenance["vendored_game"])

    family = provenance["family_modules"]
    assert f"pebby.games.{slug}.env" in family
    assert f"pebby.games.{slug}.generate" in family
    assert f"pebby.games.{slug}.plan" in family
    for identity in family.values():
        _assert_file_identity(identity)
    if slug == "ls20":
        assert "pebby.ls20.reference_generator_v2" in family

    shared = provenance["shared_modules"]
    assert set(shared) == {
        "pebby.multigame",
        "pebby.multigame_variants",
        "tools.collect_multigame_games",
    }
    for identity in shared.values():
        _assert_file_identity(identity)
    installed = provenance["installed_packages"]
    assert installed["arcengine"]["distribution_version"] == importlib_metadata.version("arcengine")
    assert installed["arc-agi"]["distribution_version"] == importlib_metadata.version("arc-agi")

    # Preflight callers receive independent documents rather than the mutable
    # cached source object used for subsequent records and manifests.
    provenance["status"] = "tampered"
    assert M.preflight([slug])[0].provenance["status"] == "available"


def test_synthetic_modules_mark_provenance_unavailable_without_public_leakage():
    collected = M.collect_generated_game(
        _toy_modules(),
        master_seed=1,
        game_index=0,
        difficulties=(1,),
        limits=M.SearchLimits(4, 10),
        outer_generation_attempts=1,
        generator_attempts=1,
    )
    assert collected.record["status"] == "won"
    provenance = collected.record["provenance"]
    assert provenance["format"] == M.PROVENANCE_FORMAT
    assert provenance["status"] == "unavailable"
    assert "without production preflight" in provenance["reason"]
    assert "provenance" not in collected.public
    assert "provenance" not in collected.teacher


def test_production_preflight_rejects_an_unusable_code_identity(monkeypatch):
    def broken(_source_id):
        raise OSError("unreadable source bytes")

    monkeypatch.setattr(M, "_production_provenance", broken)
    with pytest.raises(M.PreflightError, match="unusable code provenance") as caught:
        M.preflight(["cd82"])
    assert "unreadable source bytes" in caught.value.availability[0].errors[0]


@pytest.mark.parametrize("slug,seed", (("ls20", 0), ("bp35", 3), ("ar25", 0)))
def test_level_count_is_correct_for_official_generated_reset_and_clone(slug, seed):
    modules = M.preflight([slug])[0]
    official = modules.env.Env()
    assert official.level_count == len(modules.env.official_levels())
    assert official.clone().level_count == official.level_count

    spec = modules.generate.generate(seed, 1)
    assert spec is not None
    generated = modules.env.Env([
        modules.generate.build_level(spec),
        modules.generate.build_level(spec),
    ])
    assert generated.level_count == 2
    assert generated.clone().level_count == 2
    generated.set_level(1)
    assert generated.level_index == 1 and generated.level_count == 2
    assert generated.clone().level_count == 2
    generated.reset()
    assert generated.level_index == 0 and generated.level_count == 2


@pytest.mark.parametrize(
    "slug,required",
    (
        ("sp80", ("vertical", "cyclic", "inconclusive")),
        ("sk48", ("crossing", "official level 1", "inconclusive")),
        ("bp35", ("remote hazards", "undo branches", "inconclusive")),
        ("ka59", ("player routing", "explosives", "official level 1")),
        ("ar25", ("fixed horizontal or vertical mirror", "recursive mirrors", "not claimed")),
        ("vc33", ("both gravity axes", "swap bars", "official level 1")),
        ("su15", ("magnetic full-pixel movement", "equal-tier merging", "pursuers", "inconclusive")),
        ("dc22", ("movement-only perfect-tree mazes", "keys/colour gates", "crushers", "unsupported")),
        ("lp85", ("disjoint bidirectional rectangle cycles", "stacked multi-group controls", "inconclusive")),
        ("re86", ("independently translating rigid shapes", "dye/flood fill", "restored intrinsic selection-marker centres", "unsupported")),
        ("lf52", ("static single-colour peg jumps", "scripted trap/reset cells", "live-history", "truncated")),
        ("sb26", ("one ordinary 3/5/7-cell frame", "recursive link tiles", "official level 1", "inconclusive")),
        ("tn36", ("one editable program", "destructive/toggling gates", "manual dual-panel execution", "official level 1")),
        ("g50t", ("randomized induced-tree regions", "dead-end pressure switch", "teleports", "truncated/inconclusive")),
    ),
)
def test_reviewed_coverage_notes_preserve_generator_limits(slug, required):
    note = M.KNOWN_COVERAGE[slug]
    assert all(fragment in note for fragment in required)


def test_all_training_families_have_reviewed_coverage_notes():
    train_slugs = {source.slug for source in M.TRAIN_SOURCES}
    assert set(M.KNOWN_COVERAGE) == train_slugs
    assert M.SOURCE_BY_SLUG["m0r0"].held_out
    assert "m0r0" not in M.KNOWN_COVERAGE


def test_action_objects_and_triples_are_validated_canonically():
    assert M.Action.parse((6, 1, 2)) == M.Action(6, 1, 2)
    assert M.Action.parse(M.Action(4, None, None)) == M.Action(4, None, None)
    for bad in (M.Action(6, 1.5, 2), M.Action(True, None, None), 3, (1, None)):
        with pytest.raises(M.IllegalActionError):
            M.Action.parse(bad)


@pytest.mark.parametrize("slug", ("cd82", "ft09", "tr87"))
def test_two_generated_levels_are_solved_from_each_live_real_engine_state(slug, real_collections):
    collected = real_collections[slug]
    assert collected.record["status"] == "won", collected.record["errors"]
    assert collected.record["levels_generated"] == 2
    assert collected.record["levels_completed"] == 2
    assert collected.record["final_state"] == "WIN"
    assert collected.record["level_boundaries"] == 2
    assert collected.record["random_steps"] == 0
    assert collected.record["rollout_mode"] == "teacher_only"
    assert not collected.record["random_transition_scoring_available"]

    public, teacher = collected.public, collected.teacher
    assert public is not None and teacher is not None
    steps = collected.record["steps"]
    assert public["frames"].shape == (steps + 1, 64, 64)
    assert public["frames"].dtype == np.uint8
    assert public["legal_action_mask"].shape == (steps + 1, 8)
    assert public["action_id"].shape == (steps,)
    assert teacher["target_action_id"].shape == (steps,)
    assert np.array_equal(public["action_id"], teacher["target_action_id"])
    assert np.all(teacher["source"] == M.TEACHER_SOURCE)
    assert public["terminal"][-1] and public["won"][-1]
    assert not public["legal_action_mask"][-1].any()
    assert np.all(public["legal_action_mask"][np.arange(steps), public["action_id"]])
    assert int(public["level_boundary"].sum()) == 2
    assert "source" not in public and "target_action_id" not in public
    assert all(search["level_index"] == index for index, search in enumerate(collected.record["searches"]))


def test_public_facade_exposes_render_state_progress_and_strict_triples():
    modules = M.preflight(["tr87"])[0]
    spec = modules.generate.generate(0, 1)
    game = M.MultiGameEnv.from_specs(modules, [spec])
    frame = game.reset()
    assert frame.shape == game.render_shape == (64, 64)
    assert game.state == "NOT_FINISHED"
    assert game.progress.level_index == 0
    assert game.progress.levels_completed == 0
    with pytest.raises(M.IllegalActionError, match="triple"):
        game.perform(3)
    with pytest.raises(M.IllegalActionError, match="coordinates"):
        game.perform((3, 4, 5))
    with pytest.raises(M.IllegalActionError, match="unavailable"):
        game.perform((6, 1, 1))


def test_generation_failure_is_a_status_with_error_and_no_fake_arrays():
    real = M.preflight(["cd82"])[0]

    def always_none(seed, difficulty, attempts=1, limit=1):
        return None

    modules = M.GameModules(
        source=real.source,
        env=real.env,
        generate=SimpleNamespace(generate=always_none, build_level=lambda spec: None),
        plan=real.plan,
        solver_adapter=real.solver_adapter,
    )
    collected = M.collect_generated_game(
        modules,
        master_seed=0,
        game_index=0,
        difficulties=(1,),
        limits=M.SearchLimits(10, 10),
        outer_generation_attempts=2,
        generator_attempts=1,
    )
    assert collected.record["status"] == "generation_failed"
    assert collected.record["levels_generated"] == 0
    assert "generator returned None" in collected.record["errors"][0]
    assert collected.public is None and collected.teacher is None


def test_teacher_plan_that_does_not_complete_never_force_advances_level():
    real = M.preflight(["tr87"])[0]
    spec = real.generate.generate(0, 1)

    def fixed_generator(seed, difficulty):
        return dict(spec)

    def one_nonwinning_move(env, limit):
        return SimpleNamespace(
            actions=[(3, None, None)], truncated=False, expanded=1, reason="test nonwinning plan"
        )

    modules = M.GameModules(
        source=real.source,
        env=real.env,
        generate=SimpleNamespace(generate=fixed_generator, build_level=real.generate.build_level),
        plan=SimpleNamespace(search=one_nonwinning_move),
        solver_adapter="work_limit",
    )
    collected = M.collect_generated_game(
        modules,
        master_seed=0,
        game_index=0,
        difficulties=(1,),
        limits=M.SearchLimits(10, 100),
        outer_generation_attempts=1,
        generator_attempts=1,
    )
    assert collected.record["status"] == "rollout_failed"
    assert collected.record["levels_completed"] == 0
    assert collected.record["level_boundaries"] == 0
    assert "without engine completion" in collected.record["errors"][0]
    assert np.all(collected.public["level_index"] == 0)
    assert np.all(collected.public["levels_completed"] == 0)


def test_wa30_adapter_never_extends_the_native_action_budget():
    captured = {}

    def search(env, limit=1, budget=None):
        captured.update(limit=limit, budget=budget)
        return SimpleNamespace(
            actions=[(1, None, None)], truncated=False, expanded=1, reason="test"
        )

    game = object.__new__(M.MultiGameEnv)
    game.modules = M.GameModules(
        source=M.source_for("wa30"), env=SimpleNamespace(), generate=SimpleNamespace(),
        plan=SimpleNamespace(search=search), solver_adapter="work_limit+action_budget",
    )
    game.source = M.source_for("wa30")
    game._env = SimpleNamespace(steps_left=lambda: 7)
    result = M.solve_live_level(game, M.SearchLimits(512, 1234))
    assert result.actions
    assert captured == {"limit": 1234, "budget": 7}


def test_manifest_roundtrip_and_public_teacher_files_are_separate(tmp_path, real_collections):
    modules = M.preflight(["cd82"])
    manifest = M.new_manifest(
        sources=[modules[0].source], explicit_subset=True, seed=17, games_per_source=1,
        difficulties=(1, 2), limits=M.SearchLimits(256, 2_000_000),
    )
    record = M.save_collected_game(tmp_path, "cd82-000000", real_collections["cd82"])
    manifest["records"].append(record)
    M.update_manifest_summary(manifest)
    path = M.save_manifest(tmp_path / "manifest.json", manifest)
    assert M.load_manifest(path) == manifest
    provenance = manifest["provenance"]
    assert provenance["status"] == "available"
    assert set(provenance["sources"]) == {modules[0].source.source_id}
    source_provenance = provenance["sources"][modules[0].source.source_id]
    _assert_file_identity(source_provenance["vendored_game"])
    assert record["provenance"]["status"] == "available"
    assert record["provenance"]["source_id"] == modules[0].source.source_id
    assert record["public_npz"] != record["teacher_npz"]
    with np.load(tmp_path / record["public_npz"], allow_pickle=False) as public:
        assert "frames" in public and "target_action_id" not in public
        assert "provenance" not in public
    with np.load(tmp_path / record["teacher_npz"], allow_pickle=False) as teacher:
        assert "target_action_id" in teacher and "frames" not in teacher


def test_collector_refuses_to_overwrite_a_populated_output(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("owned by an earlier run")
    with pytest.raises(ValueError, match="not empty"):
        collector.main(["--games", "cd82", "--difficulties", "1", "--out-dir", str(out), "--quiet"])
    assert marker.read_text() == "owned by an earlier run"
    assert not (out / "manifest.json").exists()


def test_collector_rejects_search_ceiling_below_full_tier_before_output(tmp_path, monkeypatch):
    source = M.source_for("cd82")
    contract = M.FullStandardContract(
        source.source_id,
        "ready",
        "fixture-mechanics",
        "fixture-quality",
        1,
        (M.CurriculumEntry(1, 0, 10),),
        tuple((key, f"fixture-{key}") for key in M.FULL_STANDARD_EVIDENCE),
        ("synthetic collector ceiling fixture",),
    )
    modules = M.GameModules(
        source,
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        "work_limit",
        full_standard=contract,
    )

    def preflight(_games=None, *, require_full_standard=False):
        assert require_full_standard
        return (modules,)

    monkeypatch.setattr(collector, "preflight", preflight)
    out = tmp_path / "ceiling-rejected"
    with pytest.raises(ValueError, match="below a required family tier cap"):
        collector.main([
            "--out-dir", str(out), "--max-search-work", "5", "--quiet",
        ])
    assert not out.exists()


def test_generator_declarations_and_solver_exactness_metadata_are_preserved():
    modules = _toy_modules()
    specs, levels = M.generate_levels(
        modules, master_seed=1, game_index=0, difficulties=(1,), outer_attempts=1,
        generator_attempts=1, search_work_limit=10,
    )
    assert specs
    declared = levels[0]["declared_coverage"]
    assert set(declared) == {"coverage", "omitted_mechanics", "generator_notes", "limitations"}

    game = M.MultiGameEnv.from_specs(modules, specs)
    game.reset()
    result = M.solve_live_level(game, M.SearchLimits(4, 10))
    assert result.exact is True and result.unsupported is False


def test_fixed_variant_teacher_trace_inverse_replays_raw_engine_across_levels():
    from pebby.multigame_variants import VariantOptions, WholeGameVariant

    modules = M.preflight(["cd82"])[0]
    collected = M.collect_generated_game(
        modules,
        master_seed=27,
        game_index=2,
        difficulties=(1, 2),
        limits=M.SearchLimits(128, 200_000),
        outer_generation_attempts=2,
        generator_attempts=10,
        variants=VariantOptions(enabled=True, seed=91),
    )
    assert collected.record["status"] == "won", collected.record["errors"]
    metadata = collected.record["variant"]
    variant = WholeGameVariant(
        metadata["selected"], metadata["seed"], metadata["control_raw_to_public"],
        metadata["spatial"], metadata["palette_raw_to_public"],
    )
    game = M.MultiGameEnv.from_specs(modules, collected.specs)
    game.reset()
    public = collected.public
    assert public is not None
    for step, action_id in enumerate(public["action_id"]):
        assert np.array_equal(public["frames"][step], variant.public_frame(game.render()))
        assert np.array_equal(public["legal_action_mask"][step], M.legal_mask(game, variant))
        x = int(public["action_x"][step])
        y = int(public["action_y"][step])
        raw = variant.raw_action(
            int(action_id), None if x < 0 else x, None if y < 0 else y,
        )
        game.perform(raw)
    assert game.progress.won
    assert np.array_equal(public["frames"][-1], variant.public_frame(game.render()))
    assert int(public["level_boundary"].sum()) == 2


def test_full_standard_uses_certified_raw_route_and_variant_not_alternate_search():
    from pebby.multigame_variants import VariantOptions, WholeGameVariant

    def forbidden_search(env, limit):
        raise AssertionError("pristine full-standard collection must not search another teacher")

    modules = _full_toy_modules(
        solutions=(((1, None, None),), ((1, None, None),)),
        search=forbidden_search,
    )
    collected = M.collect_generated_game(
        modules,
        master_seed=0,
        game_index=0,
        difficulties=(1, 2),
        split="train",
        require_full_standard=True,
        limits=M.SearchLimits(4, 10),
        outer_generation_attempts=1,
        generator_attempts=1,
        variants=VariantOptions(
            enabled=True, seed=0, controls=True, spatial=False, palette=False,
        ),
    )
    assert collected.record["status"] == "won", collected.record["errors"]
    assert collected.record["certified_solution_completed"]
    assert collected.record["certified_teacher_actions"] == 2
    assert collected.record["live_recovery_teacher_actions"] == 0
    assert collected.record["certified_route_levels_completed"] == 2
    assert collected.teacher["route_source"].tolist() == [
        M.CERTIFIED_ROUTE_SOURCE, M.CERTIFIED_ROUTE_SOURCE,
    ]
    assert len(collected.record["searches"]) == 2
    for level, plan in enumerate(collected.record["searches"]):
        assert plan["level_index"] == plan["context_index"] == level
        assert plan["difficulty"] == level + 1
        assert plan["trigger"] == "level_start"
        assert plan["adapter"] == plan["route_source"] == M.CERTIFIED_ROUTE_SOURCE
        assert plan["actions"] == 1 and plan["work"] is None
        assert plan["exact"] and not plan["truncated"] and not plan["unsupported"]

    metadata = collected.record["variant"]
    variant = WholeGameVariant(
        metadata["selected"], metadata["seed"], metadata["control_raw_to_public"],
        metadata["spatial"], metadata["palette_raw_to_public"],
    )
    assert collected.public["action_id"].tolist() == [2, 2]
    public_action = (
        int(collected.public["action_id"][0]), None, None,
    )
    assert variant.raw_action(*public_action) == (1, None, None)


def test_full_standard_random_perturbation_discards_certificate_and_labels_recovery(monkeypatch):
    searches = 0

    def recover(env, limit):
        nonlocal searches
        searches += 1
        return SimpleNamespace(
            actions=[(1, None, None)], truncated=False, unsupported=False,
            exact=True, expanded=1, reason="actual-state recovery",
        )

    def random_noop(game, variant, rng):
        raw = game.validate_action(M.Action(2))
        return M.Action(*variant.public_action(*raw.as_tuple())), raw

    monkeypatch.setattr(M, "_sample_random_action", random_noop)
    collected = M.collect_generated_game(
        _full_toy_modules(search=recover),
        master_seed=2,
        game_index=0,
        difficulties=(1,),
        split="validation",
        require_full_standard=True,
        limits=M.SearchLimits(4, 10),
        outer_generation_attempts=1,
        generator_attempts=1,
        rollout=M.RolloutOptions(random_action_probability=0.5, max_game_steps=4),
    )
    assert collected.record["status"] == "won", collected.record["errors"]
    assert searches == 1
    assert collected.teacher["route_source"].tolist() == [
        M.RANDOM_ROUTE_SOURCE, M.RECOVERY_ROUTE_SOURCE,
    ]
    assert [item["route_source"] for item in collected.record["searches"]] == [
        M.CERTIFIED_ROUTE_SOURCE, M.RECOVERY_ROUTE_SOURCE,
    ]
    assert collected.record["certified_teacher_actions"] == 0
    assert collected.record["live_recovery_teacher_actions"] == 1
    assert not collected.record["certified_solution_completed"]
    assert not collected.record["actual_trajectory_coverage"][
        "stored_mechanic_certificate_inherited_by_recovery"
    ]


def test_random_action_that_completes_level_is_never_counted_as_certified(monkeypatch):
    def random_win(game, variant, rng):
        raw = game.validate_action(M.Action(1))
        return M.Action(*variant.public_action(*raw.as_tuple())), raw

    monkeypatch.setattr(M, "_sample_random_action", random_win)
    collected = M.collect_generated_game(
        _full_toy_modules(),
        master_seed=0,
        game_index=0,
        difficulties=(1,),
        split="test",
        require_full_standard=True,
        limits=M.SearchLimits(4, 10),
        outer_generation_attempts=1,
        generator_attempts=1,
        rollout=M.RolloutOptions(random_action_probability=1.0, max_game_steps=2),
    )
    assert collected.record["status"] == "won", collected.record["errors"]
    assert collected.teacher["route_source"].tolist() == [M.RANDOM_ROUTE_SOURCE]
    assert collected.record["random_steps"] == 1
    assert collected.record["certified_teacher_actions"] == 0
    assert collected.record["certified_route_levels_completed"] == 0
    assert not collected.record["certified_solution_completed"]


def test_full_standard_rejects_post_win_certificate_without_search_or_partial_trace():
    searches = 0

    def forbidden_search(env, limit):
        nonlocal searches
        searches += 1
        raise AssertionError("invalid certificates must not be silently replaced")

    collected = M.collect_generated_game(
        _full_toy_modules(solution=((1, None, None), (2, None, None)), search=forbidden_search),
        master_seed=0,
        game_index=0,
        difficulties=(1,),
        split="train",
        require_full_standard=True,
        limits=M.SearchLimits(4, 10),
        outer_generation_attempts=1,
        generator_attempts=1,
    )
    assert collected.record["status"] == "rollout_failed"
    assert searches == 0
    assert "completes before its final action" in collected.record["errors"][0]
    assert collected.record["steps"] == 0
    assert collected.record["searches"] == []


@pytest.mark.parametrize(
    "solution,error",
    (
        (((True, None, None),), "malformed action triples"),
        (((6, 64, 0),), "native preflight"),
    ),
)
def test_full_standard_rechecks_strict_stored_action_triples(solution, error):
    class ClickToyEnv(_ToyEnv):
        @property
        def available_actions(self):
            return (1, 2, 6)

    collected = M.collect_generated_game(
        _full_toy_modules(solution=solution, env_cls=ClickToyEnv),
        master_seed=0,
        game_index=0,
        difficulties=(1,),
        split="train",
        require_full_standard=True,
        limits=M.SearchLimits(4, 10),
        outer_generation_attempts=1,
        generator_attempts=1,
    )
    assert collected.record["status"] == "rollout_failed"
    assert error in collected.record["errors"][0]
    assert collected.record["steps"] == 0


def test_variant_and_mixed_options_cannot_bypass_heldout_collection():
    from pebby.multigame_variants import VariantOptions

    heldout = M.GameModules(
        source=M.source_for("m0r0"), env=SimpleNamespace(), generate=SimpleNamespace(),
        plan=SimpleNamespace(), solver_adapter="work_limit",
    )
    with pytest.raises(ValueError, match="held-out"):
        M.collect_generated_game(
            heldout, master_seed=0, game_index=0, difficulties=(1,),
            variants=VariantOptions(enabled=True),
            rollout=M.RolloutOptions(random_action_probability=1.0, max_game_steps=2),
        )


def test_controlled_random_rollout_replans_and_preserves_true_boundaries(monkeypatch):
    sampled = iter((2, 1, 2, 1))

    def scripted_random(game, variant, rng):
        raw_id = next(sampled)
        raw = game.validate_action(M.Action(raw_id))
        return M.Action(*variant.public_action(raw.id, raw.x, raw.y)), raw

    monkeypatch.setattr(M, "_sample_random_action", scripted_random)
    collected = M.collect_generated_game(
        _toy_modules(),
        master_seed=3,
        game_index=0,
        difficulties=(1, 2),
        limits=M.SearchLimits(4, 20),
        outer_generation_attempts=1,
        generator_attempts=1,
        rollout=M.RolloutOptions(random_action_probability=1.0, max_game_steps=8),
    )
    assert collected.record["status"] == "won", collected.record["errors"]
    assert collected.record["teacher_steps"] == 0
    assert collected.record["random_steps"] == 4
    assert collected.record["random_metric_eligible_steps"] == 4
    assert collected.record["random_transition_scoring_available"]
    assert collected.public["action_id"].tolist() == [2, 1, 2, 1]
    assert collected.public["level_boundary"].tolist() == [False, True, False, True]
    assert collected.teacher["target_action_id"].tolist() == [1, 1, 1, 1]
    assert collected.teacher["source"].tolist() == [M.RANDOM_SOURCE] * 4
    assert [search["trigger"] for search in collected.record["searches"]] == [
        "level_start", "after_perturbation", "level_start", "after_perturbation",
    ]


def test_random_recovery_unsupported_retains_actual_partial_n_plus_one_trace(monkeypatch):
    searches = 0

    def search(env, limit):
        nonlocal searches
        searches += 1
        if searches == 1:
            return SimpleNamespace(
                actions=[(1, None, None)], truncated=False, unsupported=False,
                exact=True, expanded=1, reason="initial",
            )
        return SimpleNamespace(
            actions=None, truncated=False, unsupported=True,
            exact=False, expanded=1, reason="random state unsupported",
        )

    def random_noop(game, variant, rng):
        raw = game.validate_action(M.Action(2))
        return M.Action(*variant.public_action(2, None, None)), raw

    monkeypatch.setattr(M, "_sample_random_action", random_noop)
    collected = M.collect_generated_game(
        _toy_modules(search=search), master_seed=0, game_index=0, difficulties=(1,),
        limits=M.SearchLimits(5, 10), outer_generation_attempts=1, generator_attempts=1,
        rollout=M.RolloutOptions(random_action_probability=1.0, max_game_steps=5),
    )
    assert collected.record["status"] == "rollout_failed"
    assert collected.record["steps"] == collected.record["random_steps"] == 1
    assert collected.public["frames"].shape[0] == collected.record["steps"] + 1
    assert not collected.public["level_boundary"].any()
    assert collected.public["levels_completed"].tolist() == [0, 0]
    assert "unsupported=True" in collected.record["errors"][0]
    assert collected.record["searches"][-1]["unsupported"] is True
    assert collected.record["searches"][-1]["exact"] is False


def test_random_replanning_respects_actual_per_level_action_cap(monkeypatch):
    def random_noop(game, variant, rng):
        raw = game.validate_action(M.Action(2))
        return M.Action(*variant.public_action(2, None, None)), raw

    monkeypatch.setattr(M, "_sample_random_action", random_noop)
    collected = M.collect_generated_game(
        _toy_modules(), master_seed=0, game_index=0, difficulties=(1,),
        limits=M.SearchLimits(3, 10), outer_generation_attempts=1, generator_attempts=1,
        rollout=M.RolloutOptions(random_action_probability=1.0, max_game_steps=20),
    )
    assert collected.record["status"] == "rollout_failed"
    assert collected.record["steps"] == 3
    assert collected.public["frames"].shape[0] == 4
    assert len(collected.record["searches"]) == 3
    assert "action cap 3 reached" in collected.record["errors"][0]


def test_post_action_successor_failure_rolls_back_transition_and_partial_state_columns():
    collected = M.collect_generated_game(
        _toy_modules(env_cls=_RenderFailsAfterAction),
        master_seed=0,
        game_index=0,
        difficulties=(1,),
        limits=M.SearchLimits(2, 10),
        outer_generation_attempts=1,
        generator_attempts=1,
    )
    assert collected.record["status"] == "rollout_failed"
    assert collected.record["steps"] == 0
    assert collected.public["frames"].shape == (1, 64, 64)
    assert collected.public["action_id"].shape == (0,)
    assert collected.record["final_state"] == "NOT_FINISHED"
    assert collected.record["levels_completed"] == 0
    assert collected.record["unrecorded_engine_state_after_error"]["state"] == "WIN"


def test_mixed_manifest_reports_actual_counts_and_rollout_failures(monkeypatch):
    manifest = M.new_manifest(
        sources=[M.source_for("cd82")], explicit_subset=True, seed=0, games_per_source=1,
        difficulties=(1,), limits=M.SearchLimits(),
        rollout=M.RolloutOptions(random_action_probability=0.5, max_game_steps=10),
    )
    manifest["records"].append({
        "source": "cd82", "game_index": 0, "status": "rollout_failed",
        "levels_completed": 0, "steps": 3, "teacher_steps": 1, "random_steps": 2,
        "random_metric_eligible_steps": 2, "record": "records/cd82-000000.json",
    })
    M.update_manifest_summary(manifest)
    assert manifest["random_transition_scoring_available"]
    assert manifest["teacher_steps"] == 1 and manifest["random_steps"] == 2
    assert manifest["rollout_failures"] == ["records/cd82-000000.json"]
    monkeypatch.setattr(collector, "main", lambda argv=None: manifest)
    assert collector.cli([]) == 0


def test_random_sampler_uses_public_legal_ids_and_full_pixel_coordinate_bounds():
    from pebby.multigame_variants import WholeGameVariant

    class ScriptedRng:
        def __init__(self):
            self.coordinates = iter((63, 0))

        def choice(self, values):
            assert np.asarray(values).tolist() == [1, 6]
            return 6

        def integers(self, low, high):
            assert (low, high) == (0, 64)
            return next(self.coordinates)

    class ClickGame:
        legal_action_ids = (1, 6)

        def validate_action(self, action):
            return action

    variant = WholeGameVariant(
        True, 1, tuple(range(8)), "rot90", tuple(range(16)),
    )
    public, raw = M._sample_random_action(ClickGame(), variant, ScriptedRng())
    assert public == M.Action(6, 63, 0)
    assert variant.public_action(*raw.as_tuple()) == public.as_tuple()
    assert all(0 <= value < 64 for value in (raw.x, raw.y))


def test_manifest_scope_and_all_persistence_paths_reject_contradictory_or_heldout_sources(tmp_path):
    with pytest.raises(ValueError, match="all 24"):
        M.new_manifest(
            sources=[M.source_for("cd82")], explicit_subset=False, seed=0,
            games_per_source=1, difficulties=(1,), limits=M.SearchLimits(),
        )

    heldout_record = {
        "format": M.FORMAT,
        "source": "m0r0",
        "source_id": M.HELD_OUT_SOURCE_ID,
        "status": "rollout_failed",
    }
    with pytest.raises(ValueError, match="held-out"):
        M.save_collected_game(tmp_path, "forged", M.CollectedGame(heldout_record, None, None, []))

    manifest = M.new_manifest(
        sources=[M.source_for("cd82")], explicit_subset=True, seed=0,
        games_per_source=1, difficulties=(1,), limits=M.SearchLimits(),
    )
    manifest["records"] = [heldout_record]
    with pytest.raises(ValueError, match="held out or was not requested"):
        M.save_manifest(tmp_path / "bad-save.json", manifest)
    (tmp_path / "bad-load.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="held out or was not requested"):
        M.load_manifest(tmp_path / "bad-load.json")


def test_cli_parses_explicit_variant_and_mixed_rollout_controls():
    args = collector.parse_args([
        "--games", "cd82", "--out-dir", "/tmp/not-created-by-parse",
        "--variants", "--variant-components", "spatial", "palette",
        "--variant-mix", "0.25", "--variant-seed", "91",
        "--random-action-probability", "0.4", "--max-game-steps", "77",
    ])
    assert args.variants and args.variant_components == ["spatial", "palette"]
    assert args.variant_mix == 0.25 and args.variant_seed == 91
    assert args.random_action_probability == 0.4 and args.max_game_steps == 77
