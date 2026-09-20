"""Bounded split-preparation tests using injected synthetic collectors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pebby import multigame as M
from pebby import multigame_dataset as D
from pebby.multigame_dataset import (
    DatasetPreparationConfig,
    prepare_multigame_dataset,
)
from pebby.multigame_variants import WholeGameVariant
from tools import prepare_multigame_dataset as cli


def _modules(source: str | M.Source, *, full_standard: bool = False) -> M.GameModules:
    resolved = M.source_for(source)
    if not full_standard:
        component = SimpleNamespace()
        return M.GameModules(
            source=resolved,
            env=component,
            generate=component,
            plan=component,
            solver_adapter="work_limit",
        )

    env = SimpleNamespace(official_levels=lambda: [object()])

    def generate(seed, difficulty, *, split):
        return {"seed": seed, "difficulty": difficulty, "split": split}

    def generate_game(seed, *, split, difficulties=None):
        return [generate(seed, 1, split=split)]

    generator = SimpleNamespace(
        DIFFICULTIES=(1,),
        FULL_STANDARD_CONTRACT={
            "format": M.FULL_STANDARD_FORMAT,
            "status": "ready",
            "source_id": resolved.source_id,
            "mechanics_inventory_version": "synthetic-mechanics-v1",
            "quality_profile_version": "synthetic-quality-v1",
            "curriculum": [{"difficulty": 1, "context_index": 0, "search_work": 2}],
            "evidence": {key: f"synthetic-{key}" for key in M.FULL_STANDARD_EVIDENCE},
            "caveats": ["synthetic one-level fixture; not production evidence"],
        },
        generate=generate,
        generate_game=generate_game,
        build_level=lambda spec: dict(spec),
        build_game=lambda specs: [dict(spec) for spec in specs],
        validate_full_standard=lambda spec, entry: [],
    )
    contract = M._full_standard_contract(resolved, env, generator)
    return M.GameModules(
        source=resolved,
        env=env,
        generate=generator,
        plan=SimpleNamespace(),
        solver_adapter="work_limit",
        full_standard=contract,
    )


def _collected(
    source: M.Source,
    *,
    master_seed: int,
    game_index: int,
    effective_seed: int,
    puzzle: str,
    frame_puzzle: str | None = None,
    mixed: bool,
    status: str = "won",
) -> M.CollectedGame:
    if status == "generation_failed":
        return M.CollectedGame(
            {
                "format": M.FORMAT,
                "source": source.slug,
                "source_id": source.source_id,
                "master_seed": master_seed,
                "game_index": game_index,
                "status": status,
                "steps": 0,
                "levels_generated": 0,
                "levels_completed": 0,
                "teacher_steps": 0,
                "random_steps": 0,
                "errors": ["synthetic bounded generation failure"],
            },
            None,
            None,
            [],
        )

    won = status == "won"
    digest = hashlib.sha256((frame_puzzle or puzzle).encode()).digest()
    color = 1 + digest[0] % 14
    frames = np.zeros((2, 64, 64), dtype=np.uint8)
    frames[:, 2:4, 3:5] = color
    frames[:, 0, :32] = np.frombuffer(digest, dtype=np.uint8) % 16
    frames[1, 8, 9] = (color + 1) % 16
    legal = np.zeros((2, M.ACTION_COUNT), dtype=np.bool_)
    legal[0, 1] = True
    if not won:
        legal[1, 1] = True
    boundary = np.asarray([won], dtype=np.bool_)
    terminal = np.asarray([False, won], dtype=np.bool_)
    completed = np.asarray([0, int(won)], dtype=np.int16)
    public = {
        "frames": frames,
        "legal_action_mask": legal,
        "state": np.asarray(["NOT_FINISHED", "WIN" if won else "NOT_FINISHED"]),
        "level_index": np.asarray([0, 0], dtype=np.int16),
        "levels_completed": completed,
        "terminal": terminal,
        "won": terminal.copy(),
        "action_id": np.asarray([1], dtype=np.int8),
        "action_x": np.asarray([-1], dtype=np.int16),
        "action_y": np.asarray([-1], dtype=np.int16),
        "level_boundary": boundary,
    }
    source_value = M.RANDOM_SOURCE if mixed else M.TEACHER_SOURCE
    teacher = {
        "target_action_id": np.asarray([1], dtype=np.int8),
        "target_action_x": np.asarray([-1], dtype=np.int16),
        "target_action_y": np.asarray([-1], dtype=np.int16),
        "source": np.asarray([source_value], dtype=np.int8),
    }
    specs = [{
        "generator_version": 1,
        "effective_seed": effective_seed,
        "puzzle_token": puzzle,
    }]
    record = {
        "format": M.FORMAT,
        "source": source.slug,
        "source_id": source.source_id,
        "master_seed": master_seed,
        "game_index": game_index,
        "status": status,
        "steps": 1,
        "levels_generated": 1,
        "levels_completed": int(won),
        "level_boundaries": int(won),
        "final_state": "WIN" if won else "NOT_FINISHED",
        "teacher_steps": int(not mixed),
        "random_steps": int(mixed),
        "random_metric_eligible_steps": int(mixed),
        "random_transition_scoring_available": mixed,
        "errors": [] if won else ["synthetic unsupported recovery retained honestly"],
        "levels": [{"generator_version": 1}],
        "variant": WholeGameVariant.identity().private_metadata(),
    }
    return M.CollectedGame(record, public, teacher, specs)


class _ScriptedCollector:
    def __init__(
        self, *, fail_all=False, overlap_first_validation=False,
        duplicate_second=False, finite_tutorial_geometry=False,
    ):
        self.fail_all = fail_all
        self.overlap_first_validation = overlap_first_validation
        self.duplicate_second = duplicate_second
        self.finite_tutorial_geometry = finite_tutorial_geometry
        self.calls = []

    def __call__(self, package, **kwargs):
        source = M.source_for(package.source)
        master_seed = kwargs["master_seed"]
        game_index = kwargs["game_index"]
        mixed = kwargs["rollout"].random_action_probability > 0
        self.calls.append((source.source_id, master_seed, game_index, mixed))
        if self.fail_all:
            return _collected(
                source,
                master_seed=master_seed,
                game_index=game_index,
                effective_seed=0,
                puzzle="failure",
                mixed=mixed,
                status="generation_failed",
            )
        identity_master = master_seed
        identity_index = game_index
        if self.overlap_first_validation and master_seed == 200 and game_index == 0:
            identity_master = 100
            identity_index = 0
        if self.duplicate_second and game_index == 1:
            identity_index = 0
        effective = identity_master * 1000 + identity_index
        puzzle = f"{source.source_id}:{identity_master}:{identity_index}"
        collected = _collected(
            source,
            master_seed=master_seed,
            game_index=game_index,
            effective_seed=effective,
            puzzle=puzzle,
            frame_puzzle=f"public:{source.source_id}:{master_seed}:{game_index}",
            mixed=mixed,
            status="rollout_failed" if mixed else "won",
        )
        if self.finite_tutorial_geometry:
            collected.specs[0]["geometry_d4_sha256"] = hashlib.sha256(
                f"tutorial-class:{source.source_id}:{master_seed}".encode()
            ).hexdigest()
        if kwargs.get("require_full_standard"):
            curriculum = tuple(kwargs["curriculum"])
            split = kwargs["split"]
            contract = package.full_standard
            assert contract is not None
            collected.record.update({
                "requested_curriculum": [entry.to_dict() for entry in curriculum],
                "generation_split": split,
                "full_standard_required": True,
                "full_standard_validated": True,
                "full_standard_contract_hash": contract.sha256,
            })
            for level, entry in zip(collected.record["levels"], curriculum):
                level.update(
                    entry.to_dict(), generation_split=split,
                    full_standard_validated=True,
                )
            for spec, entry in zip(collected.specs, curriculum):
                spec.update({
                    "difficulty": entry.difficulty,
                    "context_index": entry.context_index,
                    "split": split,
                    "geometry_d4_sha256": hashlib.sha256(
                        f"geometry:{puzzle}".encode()
                    ).hexdigest(),
                    "gameplay_sha256": hashlib.sha256(
                        f"gameplay:{puzzle}".encode()
                    ).hexdigest(),
                })
        return collected


def _smoke_config(root: Path, **overrides) -> DatasetPreparationConfig:
    values = {
        "output_root": root,
        "train_master_seed": 100,
        "validation_master_seed": 200,
        "games": ("cd82",),
        "smoke": True,
        "completed_teacher_games_per_family": 1,
        "mixed_games_per_family": 1,
        "mixed_epsilon": 0.25,
        "difficulties": (1,),
        "limits": M.SearchLimits(4, 8),
        "outer_generation_attempts": 1,
        "generator_attempts": 1,
        "max_candidate_attempts": 3,
        "max_game_steps": 4,
    }
    values.update(overrides)
    return DatasetPreparationConfig(**values)


def test_deterministic_overlap_rejection_resamples_and_retains_mixed_prefixes(tmp_path):
    collector = _ScriptedCollector(overlap_first_validation=True)
    result = prepare_multigame_dataset(
        _smoke_config(tmp_path / "dataset"),
        modules=[_modules("cd82")],
        collect_fn=collector,
        progress=lambda message: None,
    )
    assert result.complete
    assert result.summary["strict_audit"]["smoke"]
    assert set(result.summary["manifests"]) == {"train", "validation"}
    assert len(result.summary["manifests"]["train"]) == 2
    assert len(result.summary["manifests"]["validation"]) == 2
    assert len(result.summary["parameter_hash"]) == 64
    assert len(result.summary["source_hash"]) == 64
    assert len(result.summary["provenance_hash"]) == 64

    validation_teacher = result.summary["cohorts"]["validation-teacher"]
    validation_mixed = result.summary["cohorts"]["validation-mixed"]
    assert validation_teacher["rejected_overlap"] == 1
    assert validation_mixed["rejected_overlap"] == 1
    first_rejection = validation_teacher["sources"][M.source_for("cd82").source_id][
        "attempts"
    ][0]
    assert {item[1] for item in first_rejection["overlap_sample"]} == {
        "canonical_spec", "effective_seed",
    }  # public initial frames were deliberately different
    assert validation_teacher["sources"][M.source_for("cd82").source_id]["attempts"][-1][
        "candidate_index"
    ] == 1
    assert result.summary["cohorts"]["train-mixed"]["retained_partial"] == 1
    assert validation_mixed["retained_partial"] == 1
    assert result.summary["strict_audit"]["train"]["random_rows"] == 1
    assert result.summary["strict_audit"]["validation"]["random_rows"] == 1
    assert collector.calls == [
        (M.source_for("cd82").source_id, 100, 0, False),
        (M.source_for("cd82").source_id, 100, 0, True),
        (M.source_for("cd82").source_id, 200, 0, False),
        (M.source_for("cd82").source_id, 200, 1, False),
        (M.source_for("cd82").source_id, 200, 0, True),
        (M.source_for("cd82").source_id, 200, 1, True),
    ]

    for relative in (
        *result.summary["manifests"]["train"],
        *result.summary["manifests"]["validation"],
    ):
        manifest = M.load_manifest(result.output_root / relative)
        assert manifest["preparation_complete"]
        assert manifest["provenance"]["status"] == "synthetic_test_fixture"
        assert set(manifest["provenance"]["sources"]) == {M.source_for("cd82").source_id}
        for record in manifest["records"]:
            assert set(record["file_hashes"]) == {
                "public_npz", "teacher_npz", "generated_specs",
            }
            private_record = json.loads((result.output_root / relative).parent.joinpath(
                record["record"]
            ).read_text())
            assert private_record["file_hashes"] == record["file_hashes"]
            assert private_record["provenance"]["status"] == "synthetic_test_fixture"


def test_candidate_failure_is_bounded_and_never_stamped_ready(tmp_path):
    collector = _ScriptedCollector(fail_all=True)
    result = prepare_multigame_dataset(
        _smoke_config(
            tmp_path / "failed", mixed_games_per_family=0, max_candidate_attempts=2,
        ),
        modules=[_modules("cd82")],
        collect_fn=collector,
        progress=lambda message: None,
    )
    assert not result.complete
    assert result.summary["strict_audit"] is None
    assert len(collector.calls) == 4  # two bounded attempts in each split
    for name in ("train-teacher", "validation-teacher"):
        cohort = result.summary["cohorts"][name]
        assert cohort["attempted"] == cohort["failed"] == 2
        manifest = M.load_manifest(result.output_root / name / "manifest.json")
        assert not manifest["preparation_complete"]
        assert not manifest["is_full_experiment_collection"]
        assert manifest["scope"].startswith("incomplete_")


def test_finite_tutorial_geometry_may_repeat_but_duplicate_whole_games_resample(tmp_path):
    collector = _ScriptedCollector(
        duplicate_second=True, finite_tutorial_geometry=True,
    )
    result = prepare_multigame_dataset(
        _smoke_config(
            tmp_path / "finite-tutorial",
            completed_teacher_games_per_family=2,
            mixed_games_per_family=0,
            max_candidate_attempts=2,
        ),
        modules=[_modules("cd82")],
        collect_fn=collector,
        progress=lambda message: None,
    )
    assert result.complete
    for cohort_name in ("train-teacher", "validation-teacher"):
        report = result.summary["cohorts"][cohort_name]
        assert report["accepted"] == 2
        assert report["rejected_duplicate"] == 1
        source_report = report["sources"][M.source_for("cd82").source_id]
        attempts = source_report["attempts"]
        assert any(item["outcome"] == "rejected_duplicate" for item in attempts)
        assert source_report["accepted_fingerprint_diversity"]["geometry_d4"] == {
            "unique": 1, "observations": 2,
        }


def test_whole_game_dedup_uses_ordered_semantics_not_generation_or_rollout_metadata():
    source_id = M.source_for("cd82").source_id
    game = [
        {
            "gameplay_sha256": "a" * 64,
            "effective_seed": 10,
            "generation_exclusions": {"geometry_split": 2},
            "observed_rollout_prefix": [1, 2, 3],
        },
        {
            "puzzle_hash": "b" * 64,
            "seed": 20,
            "generation_attempt": 3,
        },
    ]
    same_game_other_run = [
        {
            "gameplay_sha256": "a" * 64,
            "effective_seed": 999,
            "generation_exclusions": {"geometry_split": 17},
            "observed_rollout_prefix": [1],
        },
        {
            "puzzle_hash": "b" * 64,
            "seed": 777,
            "generation_attempt": 40,
        },
    ]
    assert D._whole_game_fingerprint(source_id, game) == D._whole_game_fingerprint(
        source_id, same_game_other_run,
    )
    swapped = list(reversed(same_game_other_run))
    assert D._whole_game_fingerprint(source_id, game) != D._whole_game_fingerprint(
        source_id, swapped,
    )


def test_stale_won_status_cannot_count_as_a_completed_teacher_game(tmp_path):
    def stale_winner(package, **kwargs):
        source = M.source_for(package.source)
        collected = _collected(
            source,
            master_seed=kwargs["master_seed"],
            game_index=kwargs["game_index"],
            effective_seed=kwargs["master_seed"] * 1000 + kwargs["game_index"],
            puzzle=f"stale:{kwargs['master_seed']}:{kwargs['game_index']}",
            mixed=False,
            status="rollout_failed",
        )
        collected.record["status"] = "won"
        return collected

    result = prepare_multigame_dataset(
        _smoke_config(
            tmp_path / "stale", mixed_games_per_family=0, max_candidate_attempts=1,
        ),
        modules=[_modules("cd82")],
        collect_fn=stale_winner,
        progress=lambda message: None,
    )
    assert not result.complete
    for cohort in result.summary["cohorts"].values():
        assert cohort["accepted"] == 0
        assert cohort["failed"] == 1


def test_nonempty_output_and_heldout_are_rejected_before_collection(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep")
    collector = _ScriptedCollector()
    with pytest.raises(ValueError, match="not empty"):
        prepare_multigame_dataset(
            _smoke_config(occupied),
            modules=[_modules("cd82")],
            collect_fn=collector,
        )
    assert not collector.calls

    heldout = tmp_path / "heldout"
    with pytest.raises(ValueError, match="held-out"):
        prepare_multigame_dataset(
            DatasetPreparationConfig(
                output_root=heldout,
                train_master_seed=1,
                validation_master_seed=2,
                games=("m0r0",),
                smoke=True,
                mixed_games_per_family=0,
                difficulties=(1,),
            ),
            modules=[_modules("m0r0")],
            collect_fn=collector,
        )
    assert not heldout.exists()


def test_default_full_scope_requires_all_24_and_is_strict_audit_compatible(tmp_path):
    collector = _ScriptedCollector()
    result = prepare_multigame_dataset(
        DatasetPreparationConfig(
            output_root=tmp_path / "full",
            train_master_seed=11,
            validation_master_seed=12,
            completed_teacher_games_per_family=1,
            mixed_games_per_family=0,
            limits=M.SearchLimits(2, 2),
            outer_generation_attempts=1,
            generator_attempts=1,
            max_candidate_attempts=1,
            max_game_steps=2,
        ),
        modules=[_modules(source, full_standard=True) for source in M.TRAIN_SOURCES],
        collect_fn=collector,
        progress=lambda message: None,
    )
    assert result.complete
    assert result.summary["scope"] == "full_24_family_experiment"
    assert result.summary["strict_audit"]["scope"] == "full_24_family_experiment"
    assert len(result.summary["source_ids"]) == 24
    for relative in result.summary["manifests"]["train"] + result.summary["manifests"]["validation"]:
        manifest = M.load_manifest(result.output_root / relative)
        assert manifest["is_full_experiment_collection"]
        for source_id, coverage in manifest["source_coverage_notes"].items():
            assert coverage["admission"] == "ready"
            assert coverage["official_level_count"] == 1
            assert isinstance(manifest["historical_core_coverage_notes"][source_id], str)


def test_default_preflight_failure_and_cli_scope_errors_leave_no_output(tmp_path, monkeypatch):
    output = tmp_path / "preflight"

    def rejected(_games=None, **_kwargs):
        raise M.PreflightError("missing family")

    monkeypatch.setattr(M, "preflight", rejected)
    with pytest.raises(M.PreflightError):
        prepare_multigame_dataset(DatasetPreparationConfig(
            output_root=output,
            train_master_seed=1,
            validation_master_seed=2,
            mixed_games_per_family=0,
        ))
    assert not output.exists()
    with pytest.raises(SystemExit):
        cli.parse_args([
            "--output-root", str(tmp_path / "bad-cli"),
            "--train-master-seed", "1", "--validation-master-seed", "2",
            "--games", "cd82",
        ])


def test_cli_returns_nonzero_for_a_bounded_incomplete_preparation(tmp_path, monkeypatch):
    fake = SimpleNamespace(
        complete=False,
        summary_path=tmp_path / "preparation.json",
        summary={
            "scope": "smoke",
            "training_cli_arguments": {},
            "strict_audit_error": None,
        },
    )
    monkeypatch.setattr(cli, "prepare_multigame_dataset", lambda config, progress: fake)
    code = cli.cli([
        "--output-root", str(tmp_path / "incomplete"),
        "--train-master-seed", "1",
        "--validation-master-seed", "2",
        "--games", "cd82",
        "--smoke",
        "--mixed-games-per-family", "0",
        "--quiet",
    ])
    assert code == 1


def test_explicit_fixture_provenance_validates_sources_and_never_relabels_production_modules():
    source = M.source_for("cd82")
    production = _modules(source)
    production = M.GameModules(
        production.source,
        production.env,
        production.generate,
        production.plan,
        production.solver_adapter,
        provenance={"status": "available"},
    )
    assert D._manifest_provenance_override((production,)) is None

    fixture = D._manifest_provenance_override((_modules(source),))
    assert fixture is not None and fixture["status"] == "synthetic_test_fixture"
    fixture["sources"] = {}
    with pytest.raises(ValueError, match="source IDs"):
        M.new_manifest(
            sources=(source,),
            explicit_subset=True,
            seed=1,
            games_per_source=1,
            difficulties=(1,),
            limits=M.SearchLimits(1, 1),
            provenance=fixture,
        )
