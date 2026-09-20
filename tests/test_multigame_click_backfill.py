"""Replay-verified backfill commits, bounds, source isolation, and crash recovery."""

import json
import time

import numpy as np
import pytest

from pebby import multigame as M
from pebby.agent.multigame_model import load_supervised_game
from pebby.agent.multigame_training import sha256_file
from pebby.multigame_variants import VariantOptions
from tests.test_multigame_click_regions import _click_modules
from tools import backfill_multigame_click_regions as B


def _source(root):
    modules = _click_modules()
    records = []
    for index in range(2):
        collected = M.collect_generated_game(
            modules, master_seed=3, game_index=index, difficulties=(1, 2),
            variants=VariantOptions(enabled=True, seed=4),
            rollout=M.RolloutOptions(click_region_probe_limit=0),
        )
        assert collected.record["status"] == "won"
        records.append(M.save_collected_game(root, f"game-{index}", collected))
    manifest = M.new_manifest(sources=[modules.source], explicit_subset=True, seed=3,
                              games_per_source=2, difficulties=(1, 2), limits=M.SearchLimits())
    manifest["records"] = records
    M.update_manifest_summary(manifest)
    M.save_manifest(root / "manifest.json", manifest)
    return root / "manifest.json", {modules.source.source_id: modules}


def test_backfill_resume_keeps_source_immutable_and_restores_variant_regions(tmp_path):
    source, modules = _source(tmp_path / "source")
    before = {str(path): sha256_file(path) for path in source.parent.rglob("*") if path.is_file()}
    output = tmp_path / "repaired"
    first = B.backfill(source, output, max_games=1, modules_by_source=modules)
    assert first["committed_records"] == 1 and not first["complete"]
    partial = M.load_manifest(output / "manifest.json")
    assert partial["preparation_complete"] is False
    assert partial["is_full_experiment_collection"] is False
    second = B.backfill(source, output, max_games=1, modules_by_source=modules)
    assert second["complete"] and second["committed_records"] == 2
    assert B.backfill(source, output, modules_by_source=modules)["new_records"] == 0
    final = M.load_manifest(output / "manifest.json")
    for record in final["records"]:
        game = load_supervised_game(output / record["public_npz"], output / record["teacher_npz"])
        assert game.target_click_region.sum(axis=(1, 2)).tolist() == [16, 16]
        assert record["click_backfill"]["verified_transitions"] == 2
        assert record["backfill_source_manifest_sha256"] == before[str(source)]
    assert before == {str(path): sha256_file(path) for path in source.parent.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="options changed"):
        B.backfill(source, output, probe_limit=32, modules_by_source=modules)


def test_backfill_recovers_record_committed_before_manifest_crash(tmp_path, monkeypatch):
    source, modules = _source(tmp_path / "source")
    output = tmp_path / "repaired"
    save = M.save_manifest
    monkeypatch.setattr(M, "save_manifest", lambda *a, **k: (_ for _ in ()).throw(OSError("interrupt")))
    with pytest.raises(OSError, match="interrupt"):
        B.backfill(source, output, max_games=1, modules_by_source=modules)
    assert (output / "records/backfill-000000.json").exists()
    assert not (output / "manifest.json").exists()
    monkeypatch.setattr(M, "save_manifest", save)
    original = B._backfill_record
    calls = []

    def record(*args, **kwargs):
        calls.append(args[0]["game_index"])
        return original(*args, **kwargs)

    monkeypatch.setattr(B, "_backfill_record", record)
    assert B.backfill(source, output, max_games=1, modules_by_source=modules)["complete"]
    assert calls == [1]


def test_replay_mismatch_and_step_bound_never_commit_labels(tmp_path):
    source, modules = _source(tmp_path / "source")
    with pytest.raises(ValueError, match="max-steps"):
        B.backfill(source, tmp_path / "too-short", max_steps_per_game=1, modules_by_source=modules)
    record = M.load_manifest(source)["records"][0]
    path = source.parent / record["public_npz"]
    with np.load(path) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    arrays["frames"][1, 0, 0] = (int(arrays["frames"][1, 0, 0]) + 1) % 16
    M._atomic_npz(path, arrays)
    with pytest.raises(ValueError, match="replay frame mismatch"):
        B.backfill(source, tmp_path / "mismatch", modules_by_source=modules)
    assert not (tmp_path / "mismatch/records/backfill-000000.json").exists()


def test_deadline_escapes_probe_exception_handlers_and_cleans_timer():
    import signal

    def swallowed_exception():
        try:
            time.sleep(.1)
        except Exception:
            pytest.fail("deadline was swallowed by candidate validation")

    with pytest.raises(ValueError, match="exceeded"):
        B._bounded_call(swallowed_exception, .01)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


@pytest.mark.parametrize("mutation", ["output", "source", "manifest"])
def test_resume_rejects_changed_committed_artifact(tmp_path, mutation):
    source, modules = _source(tmp_path / "source")
    output = tmp_path / "repaired"
    B.backfill(source, output, max_games=1, modules_by_source=modules)
    record = json.loads((output / "records/backfill-000000.json").read_text())
    if mutation == "output":
        (output / record["teacher_npz"]).write_bytes(b"changed")
        message = "output .* changed"
    elif mutation == "source":
        original = M.load_manifest(source)["records"][0]
        (source.parent / original["teacher_npz"]).write_bytes(b"changed")
        message = "source .* changed"
    else:
        source.write_text(source.read_text() + "\n")
        message = "source manifest or label options changed"
    with pytest.raises(ValueError, match=message):
        B.backfill(source, output, modules_by_source=modules)


def test_concurrent_backfill_writer_is_rejected(tmp_path):
    import fcntl

    source, modules = _source(tmp_path / "source")
    output = tmp_path / "repaired"
    output.mkdir()
    with (output / ".backfill.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="another backfill writer"):
            B.backfill(source, output, modules_by_source=modules)
    assert B.backfill(source, output, modules_by_source=modules)["new_records"] == 1


def test_backfill_can_resume_with_larger_runtime_bounds(tmp_path):
    source, modules = _source(tmp_path / "source")
    output = tmp_path / "repaired"
    with pytest.raises(ValueError, match="max-steps"):
        B.backfill(source, output, max_steps_per_game=1, game_seconds=1, modules_by_source=modules)
    assert B.backfill(source, output, max_steps_per_game=2, game_seconds=2,
                      modules_by_source=modules)["new_records"] == 1
    assert B.backfill(source, output, max_steps_per_game=4, game_seconds=3,
                      modules_by_source=modules)["complete"]
    records = M.load_manifest(output / "manifest.json")["records"]
    assert records[0]["backfill_runtime_bounds"] == {"max_steps_per_game": 2, "game_seconds": 2}
    assert records[1]["backfill_runtime_bounds"] == {"max_steps_per_game": 4, "game_seconds": 3}
