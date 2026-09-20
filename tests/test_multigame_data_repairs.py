"""Regression checks for learner input contracts and honest corpus coverage."""

from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from pebby import multigame as M
from pebby.agent import multigame_training as T
from pebby.multigame_dataset import LearnerPerturber
from pebby.multigame_variants import WholeGameVariant, VariantOptions
from tests.test_multigame_click_regions import _click_modules
from tests.test_multigame_recovery import _collect, _full_modules, _ScriptedLearner
from tools.audit_multigame_coverage import audit_manifest, coverage_failures, cli


@pytest.mark.parametrize("canonical", [True, False])
@pytest.mark.parametrize("action_id", [1, 6])
def test_learner_preserves_checkpoint_contract_and_transforms_entire_adapter(
    tmp_path, monkeypatch, canonical, action_id,
):
    class Policy:
        def parameters(self):
            yield torch.zeros(1)

        def initial_memory(self, *args, **kwargs):
            return "memory"

        def policy_step(self, **kwargs):
            self.inputs = kwargs
            logits = torch.zeros(1, 8)
            logits[0, action_id] = 10
            return SimpleNamespace(action_logits=logits, click_logits=None), "next"

        def decode_click(self, logits):
            return torch.tensor([12]), torch.tensor([23])

    model = Policy()
    payload = {"history_mode": "full", "canonical_inputs": not canonical,
               "training_config": {"history_mode": "none", "canonical_inputs": canonical}}
    monkeypatch.setattr(T, "model_from_training_checkpoint", lambda *a, **k: (model, payload))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test")
    learner = LearnerPerturber(checkpoint)
    variant = WholeGameVariant(True, 2, (0, 3, 7, 1, 4, 5, 6, 2), "rot90", tuple(reversed(range(16))))
    learner.configure_variant(variant)
    learner.reset()
    frame = np.arange(4096, dtype=np.uint8).reshape(64, 64) % 16
    legal = np.asarray([False, True, False, False, False, False, True, True])
    previous = M.Action(6, 17, 29) if action_id == 6 else M.Action(7)
    result = learner.step(variant.public_frame(frame), variant.public_legal_mask(legal),
                          M.Action(*variant.public_action(*previous.as_tuple())), True)
    inputs = model.inputs
    assert np.array_equal(inputs["frame"][0].numpy(), frame if canonical else variant.public_frame(frame))
    assert np.array_equal(inputs["legal_action_mask"][0].numpy(),
                          legal if canonical else variant.public_legal_mask(legal))
    expected_previous = previous.as_tuple() if canonical else variant.public_action(*previous.as_tuple())
    assert tuple(int(inputs[f"previous_action_{name}"].item()) for name in ("id", "x", "y")) == tuple(
        -1 if value is None else value for value in expected_previous
    )
    assert inputs["history_keep"].tolist() == [False]
    assert inputs["previous_level_boundary"].tolist() == [True]
    expected = (action_id, 12, 23) if action_id == 6 else (action_id, None, None)
    assert result.as_tuple() == (variant.public_action(*expected) if canonical else expected)


def test_collector_binds_variant_before_learner_reset():
    class Learner(_ScriptedLearner):
        def configure_variant(self, variant):
            self.variant = variant

        def reset(self):
            assert isinstance(self.variant, WholeGameVariant)
            super().reset()

    learner = Learner(2)
    collected = _collect(
        _full_modules(), perturber=learner,
        variants=VariantOptions(enabled=True, controls=False, palette=True, spatial=True),
        rollout=M.RolloutOptions(.5, max_game_steps=8, perturbation="learner", learner_checkpoint="test"),
    )
    assert collected.record["variant"] == learner.variant.private_metadata()
    assert learner.resets == 1


def _click_manifest(tmp_path):
    modules = _click_modules()
    records = []
    for index, probe_limit in enumerate((0, 64)):
        collected = M.collect_generated_game(
            modules, master_seed=3, game_index=index, difficulties=(1, 2),
            limits=M.SearchLimits(8, 100),
            rollout=M.RolloutOptions(max_game_steps=1, click_region_probe_limit=probe_limit),
        )
        if probe_limit == 0:
            assert "click_region_mask" not in collected.teacher
            assert "click_region_size" not in collected.teacher
        records.append(M.save_collected_game(tmp_path, f"game-{index}", collected))
    manifest = M.new_manifest(sources=[modules.source], explicit_subset=True, seed=3, games_per_source=2,
                                   difficulties=(1, 2), limits=M.SearchLimits(8, 100))
    manifest["records"] = records
    M.update_manifest_summary(manifest)
    path = tmp_path / "manifest.json"
    M.save_manifest(path, manifest)
    return path


def test_coverage_reports_actual_click_labels_and_zero_observed_later_levels(tmp_path):
    path = _click_manifest(tmp_path)
    report = audit_manifest(path)
    totals = report["totals"]
    assert totals["transitions"] == totals["click_targets"] == 2
    assert totals["stored_region_clicks"] == totals["exact_fallback_clicks"] == 1
    assert totals["multi_pixel_region_clicks"] == 1
    assert totals["stored_region_fraction"] == .5
    assert totals["target_actions_canonical"] == {"6": 2}
    assert report["by_family"][M.source_for("cd82").source_id]["by_level_index"]["1"]["targets"] == 0
    assert sum(totals["route_source_counts"].values()) == totals["transitions"]
    failures = coverage_failures(report, min_click_region_fraction=1, min_targets_per_level=1,
                                 min_undo_targets_per_family=1, min_learner_steps_per_family=1)
    assert len(failures) == 4
    assert cli(["--manifest", str(path), "--require-level-coverage", "--output", str(tmp_path / "audit.json")]) == 1
    assert cli(["--manifest", str(path), "--output", str(path)]) == 2


def test_route_summary_accounts_for_unknown_legacy_and_rejects_bad_totals():
    manifest = {"records": [{"status": "won", "steps": 3}]}
    M.update_manifest_summary(manifest)
    assert manifest["route_source_counts"] == {"unknown": 3}
    manifest["records"][0]["route_source_counts"] = {M.CERTIFIED_ROUTE_SOURCE: 1}
    with pytest.raises(ValueError, match="sum to steps"):
        M.update_manifest_summary(manifest)


def test_merge_reconstructs_missing_counts_without_mutating_source(tmp_path):
    from tools.merge_multigame_manifests import _load_side
    from tests.test_multigame_dataset import _collected
    collected = _collected(M.source_for("ar25"), master_seed=1, game_index=0,
                           effective_seed=10, puzzle="test", mixed=False)
    collected.teacher["route_source"] = np.asarray([M.CERTIFIED_ROUTE_SOURCE])
    record = M.save_collected_game(tmp_path, "game", collected)
    manifest = M.new_manifest(sources=[M.source_for("ar25")], explicit_subset=True, seed=1,
                                   games_per_source=1, difficulties=(1,), limits=M.SearchLimits())
    manifest["records"] = [record]
    path = tmp_path / "manifest.json"
    M.save_manifest(path, manifest)
    before = path.read_bytes()
    _, games = _load_side([path], side="train")
    assert games[0].record["route_source_counts"] == {M.CERTIFIED_ROUTE_SOURCE: 1}
    assert path.read_bytes() == before
    assert "route_source_counts" not in json.loads((tmp_path / record["record"]).read_text())
