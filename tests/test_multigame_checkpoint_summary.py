"""Never report an earlier epoch's metrics as those of new checkpoint weights."""
import torch

from pebby.agent.multigame_training import TRAINING_FORMAT
from tools.summarize_multigame_checkpoint import summarize


def test_selected_weights_metrics_are_not_historical_peaks(tmp_path):
    path = tmp_path / "latest.pt"
    torch.save({
        "format": TRAINING_FORMAT, "epoch": 2,
        "logs": [
            {"epoch": 1, "generated_train_closed_loop": {"games_won": 2, "levels_completed": 51}},
            {"epoch": 2, "generated_train_closed_loop": {"games_won": 1, "levels_completed": 25}},
        ],
    }, path)
    result = summarize(path)
    key = "generated_train_closed_loop"
    assert result["selected_weights_metrics"][key]["levels_completed"] == 25
    peak = result["historical_peaks_in_this_checkpoint_only"][key]
    assert (peak["epoch_zero_based"], peak["levels_completed"]) == (1, 51)


def test_mid_epoch_checkpoint_has_no_borrowed_metrics(tmp_path):
    path = tmp_path / "latest.pt"
    torch.save({
        "format": TRAINING_FORMAT, "epoch": 2, "epoch_progress": {"cursor": 1},
        "training_config": {"history_dropout": 0.5}, "model_config": {"history_dropout": 0},
        "logs": [{"epoch": 1, "generated_validation_offline": {"action_accuracy": 0.9}}],
    }, path)
    result = summarize(path)
    assert result["selected_weights_metrics"]["generated_validation_offline"] is None
    assert result["history_dropout"] == 0.5
    assert len(result["warnings"]) == 2
