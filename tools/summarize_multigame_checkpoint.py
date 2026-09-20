#!/usr/bin/env python3
"""Report metrics belonging to these exact checkpoint weights, separately from peaks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from pebby.agent.multigame_training import TRAINING_FORMAT


def summarize(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != TRAINING_FORMAT:
        raise ValueError("expected a multigame training checkpoint")
    epoch = payload["epoch"]
    logs = payload.get("logs", [])
    selected = next((row for row in reversed(logs) if row["epoch"] == epoch), None)
    panels = ("generated_train_closed_loop", "generated_validation_closed_loop")
    fields = ("games_won", "games_played", "levels_completed", "levels_requested")

    def panel(row, key):
        value = row.get(key) if row else None
        return {field: value.get(field) for field in fields} if value else None

    training = payload.get("training_config", {})
    dropout = training.get("history_dropout", payload.get("history_dropout", 0.0))
    warnings = []
    if payload.get("model_config", {}).get("history_dropout", dropout) != dropout:
        warnings.append("Legacy model_config history_dropout disagrees; training_config is authoritative.")
    if selected is None:
        warnings.append("No completed evaluation belongs to these weights (e.g. a mid-epoch checkpoint).")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "checkpoint": str(path.resolve()), "sha256": digest,
        "epoch_zero_based": epoch, "checkpoint_role": payload.get("checkpoint_role"),
        "scope": payload.get("scope"),
        "history_mode": training.get("history_mode", payload.get("history_mode", "full")),
        "history_dropout": dropout,
        "selected_weights_metrics": {
            **{key: panel(selected, key) for key in panels},
            "generated_validation_offline": selected.get("generated_validation_offline") if selected else None,
        },
        "historical_peaks_in_this_checkpoint_only": {
            key: max((
                {"epoch_zero_based": row["epoch"], **panel(row, key)}
                for row in logs if row.get(key)
            ), key=lambda value: (value["levels_completed"], value["games_won"]), default=None)
            for key in panels
        },
        "limitations": [
            "Generated development panels are not unseen confirmation panels or official scores.",
            "The auxiliary dynamics head is not used for planning by this policy.",
            "Historical peaks may belong to different weights and must not be compared as one checkpoint.",
        ],
        "warnings": warnings,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = summarize(args.checkpoint)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
