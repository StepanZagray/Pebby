"""Matched on-policy continuation with prediction-only target stop-gradient.

Experimental variant, not a LeWM replication or a guaranteed repair. The existing
on-policy and repair runners retain their exact data, initialization, optimizer,
318-update budget, weights and source guards. No training occurs on import.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from pebby.agent import world_detached_objective as objective
from pebby.agent import world_train, world_training_objectives
from tools import train_reference_onpolicy as onpolicy
from tools import train_reference_repair as repair


def source_record(module):
    path = Path(module.__file__).resolve()
    return {"module": module.__name__, "path": str(path), "sha256": repair.digest(path)}


def training_objective_source():
    return {**source_record(objective),
            "variant": "prediction_target_stop_gradient",
            "base_objective": source_record(world_training_objectives),
            "wrapper": {"path": str(Path(__file__).resolve()), "sha256": repair.digest(__file__)},
            "prediction_target_detached": True,
            "actual_state_objectives_detached": False,
            "lewm_replication": False}


@contextmanager
def objective_context():
    """Scope trainer hooks and bind both implementations in every receipt."""
    original_guard, original_write = repair.source_guard, repair.write

    def guard():
        result = original_guard()
        paths = (Path(__file__).resolve(), Path(objective.__file__).resolve(),
                 Path(world_training_objectives.__file__).resolve())
        result.hashes.update({str(path): repair.digest(path) for path in paths})
        return result

    def write(path, value):
        if path.name == "provenance.json":
            value = {**value, "training_objective_source": training_objective_source()}
        return original_write(path, value)

    with patch.object(world_train, "world_losses", objective.world_losses), \
            patch.object(world_train, "training_objective_source", training_objective_source), \
            patch.object(repair, "source_guard", guard), patch.object(repair, "write", write):
        yield


def main(argv=None):
    with objective_context():
        return onpolicy.main(argv)


if __name__ == "__main__":
    main()
