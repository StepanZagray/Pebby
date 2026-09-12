"""Keep current loss wiring independent of frozen encoder source identity."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from pebby.agent import batch_probe, world_model, world_train, world_training_objectives
from tools import preflight_world_cell_recall, score_world_sequences, world_continuation_preflight


class TrainingObjectiveBoundaryTests(unittest.TestCase):
    def test_current_consumers_use_failure_aware_objectives(self):
        for consumer in (world_train, batch_probe, score_world_sequences,
                         preflight_world_cell_recall, world_continuation_preflight):
            with self.subTest(consumer=consumer.__name__):
                self.assertIs(consumer.world_losses, world_training_objectives.world_losses)
        self.assertIsNot(world_training_objectives.world_losses, world_model.world_losses)

    def test_existing_checkpoint_encoder_source_remains_byte_identical(self):
        # Existing structured checkpoints bind this full source hash. Training
        # changes belong in objectives; encoder changes need an explicit new
        # checkpoint lineage rather than silently invalidating frozen policies.
        self.assertEqual(hashlib.sha256(Path(world_model.__file__).read_bytes()).hexdigest(),
                         "1be9b00c37f758d9975da981cd92e9d6dac76a17ae3aef7023ebb2647aec1353")

    def test_objective_source_round_trips_without_changing_frozen_model(self):
        from tests.test_policy_history import tiny_world
        record = world_train.training_objective_source()
        self.assertEqual(record["module"], world_training_objectives.__name__)
        self.assertEqual(record["sha256"], hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            world_model.save_world_checkpoint(path, tiny_world(), training_objective_source=record)
            _, loaded = world_model.load_world_checkpoint(path)
            self.assertEqual(loaded["training_objective_source"], record)


if __name__ == "__main__":
    unittest.main()
