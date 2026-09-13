"""CPU CLI loader dispatch checks; official sessions and gameplay are mocked."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from pebby.agent import competition, model
from pebby.agent import neural_outcome_policy as outcomes
from pebby.agent import spatial_outcome_planner
from pebby.agent import spatial_outcome_policy as spatial_outcomes
from pebby.agent import world_position_recall as position
from tools import evaluate_reference_competition as evaluator


class EvaluateReferenceCompetitionTests(unittest.TestCase):
    def test_format_dispatch_and_spatial_source_bindings_without_gameplay(self):
        formats = ((spatial_outcomes.FORMAT, spatial_outcomes),
                   (outcomes.FORMAT, outcomes),
                   (position.POSITION_RECALL_FORMAT, position),
                   ('existing-base-checkpoint', model))
        for checkpoint_format, expected_module in formats:
            with self.subTest(format=checkpoint_format), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / 'model.pt'
                output = Path(directory) / 'report.json'
                metadata = {'format': checkpoint_format}
                torch.save(metadata, checkpoint)
                policy = Mock()
                policy.config.return_value = {'test_policy': True}
                session = SimpleNamespace(level_count=7)
                with ExitStack() as stack:
                    loaders = {module: stack.enter_context(patch.object(
                        module, 'load_checkpoint', return_value=(policy, metadata)))
                        for module in (model, outcomes, position, spatial_outcomes)}
                    constructor = stack.enter_context(patch.object(
                        competition, 'CompetitionSession', return_value=session))
                    runner = stack.enter_context(patch.object(
                        competition, 'run_competition', return_value=dict(
                            completed=False, levels_completed=0, actions=0, resets=0)))
                    stack.enter_context(patch.object(evaluator, 'require_memory_reserve', return_value=8 * 1024**3))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    original_threads = torch.get_num_threads()
                    try:
                        result = evaluator.main([
                            '--checkpoint', str(checkpoint),
                            '--checkpoint-sha256', evaluator.digest(checkpoint),
                            '--report-out', str(output), '--device', 'cpu',
                            '--per-level-max-actions', '1'])
                    finally:
                        torch.set_num_threads(original_threads)
                loaders[expected_module].assert_called_once_with(checkpoint.resolve(), device='cpu')
                for module, loader in loaders.items():
                    if module is not expected_module:
                        loader.assert_not_called()
                constructor.assert_called_once_with()
                self.assertIs(runner.call_args.args[1], session)
                self.assertEqual(runner.call_args.kwargs['per_level_caps'], [1] * 7)
                self.assertEqual(result['checkpoint_format'], checkpoint_format)
                self.assertEqual(json.loads(output.read_text()), result)
                for module in (spatial_outcomes, spatial_outcome_planner):
                    path = str(Path(module.__file__).resolve())
                    self.assertEqual(result['source_sha256'][path], evaluator.digest(path))


if __name__ == '__main__':
    unittest.main()
