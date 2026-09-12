"""Tiny synthetic fixtures exercise provenance through actual optimizer/checkpoint code."""
import contextlib
import io
import json
import unittest

import torch

from pebby.agent.world_train import main
from tests import test_merge_onpolicy_world as fixtures
from tools.merge_onpolicy_world import merge


class AggregateCliTests(unittest.TestCase):
    def test_two_behaviors_survive_training_checkpoint(self):
        torch.set_num_threads(1)
        fixture = fixtures.OnPolicyMergeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        first, second = fixture.source('first', 8), fixture.source('second', 9)
        aggregate = fixture.root / 'aggregate.npz'
        merge([first, second], aggregate)
        checkpoint = fixture.root / 'tiny.pt'
        with contextlib.redirect_stdout(io.StringIO()):
            main(['--train', str(aggregate), '--on-policy-data', str(aggregate),
                  '--on-policy-fraction', '.5', '--epochs', '1', '--batch-size', '2',
                  '--curriculum', '--curriculum-start', '1', '0', '0', '0', '0',
                  '--curriculum-end', '1', '0', '0', '0', '0', '--drop-last',
                  '--require-verified-data', '--require-winning-coverage',
                  '--device', 'cpu', '--channels', '4', '--heads', '1', '--blocks', '1',
                  '--loops', '1', '--history', '8', '--hud-channels', '2', '--hud-tokens', '1',
                  '--latent', '8', '--predictor-hidden', '8', '--value-hidden', '8',
                  '--sigreg-projections', '4', '--checkpoint-out', str(checkpoint)])
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        report = json.loads(checkpoint.with_suffix('.training.json').read_text())
        for source in (saved['on_policy_source'], report['on_policy_source']):
            self.assertEqual(len(source['behavior_checkpoints']), 2)
            self.assertNotIn('behavior_checkpoint', source)
            self.assertEqual(source['on_policy_sources'][1]['on_policy_rows'], [3, 5])
        self.assertEqual(saved['optimizer_steps'], 3)
        self.assertEqual(report['history'][0]['train']['on_policy_samples'], 3)


if __name__ == '__main__':
    unittest.main()
