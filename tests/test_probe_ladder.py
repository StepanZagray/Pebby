"""The probe ladder judges both primitives and fails loudly."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from pebby.agent.model import build_policy, save_checkpoint
from tools import probe_ladder


def tiny_world():
    return build_policy({"architecture": "world", "channels": 8, "heads": 2,
                         "blocks": 1, "loops": 1, "history": 3, "expansion": 1,
                         "temporal_layers": 1, "hud_channels": 4, "hud_tokens": 2,
                         "latent": 8, "reduce": 1, "predictor_blocks": 1,
                         "predictor_hidden": 8, "value_hidden": 8, "max_distance": 8,
                         "lookahead_depth": 1, "summary": 2, "readout_hidden": 8,
                         "ranker_hidden": 8, "sigreg_projections": 4, "sigreg_knots": 3})


class ProbeLadderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.directory.name) / 'tiny.pt'
        torch.manual_seed(0)
        save_checkpoint(cls.checkpoint, tiny_world())

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()
        torch.set_num_threads(cls.old_threads)

    def test_judge_applies_each_threshold(self):
        navigation = dict(first_action_accuracy=0.8)
        cycler = dict(per_type=dict(leave=dict(accuracy=0.85)), overall=dict(spoil_rate=0.05))
        rows = probe_ladder.judge(navigation, cycler, probe_ladder.DEFAULT_THRESHOLDS)
        self.assertEqual([r['name'] for r in rows],
                         ['empty_room_first_action_accuracy', 'cycler_leave_accuracy', 'cycler_spoil_rate'])
        self.assertTrue(all(r['passed'] for r in rows))
        cycler['overall']['spoil_rate'] = 0.2
        rows = probe_ladder.judge(navigation, cycler, probe_ladder.DEFAULT_THRESHOLDS)
        self.assertEqual([r['passed'] for r in rows], [True, True, False])
        navigation['first_action_accuracy'] = None
        self.assertFalse(probe_ladder.judge(navigation, cycler, probe_ladder.DEFAULT_THRESHOLDS)[0]['passed'])
        self.assertIn('FAIL', probe_ladder.format_table(rows, {'x': None}))

    def test_cli_runs_both_probes_and_exits_on_failure(self):
        out = Path(self.directory.name) / 'ladder.json'
        base = ['--checkpoint', str(self.checkpoint), '--out', str(out), '--count', '4', '--nav-groups', '1',
                '--seed', '1', '--threads', '1']
        code = probe_ladder.main(base + ['--min-empty-room-accuracy', '1.1', '--min-leave-accuracy', '0',
                                         '--max-spoil-rate', '1'])
        self.assertEqual(code, 1)
        report = json.loads(out.read_text())
        self.assertEqual(report['format'], probe_ladder.FORMAT)
        self.assertFalse(report['passed'])
        self.assertEqual([r['passed'] for r in report['checks']], [False, True, True])
        self.assertEqual(report['navigation']['cases'], 16)
        self.assertEqual(report['cycler']['overall']['cases'], 12)
        self.assertEqual(len(report['cycler']['cases']), 12)
        self.assertIn('not a benchmark', report['gate'])
        self.assertEqual(report['thresholds']['min_empty_room_accuracy'], 1.1)
        code = probe_ladder.main(base + ['--min-empty-room-accuracy', '0', '--min-leave-accuracy', '0',
                                         '--max-spoil-rate', '1'])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out.read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
