import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
import torch
from tools.evaluate_structured_workspace_gameplay import validate_fit, paired, summarize, DecisionTimer, checked_bank, checked_fit, digest
from pebby.agent.evaluate import rollout
from pebby.ls20.env import Ls20Scenario


class GameplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def fit(self):
        report = {'status': 'complete', 'smoke': False, 'source': 'generated_only', 'official_inputs_used': False,
                  'sources_unchanged': True, 'paired_selections_exact': True, 'primary_depth': 2,
                  'trained_depths': [1, 2, 4], 'args': {'batch_size': 1024, 'updates': 600}, 'arms': {}}
        for arm, count in [('static', 108097), ('evolving', 201315)]:
            report['arms'][arm] = {'status': 'complete', 'completed_updates': 600, 'active_parameters': count,
                'parameters': 201315, 'depth_draws': {'1': 200, '2': 200, '4': 200},
                'selection_sha256': 'a' * 64, 'checkpoint': {'strict_reload_exact': True}}
        return report

    def test_reject_incomplete_smoke_batch_or_sampling(self):
        validate_fit(self.fit())
        for key, value in [('status', 'running'), ('smoke', True), ('official_inputs_used', True), ('primary_depth', 4)]:
            r = self.fit(); r[key] = value
            with self.assertRaises(ValueError): validate_fit(r)
        r = self.fit(); r['args']['batch_size'] = 512
        with self.assertRaises(ValueError): validate_fit(r)
        r = self.fit(); r['arms']['evolving']['selection_sha256'] = 'b' * 64
        with self.assertRaises(ValueError): validate_fit(r)
        r = self.fit(); r['arms']['evolving']['depth_draws'] = {'1': 0, '2': 300, '4': 300}
        with self.assertRaises(ValueError): validate_fit(r)

    def test_checkpoint_and_source_hash_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); actor = root / 'actor.pt'; actor.write_bytes(b'fixture actor')
            report = self.fit(); report['sources'] = {str(actor): digest(actor)}
            for arm in ('static', 'evolving'):
                provenance = {'arm': arm, 'updates': 600, 'batch_size': 1024, 'smoke': False,
                    'fixed_final': True, 'primary_depth': 2, 'depths': [1, 2, 4],
                    'depth_draws': report['arms'][arm]['depth_draws'], 'selection_sha256': 'a' * 64,
                    'encoder_and_dynamics_frozen': True, 'official_inputs_used': False, 'source': 'generated_only'}
                saved = {'format': 'pebby.structured-workspace-readout.v1', 'actor_checkpoint': str(actor),
                    'actor_sha256': digest(actor), 'sources': report['sources'], 'source_unchanged': True,
                    'official_inputs_used': False, 'config': {'memory_mode': arm}, 'training_provenance': provenance}
                checkpoint = root / (arm + '.pt'); torch.save(saved, checkpoint)
                report['arms'][arm]['checkpoint'] = {'path': str(checkpoint), 'sha256': digest(checkpoint), 'strict_reload_exact': True}
            path = root / 'fit.json'; path.write_text(json.dumps(report))
            with patch('tools.evaluate_structured_workspace_gameplay.ACTOR', str(actor)):
                checked_fit(path)
                (root / 'evolving.pt').write_bytes(b'replaced')
                with self.assertRaisesRegex(ValueError, 'checkpoint SHA'): checked_fit(path)
                actor.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'source changed'): checked_fit(path)

    def test_paired_gains_and_long_route_stratum(self):
        def row(seed, win, optimum):
            return {'seed': seed, 'completed': win, 'optimal': optimum, 'losses': 0 if win else 3,
                    'stalls': 0, 'goals_cleared': int(win), 'goals_total': 1, 'actions': 5,
                    'ending': 'win' if win else 'game_over'}
        a = [row(1, True, 5), row(2, False, 17)]
        b = [row(1, False, 5), row(2, True, 17)]
        result = paired(a, b)
        self.assertEqual(result['all']['gained_wins'], [1])
        self.assertEqual(result['initial_optimal_ge17']['lost_wins'], [2])
        self.assertEqual(summarize(a)['initial_optimal_ge17']['levels'], 1)
        with self.assertRaises(ValueError): paired(a, b[::-1])

    def test_actual_generated_public_h8_one_action_software_only(self):
        # A deterministic stub checks evaluator plumbing, NOT model performance.
        class PublicStub(torch.nn.Module):
            def __init__(self): super().__init__(); self.calls = []
            def config(self): return {'architecture': 'structured', 'history': 8}
            def forward(self, frames, history_valid=None, previous_actions=None):
                self.calls.append((frames.shape, history_valid.clone(), previous_actions.clone()))
                return torch.tensor([[1., 0., 0., 0.]])
        levels, optima, specs = checked_bank()
        self.assertEqual(len(levels), 100)
        model = PublicStub(); timer = DecisionTimer(model, 'cpu')
        try:
            run = rollout(model, Ls20Scenario(levels[0], specs[0]['training_context_index']),
                          1, 'cpu', optima[0], 'repeat', temperature=0.)
        finally: timer.close()
        self.assertEqual(run['actions'], 1)
        self.assertEqual(len(timer.durations), 1)
        self.assertEqual(len(model.calls), 1)
        shape, valid, actions = model.calls[0]
        self.assertEqual(tuple(shape), (1, 8, 64, 64))
        self.assertEqual(int(valid.sum()), 1)
        self.assertTrue((actions == -1).all())
        self.assertFalse(model._forward_hooks)
        self.assertFalse(model._forward_pre_hooks)


if __name__ == '__main__': unittest.main()
