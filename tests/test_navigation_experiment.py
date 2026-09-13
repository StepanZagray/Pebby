"""Small real-engine integration checks for the research experiment boundary."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pebby.agent.navigation_diagnostics import make_cases, collect_examples
from tests.test_spatial_outcome_policy import checkpoint, make_policy
from tools import train_navigation_probe as train
from tools import evaluate_navigation_probe as evaluate


class NavigationExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_sampler_balances_cases_and_stages_instead_of_long_paths(self):
        cases = [dict(stage='adjacent'),dict(stage='adjacent'),dict(stage='detour')]
        arrays = dict(case_index=np.array([0]+[1]*20+[2]*40))
        selected = train.sample_indices(arrays,cases,8000,np.random.default_rng(3),['adjacent','detour'])
        counts = np.bincount(arrays['case_index'][selected],minlength=3)/len(selected)
        np.testing.assert_allclose(counts,[.25,.25,.5],atol=.025)
        other = train.sample_indices(arrays,cases,8000,np.random.default_rng(3),['adjacent','detour'])
        np.testing.assert_array_equal(selected,other)

    def test_mastery_requires_heldout_actions_and_gameplay(self):
        roots = dict(per_stage=dict(adjacent=dict(cases=4,first_accuracy=1.,case_accuracy=1.)))
        games = dict(per_stage=dict(adjacent=dict(episodes=4,win_rate=.5)))
        self.assertFalse(train.mastered('adjacent',roots,games,.95,.9))
        games['per_stage']['adjacent']['win_rate'] = 1.
        self.assertTrue(train.mastered('adjacent',roots,games,.95,.9))
        roots['per_stage']['adjacent']['case_accuracy'] = .8
        self.assertFalse(train.mastered('adjacent',roots,games,.95,.9))
        self.assertFalse(train.mastered('missing',roots,games,.95,.9))

    def test_cli_rejects_confounded_or_unbounded_comparisons(self):
        base = ['--parent','unused','--parent-sha256','0'*64,'--out','unused']
        for options in (['--objective','outcomes'],['--schedule','mastery'],['--recollect-every','2'],
                        ['--controller-init','retained'],['--max-seconds','nan'],['--steps','0'],['--width','0']):
            with self.subTest(options=options),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                train.arguments(base+options)

    def test_rollout_selection_covers_opposite_cardinal_and_diagonal_goals(self):
        cases=make_cases(seed=5,groups_per_split=2)['development']
        selected=train.select_rollout_cases(cases,4)
        directions={c['direction'] for c in selected if c['stage']=='open'}
        self.assertEqual(directions,{'up','down','upper-left','lower-right'})
        self.assertEqual(len({c['id'] for c in selected}),len(selected))
        self.assertEqual(selected,train.select_rollout_cases(cases,4))

    def test_hash_gate_precedes_model_loading_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); parent=root/'parent.pt'; parent.write_bytes(b'bad checkpoint')
            with patch('pebby.agent.spatial_outcome_policy.load_checkpoint') as loader:
                with self.assertRaisesRegex(ValueError,'SHA256'):
                    train.main(['--parent',str(parent),'--parent-sha256','0'*64,'--out',str(root/'run')])
                loader.assert_not_called()
            self.assertFalse((root/'run').exists())

    def test_matched_real_training_roundtrip_and_separate_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); parent=root/'parent.pt'; out=root/'run'
            torch.manual_seed(22)
            torch.save(checkpoint(make_policy()),parent)
            calls=[]
            def tracked_collect(cases,**kwargs):
                calls.append({c['split'] for c in cases})
                return collect_examples(cases,**kwargs)
            argv=['--parent',str(parent),'--parent-sha256',train.digest(parent),'--out',str(out),
                  '--steps','2','--eval-every','2','--batch-size','4','--groups-per-split','1',
                  '--stages','adjacent','--rollouts-per-stage','1','--rollout-cap','2','--max-seconds','90']
            with patch('pebby.agent.navigation_diagnostics.collect_examples',side_effect=tracked_collect),contextlib.redirect_stdout(io.StringIO()):
                report=train.main(argv)
            self.assertEqual(calls,[{'train'},{'development'}])
            self.assertEqual(report['status'],'complete')
            self.assertFalse(report['confirmation_used'])
            self.assertFalse(report['promoted'])
            self.assertTrue(report['matched_initialization_and_batches'])
            for name,arm in report['arms'].items():
                self.assertEqual(len(arm['steps']),2)
                self.assertEqual([x['step'] for x in arm['evaluations']],[0,2])
                self.assertEqual(bool(arm['updates']['encoder']['changed_tensors']),name.startswith('finetune'))
                self.assertEqual(train.digest(arm['checkpoint']),arm['checkpoint_sha256'])
                policy,metadata=evaluate.load_policy(arm['checkpoint'],'cpu')
                self.assertEqual(metadata['arm'],name)
                self.assertEqual(policy.config()['decision_architecture'],'navigation_probe')
            arm=report['arms']['frozen-direct']
            with contextlib.redirect_stdout(io.StringIO()):
                confirmation=evaluate.main(['--checkpoint',arm['checkpoint'],'--checkpoint-sha256',arm['checkpoint_sha256'],
                    '--bank',str(out/'bank.json'),'--bank-sha256',report['bank_sha256'],'--split','confirmation',
                    '--out',str(root/'confirmation.json'),'--rollout-cap','2','--max-seconds','60'])
            self.assertEqual(confirmation['status'],'complete')
            self.assertTrue(confirmation['confirmation_exposure'])
            self.assertEqual(confirmation['rollouts']['episodes'],4)
            # No mutation of the training record by separate confirmation.
            self.assertFalse(json.loads((out/'report.json').read_text())['confirmation_used'])

    def test_failed_run_keeps_machine_readable_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); parent=root/'parent.pt'; out=root/'run'
            torch.save(checkpoint(make_policy()),parent)
            with patch('pebby.agent.navigation_diagnostics.make_cases',side_effect=RuntimeError('test interruption')):
                with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'test interruption'):
                    train.main(['--parent',str(parent),'--parent-sha256',train.digest(parent),'--out',str(out)])
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertIn('test interruption',report['error'])
            self.assertIn('pid',report)

    def test_mastery_recollection_runs_with_public_learner_and_joint_objective(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent=root/'parent.pt';out=root/'run'
            torch.save(checkpoint(make_policy()),parent)
            with contextlib.redirect_stdout(io.StringIO()):
                report=train.main(['--parent',str(parent),'--parent-sha256',train.digest(parent),'--out',str(out),
                    '--arms','finetune-outcomes','--objective','outcomes','--schedule','mastery',
                    '--stages','adjacent','open','--groups-per-split','1',
                    '--steps','2','--eval-every','2','--batch-size','4','--rollouts-per-stage','1','--rollout-cap','2',
                    '--recollect-every','1','--recollect-cases','1','--recollect-cap','2','--max-seconds','90'])
            arm=report['arms']['finetune-outcomes']
            self.assertEqual(report['status'],'complete')
            self.assertEqual(len(arm['recollections']),1)
            self.assertGreater(arm['recollections'][0]['roots'],0)
            self.assertEqual(len(arm['sampled_batches'][1]['recent']),1)
            self.assertEqual(set(arm['steps'][-1]['losses']),{'policy','physical','value','events'})
            self.assertEqual(set(arm['final_mastery']),{'adjacent','open'})

    def test_evaluator_nonfinite_budget_and_partial_confirmation_exposure(self):
        base=['--checkpoint','unused','--checkpoint-sha256','0'*64,'--out','unused','--sequential']
        for value in ('nan','inf'):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                evaluate.main(base+['--max-seconds',value])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent=root/'parent.pt';bank=root/'bank.json';output=root/'report.json'
            torch.save(checkpoint(make_policy()),parent)
            train.write_json(bank,dict(format='pebby.navigation-bank.v1',splits=make_cases(groups_per_split=1,stages=('adjacent',))))
            with patch('pebby.agent.navigation_diagnostics.collect_examples',side_effect=TimeoutError('partial confirmation')):
                with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(TimeoutError):
                    evaluate.main(['--checkpoint',str(parent),'--checkpoint-sha256',train.digest(parent),
                        '--bank',str(bank),'--bank-sha256',train.digest(bank),'--split','confirmation','--out',str(output)])
            report=json.loads(output.read_text())
            self.assertEqual(report['status'],'failed')
            self.assertTrue(report['confirmation_exposure'])


if __name__ == '__main__':
    unittest.main()
