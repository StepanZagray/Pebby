"""Read-only generated-bank auditing, including real contextual spotchecks."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from pebby.ls20 import extended_curriculum as extended
from tools import audit_generated_banks as bank_audit


class GeneratedBankAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = extended.generate_level(20001, 1)[0]
        cls.validation = extended.generate_level(1020001, 1)[0]
        assert cls.train and cls.validation

    def test_real_rows_pass_and_default_validation_floor_is_not_silently_relaxed(self):
        result = bank_audit.audit([self.train], [self.validation], min_validation=1)
        self.assertEqual(result['status'], 'complete', result['errors'])
        self.assertEqual(result['overlap'], {'seeds': 0, 'gameplay': 0, 'geometry': 0})
        self.assertEqual(result['splits']['train']['quality_profiles']['learning']['floor'], 8)
        self.assertEqual(result['splits']['train']['per_mode']['extended']['distinct_starts'], 1)
        self.assertEqual(bank_audit.audit([self.train], [self.validation])['status'], 'failed_closed')
        self.assertEqual(bank_audit.audit([None], [self.validation], 1)['status'], 'failed_closed')

    def test_incompatible_proofs_and_spawn_goals_fail_closed(self):
        changes = ({'extended_curriculum_version': 1}, {'generator_version': -1},
                   {'search_truncated': True}, {'engine_verified': False},
                   {'minimum_slack_moves': -1}, {'minimum_slack_moves': 7},
                   {'context_optimal_actions': self.train['optimal_actions'] + 1},
                   {'geometry_sha256': 'wrong'}, {'step_counter': 36},
                   {'training_context_index': (self.train['seed'] + 1) % 7},
                   {'goals': [{**self.train['goals'][0], 'triple': self.train['start_triple']}]})
        for change in changes:
            with self.subTest(change=change):
                result = bank_audit.audit([{**self.train, **change}], [self.validation], 1)
                self.assertEqual(result['status'], 'failed_closed')
                self.assertTrue(result['errors'])

    def test_overlap_is_reported_even_when_the_reused_row_has_invalid_split_metadata(self):
        result = bank_audit.audit([self.train], [self.train], 1)
        self.assertEqual(result['status'], 'failed_closed')
        self.assertEqual(result['overlap'], {'seeds': 1, 'gameplay': 1, 'geometry': 1})
        result = bank_audit.audit([self.train, self.train], [self.validation], 1)
        self.assertTrue(any('duplicate seeds' in error for error in result['errors']))

    def test_pilot_and_training_strings_share_the_same_mechanism_quality_version(self):
        rows = []
        for row, label in ((self.train, 'mechanism-training-v2'), (self.validation, 'mechanism-pilot-v2')):
            row = {**row, 'quality_version': 2, 'curriculum_version': label}
            del row['extended_curriculum_version']
            rows.append(row)
        result = bank_audit.audit(rows[:1], rows[1:], 1)
        self.assertEqual(result['status'], 'complete', result['errors'])
        self.assertEqual(result['versions']['train'], result['versions']['validation'])

    def test_challenge_rows_use_their_own_floor(self):
        challenge = extended.generate_level(20001, 1, quality_profile='challenge')[0]
        self.assertIsNotNone(challenge)
        result = bank_audit.audit([challenge], [self.validation], 1)
        self.assertEqual(result['status'], 'complete', result['errors'])
        self.assertEqual(result['splits']['train']['quality_profiles']['challenge']['floor'], 0)

    def test_previous_action_statistics_never_connect_episodes(self):
        result = bank_audit._actions([{'solution': [1, 1]}, {'solution': [2, 2]}])
        self.assertEqual(result['within_episode_transition_count'], 2)
        self.assertEqual(result['transition_counts'], {'1': {1: 1}, '2': {2: 1}})
        self.assertEqual(result['best_constant_accuracy'], .5)
        self.assertEqual(result['previous_action_markov_accuracy'], 1.)
        self.assertEqual(result['first'], {1: 1, 2: 1})
        self.assertEqual(result['last'], {1: 1, 2: 1})

    def test_spotchecks_research_optimal_length_and_replay_stored_routes(self):
        rows = [self.train, self.validation]
        proofs = bank_audit.spotcheck(rows, 2)
        self.assertEqual(len(proofs), 2)
        self.assertTrue(all(proof['engine_win_three_lives'] for proof in proofs))
        corrupted = {**self.train, 'solution': [1] * len(self.train['solution'])}
        with self.assertRaises(ValueError):
            bank_audit.spotcheck([corrupted], 1)

    def test_cli_writes_exclusive_report_and_never_changes_banks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root/'train.jsonl', root/'validation.jsonl']
            for path, row in zip(paths, (self.train, self.validation)):
                path.write_text(json.dumps(row)+'\n')
            original = [path.read_bytes() for path in paths]
            report = root/'audit.json'
            args = ['--train', str(paths[0]), '--validation', str(paths[1]), '--report', str(report),
                    '--min-validation', '1', '--spotcheck', '1']
            with redirect_stdout(io.StringIO()):
                self.assertEqual(bank_audit.main(args), 0)
            result = json.loads(report.read_text())
            self.assertTrue(result['input_hashes_unchanged'])
            self.assertEqual(result['status'], 'complete')
            self.assertEqual([path.read_bytes() for path in paths], original)
            report_bytes = report.read_bytes()
            with self.assertRaises(FileExistsError):
                bank_audit.main(args)
            self.assertEqual(report.read_bytes(), report_bytes)


if __name__ == '__main__':
    unittest.main()
