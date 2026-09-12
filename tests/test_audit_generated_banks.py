"""Read-only generated-bank auditing, including real contextual spotchecks."""
from contextlib import redirect_stdout
import io
import copy
import os
import json
from pathlib import Path
import tempfile
import unittest

from pebby.ls20 import extended_curriculum as extended
from tools import audit_generated_banks as bank_audit


class GeneratedBankAuditTests(unittest.TestCase):
    def audit_sample(self, *args, **kwargs):
        return bank_audit.audit(*args, coverage_policy='sample', **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.train = extended.generate_legacy_level(20001, 1)[0]
        cls.validation = extended.generate_legacy_level(1020001, 1)[0]
        assert cls.train and cls.validation

    def test_real_rows_pass_and_default_validation_floor_is_not_silently_relaxed(self):
        result = self.audit_sample([self.train], [self.validation], min_validation=1)
        self.assertEqual(result['status'], 'complete', result['errors'])
        self.assertEqual(result['overlap'], {'seeds': 0, 'gameplay': 0, 'geometry': 0, 'geometry_d4': 0})
        self.assertEqual(result['splits']['train']['quality_profiles']['learning']['floor'], 8)
        self.assertEqual(result['splits']['train']['per_mode']['extended']['distinct_starts'], 1)
        self.assertEqual(self.audit_sample([self.train], [self.validation])['status'], 'failed_closed')
        self.assertEqual(self.audit_sample([None], [self.validation], 1)['status'], 'failed_closed')

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
                result = self.audit_sample([{**self.train, **change}], [self.validation], 1)
                self.assertEqual(result['status'], 'failed_closed')
                self.assertTrue(result['errors'])

    def test_overlap_is_reported_even_when_the_reused_row_has_invalid_split_metadata(self):
        result = self.audit_sample([self.train], [self.train], 1)
        self.assertEqual(result['status'], 'failed_closed')
        self.assertEqual(result['overlap'], {'seeds': 1, 'gameplay': 1, 'geometry': 1, 'geometry_d4': 1})
        result = self.audit_sample([self.train, self.train], [self.validation], 1)
        self.assertTrue(any('duplicate seeds' in error for error in result['errors']))

    def test_pilot_and_training_strings_share_the_same_mechanism_quality_version(self):
        rows = []
        for row, label in ((self.train, 'mechanism-training-v2'), (self.validation, 'mechanism-pilot-v2')):
            row = {**row, 'quality_version': 2, 'curriculum_version': label}
            del row['extended_curriculum_version']
            rows.append(row)
        result = self.audit_sample(rows[:1], rows[1:], 1)
        self.assertEqual(result['status'], 'complete', result['errors'])
        self.assertEqual(result['versions']['train'], result['versions']['validation'])

    def test_challenge_rows_use_their_own_floor(self):
        challenge = extended.generate_legacy_level(20001, 1, quality_profile='challenge')[0]
        self.assertIsNotNone(challenge)
        result = self.audit_sample([challenge], [self.validation], 1)
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

    def test_default_full_policy_does_not_certify_singleton_coverage(self):
        result = bank_audit.audit([self.train], [self.validation], 1)
        self.assertEqual(result['status'], 'failed_closed')
        self.assertTrue(any('missing difficulty coverage' in e for e in result['errors']))
        self.assertEqual(result['coverage_policy'], 'full')

    def test_reference_profile_uses_tier_context_d4_and_exact_composition(self):
        from pebby.ls20.reference_generator import generate_level
        train = generate_level(720000, 1, attempts=20, search_limit=20000)
        validation = generate_level(8100000, 1, attempts=20, search_limit=20000)
        self.assertNotEqual(train['seed'] % 7, train['training_context_index'])
        self.assertEqual(train['distractor_count'], 0)
        result = self.audit_sample([train], [validation], 1)
        self.assertEqual(result['status'], 'complete', result['errors'])
        for change in ({'difficulty': 7}, {'geometry_d4_sha256': 'wrong'},
                       {'training_context_index': train['seed'] % 7}, {'proof': None}):
            with self.subTest(change=change):
                self.assertEqual(self.audit_sample([{**train, **change}], [validation], 1)['status'], 'failed_closed')

    def test_difficulty_and_nested_proof_mutations_fail_closed(self):
        proof = dict(seed=self.train['seed'], context_index=self.train['training_context_index'],
                     context_engine_verified=True, search_truncated=False,
                     context_optimal_actions=self.train['context_optimal_actions'],
                     oracle_backend=self.train['oracle_backend'])
        valid = {**self.train, 'proof': proof}
        self.assertEqual(self.audit_sample([valid], [self.validation], 1)['status'], 'complete')
        for change in ({'difficulty': 99}, {'difficulty': True},
                       {'proof': {**proof, 'context_index': 99}},
                       {'proof': {**proof, 'context_engine_verified': 'verified'}},
                       {'proof': {**proof, 'search_truncated': 0}}, {'proof': {}}):
            with self.subTest(change=change):
                self.assertEqual(self.audit_sample([{**valid, **change}], [self.validation], 1)['status'], 'failed_closed')

    def test_degenerate_geometry_and_attribute_banks_fail_diversity_gates(self):
        # Alter only provenance seeds; retaining geometry is precisely the regression.
        rows = [{**copy.deepcopy(self.train), 'seed': self.train['seed'] + 7*i} for i in range(8)]
        result = self.audit_sample(rows, [self.validation], 1)
        self.assertTrue(any('D4 geometry diversity' in e for e in result['errors']))
        self.assertTrue(any('changing-attribute diversity collapsed' in e for e in result['errors']))

    def test_d4_canonicalization_identifies_rotated_and_mirrored_rooms(self):
        free = {(2, 2), (3, 2), (4, 2), (2, 3), (3, 3), (2, 4)}
        original = {'walls': list(bank_audit.BOARD - free)}
        for transformed in ({(9-y, x+1) for x, y in free}, {(10-x, y+2) for x, y in free}):
            other = {'walls': list(bank_audit.BOARD - transformed)}
            self.assertEqual(bank_audit._d4_hash(original), bank_audit._d4_hash(other))
        # Cross-split detection operates before any stored-proof rejection.
        rows = [{**self.train, **original}, {**self.validation, 'walls': other['walls']}]
        result = self.audit_sample(rows[:1], rows[1:], 1)
        self.assertEqual(result['overlap']['geometry_d4'], 1)

    def test_spotcheck_selection_covers_strata_before_repeating(self):
        rows = [dict(difficulty=tier, pilot_mode=f'mode{tier}', quality_profile='challenge' if tier > 5 else 'learning')
                for tier in range(1, 8) for _ in range(20)]
        indices = bank_audit._spotcheck_indices(rows, 7)
        self.assertEqual({rows[i]['difficulty'] for i in indices}, set(range(1, 8)))
        self.assertEqual(indices, bank_audit._spotcheck_indices(rows, 7))
        self.assertNotEqual(indices, sorted(indices))

    def test_generation_report_binding_and_distinct_rejection_denominators(self):
        inputs = {'train': [('train-hash', 2004)], 'validation': [('val-hash', 504)]}
        report = dict(status='complete', banks={split: dict(sha256=values[0][0], levels=values[0][1])
                                               for split, values in inputs.items()},
                      rejection_counts={'incomplete contextual oracle (search limit)': 11599,
                                        'unsolved contextual oracle': 10634, 'other': 9816})
        evidence = bank_audit._generation_evidence(report, inputs)
        self.assertEqual(evidence['search_cap_rejections'], 11599)
        self.assertEqual(evidence['unsolved_rejections'], 10634)
        self.assertAlmostEqual(evidence['search_cap_fraction_of_rejections'], 11599/32049)
        self.assertAlmostEqual(evidence['cap_or_unsolved_fraction_of_rejections'], 22233/32049)
        report['banks']['train']['sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'hash/count mismatch'):
            bank_audit._generation_evidence(report, inputs)

    def test_spotchecks_research_optimal_length_and_replay_stored_routes(self):
        rows = [self.train, self.validation]
        proofs = bank_audit.spotcheck(rows, 2)
        self.assertEqual(len(proofs), 2)
        self.assertTrue(all(proof['engine_win_three_lives'] for proof in proofs))
        self.assertTrue(all(proof['same_planner_search_agrees'] for proof in proofs))
        self.assertTrue(all(proof['independent_optimality_verified'] is False for proof in proofs))
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
                    '--min-validation', '1', '--spotcheck', '1', '--coverage-policy', 'sample']
            previous = Path.cwd()
            try:
                os.chdir(root)
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(bank_audit.main(args), 0)
            finally:
                os.chdir(previous)
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
