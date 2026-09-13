"""Actual small generated v4 proofs and fail-closed three-split admission."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from pebby.ls20.generate import generate_level
from pebby.ls20.generation_quality import geometry_d4_partition
from tools import audit_generated_banks as audit_tool


class GeneratedBankV4AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {split: generate_level(seed, 1, attempts=80, generator_version=4)
                    for split, seed in [('train', 0), ('validation', 1_000_000), ('test', 2_000_000)]}

    def audit(self, train=None, validation=None, test=None):
        return audit_tool.audit([self.rows['train']] if train is None else train,
                                [self.rows['validation']] if validation is None else validation,
                                min_validation=1, coverage_policy='sample', testrows=test)

    def test_real_v4_pair_and_third_split_keep_native_proof_and_version_contracts(self):
        for test in (None, [self.rows['test']]):
            result = self.audit(test=test)
            self.assertEqual(result['status'], 'complete', result['errors'])
            self.assertEqual(result['versions']['train'], result['versions']['validation'])
            self.assertEqual(result['overlap'], dict(seeds=0, gameplay=0, geometry=0, geometry_d4=0))
            if test:
                self.assertEqual(result['versions']['train'], result['versions']['test'])
                self.assertEqual(set(result['pairwise_overlap']), {'train/validation', 'train/test', 'validation/test'})
                self.assertTrue(all(not any(value.values()) for value in result['pairwise_overlap'].values()))
            proofs = audit_tool.spotcheck(list(self.rows.values()) if test else list(self.rows.values())[:2], 3)
            self.assertTrue(all(proof['engine_win_three_lives'] for proof in proofs))
            self.assertTrue(all(proof['same_planner_search_agrees'] for proof in proofs))

    def test_wrong_mechanics_geometry_family_and_nested_proofs_fail_closed(self):
        parent = self.rows['train']
        changes = [dict(mechanics_version='unknown'), dict(mechanics_version=None),
                   dict(geometry_version='dihedral-v1'), dict(generator_version=3),
                   dict(quality_version=2), dict(extended_curriculum_version=3),
                   dict(geometry_split='validation'), dict(split='test'),
                   dict(search_truncated=True), dict(context_engine_verified=False),
                   dict(proof={**parent['proof'], 'generator_version': 3}),
                   dict(proof={key: value for key, value in parent['proof'].items() if key != 'mechanics_version'})]
        for change in changes:
            with self.subTest(change=change):
                result = self.audit(train=[{**parent, **change}])
                self.assertEqual(result['status'], 'failed_closed')
        binary_hash, binary_split = geometry_d4_partition(parent)
        broken = copy.deepcopy(parent)
        broken.update(geometry_sha256=binary_hash, geometry_d4_sha256=binary_hash,
                      geometry_version='dihedral-v1', geometry_split=binary_split)
        broken['proof'].update(geometry_version='dihedral-v1', geometry_split=binary_split)
        self.assertEqual(self.audit(train=[broken])['status'], 'failed_closed')

    def test_mixed_v3_v4_splits_fail_even_with_each_rows_valid_native_proof(self):
        legacy = generate_level(720000, 1, attempts=20, search_limit=20000)
        result = self.audit(train=[legacy])
        self.assertEqual(result['status'], 'failed_closed')
        self.assertTrue(any('versions do not match' in error for error in result['errors']))

    def test_four_fingerprint_duplicates_are_found_for_every_split_pair_before_metadata_rejection(self):
        expected = dict(seeds=1, gameplay=1, geometry=1, geometry_d4=1)
        # Copies are deliberately invalid for their destination split. Duplicate
        # detection must still report the actual identity, before proof rejection.
        result = self.audit(validation=[self.rows['train']], test=[self.rows['train']])
        self.assertEqual(result['status'], 'failed_closed')
        for pair in ('train/validation', 'train/test', 'validation/test'):
            self.assertEqual(result['pairwise_overlap'][pair], expected)
        result = self.audit(test=[self.rows['validation']])
        self.assertEqual(result['pairwise_overlap']['validation/test'], expected)

    def test_test_seed_mislabel_and_empty_or_undersized_test_are_rejected(self):
        parent = copy.deepcopy(self.rows['test'])
        parent['seed'] = 17
        parent['proof']['seed'] = 17
        result = self.audit(test=[parent])
        self.assertTrue(any('seed range' in error for error in result['errors']))
        missing = {key: value for key, value in self.rows['test'].items() if key != 'split'}
        self.assertEqual(self.audit(test=[missing])['status'], 'failed_closed')
        self.assertEqual(self.audit(test=[])['status'], 'failed_closed')
        result = audit_tool.audit([self.rows['train']], [self.rows['validation']], 1,
                                 coverage_policy='sample', testrows=[self.rows['test']], min_test=2)
        self.assertTrue(any('test count 1 below required 2' in error for error in result['errors']))

    def test_three_bank_cli_replays_all_splits_without_changing_any_input(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            args = []
            snapshots = {}
            for split, row in self.rows.items():
                path = folder / (split + '.jsonl')
                path.write_text(json.dumps(row) + '\n')
                snapshots[path] = path.read_bytes()
                args.extend(['--' + split, str(path)])
            report = folder / 'report.json'
            args.extend(['--report', str(report), '--min-validation', '1', '--min-test', '1',
                         '--coverage-policy', 'sample', '--spotcheck', '1'])
            with redirect_stdout(io.StringIO()):
                self.assertEqual(audit_tool.main(args), 0)
            result = json.loads(report.read_text())
            self.assertEqual(result['status'], 'complete', result['errors'])
            self.assertEqual(set(result['spotchecks']), {'train', 'validation', 'test'})
            self.assertTrue(result['input_hashes_unchanged'])
            self.assertTrue(any(name.endswith('reference_generator_v2.py') for name in result['audit_code_sha256']))
            self.assertTrue(all(path.read_bytes() == value for path, value in snapshots.items()))


if __name__ == '__main__':
    unittest.main()
