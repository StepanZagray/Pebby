import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'build_spatial_confirmation_under_test',
    ROOT / 'tools/build_spatial_confirmation.py',
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class ConfirmationBuilderHelpersTest(unittest.TestCase):
    def test_development_subset_checks_all_fingerprint_fields(self):
        baseline = {field: {f'{field}-known'} for field in BUILDER.FINGERPRINTS}
        candidate = {field: set(values) for field, values in baseline.items()}
        candidate['seed'].add(101)
        candidate['gameplay_sha256'].add('gameplay-new')
        candidate['geometry_sha256'].add('geometry-new')
        candidate['geometry_d4_sha256'].add('d4-new')

        errors = BUILDER.fingerprint_subset_errors(candidate, baseline)

        self.assertEqual(set(errors), set(BUILDER.FINGERPRINTS))

    def test_incomplete_final_target_is_failure_exit(self):
        self.assertEqual(BUILDER.exit_code_for('complete', 70), 0)
        self.assertEqual(BUILDER.exit_code_for('partial', 70), 1)
        self.assertEqual(BUILDER.exit_code_for('smoke_complete', 14), 0)
        self.assertEqual(BUILDER.exit_code_for('partial', 14), 1)


if __name__ == '__main__':
    unittest.main()
