"""Public generation is seven-tier; historical regression fixtures opt in explicitly."""
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
import io

from pebby.ls20 import generate, curriculum, extended_curriculum as extended
from pebby.ls20.reference_profiles import DIFFICULTY_VERSION, profile_errors


class PublicGenerationTests(unittest.TestCase):
    def test_all_public_entrypoints_route_every_tier_to_reference_generator(self):
        self.assertEqual(tuple(generate.DIFFICULTY), tuple(range(1, 8)))
        for module in (generate, curriculum, extended):
            self.assertEqual(module.DIFFICULTIES, tuple(range(1, 8)))
            self.assertEqual(module.LEGACY_DIFFICULTIES, tuple(range(1, 6)))
            for difficulty in module.DIFFICULTIES:
                expected = {'difficulty': difficulty, 'difficulty_version': DIFFICULTY_VERSION}
                with self.subTest(module=module.__name__, difficulty=difficulty), patch(
                        'pebby.ls20.reference_generator.generate_level', return_value=expected) as factory:
                    actual = module.generate_level(720000, difficulty)
                    self.assertEqual(actual[0] if module is extended else actual, expected)
                    self.assertEqual(factory.call_args.args, (720000, difficulty))
                    self.assertIsNone(factory.call_args.kwargs['search_limit'])
                    self.assertEqual(factory.call_args.kwargs['attempts'], 400)

    def test_public_defaults_generate_real_reference_proofs(self):
        for module in (generate, curriculum, extended):
            with self.subTest(module=module.__name__):
                result = module.generate_level(720000, 1, attempts=20, search_limit=20000)
                row = result[0] if module is extended else result
                self.assertEqual(row['difficulty_version'], DIFFICULTY_VERSION)
                self.assertEqual(row['training_context_index'], 0)
                self.assertEqual(profile_errors(row), [])
                self.assertIs(row['context_engine_verified'], True)

    def test_extended_namespace_profile_and_failure_contracts(self):
        for seed in (0, 1000000, 2000000):
            with self.assertRaisesRegex(ValueError, 'namespace'):
                extended.generate_level(seed)
        with self.assertRaisesRegex(ValueError, 'fixed by'):
            extended.generate_level(720000, 2, quality_profile='learning')
        with patch('pebby.ls20.reference_generator.generate_level',
                   side_effect=RuntimeError('no reference-profile level seed=720000')):
            self.assertEqual(extended.generate_level(720000), (None, {}))
        for error in (extended.ContractMismatch('engine mismatch'), RuntimeError('unexpected failure')):
            with patch('pebby.ls20.reference_generator.generate_level', side_effect=error):
                with self.assertRaisesRegex(RuntimeError, str(error)):
                    extended.generate_level(720000)

    def test_extended_cli_defaults_to_resumable_reference_path(self):
        with patch('tools.regenerate_mechanism_banks.main', return_value=0) as public:
            self.assertEqual(extended.main(['--train-count', '7']), 0)
            public.assert_called_once_with(['--train-count', '7'])
        with patch.object(extended, 'legacy_main', return_value=0) as legacy:
            self.assertEqual(extended.main(['--legacy', '--train-count', '5']), 0)
            legacy.assert_called_once_with(['--train-count', '5'])

    def test_curriculum_cli_cycles_all_seven_with_tier_specific_defaults(self):
        def row(seed, difficulty, attempts, min_slack, search_limit):
            self.assertEqual(attempts, 400)
            self.assertIsNone(min_slack)
            self.assertIsNone(search_limit)
            return dict(seed=seed, difficulty=difficulty, reachable_states=1, optimal_actions=10)
        with patch('sys.argv', ['curriculum', '--levels', '7', '--out', '/unused.jsonl']), \
                patch.object(curriculum, 'generate_level', side_effect=row) as factory, \
                patch('pebby.ls20.bank.save'), redirect_stdout(io.StringIO()):
            curriculum.main()
        self.assertEqual([call.args[1] for call in factory.call_args_list], list(range(1, 8)))

    def test_historical_extender_requires_explicit_legacy_mode(self):
        from tools import extend_extended_bank
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            extend_extended_bank.main([])
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
