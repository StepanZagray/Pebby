import unittest

import numpy as np

from pebby.agent import world_train as train
from pebby.agent.curriculum_sampling import DEFAULT_END, DEFAULT_START, CurriculumSampler


def coverage_data(seeds=(10, 10, 20, 20), wins=(True, False, False, True), terminals=None,
                  metadata_wins=None):
    seeds = np.asarray(seeds, dtype=np.int32)
    wins = np.asarray(wins, dtype=bool)[:, None]
    won = np.repeat(wins, 4, axis=1)
    if terminals is None:
        terminals = won.copy()
    else:
        terminals = np.asarray(terminals, dtype=bool)[:, None]
        terminals = np.repeat(terminals, 4, axis=1)
    return {"seeds": seeds, "won": won, "terminal": terminals,
            "meta": {"levels": metadata_wins or []}}


class VerifiedProvenanceTests(unittest.TestCase):
    def data(self, **changes):
        proof = {"seed": 10, "context_index": 3, "context_engine_verified": True,
                 "search_truncated": False}
        proof.update(changes)
        return {"seeds": np.array([10]), "meta": {"source": "generated_only",
                "oracle_search": "complete_only", "levels": [proof]}}

    def test_explicit_boolean_complete_proof_is_accepted(self):
        train.require_verified_data(self.data())

    def test_missing_or_nonboolean_proof_flags_are_rejected(self):
        for field in ("context_engine_verified", "search_truncated"):
            for value in (None, "false", "true", 0, 1):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, "untruncated contextual"):
                        train.require_verified_data(self.data(**{field: value}))
            data = self.data()
            del data["meta"]["levels"][0][field]
            with self.subTest(missing=field), self.assertRaisesRegex(ValueError, "untruncated contextual"):
                train.require_verified_data(data)


class WinningCoverageTests(unittest.TestCase):
    def test_actual_winning_successor_covers_every_seed(self):
        train.require_winning_coverage(coverage_data(), "train")

    def test_metadata_claim_does_not_replace_a_missing_won_row(self):
        metadata = [{"seed": 10, "won": True}, {"seed": 20, "won": True}]
        data = coverage_data(wins=(False, False, False, False), metadata_wins=metadata)
        with self.assertRaisesRegex(ValueError, "20"):
            train.require_winning_coverage(data, "training data")

    def test_missing_seed_is_reported_even_when_another_seed_wins(self):
        data = coverage_data(seeds=(10, 10, 20, 20), wins=(True, False, False, False))
        with self.assertRaisesRegex(ValueError, "20"):
            train.require_winning_coverage(data, "validation data")

    def test_won_must_imply_terminal(self):
        data = coverage_data(terminals=(False, False, True, True))
        with self.assertRaisesRegex(ValueError, "also be terminal"):
            train.require_winning_coverage(data)


class CurriculumScheduleTests(unittest.TestCase):
    def test_parser_defaults_select_schedule_from_data_version(self):
        args = train.build_parser().parse_args(["--train", "train.npz"])
        self.assertIsNone(args.curriculum_start)
        self.assertIsNone(args.curriculum_end)
        self.assertFalse(args.require_winning_coverage)
        self.assertFalse(args.state_recall)
        self.assertFalse(args.grounding)

    def test_state_recall_flag_is_opt_in(self):
        args = train.build_parser().parse_args(["--train", "train.npz", "--state-recall"])
        self.assertTrue(args.state_recall)
        self.assertFalse(args.grounding)

    def test_custom_schedule_is_normalized_in_checkpoint_metadata(self):
        data = {"seeds": np.arange(10), "meta": {"levels": [
            {"seed": int(seed), "difficulty": int(seed % 5 + 1)}
            for seed in range(10)]}}
        sampler = CurriculumSampler(data, start=[2, 0, 0, 0, 0],
                                    end=[0, 0, 0, 0, 4])
        metadata = train.curriculum_metadata(sampler)
        self.assertEqual(metadata["unit"], "distinct_level")
        self.assertAlmostEqual(sum(metadata["start"]), 1.)
        self.assertAlmostEqual(sum(metadata["end"]), 1.)
        self.assertGreater(metadata["start"][0], metadata["end"][0])
        self.assertLess(metadata["start"][4], metadata["end"][4])


if __name__ == "__main__":
    unittest.main()
