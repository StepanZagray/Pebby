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
    def test_parser_defaults_preserve_existing_schedule(self):
        args = train.build_parser().parse_args(["--train", "train.npz"])
        self.assertEqual(tuple(args.curriculum_start), DEFAULT_START)
        self.assertEqual(tuple(args.curriculum_end), DEFAULT_END)
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
