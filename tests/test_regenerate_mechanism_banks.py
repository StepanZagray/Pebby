import unittest

from tools.generate_mechanism_pilot import MODES
from tools.regenerate_mechanism_banks import jobs_for


class RegenerationJobsTests(unittest.TestCase):
    def test_reserved_blocks_do_not_overlap_existing_seeds_or_each_other(self):
        occupied = {720000, 720100, 8100001}
        train = jobs_for(24, 'train', occupied)
        validation = jobs_for(24, 'validation', occupied)
        blocks = [set(range(job[1], job[1]+64)) for job in train+validation]
        seen = set(occupied)
        for block in blocks:
            self.assertFalse(block & seen)
            seen.update(block)
        self.assertEqual([job[2] for job in train], list(MODES)*2)
        self.assertEqual(train, jobs_for(24, 'train', occupied))

    def test_oversized_request_fails_before_starting_workers(self):
        with self.assertRaises(ValueError):
            jobs_for(3000, 'train', set())


if __name__ == '__main__':
    unittest.main()
