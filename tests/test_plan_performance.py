"""The fast exact planner must be the reference planner, only quicker.

Every case here is generated (curriculum lessons and generator drafts); no
shipped level, layout, route or label is loaded. `plan.simulate` and the
pure-Python `Oracle._search` are the authority: the C kernel in `fastplan` is
checked against them transition by transition and state by state, including
where a search is truncated by its limit, because the generator's specs depend
on exactly what a truncated search reports.

Run as a script for the CPU benchmark that produced
`artifacts/planner-cpu-optimization.json`:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
        tests/test_plan_performance.py --benchmark --out artifacts/planner-cpu-optimization.json
"""

from collections import deque
import functools
import os
import time
import unittest
from unittest.mock import patch

from pebby.ls20 import curriculum, fastplan, generate, names
from pebby.ls20.env import Ls20Env, Ls20Scenario
from pebby.ls20.layout import extract
from pebby.ls20.plan import Oracle, advance, simulate

# (source, seed, difficulty): together these cover static rooms, short and ring
# rails, fog, refills, launchers, two goals, and a search that hits the limit.
CURRICULUM_CASES = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (13, 5))
GENERATOR_CASES = ((103, 3), (104, 4), (105, 5))
TRANSITION_CAP = int(os.environ.get("PEBBY_PLAN_TRANSITION_CAP", 50_000))

_cases = None


def cases():
    """Generated cases as (name, spec, layout, limit), built once per process."""
    global _cases
    if _cases is None:
        found = []
        for seed, difficulty in CURRICULUM_CASES:
            spec = curriculum.generate_level(seed, difficulty)
            found.append((f"curriculum-{seed}-d{difficulty}", spec, spec["search_limit"]))
        for seed, difficulty in GENERATOR_CASES:
            spec = generate.generate_level(seed, difficulty)
            found.append((f"generate-{seed}-d{difficulty}", spec, 600_000))
        _cases = [(name, spec, extract(Ls20Scenario(generate.build_level(spec), spec["seed"] % 7)), limit)
                  for name, spec, limit in found]
    return _cases


def reachable_states(oracle):
    """The reference's discovered set, in discovery order, up to the oracle's limit."""
    seen, queue, order = {oracle.start}, deque([oracle.start]), [oracle.start]
    while queue and len(order) < oracle.limit:
        state = queue.popleft()
        if state[4] == oracle.full_mask:
            continue
        for action in range(4):
            nxt = advance(oracle.layout, state, action, oracle.refills)
            if nxt is not None and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
                order.append(nxt)
    return order


def expanded_sample(oracle, cap):
    """Every state the search expands, or an evenly strided sample of `cap` of them.

    Won states are left out: the search never steps them (the level advances
    there), and one with an overdrawn budget is outside the packed state space.
    """
    expanded = [state for state in reachable_states(oracle) if state[4] != oracle.full_mask]
    stride = max(1, len(expanded) // cap)
    return expanded[::stride]


def same_results(fast, reference):
    """Everything an Oracle reports, compared field by field."""
    return {
        "reachable": fast._reachable == reference._reachable,
        "truncated": fast.truncated == reference.truncated,
        "distances": dict(fast._distance.items()) == reference._distance,
        "solvable": fast.solvable == reference.solvable,
        "optimal_actions": fast.optimal_actions == reference.optimal_actions,
        "solution": fast.solution() == reference.solution(),
    }


class FastPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not fastplan.available():
            raise unittest.SkipTest(f"fast planner unavailable: {fastplan.load_error()}")

    def test_kernel_covers_every_generated_case(self):
        for name, spec, layout, limit in cases():
            with self.subTest(case=name):
                oracle = Oracle(layout, limit=limit)
                self.assertEqual(oracle.engine, "fast", oracle.fallback_reason)

    def test_completed_search_matches_reference_on_every_state(self):
        for name, spec, layout, limit in cases():
            with self.subTest(case=name):
                fast = Oracle(layout, limit=limit, engine="fast")
                reference = Oracle(layout, limit=limit, engine="reference")
                self.assertEqual(same_results(fast, reference),
                                 {key: True for key in same_results(fast, reference)})
                self.assertEqual(len(fast._distance), len(reference._distance))
                self.assertEqual(set(fast._distance), set(reference._distance))
                self.assertEqual(fast._reachable, spec["reachable_states"])
                self.assertEqual(fast.truncated, spec["search_truncated"])
                self.assertEqual(fast.solution(seed=spec["seed"]), spec["solution"])

    def test_every_transition_matches_simulate(self):
        """The kernel's step against the Python rules, on every reachable state."""
        for name, spec, layout, limit in cases():
            with self.subTest(case=name):
                reference = Oracle(layout, limit=limit, engine="reference")
                tables = fastplan.tables_for(layout, reference.refills)
                self.assertIsNotNone(tables)
                compared, outcomes = 0, set()
                for state in expanded_sample(reference, TRANSITION_CAP):
                    for action in range(4):
                        expected = simulate(layout, state, action, reference.refills)
                        actual = fastplan.step(tables, state, action)
                        if expected != actual:
                            self.fail(f"{name}: {state} action {action}: "
                                      f"simulate {expected} but kernel {actual}")
                        compared += 1
                        outcomes.add(expected[1])
                self.assertGreater(compared, 0)
                self.assertIn("won", outcomes)

    def test_truncated_search_matches_reference_including_its_quirks(self):
        for name, spec, layout, limit in cases()[:4] + cases()[-3:-2]:
            reference = Oracle(layout, limit=limit, engine="reference")
            total = reference._reachable
            for cap in (1, 2, 17, 700, total - 1, total, total + 1):
                with self.subTest(case=name, limit=cap):
                    fast = Oracle(layout, limit=cap, engine="fast")
                    small = Oracle(layout, limit=cap, engine="reference")
                    self.assertEqual(same_results(fast, small),
                                     {key: True for key in same_results(fast, small)})
                    self.assertEqual(small.truncated, cap < total)
                    self.assertEqual(fast._reachable, min(cap, total))

    def test_optimal_action_choice_and_live_lookups_agree(self):
        name, spec, layout, limit = cases()[4]  # rails, fog, refill, launcher, two goals
        fast = Oracle(layout, limit=limit, engine="fast")
        reference = Oracle(layout, limit=limit, engine="reference")
        for state in reference._distance:
            self.assertEqual(fast.action_for(state), reference.action_for(state), state)
            self.assertEqual(fast.distance_for(state), reference.distance_for(state))
        env = Ls20Scenario(generate.build_level(spec), spec["seed"] % 7)
        remaining = fast.distance_for(fast.state_of(env))
        self.assertEqual(remaining, len(spec["solution"]))
        for action in spec["solution"][:-1]:
            self.assertEqual(fast.action_at(env), reference.action_at(env))
            env.perform(action)
            remaining -= 1
            self.assertEqual(fast.distance_for(fast.state_of(env)), remaining)

    def test_distance_mapping_behaves_like_the_reference_dict(self):
        name, spec, layout, limit = cases()[3]
        fast = Oracle(layout, limit=limit, engine="fast")
        reference = Oracle(layout, limit=limit, engine="reference")
        self.assertEqual(dict(fast._distance), reference._distance)
        self.assertEqual(fast._distance, reference._distance)
        self.assertIn(fast.start, fast._distance)
        self.assertEqual(fast._distance[fast.start], reference._distance[fast.start])
        cell, shape, color, rot, goals, taken, steps, tick = fast.start
        for foreign in (((99, 99), shape, color, rot, goals, taken, steps, tick),
                        (cell, names.SHAPE_COUNT, color, rot, goals, taken, steps, tick),
                        (cell, shape, color, rot, goals, taken, -steps - 9, tick),
                        (cell, shape, color, rot, goals, taken, steps, 1 << 40),
                        (cell, shape, color, rot, 1 << 40, taken, steps, tick),
                        ("not", "a", "state"), ()):
            self.assertIsNone(fast._distance.get(foreign))
            self.assertNotIn(foreign, fast._distance)
            with self.assertRaises(KeyError):
                fast._distance[foreign]

    def test_generated_specs_are_byte_identical_under_either_engine(self):
        reference_only = functools.partial(Oracle, engine="reference")
        for seed, difficulty in ((7, 3), (8, 4), (9, 5)):
            with self.subTest(source="curriculum", seed=seed):
                expected = curriculum.generate_level(seed, difficulty)
                with patch("pebby.ls20.curriculum.Oracle", reference_only):
                    self.assertEqual(curriculum.generate_level(seed, difficulty), expected)
        with self.subTest(source="generate", seed=11):
            expected = generate.generate_level(11, 3)
            with patch("pebby.ls20.generate.Oracle", reference_only):
                self.assertEqual(generate.generate_level(11, 3), expected)

    def test_auto_falls_back_to_reference_when_kernel_is_missing(self):
        name, spec, layout, limit = cases()[0]
        expected = Oracle(layout, limit=limit, engine="reference")
        with patch.object(fastplan, "available", return_value=False), \
                patch.object(fastplan, "load_error", return_value="no compiler"):
            oracle = Oracle(layout, limit=limit)
            self.assertEqual((oracle.engine, oracle.fallback_reason), ("reference", "no compiler"))
            self.assertEqual(oracle._distance, expected._distance)
            with self.assertRaises(RuntimeError):
                Oracle(layout, limit=limit, engine="fast")
        with patch.object(fastplan, "tables_for", return_value=None):
            oracle = Oracle(layout, limit=limit)
            self.assertEqual(oracle.engine, "reference")
            self.assertIn("packed", oracle.fallback_reason)
        with self.assertRaises(ValueError):
            Oracle(layout, limit=limit, engine="turbo")

    def test_fast_search_is_faster_on_cpu(self):
        name, spec, layout, limit = cases()[4]
        fast = median_cpu(lambda: Oracle(layout, limit=limit, engine="fast"))
        reference = median_cpu(lambda: Oracle(layout, limit=limit, engine="reference"))
        self.assertLess(fast * 3, reference, f"{name}: fast {fast:.4f}s vs reference {reference:.4f}s")


def median_cpu(work, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.process_time()
        work()
        times.append(time.process_time() - start)
    return sorted(times)[len(times) // 2]


def benchmark(out=None, repeats=5, curriculum_levels=20):
    """Per-case and end-to-end single-thread CPU timings, reference vs fast."""
    import json, platform, sys
    report = {"machine": {"python": sys.version.split()[0], "platform": platform.platform(),
                          "processor": platform.processor(), "threads": {
                              key: os.environ.get(key) for key in
                              ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
              "kernel": {"available": fastplan.available(), "load_error": fastplan.load_error(),
                         "library": str(fastplan.library_path())},
              "repeats": repeats, "cases": [], "curriculum_generation": None}
    for name, spec, layout, limit in cases():
        reference = Oracle(layout, limit=limit, engine="reference")
        fast = Oracle(layout, limit=limit, engine="fast")
        before = median_cpu(lambda: Oracle(layout, limit=limit, engine="reference"), repeats)
        after = median_cpu(lambda: Oracle(layout, limit=limit, engine="fast"), repeats)
        report["cases"].append({
            "case": name, "difficulty": spec["difficulty"], "rail_mode": spec.get("rail_mode"),
            "fog": spec["fog"], "refills": len(spec["refills"]), "launchers": len(spec["launchers"]),
            "goals": len(spec["goals"]), "tick_span": layout.tick_span,
            "reachable_states": reference._reachable, "truncated": reference.truncated,
            "solvable_states": len(reference._distance), "optimal_actions": reference.optimal_actions,
            "median_cpu_seconds_before": before, "median_cpu_seconds_after": after,
            "speedup": before / after if after else None, "exact_outputs": same_results(fast, reference)})
        print(f"{name}: states={reference._reachable} before={before:.4f}s after={after:.4f}s "
              f"x{before / after if after else float('inf'):.1f} exact={all(same_results(fast, reference).values())}",
              flush=True)
    seeds = [(seed, seed % 5 + 1) for seed in range(curriculum_levels)]

    def generate_all():
        return [curriculum.generate_level(seed, difficulty) for seed, difficulty in seeds]
    with patch("pebby.ls20.curriculum.Oracle", functools.partial(Oracle, engine="reference")):
        before_specs, before = generate_all(), median_cpu(generate_all, 1)
    after_specs, after = generate_all(), median_cpu(generate_all, 1)
    report["curriculum_generation"] = {
        "levels": curriculum_levels, "seeds": seeds, "cpu_seconds_before": before,
        "cpu_seconds_after": after, "speedup": before / after if after else None,
        "identical_specs": before_specs == after_specs}
    print(f"curriculum generation x{curriculum_levels}: before={before:.2f}s after={after:.2f}s "
          f"x{before / after:.1f} identical={before_specs == after_specs}", flush=True)
    if out:
        with open(out, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {out}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--levels", type=int, default=20)
    args, rest = parser.parse_known_args()
    if args.benchmark:
        benchmark(args.out, args.repeats, args.levels)
    else:
        unittest.main(argv=[__file__] + rest)
