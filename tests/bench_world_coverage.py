"""Tiny CPU comparison of prefix versus mixed transition collection.

Uses a handful of generated levels only (no bank files, no official levels):
one `generate` level and one curriculum level per difficulty. Reports wall
time and real engine actions per mode. Run with a single thread, e.g.::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python tests/bench_world_coverage.py
"""

import argparse
import json
import sys
import time
from unittest.mock import patch

from pebby.agent import world_data
from pebby.ls20 import generate
from pebby.ls20.curriculum import LEGACY_DIFFICULTIES as DIFFICULTIES, generate_legacy_level as curriculum_level
from pebby.ls20.env import Ls20Scenario


def timed(spec, **kwargs):
    original = Ls20Scenario.perform
    calls = []

    def counted(env, action):
        calls.append(action)
        return original(env, action)

    started = time.perf_counter()
    with patch.object(Ls20Scenario, "perform", counted):
        rows, meta = world_data.collect_level(spec, **kwargs)
    return {"seconds": time.perf_counter() - started, "engine_actions": len(calls),
            "rows": len(rows), "win_rows": meta.get("win_rows", 0),
            "route": meta.get("context_optimal_actions"), "excluded": meta.get("excluded")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="print one JSON document")
    args = parser.parse_args(argv)
    specs = [generate.generate_legacy_level(0, 1)] + [curriculum_level(d, d) for d in DIFFICULTIES]
    report = []
    for spec in specs:
        entry = {"seed": spec["seed"], "difficulty": spec.get("difficulty"),
                 "kind": "curriculum" if "curriculum_version" in spec else "generate"}
        for coverage in world_data.COVERAGE_MODES:
            entry[coverage] = timed(spec, samples=args.samples, history=args.history,
                                    coverage=coverage)
        report.append(entry)
    if args.json:
        json.dump(report, sys.stdout, indent=1)
        print()
        return
    print(f"{'level':>18} {'route':>5} | {'prefix rows':>11} {'actions':>7} {'s':>6} | "
          f"{'mixed rows':>10} {'actions':>7} {'s':>6} {'win rows':>8}")
    for entry in report:
        prefix, mixed = entry["prefix"], entry["mixed"]
        if prefix["excluded"]:
            print(f"{entry['kind']}:{entry['seed']:<8} excluded: {prefix['excluded']}")
            continue
        print(f"{entry['kind'] + ':' + str(entry['seed']):>18} {prefix['route']:>5} | "
              f"{prefix['rows']:>11} {prefix['engine_actions']:>7} {prefix['seconds']:>6.2f} | "
              f"{mixed['rows']:>10} {mixed['engine_actions']:>7} {mixed['seconds']:>6.2f} "
              f"{mixed['win_rows']:>8}")
    print(f"prefix win rows: {sum(e['prefix']['win_rows'] for e in report)}; "
          f"mixed win rows: {sum(e['mixed']['win_rows'] for e in report)}")


if __name__ == "__main__":
    main()
