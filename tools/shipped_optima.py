"""Re-derive the shipped-level optima cached in pebby/ls20/shipped.py.

Levels 6 and 7 need minutes and several GiB each, so they run one at a time and
are opt-in. Run with: PYTHONPATH=. .venv/bin/python tools/shipped_optima.py [--all]
"""

import argparse
import resource
import time

from pebby.ls20 import shipped
from pebby.ls20.env import Ls20Env
from pebby.ls20.plan import oracle_for


def check(index):
    env = Ls20Env()
    env.set_level(index)
    started = time.perf_counter()
    oracle = oracle_for(env, limit=shipped.search_limit(index))
    elapsed = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    if not oracle.solvable:
        return f"L{index + 1}: NO PLAN (states={oracle._reachable} truncated={oracle.truncated})"
    solution = oracle.solution()
    replay = Ls20Env()
    replay.set_level(index)
    for action in solution:
        observation = replay.perform(action)
    # Level 7 is last, so the index cannot advance; the engine reports WIN instead.
    won = replay.levels_completed == 1 and (
        replay.level_index == index + 1 or observation.won)
    expected = shipped.optimal(index)
    return (f"L{index + 1}: optimal={len(solution)} (cached {expected}, "
            f"{'MATCH' if len(solution) == expected else 'MISMATCH'}) "
            f"states={oracle._reachable} {elapsed:.0f}s peakRSS={peak:.1f}GiB "
            f"engine_confirms_win={won}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="include levels 6 and 7")
    parser.add_argument("--level", type=int, help="check one level, 1-based")
    args = parser.parse_args()
    if args.level:
        levels = [args.level - 1]
    else:
        levels = range(shipped.LEVEL_COUNT) if args.all else shipped.CHEAP_LEVELS
    for index in levels:
        print(check(index), flush=True)


if __name__ == "__main__":
    main()
