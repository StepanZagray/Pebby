"""Ablation: search over the REAL engine as world model, learned network as leaf value.

This is a tool-assisted diagnostic, not a learned controller and not a
competition entry. It answers one question: given a perfect world model (the
vendored game itself, cloned and stepped), does a depth-limited search using
the current network's value head solve the shipped levels? Two controls make
the answer interpretable:

* ``--value oracle``: the exact remaining distance from ``plan.Oracle`` scores
  the leaves. This is the upper bound; if it does not win, the search itself is
  broken.
* ``--value none``: leaves score zero, so only wins reachable within the
  horizon are found. This is what depth alone buys.

Search: every action sequence up to ``--depth`` is expanded on cloned envs,
duplicates (same planner state) are merged, a sequence that wins scores its
length, a life lost adds a large penalty, and a non-terminal leaf scores its
length plus the value estimate. The first action of the best sequence is
played, then the search is repeated from the real state.

Learned value: the network's 130-bin successor-distance head is read at the
leaf on its exact public history (the frames the engine produced along the
imagined path), and the state value is the minimum over actions of the
expected bin, so this uses the same weights the one-step comparator uses.

Example::

    PYTHONPATH=. uv run python tools/ablate_engine_search.py \
        --checkpoint artifacts/spatial-recovery-v1/quality-fit/recovery.pt \
        --value learned --depth 3 --out /tmp/ablate-learned-d3.json
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from pebby.agent.evaluate import shipped_levels
from pebby.agent.model import load_checkpoint
from pebby.agent.world_data import clone_env, history_arrays
from pebby.agent.world_features import encode_features
from pebby.ls20 import layout as layout_module, names, shipped
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.plan import Oracle

LIFE_LOSS_PENALTY = 1_000.
GAME_OVER_PENALTY = 2_000.
UNREACHABLE = 500.


class LearnedValue:
    """State value from the network's successor-distance head; batched over leaves."""

    def __init__(self, checkpoint, device='cpu'):
        self.policy, self.data = load_checkpoint(checkpoint, device)
        self.policy.eval()
        self.device = device
        self.history = self.policy.config().get('history', 8)

    def _predict(self, frames, valid, previous):
        policy = self.policy
        if hasattr(policy, 'predict'):
            return policy.predict(frames, valid, previous)
        encoded = encode_features(policy.encoder, frames, valid, previous)
        player = policy.encoder.player_weights(encoded['cells'])[1]
        return policy.planner(encoded['raw'], encoded['state'], encoded['glyph'], player)

    @torch.inference_mode()
    def values(self, nodes):
        arrays = [history_arrays(n['frames'], n['actions'], self.history) for n in nodes]
        frames = torch.as_tensor(np.stack([a[0] for a in arrays]), device=self.device).long()
        valid = torch.as_tensor(np.stack([a[1] for a in arrays]), device=self.device)
        previous = torch.as_tensor(np.stack([a[2] for a in arrays]), device=self.device)
        out = self._predict(frames, valid, previous)
        bins = torch.arange(130, device=self.device, dtype=torch.float32)
        bins[129] = UNREACHABLE
        expected = (out['value_logits'].float().softmax(-1) * bins).sum(-1)  # [B, 4]
        return expected.min(-1).values.cpu().tolist()


class OracleValue:
    def __init__(self, oracle):
        self.oracle = oracle

    def values(self, nodes):
        out = []
        for n in nodes:
            d = self.oracle.distance_for(self.oracle.state_of(n['env']))
            out.append(UNREACHABLE if d is None else float(d))
        return out


class NoValue:
    def values(self, nodes):
        return [0.] * len(nodes)


def state_key(env):
    return (env.player_cell(), tuple(env.triple()), tuple(env.goals_solved()), env.steps_left(), env.lives())


def search(env, frames, actions, depth, valuer):
    """Return (best first action index, diagnostics) for the real state `env`."""
    root_lives = env.lives()
    frontier = [dict(env=env, frames=list(frames), actions=list(actions), path=[], cost=0., done=False)]
    leaves, expansions = [], 0
    for step in range(depth):
        next_frontier, seen = [], {}
        for node in frontier:
            if node['done']:
                leaves.append(node)
                continue
            for a in range(4):
                child_env = clone_env(node['env'])
                result = child_env.perform(names.ACTION_IDS[a])
                expansions += 1
                frame = np.asarray(result.frame if result.frame is not None else child_env.render())
                cost = node['cost'] + 1.
                done = False
                if result.finished and result.won:
                    done = True
                elif result.finished:
                    cost += GAME_OVER_PENALTY
                    done = True
                elif child_env.lives() < root_lives:
                    cost += LIFE_LOSS_PENALTY
                    done = True  # do not plan through a death
                elif child_env.level_index != env.level_index:
                    done = True  # level cleared
                child_frames = [frame] if child_env.lives() < root_lives else (node['frames'] + [frame])[-8:]
                child_actions = [-1] if child_env.lives() < root_lives else (node['actions'] + [a])[-8:]
                child = dict(env=child_env, frames=child_frames, actions=child_actions,
                             path=node['path'] + [a], cost=cost, done=done, won=bool(result.finished and result.won))
                key = state_key(child_env)
                if done:
                    leaves.append(child)
                elif key not in seen or seen[key]['cost'] > cost:
                    seen[key] = child
        frontier = list(seen.values())
    leaves.extend(frontier)
    open_leaves = [n for n in leaves if not n['done']]
    if open_leaves:
        for n, v in zip(open_leaves, valuer.values(open_leaves)):
            n['cost'] += v
    for n in leaves:
        if n.get('won'):
            n['cost'] = len(n['path']) - 10_000.  # a win within the horizon beats everything
    best = min(leaves, key=lambda n: (n['cost'], n['path']))
    return best['path'][0], dict(expansions=expansions, leaves=len(leaves), best_cost=best['cost'],
                                 best_path=best['path'], won_in_horizon=bool(best.get('won')))


def make_valuer(kind, checkpoint, oracle, device):
    if kind == 'learned':
        return LearnedValue(checkpoint, device)
    if kind == 'oracle':
        return OracleValue(oracle)
    return NoValue()


def oracle_for_level(env, index):
    lay = layout_module.extract(env)
    return Oracle(lay, limit=shipped.search_limit(index), engine='fast')


def play_level(index, kind, checkpoint, depth, cap, device, learned=None):
    level = shipped_levels()[index]
    env = Ls20Scenario(level, index)
    frame = np.asarray(env.reset())
    oracle = oracle_for_level(env, index) if kind == 'oracle' else None
    valuer = learned if kind == 'learned' else make_valuer(kind, checkpoint, oracle, device)
    frames, actions = [frame], [-1]
    stats = Counter()
    trace = []
    started = time.monotonic()
    won = False
    for step in range(cap):
        a, diag = search(env, frames, actions, depth, valuer)
        lives_before = env.lives()
        result = env.perform(names.ACTION_IDS[a])
        frame = np.asarray(result.frame if result.frame is not None else env.render())
        stats['actions'] += 1
        stats['expansions'] += diag['expansions']
        trace.append(dict(step=step, action=names.ACTION_IDS[a], cell=list(env.player_cell()), triple=list(env.triple()),
                          fuel=env.steps_left(), lives=env.lives(), best_cost=round(diag['best_cost'], 2),
                          won_in_horizon=diag['won_in_horizon']))
        if result.finished:
            won = bool(result.won)
            break
        if env.lives() < lives_before:
            stats['lives_lost'] += 1
            frames, actions = [frame], [-1]
        else:
            frames, actions = (frames + [frame])[-8:], (actions + [a])[-8:]
    return dict(level=index + 1, won=won, actions=stats['actions'], lives_lost=stats['lives_lost'],
                optimal=shipped.optimal(index), expansions=stats['expansions'],
                seconds=round(time.monotonic() - started, 1), trace=trace)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/spatial-recovery-v1/quality-fit/recovery.pt'))
    parser.add_argument('--value', choices=('learned', 'oracle', 'none'), default='learned')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--levels', type=int, nargs='*', default=list(range(1, 8)))
    parser.add_argument('--cap', type=int, default=300)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    learned = LearnedValue(args.checkpoint, args.device) if args.value == 'learned' else None
    report = dict(tool='ablate_engine_search', tool_assisted=True, learned_controller=False,
                  value=args.value, depth=args.depth, cap=args.cap,
                  checkpoint=str(args.checkpoint) if args.value == 'learned' else None, levels=[])
    for index in args.levels:
        result = play_level(index - 1, args.value, args.checkpoint, args.depth, args.cap, args.device, learned)
        report['levels'].append(result)
        print(json.dumps({k: v for k, v in result.items() if k != 'trace'}), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
    wins = sum(r['won'] for r in report['levels'])
    report['wins'] = wins
    args.out.write_text(json.dumps(report, indent=1))
    print(f"value={args.value} depth={args.depth}: {wins}/{len(report['levels'])} levels won (isolated, 3 lives, cap {args.cap})")


if __name__ == '__main__':
    main()
