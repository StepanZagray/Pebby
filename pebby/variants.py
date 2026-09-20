"""A family of LS20 games that differ only in how agent actions map to moves.

A *variant* is one of the 24 permutations of the four movement actions. A
*game* is one variant, fixed for its whole duration, plus a sequence of
generated levels (tiers 1..7, one per tier, like the shipped game) played
sequentially on the real engine with the competition reset semantics of
``pebby.agent.competition``: three native lives per level, a GAME_OVER is
answered with a level-only RESET, and a full restart of the engine is
forbidden. The agent never sees engine action ids; it emits agent actions
0..3 and ``VariantGame`` translates them through the permutation.

Coordinates follow ``pebby.ls20.names``: cells are ``(x, y)`` with ``x`` the
column and ``y`` the row on the 12x12 lattice. ``tile_map()`` is indexed
``tiles[y][x]`` (row-major, image convention).
"""

import itertools
import json
from pathlib import Path
from types import MethodType

from arcengine import GameState
import numpy as np

from .ls20 import names
from .ls20.env import Ls20Env
from .ls20.generate import build_level
from .ls20.layout import extract

PERMUTATIONS = tuple(itertools.permutations(range(4)))
# INVERSE[v][engine_action] == agent_action such that PERMUTATIONS[v][agent_action] == engine_action.
INVERSE_PERMUTATIONS = tuple(tuple(perm.index(e) for e in range(4)) for perm in PERMUTATIONS)
VARIANT_COUNT = len(PERMUTATIONS)
IDENTITY_VARIANT = 0

TILE_CLASSES = ('free', 'wall', 'cycler_shape', 'cycler_color', 'cycler_rotation', 'refill',
                'launcher', 'goal', 'rail', 'outside')
TILE_INDEX = {name: index for index, name in enumerate(TILE_CLASSES)}
FREE, WALL, CYCLER_SHAPE, CYCLER_COLOR, CYCLER_ROTATION, REFILL, LAUNCHER, GOAL, RAIL, OUTSIDE = range(10)
_CYCLER_CLASS = {'shape': CYCLER_SHAPE, 'color': CYCLER_COLOR, 'rotation': CYCLER_ROTATION}

DEFAULT_TIERS = tuple(range(1, 8))

STATE_FIELDS = ('level_index', 'player_x', 'player_y', 'shape', 'color', 'rotation', 'steps_left',
                'lives', 'goals_mask', 'finished', 'won')


def check_variant(variant_id):
    if isinstance(variant_id, bool) or not isinstance(variant_id, (int, np.integer)) \
            or not 0 <= int(variant_id) < VARIANT_COUNT:
        raise ValueError(f'variant_id must be an integer in 0..{VARIANT_COUNT - 1}')
    return int(variant_id)


def factored_state(env):
    """The logical state of a running ``Ls20Env`` as a dict of plain ints."""
    x, y = env.player_cell()
    shape, color, rotation = env.triple()
    goals_mask = sum(1 << i for i, solved in enumerate(env.goals_solved()) if solved)
    won = env.state == GameState.WIN
    finished = won or env.state == GameState.GAME_OVER
    return dict(level_index=int(env.level_index), player_x=int(x), player_y=int(y),
                shape=int(shape), color=int(color), rotation=int(rotation),
                steps_left=int(env.steps_left()), lives=int(env.lives()), goals_mask=int(goals_mask),
                finished=int(finished), won=int(won))


def tile_map_from_layout(layout):
    """Static tile classes of one level, ``tiles[y][x]``. Later categories overwrite earlier."""
    tiles = np.full((names.GRID_ROWS, names.GRID_COLS), FREE, dtype=np.int8)

    def paint(cell, value):
        x, y = cell
        if 0 <= x < names.GRID_COLS and 0 <= y < names.GRID_ROWS:
            tiles[y, x] = value

    for cell in layout.walls:
        paint(cell, WALL)
    for cell in layout.rails:            # rail cells (where riding cyclers walk)
        paint(cell, RAIL)
    for cell in layout.refills:
        paint(cell, REFILL)
    for cell, kind in layout.cyclers.items():   # static cyclers only; riders are on rails
        paint(cell, _CYCLER_CLASS[kind])
    for launcher in layout.launchers:
        paint(launcher['cell'], LAUNCHER)
    for cell, _triple in layout.goals:
        paint(cell, GOAL)
    return tiles


def _level_only_reset(game):
    game.level_reset()


def _forbid_full_reset(game):
    raise RuntimeError('a variant game cannot restart the full engine; build a new VariantGame')


class VariantGame:
    """One action permutation over a sequence of generated levels on the real engine.

    ``specs`` are bank level specs in play order (normally tiers 1..7). The
    match-cycle hint LS20 shows only at engine index zero therefore lands on
    the first spec, exactly as in the shipped game and in the bank's
    ``training_context_index`` convention (tier ``d`` at index ``d-1``).
    """

    def __init__(self, specs, variant_id):
        specs = list(specs)
        if not specs:
            raise ValueError('a game needs at least one level spec')
        self.variant_id = check_variant(variant_id)
        self.action_map = PERMUTATIONS[self.variant_id]
        self.inverse_map = INVERSE_PERMUTATIONS[self.variant_id]
        self.specs = specs
        self.levels = [build_level(spec) for spec in specs]
        self.env = None
        self._layouts = {}
        self._tile_maps = {}
        self.steps = 0
        self.reset()

    # -- lifecycle ----------------------------------------------------------

    def reset(self):
        """Start (or restart) the game on a fresh engine and return the factored state.

        The engine itself never full-resets after its first action; a restart
        builds a new engine, mirroring the competition protocol where one game
        is one engine initialisation.
        """
        self.env = Ls20Env(levels=self.levels)
        self.env.reset()
        game = self.env.game
        game.handle_reset = MethodType(_level_only_reset, game)
        game.full_reset = MethodType(_forbid_full_reset, game)
        self._layouts = {}
        self._tile_maps = {}
        self.steps = 0
        self._enter_level()
        return self.state()

    def _enter_level(self):
        """Snapshot the static layout the moment a level is entered, while every tile is present."""
        index = self.env.level_index
        if index not in self._layouts:
            layout = extract(self.env)
            self._layouts[index] = layout
            self._tile_maps[index] = tile_map_from_layout(layout)

    # -- properties ---------------------------------------------------------

    @property
    def level_count(self):
        return len(self.levels)

    @property
    def level_index(self):
        return int(self.env.level_index)

    @property
    def levels_completed(self):
        return int(self.env.levels_completed)

    @property
    def won(self):
        return self.env.state == GameState.WIN

    @property
    def finished(self):
        return self.env.state in (GameState.WIN, GameState.GAME_OVER)

    def current_spec(self):
        return self.specs[self.level_index]

    def layout(self):
        """The ``Layout`` of the current level, extracted when the level was entered."""
        return self._layouts[self.level_index]

    # -- play ---------------------------------------------------------------

    def step(self, agent_action):
        """Apply one agent action. Returns ``(state_after, info)``.

        ``info`` carries ``engine_action`` (0..3, index into
        ``names.ACTION_IDS``), ``life_lost``, ``level_changed``, ``reset``
        (a GAME_OVER happened and the game answered it with a level reset,
        so ``state_after`` is the restored level), ``won`` and ``finished``.
        """
        if isinstance(agent_action, bool) or not isinstance(agent_action, (int, np.integer)) \
                or not 0 <= int(agent_action) < 4:
            raise ValueError('agent_action must be an integer in 0..3')
        if self.finished:
            raise RuntimeError('the game is finished; no action is legal')
        env = self.env
        engine_action = self.action_map[int(agent_action)]
        before_level, before_lives = env.level_index, env.lives()
        env.perform(names.ACTION_IDS[engine_action])
        reset = False
        if env.state == GameState.GAME_OVER:
            env.perform(0)           # RESET -> patched handle_reset -> level_reset
            reset = True
            if env.state != GameState.NOT_FINISHED or env.level_index != before_level:
                raise RuntimeError('level reset after GAME_OVER did not restore the level')
        if env.level_index < before_level:
            raise RuntimeError('sequential game progress regressed')
        level_changed = env.level_index != before_level
        if level_changed:
            self._enter_level()
        life_lost = reset or env.lives() < before_lives
        won = env.state == GameState.WIN
        self.steps += 1
        info = dict(engine_action=int(engine_action), life_lost=bool(life_lost),
                    level_changed=bool(level_changed), reset=bool(reset), won=bool(won),
                    finished=bool(won))
        return self.state(), info

    # -- observations -------------------------------------------------------

    def state(self):
        return factored_state(self.env)

    def tile_map(self):
        """12x12 int8 array of ``TILE_CLASSES`` for the current level, indexed ``[y][x]``."""
        return self._tile_maps[self.level_index]

    def neighbours(self):
        """Tile classes of the cells up/down/left/right of the player, in ENGINE direction order."""
        tiles = self.tile_map()
        x, y = self.env.player_cell()
        out = []
        for dx, dy in names.ACTION_DELTAS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < names.GRID_COLS and 0 <= ny < names.GRID_ROWS:
                out.append(int(tiles[ny, nx]))
            else:
                out.append(OUTSIDE)
        return out

    def goal_triples(self):
        return [tuple(int(v) for v in triple) for triple in self.env.goal_triples()]

    def frame(self):
        """The current 64x64 frame as a uint8 array of colour indices."""
        return np.asarray(self.env.render(), dtype=np.uint8)


# -- oracle bridge --------------------------------------------------------------

def oracle_agent_action(oracle, env_state, variant_id):
    """The AGENT action whose engine action the oracle deems optimal, or None."""
    engine_action = oracle.action_for(env_state)
    if engine_action is None:
        return None
    return INVERSE_PERMUTATIONS[check_variant(variant_id)][int(engine_action)]


# -- level selection ------------------------------------------------------------

def _specs_of(bank):
    if isinstance(bank, (str, Path)):
        from .ls20.bank import load
        return load(bank)
    return list(bank)


def specs_by_tier(bank_specs, tiers=DEFAULT_TIERS):
    """Group specs by difficulty, each tier sorted by seed for reproducible sampling."""
    grouped = {tier: [] for tier in tiers}
    for spec in bank_specs:
        tier = spec.get('difficulty')
        if tier in grouped:
            grouped[tier].append(spec)
    for tier in tiers:
        grouped[tier].sort(key=lambda spec: int(spec['seed']))
        if not grouped[tier]:
            raise ValueError(f'the bank has no levels of tier {tier}')
    return grouped


def pool_specs(bank_specs, levels_per_tier, rng, tiers=DEFAULT_TIERS):
    """Choose a per-tier pool of ``levels_per_tier`` specs (fewer if the tier is smaller)."""
    grouped = specs_by_tier(bank_specs, tiers)
    pool = []
    for tier in tiers:
        candidates = grouped[tier]
        pool.extend(rng.sample(candidates, min(levels_per_tier, len(candidates))))
    return pool


def game_specs(bank, tier_seeds, tiers=None):
    """Specs for one game: the level of each tier whose seed is given, in tier order."""
    tier_seeds = [int(seed) for seed in tier_seeds]
    tiers = tuple(tiers) if tiers is not None else tuple(range(1, len(tier_seeds) + 1))
    if len(tiers) != len(tier_seeds):
        raise ValueError('tier_seeds must give one seed per tier')
    index = {(spec['difficulty'], int(spec['seed'])): spec for spec in _specs_of(bank)}
    chosen = []
    for tier, seed in zip(tiers, tier_seeds):
        spec = index.get((tier, seed))
        if spec is None:
            raise KeyError(f'no tier {tier} level with seed {seed} in the bank')
        chosen.append(spec)
    return chosen


def sample_games(bank_specs, count, variant_ids, rng, tiers=DEFAULT_TIERS):
    """``count`` games per variant; each picks one level per tier uniformly from ``bank_specs``.

    Returns a list of ``(variant_id, [seed per tier])`` in variant order.
    """
    grouped = specs_by_tier(bank_specs, tiers)
    games = []
    for variant_id in variant_ids:
        variant_id = check_variant(variant_id)
        for _ in range(count):
            seeds = [int(rng.choice(grouped[tier])['seed']) for tier in tiers]
            games.append((variant_id, seeds))
    return games


def describe_variant(variant_id):
    perm = PERMUTATIONS[check_variant(variant_id)]
    return {f'agent_{a}': names.ACTION_NAMES[perm[a]] for a in range(4)}


def dumps_variant_table():
    return json.dumps({v: describe_variant(v) for v in range(VARIANT_COUNT)}, indent=1)
