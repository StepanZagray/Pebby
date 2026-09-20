"""Spatial outcome v2 training over per-level full-route rows.

Fixes relative to ``tools/train_spatial_recovery_comparison.py``:

* Objective weights come from ``training_weights`` over EXACTLY the rows the
  sampler can draw (row kinds with zero weight are excluded from both).
* Rows with ``optimal == 0`` stay in training. ``neural_outcome_losses`` and
  ``spatial_outcome_losses`` mask such roots out of the policy and teacher
  terms (``valid = bits.any(-1)``, divided by ``valid.sum()``); the physical,
  value and event heads still learn from them.
* Plateau stopping on held-out validation loss, plus predeclared gameplay-based
  checkpoint selection (``SELECTION_RULE``) instead of a fixed update count.
* ``encoder_mode='finetune'`` trains the encoder with its own learning rate.

Data contract: ``DIR/levels/<seed>.npz`` with the ``world_data._row`` keys.
Frames are loaded lazily per batch through an LRU cache of per-level arrays;
all other columns are concatenated in memory.
"""
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .neural_outcome_planner import EVENT_NAMES
from .spatial_outcome_objective import spatial_outcome_losses, training_weights

FRAME_KEY = 'frames'
HISTORY_KEYS = ('history_valid', 'previous_actions')
TARGET_KEYS = ('next_player_cell', 'next_triple', 'next_steps', 'next_lives', 'distances',
               'lost_life', 'terminal', 'won', 'optimal', 'player_cell', 'current_triple',
               'current_steps', 'current_lives')
INFO_KEYS = ('seeds', 'context_index', 'row_kind', 'trajectory_id', 'step', 'chosen_action', 'solvable')
OPTIONAL_INFO = dict(context_index=('int8', 0), row_kind=('uint8', 0), trajectory_id=('int64', -1),
                     step=('int64', -1), chosen_action=('int64', -1), solvable=('bool', True))
ROW_KINDS = ('route', 'learner', 'recovery')
WEIGHT_KEYS = ('next_triple', 'current_triple', *EVENT_NAMES)
SELECTION_RULE = ('best = max(levels_completed_sequential, then generated_wins, then lowest validation '
                  'total loss) over gameplay-scored evaluations and the final weights; when periodic '
                  'gameplay is disabled every evaluation is a candidate and those without gameplay '
                  'count 0 levels and 0 wins')
HISTORY = 8


# ----------------------------------------------------------------------------- data
def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class LevelStore:
    """Per-level NPZ rows: metadata concatenated eagerly, frames fetched per batch."""

    def __init__(self, directory, *, frame_cache_bytes=2 << 30, hash_files=True):
        self.directory = Path(directory)
        self.paths = sorted((self.directory / 'levels').glob('*.npz'))
        if not self.paths:
            raise ValueError(f'no levels/*.npz under {self.directory}')
        columns = {key: [] for key in (*HISTORY_KEYS, *TARGET_KEYS, *INFO_KEYS)}
        counts, files, self.missing_keys = [], [], set()
        for path in self.paths:
            with np.load(path, allow_pickle=False) as data:
                names = set(data.files)
                required = {FRAME_KEY, *HISTORY_KEYS, *TARGET_KEYS, 'seeds'}
                if not required <= names:
                    raise ValueError(f'{path.name} lacks {sorted(required - names)}')
                count = int(data['seeds'].shape[0])
                for key in columns:
                    if key in names:
                        value = np.asarray(data[key])
                    else:
                        dtype, default = OPTIONAL_INFO[key]
                        value = np.full(count, default, dtype=dtype)
                        self.missing_keys.add(key)
                    if value.shape[:1] != (count,):
                        raise ValueError(f'{path.name}: {key} has {value.shape[0]} rows, expected {count}')
                    columns[key].append(value)
            counts.append(count)
            files.append(dict(name=path.name, rows=count, bytes=path.stat().st_size,
                              sha256=file_sha256(path) if hash_files else None))
        self.arrays = {key: np.concatenate(values) for key, values in columns.items()}
        self.counts = np.asarray(counts, dtype=np.int64)
        self.starts = np.concatenate(([0], np.cumsum(self.counts)[:-1]))
        self.rows = int(self.counts.sum())
        self.level_of_row = np.repeat(np.arange(len(self.paths)), self.counts)
        self.offset_of_row = np.arange(self.rows) - self.starts[self.level_of_row]
        self._validate()
        combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        self.manifest = dict(directory=str(self.directory.resolve()), levels=len(self.paths), rows=self.rows,
                             files=files, sha256=combined, hashed=hash_files,
                             missing_optional_keys=sorted(self.missing_keys),
                             row_kind_counts={ROW_KINDS[k]: int(v) for k, v in
                                              sorted(Counter(self.arrays['row_kind'].tolist()).items())
                                              if k < len(ROW_KINDS)})
        self._cache, self._cache_bytes, self._cache_limit = OrderedDict(), 0, int(frame_cache_bytes)

    def _validate(self):
        arrays = self.arrays
        shapes = dict(history_valid=(HISTORY,), previous_actions=(HISTORY,), next_player_cell=(4, 2),
                      next_triple=(4, 3), next_steps=(4,), next_lives=(4,), distances=(4,), lost_life=(4,),
                      terminal=(4,), won=(4,), optimal=(), player_cell=(2,), current_triple=(3,),
                      current_steps=(), current_lives=(), row_kind=())
        for key, tail in shapes.items():
            if arrays[key].shape[1:] != tail:
                raise ValueError(f'{key} must have per-row shape {tail}, got {arrays[key].shape[1:]}')
        if not np.isin(arrays['row_kind'], np.arange(len(ROW_KINDS))).all():
            raise ValueError('row_kind must be 0 (route), 1 (learner) or 2 (recovery)')
        if arrays['optimal'].min() < 0 or arrays['optimal'].max() > 15:
            raise ValueError('optimal masks must be in 0..15')

    def __len__(self):
        return self.rows

    @property
    def level_count(self):
        return len(self.paths)

    def level_frames(self, level):
        if level in self._cache:
            self._cache.move_to_end(level)
            return self._cache[level]
        with np.load(self.paths[level], allow_pickle=False) as data:
            frames = np.ascontiguousarray(data[FRAME_KEY])
        if frames.dtype != np.uint8 or frames.shape != (int(self.counts[level]), HISTORY, 64, 64):
            raise ValueError(f'{self.paths[level].name}: frames must be uint8 [{self.counts[level]},{HISTORY},64,64]')
        if self._cache_limit > 0:
            self._cache[level] = frames
            self._cache_bytes += frames.nbytes
            while self._cache_bytes > self._cache_limit and len(self._cache) > 1:
                _, evicted = self._cache.popitem(last=False)
                self._cache_bytes -= evicted.nbytes
        return frames

    def frames(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        out = np.empty((len(indices), HISTORY, 64, 64), dtype=np.uint8)
        levels, offsets = self.level_of_row[indices], self.offset_of_row[indices]
        for level in np.unique(levels):
            where = np.flatnonzero(levels == level)
            out[where] = self.level_frames(int(level))[offsets[where]]
        return out

    def batch(self, indices, device):
        """Tensors for the policy and objective: inputs, targets and row info."""
        indices = np.asarray(indices, dtype=np.int64)
        items = {FRAME_KEY: torch.from_numpy(self.frames(indices))}
        for key in (*HISTORY_KEYS, *TARGET_KEYS, *INFO_KEYS):
            items[key] = torch.from_numpy(np.array(self.arrays[key][indices], copy=True))
        return {key: value.to(device) for key, value in items.items()}


def row_weights(store, kind_weights):
    kind_weights = np.asarray(kind_weights, dtype=np.float64)
    if kind_weights.shape != (len(ROW_KINDS),) or not np.isfinite(kind_weights).all() or (kind_weights < 0).any():
        raise ValueError('kind_weights must be three finite nonnegative values (route, learner, recovery)')
    if not kind_weights.any():
        raise ValueError('at least one row kind needs positive weight')
    return kind_weights[store.arrays['row_kind']]


def objective_weights(store, kind_weights=(1., 1., 1.)):
    """``training_weights`` over exactly the rows the sampler can draw."""
    eligible = row_weights(store, kind_weights) > 0
    if not eligible.any():
        raise ValueError('no training rows have positive kind weight')
    return training_weights({key: store.arrays[key][eligible] for key in WEIGHT_KEYS}), int(eligible.sum())


class LevelSampler:
    """Level-first batches: distinct levels uniformly, ``rows_per_level`` rows each."""

    def __init__(self, store, *, batch_size, rows_per_level=1, kind_weights=(1., 1., 1.), seed=0):
        if batch_size < 1 or rows_per_level < 1:
            raise ValueError('batch_size and rows_per_level must be positive')
        self.store, self.batch_size, self.rows_per_level = store, int(batch_size), int(rows_per_level)
        self.weights = row_weights(store, kind_weights)
        self.rng = np.random.default_rng(seed)
        self.level_rows, self.level_probabilities = {}, {}
        for level in range(store.level_count):
            rows = np.arange(store.starts[level], store.starts[level] + store.counts[level])
            rows = rows[self.weights[rows] > 0]
            if len(rows):
                self.level_rows[level] = rows
                probabilities = self.weights[rows]
                self.level_probabilities[level] = probabilities / probabilities.sum()
        self.levels = np.array(sorted(self.level_rows), dtype=np.int64)
        if not len(self.levels):
            raise ValueError('no levels have rows with positive kind weight')
        self.eligible_rows = int(sum(len(rows) for rows in self.level_rows.values()))
        self.levels_per_batch = math.ceil(self.batch_size / self.rows_per_level)
        self.levels_short = len(self.levels) < self.levels_per_batch

    def sample(self):
        """Row indices for one batch and a record of the level draw."""
        chosen, order = [], self.rng.permutation(self.levels)
        cursor = 0
        while len(chosen) < self.batch_size:
            if cursor >= len(order):  # fewer distinct levels than needed: another pass
                order, cursor = self.rng.permutation(self.levels), 0
            level = int(order[cursor])
            cursor += 1
            rows, probabilities = self.level_rows[level], self.level_probabilities[level]
            take = min(self.rows_per_level, len(rows), self.batch_size - len(chosen))
            chosen.extend(self.rng.choice(rows, take, replace=False, p=probabilities).tolist())
        indices = np.asarray(chosen, dtype=np.int64)
        levels = self.store.level_of_row[indices]
        return indices, dict(distinct_levels=int(len(np.unique(levels))), rows=int(len(indices)),
                             optimal_zero_rows=int((self.store.arrays['optimal'][indices] == 0).sum()),
                             row_kinds={ROW_KINDS[k]: int(v) for k, v in
                                        sorted(Counter(self.store.arrays['row_kind'][indices].tolist()).items())})


# --------------------------------------------------------------------- objective
def autocast(device, precision):
    device_type = torch.device(device).type
    return torch.autocast(device_type, dtype=torch.bfloat16, enabled=precision == 'bf16')


def compute_losses(policy, items, weights, *, precision='fp32', distance_ordering=0.):
    with autocast(items[FRAME_KEY].device, precision):
        predicted = policy.predict(items[FRAME_KEY], items['history_valid'], items['previous_actions'])
    return spatial_outcome_losses(policy.planner, predicted, items, weights, distance_ordering=distance_ordering)


def scalar_losses(record):
    return {name: float(value.detach().float()) for name, value in record['losses'].items()}


def scalar_metrics(record):
    return {key: None if value is None else float(value.detach().float())
            for key, value in record['diagnostics'].items()}


@torch.inference_mode()
def validate(policy, store, weights, *, batch_size, device, precision='fp32', distance_ordering=0.):
    """All held-out rows, no filtering; losses weighted by rows, metrics by their support."""
    was_training = policy.training
    policy.eval()
    losses, metric_totals, supports, rows = Counter(), Counter(), Counter(), 0
    started = time.monotonic()
    for start in range(0, len(store), batch_size):
        indices = np.arange(start, min(start + batch_size, len(store)))
        record = compute_losses(policy, store.batch(indices, device), weights,
                                precision=precision, distance_ordering=distance_ordering)
        if not bool(torch.isfinite(record['total'])):
            raise ValueError('nonfinite validation loss')
        count = len(indices)
        for name, value in scalar_losses(record).items():
            losses[name] += value * count
        rows += count
        for name, value in record['diagnostics'].items():
            if value is None:
                continue
            weight = float(record['diagnostic_weights'][name])
            counted = name.endswith('_support') or name.endswith('_count')
            metric_totals[name] += float(value) if counted else float(value) * weight
            supports[name] += weight
    policy.train(was_training)
    losses = {name: value / rows for name, value in losses.items()}
    metrics = {name: metric_totals[name] if name.endswith('_support') or name.endswith('_count')
               else metric_totals[name] / supports[name] for name in metric_totals if supports[name] > 0}
    return dict(total=sum(losses.values()), losses=losses, metrics=metrics, rows=rows,
                elapsed_seconds=time.monotonic() - started)


# ---------------------------------------------------------------------- gameplay
class GeneratedPanel:
    """A small fixed generated bank played one isolated three-life episode per level."""

    def __init__(self, bank, count=14, max_actions=120):
        from . import evaluate
        self.bank, self.count, self.max_actions = Path(bank), int(count), int(max_actions)
        if self.count < 1 or self.max_actions < 1:
            raise ValueError('generated panel needs positive count and max_actions')
        self.levels, self.optima, self.specs = evaluate.bank_levels(self.bank, self.count)
        self.sha256 = file_sha256(self.bank)

    def describe(self):
        return dict(bank=str(self.bank.resolve()), bank_sha256=self.sha256, levels=len(self.levels),
                    max_actions=self.max_actions, seeds=[spec.get('seed') for spec in self.specs])

    def play(self, policy, device):
        from . import evaluate
        from ..ls20.env import Ls20Scenario
        was_training = getattr(policy, 'training', False)
        runs, started = [], time.monotonic()
        with torch.inference_mode():
            for level, optimum, spec in zip(self.levels, self.optima, self.specs):
                env = Ls20Scenario(level, int(spec.get('training_context_index', 0)))
                run = evaluate.rollout(policy, env, self.max_actions, torch.device(device), optimum)
                runs.append(dict(seed=spec.get('seed'), difficulty=spec.get('difficulty'),
                                 completed=bool(run['completed']), actions=int(run['actions']),
                                 ending=run['ending'], optimal=optimum, lives_left=int(run['lives_left'])))
        if hasattr(policy, 'train'):
            policy.train(was_training)
        return dict(wins=sum(run['completed'] for run in runs), levels=len(runs), runs=runs,
                    actions=sum(run['actions'] for run in runs), elapsed_seconds=time.monotonic() - started)


def sequential_summary(report):
    keys = ('levels_completed', 'completed', 'actions', 'per_level_actions', 'per_level_caps', 'lives_left',
            'ending', 'resets', 'protocol_identity', 'execution_device')
    return {key: report.get(key) for key in keys}


def gameplay_evaluation(policy, device, *, per_level_cap=300, panel=None, sequential=None, full_report_path=None):
    """Sequential shipped-game gate plus the optional generated panel."""
    if sequential is None:
        from .gameplay_gate import evaluate_sequential as sequential
    was_training = getattr(policy, 'training', False)
    policy.eval()
    started = time.monotonic()
    report = sequential(policy, device, per_level_cap=per_level_cap)
    if full_report_path is not None:
        write_json(full_report_path, report)
    result = dict(sequential=dict(**sequential_summary(report), elapsed_seconds=time.monotonic() - started,
                                  full_report=None if full_report_path is None else str(full_report_path)),
                  generated=None if panel is None else panel.play(policy, device))
    if hasattr(policy, 'train'):
        policy.train(was_training)
    return result


# --------------------------------------------------------------------- selection
def selection_key(evaluation):
    """(levels_completed_sequential, generated_wins, -validation_total); see SELECTION_RULE."""
    gameplay = evaluation.get('gameplay') or {}
    sequential, generated = gameplay.get('sequential') or {}, gameplay.get('generated') or {}
    return (int(sequential.get('levels_completed') or 0), int(generated.get('wins') or 0),
            -float(evaluation['validation']['total']))


def select_best(evaluations):
    """Best evaluation under ``SELECTION_RULE``; ties keep the earliest."""
    best = None
    for evaluation in evaluations:
        if best is None or selection_key(evaluation) > selection_key(best):
            best = evaluation
    return best


class PlateauStopper:
    """Stop when the tracked value has not improved by ``min_delta`` for ``patience`` evaluations."""

    def __init__(self, patience, min_delta=0.):
        if patience < 0 or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError('patience must be nonnegative and min_delta finite nonnegative')
        self.patience, self.min_delta = int(patience), float(min_delta)
        self.best, self.stale, self.evaluations = math.inf, 0, 0

    def update(self, value):
        self.evaluations += 1
        improved = value < self.best - self.min_delta
        if improved:
            self.best, self.stale = value, 0
        else:
            self.stale += 1
        return improved

    @property
    def should_stop(self):
        return self.patience > 0 and self.stale >= self.patience

    def state(self):
        return dict(best=None if math.isinf(self.best) else self.best, stale=self.stale,
                    patience=self.patience, min_delta=self.min_delta)


# ----------------------------------------------------------------------- trainer
@dataclass
class TrainConfig:
    updates: int = 1000
    batch_size: int = 256
    rows_per_level: int = 1
    kind_weights: tuple = (1., 1., 1.)
    lr: float = 1e-4
    encoder_lr: float = 1e-5
    warmup: int = 30
    schedule: str = 'cosine'
    weight_decay: float = .05
    grad_clip: float = 1.
    precision: str = 'fp32'
    distance_ordering: float = 0.
    eval_every: int = 50
    gameplay_every: int = 200
    patience: int = 5
    min_delta: float = 1e-3
    per_level_cap: int = 300
    validation_batch_size: int = 256
    log_every: int = 10
    seed: int = 0

    def __post_init__(self):
        if self.schedule not in ('cosine', 'constant'):
            raise ValueError('schedule must be cosine or constant')
        if self.precision not in ('fp32', 'bf16'):
            raise ValueError('precision must be fp32 or bf16')
        for name in ('updates', 'batch_size', 'rows_per_level', 'eval_every', 'per_level_cap',
                     'validation_batch_size', 'log_every'):
            if int(getattr(self, name)) < 1:
                raise ValueError(f'{name} must be positive')
        for name in ('warmup', 'gameplay_every', 'patience'):
            if int(getattr(self, name)) < 0:
                raise ValueError(f'{name} must be nonnegative')
        for name in ('lr', 'encoder_lr', 'grad_clip'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.distance_ordering) or self.distance_ordering < 0:
            raise ValueError('distance_ordering must be finite and nonnegative')
        self.kind_weights = tuple(float(w) for w in self.kind_weights)


def _decay_split(parameters, weight_decay):
    decay = [p for name, p in parameters if p.ndim >= 2 and 'position' not in name and 'embedding' not in name]
    no_decay = [p for name, p in parameters if not (p.ndim >= 2 and 'position' not in name and 'embedding' not in name)]
    groups = []
    if decay:
        groups.append(dict(params=decay, weight_decay=weight_decay))
    if no_decay:
        groups.append(dict(params=no_decay, weight_decay=0.))
    return groups


def parameter_groups(policy, config):
    """Planner at ``lr``, trainable encoder tensors at ``encoder_lr``; matrix-only decay."""
    groups = []
    for prefix, rate in (('encoder.', config.encoder_lr), ('planner.', config.lr)):
        named = [(name, p) for name, p in policy.named_parameters() if p.requires_grad and name.startswith(prefix)]
        for group in _decay_split(named, config.weight_decay):
            groups.append(dict(name=prefix[:-1], base_lr=rate, lr=rate, **group))
    if not groups:
        raise ValueError('policy has no trainable parameters')
    return groups


class Trainer:
    def __init__(self, policy, train_store, config, device, *, weights=None):
        self.policy, self.store, self.config, self.device = policy, train_store, config, torch.device(device)
        torch.manual_seed(config.seed)
        self.sampler = LevelSampler(train_store, batch_size=config.batch_size, rows_per_level=config.rows_per_level,
                                    kind_weights=config.kind_weights, seed=config.seed)
        self.weights, self.weight_rows = (weights, None) if weights is not None else objective_weights(train_store, config.kind_weights)
        self.optimizer = torch.optim.AdamW(parameter_groups(policy, config), lr=config.lr)
        self.update = 0
        self.policy.to(self.device).train()

    def lr_scale(self, update):
        """Scale for ``update`` (1-based); warmup then cosine to zero or constant."""
        config = self.config
        if config.warmup and update <= config.warmup:
            return update / config.warmup
        if config.schedule == 'constant':
            return 1.
        progress = (update - config.warmup) / max(1, config.updates - config.warmup)
        return .5 * (1 + math.cos(math.pi * min(1., progress)))

    def step(self):
        config, started = self.config, time.monotonic()
        self.update += 1
        scale = self.lr_scale(self.update)
        for group in self.optimizer.param_groups:
            group['lr'] = group['base_lr'] * scale
        indices, draw = self.sampler.sample()
        items = self.store.batch(indices, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        record = compute_losses(self.policy, items, self.weights, precision=config.precision,
                                distance_ordering=config.distance_ordering)
        if not bool(torch.isfinite(record['total'])):
            raise ValueError('nonfinite training loss')
        record['total'].backward()
        trainable = [p for p in self.policy.parameters() if p.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        return dict(update=self.update, loss=float(record['total'].detach()), losses=scalar_losses(record),
                    metrics=scalar_metrics(record), unclipped_gradient_norm=float(norm), lr_scale=scale,
                    learning_rates={group['name']: group['lr'] for group in self.optimizer.param_groups},
                    batch=draw, elapsed_seconds=time.monotonic() - started)


# --------------------------------------------------------------------------- run
def git_state(root):
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True,
                              check=True, timeout=30).stdout.strip()
        status = subprocess.run(['git', 'status', '--porcelain'], cwd=root, capture_output=True, text=True,
                                check=True, timeout=60).stdout
        return dict(head=head, dirty=bool(status.strip()))
    except (OSError, subprocess.SubprocessError):
        return dict(head=None, dirty=None)


def _policy_description(policy):
    provenance = getattr(policy, 'provenance', None) or {}
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    return dict(config=policy.config() if hasattr(policy, 'config') else None,
                parameters=sum(p.numel() for p in policy.parameters()), trainable_parameters=trainable,
                encoder_mode=getattr(policy, 'encoder_mode', None),
                provenance={k: (v if not isinstance(v, list) else len(v)) for k, v in provenance.items()})


def _default_saver(path, policy, **metadata):
    from .spatial_v2_policy import save_checkpoint
    return save_checkpoint(path, policy, **metadata)


def run(policy, train_store, validation_store, config, out_dir, *, device='cpu', panel=None,
        sequential=None, saver=_default_saver, qualify=False, argv=None, root=None):
    """Train with periodic validation/gameplay, plateau stop and gameplay selection.

    ``report.json`` is rewritten after every logged step and evaluation, so an
    interrupted run leaves its evidence. ``best.pt`` follows ``SELECTION_RULE``;
    ``final.pt`` is the last weights. ``qualify`` runs three tiny updates and one
    gameplay evaluation, saving no checkpoints.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    started = time.monotonic()
    trainer = Trainer(policy, train_store, config, device)
    report = dict(status='running', mode='qualify' if qualify else 'training', pid=None,
                  started_local=datetime.now().astimezone().isoformat(),
                  argv=list(sys.argv if argv is None else argv), git=git_state(root), device=str(device),
                  selection_rule=SELECTION_RULE, config=asdict(config),
                  data=dict(train=train_store.manifest, validation=validation_store.manifest,
                            train_rows_with_positive_weight=trainer.weight_rows,
                            train_optimal_zero_rows=int((train_store.arrays['optimal'] == 0).sum()),
                            optimal_zero_rows_kept=True,
                            weights_source='training_weights over exactly the sampler-eligible train rows'),
                  objective_weights=trainer.weights, policy=_policy_description(policy),
                  sampler=dict(levels=int(len(trainer.sampler.levels)), eligible_rows=trainer.sampler.eligible_rows,
                               levels_per_batch=trainer.sampler.levels_per_batch,
                               levels_short=trainer.sampler.levels_short),
                  generated_panel=None if panel is None else panel.describe(),
                  train_log=[], evaluations=[], best=None, final=None, stop_reason=None)
    report_path = out_dir / 'report.json'
    write_json(report_path, report)
    stopper = PlateauStopper(config.patience, config.min_delta)
    best_key = None

    def checkpoint_metadata(evaluation, kind):
        return dict(kind=kind, update=trainer.update, selection_rule=SELECTION_RULE,
                    selection_key=list(selection_key(evaluation)), evaluation=evaluation,
                    objective_weights=trainer.weights, data_manifests=dict(
                        train_sha256=train_store.manifest['sha256'], validation_sha256=validation_store.manifest['sha256'],
                        train_directory=train_store.manifest['directory'],
                        validation_directory=validation_store.manifest['directory']),
                    argv=report['argv'], git=report['git'], config=asdict(config), encoder_mode=report['policy']['encoder_mode'])

    def play(evaluation):
        evaluation['gameplay'] = gameplay_evaluation(
            policy, device, per_level_cap=config.per_level_cap, panel=panel, sequential=sequential,
            full_report_path=out_dir / f'gameplay-{trainer.update:06d}.json')
        evaluation['selection_candidate'] = True
        evaluation['selection_key'] = list(selection_key(evaluation))

    def evaluate(with_gameplay):
        evaluation = dict(update=trainer.update, elapsed_seconds=time.monotonic() - started,
                          validation=validate(policy, validation_store, trainer.weights, batch_size=config.validation_batch_size,
                                              device=device, precision=config.precision,
                                              distance_ordering=config.distance_ordering),
                          gameplay=None)
        evaluation['improved_validation'] = stopper.update(evaluation['validation']['total'])
        evaluation['plateau'] = stopper.state()
        evaluation['selection_candidate'] = not config.gameplay_every
        evaluation['selection_key'] = list(selection_key(evaluation))
        if with_gameplay:
            play(evaluation)
        return evaluation

    def consider_best(evaluation):
        nonlocal best_key
        if qualify or not evaluation['selection_candidate']:
            return
        if best_key is None or tuple(evaluation['selection_key']) > best_key:
            best_key = tuple(evaluation['selection_key'])
            saver(out_dir / 'best.pt', policy, **checkpoint_metadata(evaluation, 'best'))
            evaluation['saved_best'] = True
            report['best'] = dict(update=trainer.update, selection_key=list(best_key), path=str(out_dir / 'best.pt'))

    try:
        updates = 3 if qualify else config.updates
        for _ in range(updates):
            record = trainer.step()
            if trainer.update % config.log_every == 0 or trainer.update <= 3 or trainer.update == updates:
                report['train_log'].append(record)
                write_json(report_path, report)
            if qualify:
                continue
            # The last scheduled update is always a gameplay evaluation, so the
            # final weights are scored exactly once (no post-loop repeat).
            final_update = trainer.update == updates
            with_gameplay = final_update or (bool(config.gameplay_every) and trainer.update % config.gameplay_every == 0)
            if not (with_gameplay or trainer.update % config.eval_every == 0):
                continue
            evaluation = evaluate(with_gameplay)
            consider_best(evaluation)
            report['evaluations'].append(evaluation)
            write_json(report_path, report)
            if stopper.should_stop:
                report['stop_reason'] = 'plateau'
                break
        else:
            report['stop_reason'] = 'qualified' if qualify else 'max_updates'
        last = report['evaluations'][-1] if report['evaluations'] else None
        if last is None or last['update'] != trainer.update:
            # Qualify, or a stop between evaluation steps: score the final weights once.
            evaluation = evaluate(True)
            consider_best(evaluation)
            report['evaluations'].append(evaluation)
        elif last['gameplay'] is None:
            # A plateau stop on a validation-only step: add gameplay to that same
            # entry (no second validation, no duplicate evaluation record).
            play(last)
            consider_best(last)
        final = report['evaluations'][-1]
        if not qualify:
            saver(out_dir / 'final.pt', policy, **checkpoint_metadata(final, 'final'))
        report['final'] = dict(update=trainer.update, selection_key=final['selection_key'],
                               path=None if qualify else str(out_dir / 'final.pt'))
        report.update(status='qualified' if qualify else 'complete', updates=trainer.update,
                      elapsed_seconds=time.monotonic() - started,
                      finished_local=datetime.now().astimezone().isoformat())
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', updates=trainer.update,
                      elapsed_seconds=time.monotonic() - started)
        write_json(report_path, report)
        raise
    write_json(report_path, report)
    return report
