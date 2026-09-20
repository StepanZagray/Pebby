"""Train a spatial outcome v2 policy on per-level full-route rows.

Selection is predeclared (``spatial_v2_training.SELECTION_RULE``): the saved
``best.pt`` maximises sequential levels completed, then generated wins, then
lowest validation loss. Training stops at ``--updates`` or when validation loss
plateaus (``--patience`` evaluations without a ``--min-delta`` improvement).
``--qualify`` runs three tiny updates and one gameplay evaluation, then exits.
"""
import argparse
import math
from pathlib import Path
import sys

import torch

from pebby.agent import spatial_v2_policy
from pebby.agent.spatial_outcome_policy import FORMAT as V1_FORMAT
from pebby.agent.spatial_v2_training import GeneratedPanel, LevelStore, TrainConfig, run

ROOT = Path(__file__).resolve().parents[1]


def resolve_device(name):
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('cuda requested but unavailable')
    return name


def load_policy(path, encoder_mode, hud_scalars, direct_readout=False):
    """v1 checkpoints warm-start through from_v1_checkpoint; v2 checkpoints reload directly."""
    data = torch.load(Path(path), map_location='cpu', weights_only=True)
    fmt = data.get('format') if isinstance(data, dict) else None
    if fmt == V1_FORMAT:
        warm = getattr(spatial_v2_policy.SpatialOutcomePolicyV2, 'from_v1_checkpoint', None)
        warm = warm or spatial_v2_policy.from_v1_checkpoint
        return warm(path, encoder_mode=encoder_mode, hud_scalars=hud_scalars, direct_readout=direct_readout), dict(init_format='v1', init=str(path), direct_readout=direct_readout)
    if fmt == spatial_v2_policy.FORMAT:
        loaded = spatial_v2_policy.load_checkpoint(path)
        policy = loaded[0] if isinstance(loaded, tuple) else loaded
        if policy.encoder_mode != encoder_mode:
            # Rebuild so requires_grad flags follow the requested mode.
            provenance = policy.provenance
            policy = spatial_v2_policy.SpatialOutcomePolicyV2(policy.encoder, policy.planner, encoder_mode=encoder_mode)
            policy.provenance = provenance
        return policy, dict(init_format='v2', init=str(path))
    raise SystemExit(f'--init must be a v1 or v2 spatial outcome checkpoint, got format {fmt!r}')


def parse_kind_weights(text):
    parts = [float(x) for x in text.split(',')]
    if len(parts) != 3 or any(not math.isfinite(x) or x < 0 for x in parts):
        raise argparse.ArgumentTypeError('kind weights are three nonnegative floats: route,learner,recovery')
    return tuple(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--validation-dir', type=Path, required=True)
    parser.add_argument('--init', type=Path, default=ROOT / 'artifacts/spatial-recovery-v1/quality-fit/recovery.pt',
                        help='v1 (warm start via from_v1_checkpoint) or v2 checkpoint')
    parser.add_argument('--encoder-mode', choices=('frozen', 'finetune'), default='frozen')
    parser.add_argument('--hud-scalars', type=int, default=4, help='decoded HUD scalars for v1 warm starts')
    parser.add_argument('--direct-readout', action='store_true', help='add a trainable direct per-action score to the outcome comparator (v1 warm starts)')
    parser.add_argument('--updates', type=int, default=2000, help='maximum optimizer updates')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--rows-per-level', type=int, default=1)
    parser.add_argument('--kind-weights', type=parse_kind_weights, default=(1., 1., 1.),
                        help='route,learner,recovery sampling weights; 0 excludes a kind from training')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--encoder-lr', type=float, default=1e-5)
    parser.add_argument('--warmup', type=int, default=30, help='linear warmup updates')
    parser.add_argument('--schedule', choices=('cosine', 'constant'), default='cosine')
    parser.add_argument('--weight-decay', type=float, default=.05)
    parser.add_argument('--grad-clip', type=float, default=1.)
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--gameplay-every', type=int, default=200, help='0 disables periodic gameplay')
    parser.add_argument('--patience', type=int, default=5, help='0 disables plateau stopping')
    parser.add_argument('--min-delta', type=float, default=1e-3)
    parser.add_argument('--generated-bank', type=Path, default=None)
    parser.add_argument('--generated-count', type=int, default=14)
    parser.add_argument('--generated-max-actions', type=int, default=120)
    parser.add_argument('--per-level-cap', type=int, default=300)
    parser.add_argument('--validation-batch-size', type=int, default=256)
    parser.add_argument('--frame-cache-gb', type=float, default=2., help='LRU budget for per-level frame arrays')
    parser.add_argument('--no-hash-data', action='store_true', help='skip sha256 of every level file')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--distance-ordering', type=float, default=0.)
    parser.add_argument('--qualify', action='store_true', help='3 tiny updates plus one gameplay evaluation')
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    if args.qualify:
        args.batch_size, args.rows_per_level = min(args.batch_size, 8), 1
        args.validation_batch_size = min(args.validation_batch_size, 8)
    config = TrainConfig(updates=args.updates, batch_size=args.batch_size, rows_per_level=args.rows_per_level,
                         kind_weights=args.kind_weights, lr=args.lr, encoder_lr=args.encoder_lr, warmup=args.warmup,
                         schedule=args.schedule, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
                         precision=args.precision, distance_ordering=args.distance_ordering,
                         eval_every=args.eval_every, gameplay_every=args.gameplay_every, patience=args.patience,
                         min_delta=args.min_delta, per_level_cap=args.per_level_cap,
                         validation_batch_size=args.validation_batch_size, seed=args.seed)
    cache_bytes = int(args.frame_cache_gb * (1 << 30))
    train_store = LevelStore(args.train_dir, frame_cache_bytes=cache_bytes, hash_files=not args.no_hash_data)
    validation_store = LevelStore(args.validation_dir, frame_cache_bytes=cache_bytes, hash_files=not args.no_hash_data)
    panel = None
    if args.generated_bank is not None:
        panel = GeneratedPanel(args.generated_bank, args.generated_count, args.generated_max_actions)
    policy, init = load_policy(args.init, args.encoder_mode, args.hud_scalars, args.direct_readout)
    print(f'train rows={len(train_store)} levels={train_store.level_count}; validation rows={len(validation_store)} '
          f'levels={validation_store.level_count}; device={device}; init={init}', flush=True)
    report = run(policy, train_store, validation_store, config, args.out_dir, device=device, panel=panel,
                 qualify=args.qualify, argv=sys.argv if argv is None else ['train_spatial_v2.py', *argv], root=ROOT)
    final = report['evaluations'][-1]
    gameplay = final['gameplay'] or {}
    sequential, generated = gameplay.get('sequential') or {}, gameplay.get('generated')
    print(f"status={report['status']} updates={report['updates']} stop={report['stop_reason']} "
          f"validation_total={final['validation']['total']:.4f} "
          f"levels_completed={sequential.get('levels_completed')} "
          f"generated_wins={None if generated is None else generated['wins']} "
          f"best={report['best']}", flush=True)
    return report


if __name__ == '__main__':
    main()
