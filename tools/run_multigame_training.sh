#!/usr/bin/env bash
# Convenience recipe for the audited multi-game model. No training runs on import.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

action="${1:-help}"
run_dir="${RUN_DIR:-artifacts/multigame-v3/run-C}"
epochs="${EPOCHS:-30}"
device="${DEVICE:-cuda}"
python="${PYTHON:-.venv/bin/python}"
case "$action" in
  start)
    train_manifest="${TRAIN_MANIFEST:-data/multigame-v2/merged-all/train/manifest.json}"
    validation_manifest="${VALIDATION_MANIFEST:-data/multigame-v2/merged-all/validation/manifest.json}"
    printf '%s\n' 'Starting an experimental recipe. Existing Run B data has no learner-policy rows, incomplete click regions and BP35 validation-level gaps; see docs/multigame-training-operations.md.'
    exec "$python" tools/manage_multigame_training.py start --run-dir "$run_dir" -- \
      --train-manifest "$train_manifest" --validation-manifest "$validation_manifest" \
      --initialize-from artifacts/multigame-v2/frozen-v2arch-B.pt \
      --epochs "$epochs" --device "$device" --canonical --architecture v2 \
      --learning-rate 0.0001 --seed 1 --update-mode game --history-dropout 0.5 \
      --checkpoint-every-games 25 --chunk-steps 64 --auxiliary-transitions-per-chunk 8 \
      --metric-transitions-per-game 64 --closed-loop-interval 2 \
      --closed-loop-games 24 --closed-loop-train-games 24 \
      --validation-max-actions-per-level 256 --validation-max-game-actions 2048 \
      --require-click-regions --changed-pixel-weight 5 --event-positive-weight 3
    ;;
  stop)
    exec "$python" tools/manage_multigame_training.py stop --run-dir "$run_dir"
    ;;
  resume)
    exec "$python" tools/manage_multigame_training.py resume --run-dir "$run_dir" \
      --epochs "$epochs" --device "$device"
    ;;
  status)
    exec "$python" tools/manage_multigame_training.py status --run-dir "$run_dir"
    ;;
  help|--help|-h)
    printf '%s\n' 'Usage: bash tools/run_multigame_training.sh {start|stop|resume|status}' \
      'Optional environment: RUN_DIR, EPOCHS (total), DEVICE, TRAIN_MANIFEST, VALIDATION_MANIFEST, PYTHON.' \
      'Defaults: artifacts/multigame-v3/run-C, 30 epochs, cuda, existing Run B corpus.'
    ;;
  *) printf 'Unknown command: %s\n' "$action" >&2; exit 2 ;;
esac
