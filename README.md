# Pebby

Pebby trains a recurrent pixel policy on generated games from 24 ARC-AGI-3
families. The current model is **multi-game architecture v2, Run B**. It predicts
legal actions and click coordinates from public observations and recurrent
history. Its auxiliary dynamics head is not used for planning.

The frozen checkpoint completes **25/177 generated training levels and 18/177
validation levels**, one whole game on each 24-game panel. It completes three
individual levels but no whole games in the official training-family evaluation,
and zero levels in the held-out m0r0 game. Training mastery and transfer remain
incomplete. See [model status](MODEL_STATUS.md) for the exact checkpoint and
measurement boundaries.

## Install and inspect

```bash
uv sync --locked --group dev
uv run python tools/summarize_multigame_checkpoint.py \
  artifacts/multigame-v2/frozen-v2arch-B.pt
```

Weights, corpora, and evaluation reports are local files under `artifacts/` and
`data/`; they are not included in Git. A fresh checkout needs those files restored
or a newly prepared corpus and trained checkpoint.

## Train and evaluate

[Training operations](docs/multigame-training-operations.md) covers dataset repair,
training, graceful stop, and exact resume. The starter initializes a **new
experiment** from Run B; the revised objectives have not yet demonstrated an
improvement. The existing corpus has no learner-policy transitions, incomplete
click-region labels, almost no undo targets, and incomplete later-level
validation supervision.

Official evaluation runs a frozen checkpoint in two phases:

```bash
uv run python tools/evaluate_multigame.py \
  --checkpoint artifacts/multigame-v2/frozen-v2arch-B.pt \
  --phase training-families --out artifacts/multigame-v2/eval-training.json
uv run python tools/evaluate_multigame.py \
  --checkpoint artifacts/multigame-v2/frozen-v2arch-B.pt \
  --phase heldout --training-report artifacts/multigame-v2/eval-training.json \
  --out artifacts/multigame-v2/eval-heldout.json
```

For new data, follow the [generator guide](docs/multigame-generators.md) and its
[acceptance caveats](docs/generator-acceptance-caveats.md). Exact family planners
produce and verify teacher labels; the learned policy does not call them during
inference. These are local measurements, not an official benchmark scorecard.

## Development

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run --group dev python -m pytest tests -q
```

The repository contains the multi-game model, collection and training tools,
24 family adapters, vendored engines, and their tests. LS20 engine and generator
code remains because LS20 is one of those training families. The previous
standalone LS20 agents, HTTP viewer, and HostAI integration are retired in Git
history. See the [documentation index](docs/README.md) and
[development notes](docs/development.md).
