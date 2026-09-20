# Current model status

Verified 20 September 2026. The current model is **multi-game architecture v2,
Run B**, at `artifacts/multigame-v2/frozen-v2arch-B.pt`, epoch index **19**.
SHA256: `7d4151bc7fb458e3598891762ee44805fba1d81640dc1a31ff8465e1de7f4ad9`.

The controller is recurrent imitation over public pixel observations, legal
actions, and history. Its action-conditioned auxiliary dynamics head is **not
used for planning**. Training-task mastery and unseen-layout transfer remain
incomplete. The standalone LS20 controllers and viewer have been retired; they
are recoverable in Git history.

## Results and boundaries

| Evaluation | Levels completed | Whole games completed |
|---|---:|---:|
| Generated training panel, frozen epoch 19 | 25/177 | 1/24 |
| Generated validation panel, same weights | 18/177 | 1/24 |
| Official training families, recorded Run B evaluation | 3/177 | 0/24 |
| Held-out m0r0, recorded Run B evaluation | 0/6 | 0/1 |

Official completed levels are lf52, lp85, and su15 level 1. These are local
measurements, not an official benchmark scorecard. Generated validation was
used for checkpoint selection and is not untouched confirmation data.
Nine generated validation wins come from BP35, whose offline validation
trajectories contain targets only for its first level.

The historical training peak of 51/177 levels and two games belongs to epoch
index 29, **not** the frozen epoch-19 checkpoint. Do not combine it with the
frozen validation result. The current audit also found residual dependence on
recorded history, weak functional click prediction, and poor changed-pixel and
positive-event prediction. These indicate problems on training tasks as well as
on transfer to new layouts.

## Corpus and recipe

Run B used `data/multigame-v2/merged-all`: 1,286 training and 1,238 validation games,
combining the region-labelled preparations with the earlier recovery corpus.
The union supplies LF52 completed teacher games that hit collection time caps
in the new preparations.

Training routes contain 150,008 certified, 68,398 live-recovery, and 20,006 random
transitions: 238,412 total, counted from arrays. There are **zero learner-policy
transitions**. Of 68,749 click targets, 44,959 have stored regions and 23,790 use
exact-pixel fallback (65.4% region coverage). Undo targets are almost absent and
BP35 later-level validation supervision is missing. Earlier manifest summaries
under-counted certified routes by 81,702 rows.

The run used canonical inputs, architecture v2, history dropout 0.5, whole-game
updates, and region supervision where available, for 30 epochs with gameplay
panels every second epoch. Its saved inference history mode is `full`.

## Current code versus measured weights

The audit's implementation repairs and new data tools are retained. The default
training starter initializes a new experiment from Run B with revised auxiliary
objectives and fresh optimizer state. Their benefit has **not** been established
by new training. Exact resume of an old Run B checkpoint under the changed
objective is deliberately rejected; use `--initialize-from` for a new run.

See [training and remediation commands](docs/multigame-training-operations.md),
[generator acceptance caveats](docs/generator-acceptance-caveats.md), and the local
[audit report](artifacts/model-audit-20260920/REPORT.md). Weights, corpora and local
reports are not committed to Git.

Inspect checkpoint-specific recorded metrics without running a new evaluation:

```bash
uv run python tools/summarize_multigame_checkpoint.py \
  artifacts/multigame-v2/frozen-v2arch-B.pt
```

For new official gameplay evaluations, use the two-phase commands in
[the README](README.md). Earlier model results and experiment narratives remain
in Git history at `40be7fc`.
