# Development

## Repository layout

| Path | Purpose |
|---|---|
| `pebby/agent/multigame_model.py` | Recurrent pixel policy, click decoder, auxiliary objectives, checkpoint compatibility |
| `pebby/agent/multigame_training.py` | Manifest audits, fitting, gameplay selection, atomic checkpoints, resume |
| `pebby/agent/multigame_evaluation.py` | Frozen-policy inference and phased official evaluation |
| `pebby/multigame.py` | Family registry, native adapters, collection and provenance |
| `pebby/multigame_dataset.py` | Whole-game datasets, preparation, recovery and split audits |
| `pebby/multigame_variants.py` | Reversible control, spatial and palette variants |
| `pebby/games/` | 24 family engines, teachers, generators and contracts |
| `pebby/ls20/` | Shared LS20 engine, native planner and reference generator dependencies |
| `third_party/` | Vendored game sources, licenses and provenance |
| `tools/` | Multi-game collection, audits, training lifecycle and evaluation CLIs; LS20 differential check |
| `tests/` | Model, lifecycle, data, family and LS20 planner tests |

Multi-game checkpoint and corpus compatibility is retained so Run B and its
source datasets still load. Retired standalone LS20 controllers and the viewer
are no longer supported by this tree. Their source, tests and experiment notes
remain in Git history; `40be7fc` preserves the working tree before consolidation.

## Checks

```bash
uv sync --locked --group dev
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run --group dev python -m pytest tests -q
```

Tests use CPU and temporary fixtures. Some compatibility tests use optional local
checkpoint artifacts and skip when those are absent. Game tests exercise native
engine logic and pixel arrays without opening desktop windows. Full family
regressions can take several minutes because they verify real teacher routes.

For a targeted LS20 symbolic/native comparison:

```bash
PYTHONPATH=. uv run python tools/differential.py
```

Run `--help` on the multi-game tools for bounded collection and evaluation options.
Do not treat a small smoke corpus or a successful unit test as evidence of model
mastery. [Model status](../MODEL_STATUS.md) distinguishes frozen-weight results
from historical training maxima and newly implemented experiments.

## Local outputs

`data/`, `artifacts/`, checkpoints and `.scratch/` are ignored. Preserve complete
training-run directories, including hidden checkpoint generations and symlinks,
when backing up or restoring them. Do not rewrite corpus certificates after a
source cleanup: their hashes describe the code that produced those examples.
