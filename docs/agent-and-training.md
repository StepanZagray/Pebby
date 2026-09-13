# Training and model operation

This is the stable workflow for creating a checkpoint, evaluating it, and starting
the runtime. Model architecture details, experiment results, and checkpoint quality
are intentionally not recorded here; inspect the current module help and checkpoint
metadata when those details matter. The retained checkpoint's verified capability
and incomplete components are summarized in [Current model status](../MODEL_STATUS.md).

## Install

From the repository root:

```bash
uv sync --locked
mkdir -p data checkpoints
```

The commands below use generated banks. Train and validation banks must be separate
level sets; never split near-duplicate samples from one level across both sets.

For new reference banks that need train, validation, and test partitions, select
`--generator-version 4` consistently for all three. Version 4 also expands
multi-goal relations and supports disappearing goal rings. Version 3 remains the
legacy default for reproducibility; building a new version 3 test bank is rejected.
The version 4 geometry partition separates its own splits. A test bank must also
exclude fingerprints from historical training banks when comparing with a model
trained on older generator versions. `tools/audit_generated_banks.py` accepts an
optional `--test` bank and checks overlaps between every supplied pair of splits.

## Train a policy checkpoint

Build small banks for a smoke test, or increase `--levels` for a real run:

```bash
uv run python -m pebby.ls20.bank \
  --levels 50 --split train --out data/levels-train.jsonl
uv run python -m pebby.ls20.bank \
  --levels 20 --split validation --out data/levels-validation.jsonl
```

Collect oracle-labelled data from each bank:

```bash
uv run python -m pebby.agent.data \
  --bank data/levels-train.jsonl --out data/ls20-train.npz
uv run python -m pebby.agent.data \
  --bank data/levels-validation.jsonl --out data/ls20-validation.npz
```

Train the default looped policy:

```bash
uv run python -m pebby.agent.train \
  --shards data/ls20-train.npz \
  --validation-shards data/ls20-validation.npz \
  --architecture looped \
  --device auto \
  --checkpoint-out checkpoints/ls20-looped-policy.pt
```

The trainer saves a checkpoint and a JSON training report. The report is useful for
diagnostics, but its accuracy metrics are proxies; closed-loop evaluation is the
relevant result.

The experimental semantic-repair trainer supports a predeclared
`--selection validation-policy` rule: evaluate the entire existing validation
panel at fixed update intervals and retain the first minimum of the average
per-tier negative log probability assigned to any optimal action. It saves the
selected and final checkpoints separately with their actual update counts.
The default remains `--selection final`. Existing validation is development data,
not fresh confirmation, and passing the trainer's qualification remains required
before a full run. This experimental workflow has not established better gameplay.

The counterfactual world-model path uses a different NPZ contract and trainer. A
minimal shape of that workflow is:

```bash
uv run python -m pebby.agent.world_data \
  --bank data/levels-train.jsonl --limit 50 --out data/world-train.npz
uv run python -m pebby.agent.world_data \
  --bank data/levels-validation.jsonl --limit 20 --out data/world-validation.npz
uv run python -m pebby.agent.world_train \
  --train data/world-train.npz \
  --validation data/world-validation.npz \
  --device auto \
  --checkpoint-out checkpoints/ls20-world.pt
```

Use `--help` on `bank`, `data`, `train`, `world_data`, and `world_train` before
scaling a run; their options and supported objectives are implementation details.

## Evaluate a checkpoint

Evaluate the seven shipped levels in one sequential rollout:

```bash
uv run python -m pebby.agent.evaluate \
  --checkpoint checkpoints/ls20-looped-policy.pt \
  --shipped --protocol strict
```

Evaluate a generated bank instead:

```bash
uv run python -m pebby.agent.evaluate \
  --checkpoint checkpoints/ls20-looped-policy.pt \
  --bank data/levels-validation.jsonl \
  --protocol strict
```

Evaluation must use the same checkpoint and data contract intended for deployment.
Do not treat the generator's stored route, a planner result, or training accuracy as
model completion.

Also evaluate each shipped level from its own fresh three-life start so an early
sequential failure cannot hide later-level behavior. The paired spatial evaluator
supports `--mode shipped-isolated` and records the actual goal count of each level.
Report this alongside the sequential result; isolated wins are not a seven-level
session win. Its generated comparison reports paired wins, regressions, and
uncertainty, with the level as the unit of comparison. Reused or fixed-tier panels
need separate interpretation and do not become fresh confirmation through a
statistical test.

## Start the model runtime

Serve a checkpoint through the viewer and HostAI-compatible runtime:

```bash
uv run serve.py \
  --port 11435 \
  --checkpoint checkpoints/ls20-looped-policy.pt
```

Then open `http://127.0.0.1:11435/ui/index.html`, or connect HostAI to
`http://127.0.0.1:11435`. Without `--checkpoint`, the server still exposes the
environment and viewer but reports that no agent is loaded.
