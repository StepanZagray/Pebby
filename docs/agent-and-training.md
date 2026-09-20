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

## Pebby's formulation: a planner with an ideal world model

Pebby has two parts with different contracts.

1. **World model.** Learned on LS20's fixed rules until it is effectively ideal:
   given a state and an action it returns the next state as the game would.
   Because the rules never change within a game, the model may store them in its
   weights. It is trained on real-engine transitions (every state carries all
   four branches) and judged by transition accuracy, never by gameplay alone.
2. **Planner.** Goal-agnostic. It receives a goal as a test over states, not as
   a location or a distance, and searches the world model for action sequences
   whose end state passes the test. Any learned component inside the planner (a
   progress estimate that makes search tractable) must be conditioned on the
   goal, so the same planner serves a different goal on the same rules without
   redesign. Path length is the action budget the game charges, not the
   definition of the goal.

The exact LS20 planner is the reference implementation of this contract on the
real engine, and `tools/ablate_engine_search.py` measures how far a learned
progress estimate is from it. Both are teachers and instruments, not the
learned controller.

## Full-route training workflow (v2)

The retained controller was trained on eight oracle-chosen snapshots per level,
never saw an unwinnable state, read the fuel bar only through a pooled encoder,
and replayed identical trajectories on every life. The following tools address
those gaps. They are development tooling; none of them is a benchmark result.

Run the probe ladder before and after any training. It is a cheap CPU gate on
two engine-verified primitives (empty-room navigation and cycler leave/avoid)
and fails loudly when a checkpoint cannot leave a cycler whose attribute already
matches the goal:

```bash
PYTHONPATH=. uv run python tools/probe_ladder.py \
  --checkpoint artifacts/spatial-recovery-v1/quality-fit/recovery.pt \
  --out artifacts/probe-ladder.json --count 96
```

Collect full-route rows: every state along the exact oracle route, every state
the learner visits, and uncapped oracle recoveries from the learner's first
mistake in each life. Unwinnable states are kept with an empty optimal mask.
Levels whose proof already exists are skipped, so an interrupted run resumes:

```bash
PYTHONPATH=. uv run python tools/collect_full_routes.py \
  --bank data/ls20-reference-unequal-v1/train.jsonl \
  --out-dir data/full-routes-v1/train \
  --checkpoint artifacts/spatial-recovery-v1/quality-fit/recovery.pt \
  --quotas 300 300 300 300 300 300 100 --workers 6 --seconds 7200
```

Tiers 6 and 7 need up to about 2 GiB per worker during exact search; the
collector refuses to start a search below its host memory reserve rather than
swapping. Do not run it beside another multi-worker search job.

Train the v2 policy. It warm-starts from a v1 checkpoint (identical logits
until the new fuel and lives inputs learn), decodes remaining steps and lives
exactly from the HUD pixels, can fine-tune the encoder, computes loss weights
on exactly the rows it trains on, evaluates on the whole validation directory,
stops on a validation plateau, and selects `best.pt` by actual sequential
gameplay first:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
uv run python tools/train_spatial_v2.py \
  --train-dir data/full-routes-v1/train --validation-dir data/full-routes-v1/validation \
  --init artifacts/spatial-recovery-v1/quality-fit/recovery.pt \
  --encoder-mode finetune --batch-size 96 --precision bf16 \
  --updates 6000 --eval-every 250 --gameplay-every 500 --patience 6 \
  --device cuda --out-dir artifacts/<run>
```

On an 8 GB GPU, `--encoder-mode finetune` fits batch 96; `frozen` fits 256.
`--qualify` runs three updates and one gameplay evaluation to check a setup.
`--direct-readout` adds a zero-initialised per-action score to the outcome
comparator, `--kind-weights route,learner,recovery` rebalances the three row
kinds, and `--distance-ordering W` turns on the pairwise successor-distance
ranking loss. The measured effect of each is recorded in the model status.

The runtime controller in `pebby/agent/diverse_controller.py` is a heuristic
layer, not learning. The first life of every level is strict argmax with the
evaluator's unchanged-frame mask, so it can never regress a checkpoint's
baseline; later lives on the same level sample at a temperature with reversal
and repeat penalties so the three lives differ. It is the default in the
serving engine (`serve.py --controller argmax` restores plain argmax) and is
available to the evaluator as `--protocol diverse`. The serving engine now also
answers RESET on a game over and reports a finished game instead of asking the
policy for a move.

`tools/build_balanced_bank.py` builds equal-quota generator version 4 banks
with train, validation and test splits and an optional forced-goal-order quota
for tier 6. Tiers 6 and 7 cost minutes per level; check the projection printed
by its pilot before committing a full build.

## Rule-inference experiment on LS20 variants

Only LS20's engine is available locally, so the leave-one-game-out protocol is
run on a family of LS20 rule variants. A **game** is one of the 24 action
permutations (agent action to engine direction), fixed for the whole game,
plus seven generated levels played in tier order with three lives per level
and a level reset on game over. Rules are constant within a game and differ
between games only by the permutation; 18 permutations are used for training
and 6 are held out. The identity permutation stays in training because the
shipped game uses it.

The question is whether a small sequence model, trained on whole games from
the training permutations, predicts transitions on held-out permutations by
inferring the mapping from the game's own earlier steps, or whether it has
memorised the training mappings. Three predictors are compared on the same
held-out games with the same metric: prediction accuracy of the next factored
state by step index within the game, and the number of actions until twenty
consecutive predictions are correct.

- `pebby/agent/incontext_dynamics.py`: a small causal transformer over the
  game's step sequence (public fields only) and a memoryless control that
  sees just the current step.
- `pebby/agent/hypothesis_dynamics.py`: explicit inference. Knows the LS20
  rules (a hand-written local rule, or the real engine as rule model), keeps
  the permutations consistent with every observed transition, and predicts
  under the survivors. An optional prior over permutation identity learned
  from training games is included to show that priors over rule identity
  cannot help on rules never seen.
- `pebby/agent/variant_metrics.py`: the shared scoring, plus the identity
  baseline, which predicts as if controls were standard: the behaviour of a
  model that has memorised one rule set.

Data comes from `tools/collect_variant_games.py` (whole games, exploration by
an epsilon-mixed oracle so later levels appear), training from
`tools/train_incontext_dynamics.py`, and the explicit arm from
`tools/evaluate_hypothesis_dynamics.py`. Predictions from the three arms are
expected to separate: a memorising model tracks the identity baseline and
does not improve with steps; a model with learned priors starts above the
no-prior inferrer and then stalls; the inferrer keeps improving until one
hypothesis survives. Whichever pattern appears names the component to build
next. Real public games replace the variant family as soon as their engines
are available; nothing in the protocol depends on LS20.
