# Navigation research experiments

This harness separates three questions: can Pebby fit a controlled navigation
lesson, does the skill transfer to withheld settings, and does it survive actual
gameplay? It does not promote a checkpoint or claim seven-level completion.

The retained checkpoint remains
`artifacts/spatial-recovery-v1/quality-fit/recovery.pt`, SHA256
`ff88327214b6dc2d4278167e0d61edcc37c683292b788be6927b71286331a5f8`.

## First matched experiment

Run from the repository root:

```bash
.venv/bin/python -m tools.train_navigation_probe \
  --parent artifacts/spatial-recovery-v1/quality-fit/recovery.pt \
  --parent-sha256 ff88327214b6dc2d4278167e0d61edcc37c683292b788be6927b71286331a5f8 \
  --out artifacts/navigation-adjacent-seed42 \
  --stages adjacent --groups-per-split 2 \
  --steps 100 --batch-size 8 --eval-every 50 \
  --rollouts-per-stage 4 --rollout-cap 8 --max-seconds 180
```

The three default arms use the same generated roots, sampled batches, pretrained
encoder initialization and freshly initialized spatial controller trunk:

| Arm | Encoder learning | Action readout |
|---|---|---|
| `frozen-outcomes` | Frozen | Predicted outcome probabilities → comparator |
| `finetune-outcomes` | Active feature encoder learns from public frames | Same outcome/comparator architecture |
| `frozen-direct` | Frozen | Shared pre-outcome spatial summary → fresh action head |

The encoder comparison changes gradient eligibility and adds a separate encoder
learning rate, default `3e-5`; the controller rate defaults to `3e-4`. Both encoders
stay in evaluation mode. The encoder is executed live in both arms, avoiding the
disconnected cached-feature training path. Parameter eligibility, gradients and
actual weight changes are recorded. Unused legacy heads and softmax-invariant
player biases remain frozen; optional internally frozen cell-appearance weights
are explicitly reported. Stored parameter count is not active trainable capacity.

The direct readout shares the exact spatial trunk and learned raw/state/glyph/player
inputs. It introduces no true player or goal coordinates. Its terminal head has
different capacity and initialization, so a gain supports investigating the
decision interface; it does not uniquely prove which detail caused the gain.

All three default arms use **policy-only supervision**: uniform cross entropy over
the set of optimal actions. Physical, distance and event losses are deliberately
disabled for this comparison. The predictor outputs in that run are intermediate
features, not supervised physical forecasts. To compare encoder adaptation under
primitive-compatible joint supervision, use a separate output directory:

```bash
.venv/bin/python -m tools.train_navigation_probe \
  --parent artifacts/spatial-recovery-v1/quality-fit/recovery.pt \
  --parent-sha256 ff88327214b6dc2d4278167e0d61edcc37c683292b788be6927b71286331a5f8 \
  --out artifacts/navigation-outcomes-seed42 \
  --arms frozen-outcomes finetune-outcomes --objective outcomes \
  --stages adjacent open detour --groups-per-split 2 --max-seconds 600
```

`outcomes` uses unweighted policy, physical-field, remaining-distance and event
losses. Unlike the retained balancing objective it supports datasets with no
attribute changes or positive life-loss events. Expert navigation has no life-loss
positives by design; absent-event recall is undefined, not a measured failure.
This diagnostic objective is an explicit change from historical training.

## Curriculum and learner-state teaching

The generator supplies three stages: adjacent matching goals, longer cardinal and
diagonal open-room routes, and walls forcing an initially non-Manhattan detour.
Every route and action branch is checked in the actual vendored engine. Static BFS
and engine state supply labels and diagnostics only. Model calls receive only
causal public frames, validity masks and previous action indices.

Each start/glyph group contains opposite goal pairs and all requested stages. The
whole group belongs to one split. Exact initial-pixel overlap across splits is
checked by re-rendering; geometric similarity and all history overlaps are not
excluded. These controlled settings do not represent the full LS20 distribution.

`mixed` sampling balances stages, cases, then roots, so a longer trajectory gets no
extra case weight. `--schedule mastery` starts at the first supplied stage and
advances only when development first-action accuracy, mean case accuracy, and
rollout win rate meet their declared thresholds. Earlier stages must still pass.
While fitting a later stage, 25% of sampling goes to previous stages.

Optional learner recollection uses the updated policy's public observations,
labels every visited root's four engine branches, and replaces a declared fraction
of subsequent batches. Collection is capped and causal history clears on life
loss. For example, after choosing a single arm:

```bash
.venv/bin/python -m tools.train_navigation_probe \
  --parent artifacts/spatial-recovery-v1/quality-fit/recovery.pt \
  --parent-sha256 ff88327214b6dc2d4278167e0d61edcc37c683292b788be6927b71286331a5f8 \
  --out artifacts/navigation-mastered-replay-seed42 \
  --arms finetune-outcomes --schedule mastery \
  --stages adjacent open detour --steps 200 --eval-every 25 \
  --recollect-every 50 --recollect-cases 2 --recollect-cap 16 \
  --replay-fraction .25 --max-seconds 600
```

Adaptive scheduling and recollection require a single arm. They change training
exposure according to the model, so the tool rejects combining them with the
fixed-batch multi-arm comparison. An adaptive-versus-shuffled run alone does not
isolate curriculum ordering: accepted example counts and exposure must also be
matched in a later experiment. Small development panels are engineering gates,
not statistical proof of mastery.

## Read the reports

Each fresh output directory contains a bank, report and independent experimental
checkpoints. Existing output directories are rejected. Reports include checkpoint,
bank and executed-source hashes; runtime budget; PID; random seeds; losses;
sampled root indices; gradient norms; changed parameter names; stage decisions;
learner recollections; and full generated gameplay traces.

Every checkpoint is evaluated before fitting, at declared intervals and at the
final update. Reported root statistics include case and group means in addition
to micro accuracy. Goal-pair metrics require both opposite goals to be handled;
detour pairs are excluded because their optimal first moves may coincide.
Gameplay reports completion, action counts, excess actions on wins and the first
nonoptimal decision. Their labels do not enter policy inference.

Interpret the result before expanding a run:

| Result | Next question |
|---|---|
| Tiny training bank remains unfitted | Are optimization, targets, losses or the representation preventing learning? |
| Training fits, development fails | Which goal/layout variation or shortcut prevents transfer? |
| Development decisions improve, gameplay does not | Are learner-state coverage and recovery adequate? |
| Navigation transfers, composed LS20 fails | Which attribute/resource/multiple-goal dependency is missing? |

Repeat promising comparisons with different `--seed` values while retaining the
same `--data-seed` for paired evaluation. Report gameplay across seeds, not only
offline metrics. Short runs do not establish convergence or capacity ceilings.
For a separate controller-capacity comparison, use `--width 96` with fresh
initialization and otherwise identical settings; the retained default width is
48. This changes controller capacity, not encoder width or pretraining.

## Separate confirmation and sequential evaluation

Training never collects or scores confirmation roots. Their bank specifications
are saved for a separate explicit evaluation. Copy the checkpoint and bank hashes
from the training report into these commands:

```bash
.venv/bin/python -m tools.evaluate_navigation_probe \
  --checkpoint PATH_TO_CANDIDATE --checkpoint-sha256 CHECKPOINT_SHA256 \
  --bank PATH_TO_BANK_JSON --bank-sha256 BANK_SHA256 \
  --split confirmation --out artifacts/navigation-confirmation.json

.venv/bin/python -m tools.evaluate_navigation_probe \
  --checkpoint PATH_TO_CANDIDATE --checkpoint-sha256 CHECKPOINT_SHA256 \
  --sequential --per-level-cap 300 --out artifacts/navigation-sequential.json
```

The evaluator accepts either a navigation experiment checkpoint or the retained
spatial checkpoint. Compare them under the same protocol. Sequential evaluation
uses one continuous shipped game, charged current-level RESET after GAME_OVER,
and a shared action cap for all attempts within each level. It does not sum
isolated wins. This differs from older strict-no-reset results; do not compare
those scores as if the protocols were identical. No official API scorecard is
created. Voluntary RESET and persistent game memory are not learned.

A confirmation evaluation records exposure before scoring; a partly failed run
still exposes the panel. Once results inform changes, that panel is development
evidence. The shipped levels have already been inspected repeatedly and are not
an untouched generalization test.

## Remaining architecture decisions

This implementation supplies the first diagnostic experiments and the navigation
curriculum/recollection loop. It does not implement a pretrained controller,
language-model teacher, persistent memory or recurrent future-state planner.
Those are separate, conditional architecture experiments in the wayfinder map.
Primitive training alone also cannot teach attribute changes, resource management
or multiple-goal ordering, because this corpus deliberately excludes them.

For memory, qualify the information limitation with identical permitted recent
histories that require different actions due to an earlier reveal. For planning,
use fully observed decisions with different downstream consequences. For a
pretrained baseline, declare whether input includes pixels, a decoded map, longer
history or search tools, and compare complete-system cost and actual gameplay.
The acceptance target remains all seven levels in one declared sequential session.
