# Agent and training

This is the detailed research record for the learned controller. It includes
reproduction commands and negative results because a higher validation score or a
better auxiliary metric does not by itself establish official-level completion.

## Current status

The best official checkpoint is the query-plus-glyph epoch 8 model: it completes 1/7
official levels (level 1), 15/100 compact generated monitor levels, and 0/20 harder
generated levels. The seven-level target has not been achieved.

Other reported controls include:

- the initial stateless looped pilot: 0/20 held-out generated levels and failure before
  clearing the first official level;
- the best history-aware world pilot: 8/50 generated validation levels and 0/7
  official levels;
- mixed-coverage continuation: 15/100 fixed monitor levels in epoch 1 and 12/100 in
  epoch 2, with next-state prediction error still worse than copying the current latent;
- a smaller-data grounding follow-up: 5/50; these were not controlled batch-size tests.

The closing-event and cell-recall continuations described below were still in progress
at the time of their recorded preflights; they have no completion result to report.

## Looped world policy

`pebby/agent/world_model.py` combines a weight-shared transformer refinement core,
causal observation/action history, and an action-conditioned latent predictor. The same
online encoder receives gradients through current and successor states; SIGReg
regularizes the predictor's latent representation. These mechanisms adapt
[LeWorldModel](https://arxiv.org/html/2603.19312v2) and its
[official implementation](https://github.com/lucas-maes/le-wm). History fusion,
distance supervision, player-relative action readout, and imagined-successor ranking
are task-specific additions. Learned successor predictions affect action scores; no
oracle or game-state parser supplies inference actions. More loops and noncollapsed
latents do not establish better control.

The historical five-tier curriculum includes compact rail/ring, fog, independent-goal,
and variable-cost lessons. It requires complete searches and actual engine victories,
but has limited exploration and no multiple-patroller combinations. Use the
[seven-tier generation workflow](seven-reference-difficulties.md) for new
reference-calibrated banks.

The historical world-model smoke-test pipeline is:

```bash
uv run python -m pebby.ls20.curriculum --legacy --levels 200 --split train --seed 10000 --out data/curriculum-train.jsonl
uv run python -m pebby.ls20.curriculum --legacy --levels 50 --split validation --seed 10000 --out data/curriculum-validation.jsonl
uv run python -m pebby.agent.world_data --bank data/curriculum-train.jsonl --limit 200 --coverage mixed_failure --out data/world-train.npz
uv run python -m pebby.agent.world_data --bank data/curriculum-validation.jsonl --limit 50 --coverage mixed_failure --out data/world-validation.npz
uv run python -m pebby.agent.world_train --train data/world-train.npz --validation data/world-validation.npz --device cuda --batch-size 1024 --drop-last --precision bf16 --checkpoint-encoder --encoder-chunk-size 128 --grounding --channels 48 --blocks 1 --loops 4 --latent 64 --predictor-hidden 128 --value-hidden 64 --checkpoint-out checkpoints/ls20-world.pt
uv run python -m pebby.agent.evaluate --checkpoint checkpoints/ls20-world.pt --protocol strict --on-stall repeat
uv run serve.py --checkpoint checkpoints/ls20-world.pt
```

Transition collection branches all four actions in the real engine and keeps only
complete contextual oracle supervision. Later-level rules are preserved. Evaluation
and inference reconstruct public history and reset it on life/level boundaries.
Checkpoint format `pebby.ls20-world-policy.v1` is distinct from older policies.
Context-zero layouts with launchers are excluded from exact-supervision data because a
launcher can leave a matching hint pending for another submitted action, which the
logical oracle does not model. A winning initial route alone would miss this off-path
error; `artifacts/world-seed13258-pending-hint.json` records the reproduction.

On the local 8 GiB RTX 5060, this configuration fits a true batch of 1024 at about
3.57 GiB allocated memory. The encoder processes chunks of 128 with activation
recomputation; the loss and float32 SIGReg still use all 1024 states in one optimizer
update. Re-probe after changing model or history size, with a cap of 1024:

```bash
uv run python -m pebby.agent.batch_probe --data data/world-train.npz --config-checkpoint checkpoints/ls20-world.pt --grounding --precision bf16 --checkpoint-encoder --encoder-chunk-size 128 --out artifacts/world-batch-probe.json
```

The collector shares untouched templates between temporary branches and reuses the
chosen branch. This reduced measured collection CPU time by 1.3–4.6× across contexts
with exactly matching frames and labels; see `artifacts/world-data-cpu-optimization.json`.
The native exact planner matched complete reference distance maps on generated test
cases and sped up measured searches 12–36× (3.6× for complete generation); see
`artifacts/planner-cpu-optimization.json`. A Python fallback remains available.

For the large dataset, `--curriculum --require-verified-data --min-train-levels 10000`
requires verified generated levels and samples one state per distinct level in every
full batch. For historical five-tier banks, difficulty ratios shift linearly across
optimizer updates from 55/25/12/6/2 percent to 5/10/20/25/40 percent. Validation
sampling stays fixed. The small example banks above are smoke tests and are too small
for this curriculum's 1,024-distinct-level batches.

The audited banks contain 10,296 training levels and 1,998 validation levels, with no
seed or gameplay overlap. See `artifacts/world-full-context-validity.json` and
`data/ls20-verified-{train,validation}.jsonl`. Difficulty tiers reflect mechanics,
including moving cyclers, fog, and multiple goals; shortest path length does not
increase monotonically across tiers.

The first large run used 133,363 prefix-collected training states and underrepresented
endings on longer levels: only 168 of 1,313 hardest levels supplied a winning successor.
`--coverage mixed` combines exploration with states spread across a complete expert
route, always retaining a final pre-win state. The replacement mixed dataset contains
145,857 training states and 27,970 validation states. Every retained row still has four
actual engine branches, and array-level checks confirm a terminal winning successor for
every one of the 10,296 training and 1,998 validation levels. See
`artifacts/world-mixedpath-training-preflight.json`.

The eight-epoch prefix baseline has 286,611 trainable parameters, reuses its shared
transformer block four times, and uses true batches of 1,024 distinct levels. Its final
validation optimal-action accuracy is 76.7%, but official completion is still 0/7,
both sequentially and in isolated tests. Its latent prediction error also remains
worse than copying the current latent. A two-epoch mixed-coverage continuation reached
15/100 monitor levels after epoch 1 and 12/100 after epoch 2, with 0/7 official levels.
No successful official controller is established.

Frozen probes find carried attributes more readily in early HUD features than in the
global latent (`artifacts/world-frozen-attribute-probes.json`). The optional
`--state-recall` path feeds current HUD tokens and learned player-weighted board
features into that latent's projector. It increases this configuration to 391,059
parameters. Compatible warm starts zero-pad new projector columns to preserve initial
outputs. The completed four-epoch recall checkpoint reached 6/100 fixed generated
monitor levels and 0/7 official levels in sequential and isolated cap-200 runs. Current
training uses single-step world losses and learned lookahead depth 1; multi-step world
losses and learned halting are not implemented.

A separate glyph encoder prototype reached 100% joint shape, colour, and rotation
accuracy on the existing 96-template alphabet across disjoint generated train and
validation levels. The integrated glyph variant measures 326,913 parameters versus
286,611 for the base world model and 391,059 with state recall. Its four-epoch
generated-only continuation completed 11/100 fixed monitor levels and 0/7 official
levels, so it did not improve on the mixed-coverage base checkpoint's 15/100 result.
The standalone classifier covers known templates only and does not establish unseen
template generalization or controller success. See
`artifacts/world-glyph-final-monitor.json` and
`artifacts/world-glyph-final-official.json`.

The auxiliary successor-policy continuation used the 286,611-parameter base checkpoint
with glyph perception off and selected epoch 3 at 10/100 generated monitor levels and
0/7 official levels; its last epoch reached 12/100. Its sidecars are
`data/ls20-world-successor-train-labels.npz` and
`data/ls20-world-successor-validation-labels.npz`; the original rows were replayed
exactly, with 564,750/108,199 extra valid successor-action labels. The trainer flags
are `--train-successor-labels`, `--validation-successor-labels`, and
`--successor-policy-weight`.

The query-plus-glyph experiment completed eight epochs from the best base checkpoint,
using learning rate .0005, SIGReg .0125, grounding 1, glyph 1, successor-policy 0,
and true batches of 1,024. Its selected epoch 8 completes official level 1 in 134
actions, with one life left; levels 2–7 remain unsolved in isolated tests. The corrected
sequential test allows 1,400 total actions and also clears one level before stalling.
Reports are `artifacts/world-query-glyph-final-{monitor,harder20,official-sequence1400}.json`.

The model has 403,061 parameters, including four context queries and two cross-attention
blocks over raw/refined board and HUD tokens. Four encoder loops share weights; they are
not four imagined game actions. Learned player geometry and glyph probabilities support
the readout without an inference parser, teacher coordinates, or game rules. The
query-only model has 362,087 parameters and the original base has 286,611.

The expanded bank is complete: 20,296 training levels / 304,365 states and 3,998
validation levels / 59,649 states, with contextual engine winning proofs, disjoint
layouts/seeds, and winning examples for every retained level. The combined NPZ files
are `data/ls20-world-combined-{train,validation}.npz`. A two-epoch control continuation
completed 14/100 compact generated levels, 1/20 harder levels, and 0/7 official levels,
despite 83.5% validation optimal-action accuracy. The earlier query-plus-glyph
checkpoint remains the best official result at 1/7.

A matched generated-only on-policy continuation completed 16/100 compact levels and
0/7 official levels. Unchanged-frame actions fell from 81.6% to 49.6%, but compact
game-over endings increased from four to 27. Its supplemental bank contains 23,874
policy rows and 1,998 expert anchors across 1,024 existing training levels; it adds no
new level identities. See `artifacts/world-onpolicy-run.json`.

A second replay iteration trained from the first iteration's epoch 1. Its aggregate
contains 54,684 model-visited rows plus 3,975 expert anchors from 2,048 distinct
training levels. Its final checkpoint completed 20/100 compact generated levels,
1/20 harder levels, and 0/7 official levels; epoch 1 reached 21/100 compact levels,
but 62 runs ended in game over. See
`artifacts/world-onpolicy-round2-run.json` and
`artifacts/world-onpolicy-aggregate2-validation-reviewed.json`.

Four-step training is available through `--rollout-index`. A source-bound index selects
46,503 recorded sequences from 2,003 training levels. Half of each 1,024-level batch
retains ordinary four-action branches; the other half uses four chronological actions
and frames. This adds no parameters. The live-only four-step experiment finished at
18/100 compact generated levels, 1/20 harder levels, and 0/7 official levels; its
epoch 1 reached 21/100 compact. On 1,024 fixed held-out levels, horizon-four MSE was
.2903 versus .4413 for copying the initial latent. See
`artifacts/world-k4-final-{monitor,harder20,official,k4-validation}.json`.

A closing-event continuation is recorded as running from K4 epoch 1. It retains
403,061 parameters and 20,296 distinct training levels, with two epochs/594 updates
and 512 closing plus 512 ordinary levels per batch. Its 2,025 attested sequences end
in 222 wins, 879 deadends, and 924 live states; only step four may terminate or reset.
CUDA preflight measured 3.99 GiB allocated, with 68 final wins and 211 unsafe endings
in its 512 closing rows; its two updates were discarded. No closing-run completion
result exists yet. See `artifacts/world-closing-run.json` and
`artifacts/world-closing-b1024-preflight.json`.

Frozen risk probes found readable signal on risk-enriched generated-bank states, but
little improvement on fixed fatal histories. Horizon-two latent MPC reduced compact
wins from 21 to 18 for second-replay epoch 1 and from 21 to 15 for K4 epoch 1. On a
new targeted mechanism50 validation bank, second-replay epoch 1 won 8/50 versus
query8's 1/50, but suffered 42 game overs; both failed all three- and four-goal cases.
These pilots have not entered training. Best observed official completion remains
query8's 1/7.

The encoder uses four shared transformer passes; the action-conditioned predictor
remains a two-block AdaLN MLP. Inference retains depth-1 learned lookahead and has no
learned halting. Extended training levels still use cost-1 short synchronous rails and
add no launchers. A separate generator pilot broadens mechanism counts, but those new
levels have not entered training.

An optional `--cell-recall` path adds a frozen learned appearance decoder to every
frame's spatial tokens through a zero-initialized projection. The configuration has
514,283 total parameters, including 110,166 frozen decoder parameters and 404,117
trainable parameters. Import requires the exact generated decoder bank, its completed
proof, and checkpoint provenance; inherited weights are checked against a canonical
digest. Existing checkpoints keep the option disabled.

The decoder scored all 256,254 eligible role cells and 3,110 active goal readings
correctly in 3,148 replay samples from 100 generated validation levels. Hidden player
underlays were excluded. A separate ten-level reset check also passed; moving-goal and
transient-animation appearance remain untested. The separate global visibility model
scored 3,120/3,148 gameplay masks exactly and is not used by the controller. These are
perception results, not completion results. The cell-recall continuation had a full
closing-sequence batch preflight at 3.987 GiB allocated / 5.035 GiB reserved.

## Stateless and looped policy baselines

The original stateless baseline can still be reproduced with:

```bash
uv sync
uv run python -m pebby.ls20.bank --levels 5000 --split train --out data/levels-train.jsonl
uv run python -m pebby.agent.data --bank data/levels-train.jsonl --out data/ls20-train.npz
uv run python -m pebby.agent.train --architecture looped --shards data/ls20-train.npz --checkpoint-out checkpoints/ls20-looped-policy.pt
uv run python -m pebby.agent.evaluate --checkpoint checkpoints/ls20-looped-policy.pt
uv run serve.py --port 11435 --checkpoint checkpoints/ls20-looped-policy.pt
```

The default looped spatial transformer has 239,116 parameters, 64 channels, four
attention heads, and two physical blocks reused four times. A palette-specific 5×5
stride-5 stem produces 144 cell tokens; a separate HUD encoder produces 16 column
tokens. All 160 tokens participate in attention and the position-preserving action
readout. The HUD encoder still pools spatial detail; this is not lossless pixel storage.

Every loop reads both the evolving state and the original encoded frame. Input recall
enters attention and feed-forward updates, and LayerNorm normalizes each residual
result. Recurrence happens within one decision: the model carries no memory between
actions and does not simulate future game states. Fog and hidden rail state remain
possible information limits on shipped levels.

This is a task-specific adaptation of [input recall and outer normalization](https://arxiv.org/html/2604.15259v2),
with variable-depth training motivated by [recurrent-depth pretraining](https://arxiv.org/html/2502.05171v2).
The [September RecurTrace revision](https://arxiv.org/html/2609.03379v2) motivates
measuring a quality-versus-depth curve before adding a learned halter; its results are
from pretrained language models and do not establish a benefit on LS20.

Training samples one depth per minibatch, uniformly from 1 through `--loops` (default
4). `--loop-loss final` supervises that exit; `--loop-loss all` averages action
cross-entropy over every visited exit as a separate intervention. Full backpropagation
through the short loop, gradient clipping, and float32 are used.

```bash
uv run python -m pebby.agent.data --bank data/levels-train.jsonl --workers 2 --out data/ls20-train-v2.npz
uv run python -m pebby.agent.data --bank data/levels-validation.jsonl --workers 2 --out data/ls20-validation-v2.npz
uv run python -m pebby.agent.train --architecture looped --shards data/ls20-train-v2.npz --validation-shards data/ls20-validation-v2.npz --loops 4 --train-min-loops 1 --loop-loss final --checkpoint-out checkpoints/ls20-looped-policy.pt
uv run python -m pebby.agent.evaluate --checkpoint checkpoints/ls20-looped-policy.pt --bank data/levels-test.jsonl --protocol strict --loops 4 --report-out checkpoints/ls20-looped-depth4.json
```

`--architecture cnn` retains the original convolutional baseline and its defaults;
`--broadcast-hud` remains CNN-only. CNN checkpoints retain format
`pebby.ls20-policy.v2`; looped checkpoints use `pebby.ls20-looped-policy.v1`. Old
weights load exactly, without conversion. The trainer uses a separate default looped
checkpoint filename.

Behaviour cloning uses the exact planner. The planner yields the optimal action for
every reachable state, not only solution-path states, so training data covers recovery.
Off-path states come from prefix-replay deviation: replay k optimal actions, take one
deviating action, then follow the oracle. Naive ε-greedy does not work here: on 40
levels, random actions at probability 0.05 still completed 32/40, but at 0.25 only
10/40, because random detours spend budget the level cannot spare.

## Training protocol and metrics

Official levels are **evaluation-only**. Nothing in `pebby/agent/data.py` or
`pebby/agent/train.py` can construct a shipped-level environment;
`Ls20Env(levels=None)` appears only in `pebby/agent/evaluate.py`. Splits are separated
by seed range:

| Set | Source | Seeds | Used for |
|---|---|---|---|
| Train | generated | 0–4,999 | training only |
| Validation | generated | 1,000,000+ | epoch selection |
| Test | generated | 2,000,000+ | held-out completion rate |
| Shipped levels 1–7 | upstream, verbatim | — | **evaluation only** |

Verify a shard's provenance instead of taking it on trust:

```bash
uv run python -c "import numpy as np, json; z = np.load('data/ls20-train.0000.npz', allow_pickle=True); print(json.loads(str(z['meta'])))"
```

`bank.load()` accepts supported historical and current structural formats. Collection
and extension separately check contextual proofs; a readable bank is not automatically
current or verified training data.

Per-step imitation accuracy is a poor headline because the oracle often has several
equally optimal moves. Current targets use the complete optimal-action mask rather than
one arbitrary tie. The useful metric is whether distance-to-completion fell by one and,
above that, closed-loop completion on held-out levels.

Completion compounds: it is roughly per-step optimal-action rate raised to path length,
and the median generated level needs 25 actions. At that length, a per-step rate of
0.90 is worth about 7% completion and 0.98 about 60%, so small per-step gains matter
far more than they look. These are intuition-building approximations, not guarantees
for the learned controller.
