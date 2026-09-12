# Pebby

An agent that learns to complete **LS20**, one of the ARC-AGI-3 games, by training on
procedurally generated levels that are *proved* completable.

Pebby does not re-implement LS20. It vendors the official game verbatim and plays it,
so generated levels and the seven shipped levels run the same code. Retained
training levels must pass a complete oracle search and a winning replay in the
game's own implementation, in the context used for training.

The current research target is **a learned controller that completes all seven
official levels**, with official observations and solutions reserved for evaluation.
The best official checkpoint, query-plus-glyph epoch 8, completes **1/7 official levels** (level 1),
15/100 compact generated monitor levels, and 0/20 harder generated levels.
The seven-level target has not been achieved. The initial stateless looped pilot completed
0/20 held-out generated levels and failed before clearing the first official level.
The best history-aware world pilot completed 8/50 generated validation levels and
0/7 official levels. The mixed-coverage continuation peaked at 15/100 fixed
monitor levels in epoch 1 and reached 12/100 in epoch 2. Its next-state prediction
error remained worse than copying the current latent. A smaller-data grounding
follow-up completed 5/50; these comparisons were not controlled batch-size tests.

### Looped world policy

`pebby/agent/world_model.py` combines a weight-shared transformer refinement core,
causal observation/action history, and an action-conditioned latent predictor.
The same online encoder receives gradients through current and successor states;
SIGReg regularizes the predictor's latent representation. These mechanisms adapt
[LeWorldModel](https://arxiv.org/html/2603.19312v2) and its
[official implementation](https://github.com/lucas-maes/le-wm).
The history fusion, distance supervision, player-relative action readout and
imagined-successor ranking are task-specific additions. The learned successor
predictions affect action scores; no oracle or game-state parser supplies inference
actions. More loops and noncollapsed latents do not establish better control.

The separate curriculum includes compact rail/ring, fog, independent-goal and
variable-cost lessons. It requires complete searches and actual engine victories.
These lessons have limited exploration and no multiple-patroller combinations, so
they do not yet cover the full difficulty of the official game.

```bash
uv run python -m pebby.ls20.curriculum --levels 200 --split train --seed 10000 --out data/curriculum-train.jsonl
uv run python -m pebby.ls20.curriculum --levels 50 --split validation --seed 10000 --out data/curriculum-validation.jsonl
uv run python -m pebby.agent.world_data --bank data/curriculum-train.jsonl --limit 200 --coverage mixed_failure --out data/world-train.npz
uv run python -m pebby.agent.world_data --bank data/curriculum-validation.jsonl --limit 50 --coverage mixed_failure --out data/world-validation.npz
uv run python -m pebby.agent.world_train --train data/world-train.npz --validation data/world-validation.npz --device cuda --batch-size 1024 --drop-last --precision bf16 --checkpoint-encoder --encoder-chunk-size 128 --grounding --channels 48 --blocks 1 --loops 4 --latent 64 --predictor-hidden 128 --value-hidden 64 --checkpoint-out checkpoints/ls20-world.pt
uv run python -m pebby.agent.evaluate --checkpoint checkpoints/ls20-world.pt --protocol strict --on-stall repeat
uv run serve.py --checkpoint checkpoints/ls20-world.pt
```

Transition collection branches all four actions in the real engine and keeps only
complete contextual oracle supervision. Later-level rules are preserved. Evaluation
and inference reconstruct public history and reset it on life/level boundaries.
Checkpoint format `pebby.ls20-world-policy.v1` is distinct from the older policies.
Context-zero layouts with launchers are excluded from exact-supervision data:
a launcher can leave a matching hint pending for another submitted action, which
the logical oracle does not model. A winning initial route alone would miss this
off-path error; `artifacts/world-seed13258-pending-hint.json` records the reproduction.

On the local 8 GiB RTX 5060, this configuration fits a true batch of **1024** at
about 3.57 GiB allocated memory. The encoder processes chunks of 128 with activation
recomputation; the loss and float32 SIGReg still use all 1024 states in one optimizer
update. This preserves full-batch gradients. Re-probe after changing
model or history size, with a cap of 1024:

```bash
uv run python -m pebby.agent.batch_probe --data data/world-train.npz --config-checkpoint checkpoints/ls20-world.pt --grounding --precision bf16 --checkpoint-encoder --encoder-chunk-size 128 --out artifacts/world-batch-probe.json
```

The collector shares untouched templates between temporary branches and reuses
the chosen branch. This reduced measured collection CPU time by 1.3–4.6× across
contexts with exactly matching frames and labels; see
`artifacts/world-data-cpu-optimization.json`. The native exact planner matched
complete reference distance maps on generated test cases and sped up measured
searches 12–36× (3.6× for complete generation); see
`artifacts/planner-cpu-optimization.json`. A Python fallback remains available.

For the large dataset, `--curriculum --require-verified-data --min-train-levels 10000`
requires verified generated levels and samples one state per distinct level in
every full batch. Difficulty ratios shift linearly across optimizer updates from
55/25/12/6/2 percent to 5/10/20/25/40 percent. Validation sampling stays fixed.
The small example banks above are only collection smoke tests; they are too small
for this curriculum's 1024-distinct-level batches.

The audited banks contain **10,296 training levels and 1,998 validation levels**,
with no seed or gameplay overlap. See `artifacts/world-full-context-validity.json`
and `data/ls20-verified-{train,validation}.jsonl`. Difficulty tiers reflect
mechanics, including moving cyclers, fog and multiple goals; shortest path length
does not increase monotonically across the tiers.

The first large run uses 133,363 prefix-collected training states. This collection
underrepresents endings on longer levels: only 168 of 1,313 hardest levels supply
a winning successor. `--coverage mixed` remedies that limitation by combining
exploration with states spread across a complete expert route, always retaining
a final pre-win state. Every retained row still has four actual engine branches.
Unselected expert steps execute once, avoiding four-way branching everywhere.
The replacement mixed dataset contains 145,857 training states and 27,970 validation
states. Array-level checks confirm a terminal winning successor for every one of
the 10,296 training and 1,998 validation levels. `--require-winning-coverage`
enforces this independently of metadata claims. See
`artifacts/world-mixedpath-training-preflight.json` for hashes and counts.

The eight-epoch prefix baseline has 286,611 trainable parameters, reuses its shared
transformer block four times, and uses true batches of 1,024 distinct levels.
Its final validation optimal-action accuracy is 76.7%, but official completion
is still **0/7**, both sequentially and in isolated tests. Its latent prediction
error also remains worse than copying the current latent. A two-epoch continuation
on mixed coverage, with final hard-heavy curriculum proportions and SIGReg weight
0.0125, completes 15/100 monitor levels after epoch 1 and 12/100 after epoch 2.
The final checkpoint remains at **0/7 official levels**, including with the existing
public-frame stall-handling controller. No successful official controller is established.

Frozen probes find carried attributes more readily in early HUD features than in
the global latent (`artifacts/world-frozen-attribute-probes.json`). The optional
`--state-recall` path feeds current HUD tokens and learned player-weighted board
features into that latent's projector. It increases this configuration to 391,059
parameters. Compatible warm starts zero-pad the new projector columns to preserve
initial outputs. The completed four-epoch recall checkpoint reached 6/100 fixed
generated monitor levels and 0/7 official levels in both sequential and isolated
cap-200 runs; see `artifacts/world-recall-final-monitor.json` and
`artifacts/world-recall-final-official.json`. The 391,059 figure describes
architecture size, not performance. Current training uses single-step world losses
and learned lookahead depth 1; multi-step world losses and a learned halting
mechanism are not implemented.

A separate glyph encoder prototype reached 100% joint shape, colour and rotation
accuracy on the existing 96-template alphabet, across disjoint generated train and
validation levels (`artifacts/world-glyph-pretraining-prototype.json` and
`artifacts/world-glyph-data-audit.json`). The integrated glyph variant measures
326,913 parameters versus 286,611 for the base world model and 391,059 with state
recall. Integration passed 209 backend tests (one skipped) and a real-checkpoint
output-preservation check. Its four-epoch generated-only continuation completed
11/100 fixed monitor levels and 0/7 official levels, so it did not improve on the
best mixed-coverage base checkpoint's 15/100 monitor result. The standalone
classifier result covers known templates only and does not establish unseen-template
generalization or controller success; the integrated result is likewise not official
control success (`artifacts/world-glyph-final-monitor.json` and
`artifacts/world-glyph-final-official.json`).

The auxiliary successor-policy continuation used the 286,611-parameter base
checkpoint with glyph perception off and selected epoch 3 at 10/100 generated
monitor levels and 0/7 official levels in both sequential and isolated tests;
its last epoch reached 12/100 generated levels. At that stage, the best result was the
mixed-coverage base epoch 1 at 15/100 and 0/7 official. Its sidecars are
`data/ls20-world-successor-train-labels.npz` and
`data/ls20-world-successor-validation-labels.npz`; the original 145,857/27,970
rows across 10,296/1,998 levels were replayed exactly, with 564,750/108,199 extra
valid successor-action labels. Source images are unchanged, with no new levels or
multistep world loss. The trainer flags are `--train-successor-labels`,
`--validation-successor-labels` and `--successor-policy-weight`.

The query-plus-glyph experiment completed eight epochs from the best base checkpoint,
using learning rate .0005, SIGReg .0125, grounding 1, glyph 1, successor-policy 0,
and true batches of 1,024. Its selected epoch 8 completes official level 1 in 134
actions, with one life left; levels 2–7 remain unsolved in isolated tests. The
corrected sequential test allows 1,400 total actions and also clears one level
before stalling. Earlier 200-total-action sequential reports cannot certify all
seven wins; isolated tests retain a 200-action cap per level. Reports are
`artifacts/world-query-glyph-final-{monitor,harder20,official-sequence1400}.json`.

The model has **403,061 parameters**, including four context queries and two
cross-attention blocks over raw/refined board and HUD tokens. Four encoder loops
share weights; they are not four imagined game actions. Learned player geometry
and glyph probabilities support the readout without an inference parser, teacher
coordinates or game rules. The query-only model has 362,087 parameters and the
original base has 286,611. Preflight measured 3.63 GiB GPU allocation at batch
1,024 with encoder checkpointing and 128-example chunks.

The expanded bank is complete: **20,296 training levels / 304,365 states** and
**3,998 validation levels / 59,649 states**, with contextual engine winning proofs,
disjoint layouts/seeds and winning examples for every retained level. Collection
checks all four actual branches at sampled states, not every reachable engine state.
The combined NPZ files are `data/ls20-world-combined-{train,validation}.npz`;
`artifacts/world-combined-data-validation.json` records exact small-array alignment
and hashes. The optional `--data-cache-dir data/world-array-cache` extracts checked
NPY members and maps them from disk. Both-bank loader validation peaked at 1.41 GiB
RAM; active training also uses reclaimable mapped pages and GPU memory.

A two-epoch control continuation finished training on this expanded bank with the same
403,061-parameter architecture, batch 1,024, learning rate .0003 and seed 6.
Difficulty proportions shift from 35/25/20/15/5 to 5/10/20/25/40 percent.
It completed 14/100 compact generated levels, 1/20 harder levels and 0/7 official
levels, despite 83.5% validation optimal-action accuracy. The earlier query-plus-glyph
checkpoint remains the best official result at 1/7.
`artifacts/world-combined-control-run.json` records the run. A matched generated-only
on-policy continuation completed the same 594 updates from the same original
checkpoint, reserving 256/1,024 batch entries for model-visited histories.
Its final checkpoint completed 16/100 compact levels and 0/7 official levels.
Unchanged-frame actions fell from 81.6% to 49.6%, but compact game-over endings
increased from four to 27. Better recovery from repeated stalls did not establish
successful navigation. Its supplemental bank contains 23,874 policy rows and
1,998 expert anchors across 1,024 existing training levels; it adds no new level
identities. All 103,488 sampled branches passed checks. The fixed validation
history probe remains evaluation-only. `artifacts/world-onpolicy-run.json` records
the run; `--on-policy-data` and `--on-policy-fraction` expose the training path.

A second replay iteration trained from the first iteration's epoch 1, selected
by generated completion and fewer fatal failures; its official result was also 0/7.
The replay aggregate contains **54,684 model-visited rows plus 3,975 expert anchors
from 2,048 distinct training levels**. Each source retains its behavior checkpoint,
hash and exact row ranges. The second run reserves 512/1,024 batch entries for
these histories, with two epochs and 594 updates. Independent checks matched all
20 array payloads, including images, against the raw source banks; the sampler
retains 1,024 distinct levels and exact difficulty quotas. See
`artifacts/world-onpolicy-round2-run.json` and
`artifacts/world-onpolicy-aggregate2-validation-reviewed.json`. Its final checkpoint
completed 20/100 compact generated levels, 1/20 harder levels and 0/7 official
levels. Epoch 1 reached 21/100 compact levels, but 62 runs ended in game over.

Four-step training is now available through `--rollout-index`. A source-bound
index selects 46,503 recorded sequences from 2,003 training levels. Half of each
1,024-level batch retains ordinary four-action branches; the other half uses
four chronological actions and frames. Predictions feed back into the predictor
without detaching, and the same online encoder receives target gradients.
Current action scores still use only the current public history and four imagined
first actions. Sequence clips exclude terminal and life-reset boundaries; ordinary
rows retain winning and reset examples. This adds no parameters.

The live-only four-step experiment finished at 18/100 compact generated levels,
1/20 harder levels and 0/7 official levels; its epoch 1 reached 21/100 compact.
On 1,024 fixed held-out levels, horizon-four MSE was .2903 versus .4413 for copying
the initial latent. Better latent prediction did not establish better completion.
See `artifacts/world-k4-final-{monitor,harder20,official,k4-validation}.json`.

A closing-event continuation is **running**, started 12 September at 02:30 IST from
K4 epoch 1. It retains 403,061 parameters and 20,296 distinct training levels,
with two epochs/594 updates and 512 closing plus 512 ordinary levels per batch.
The new `--closing-rollout-index`, `--closing-action-attestation` and
`--closing-action-sha256` flags are required together and exclude `--rollout-index`.
Its 2,025 attested sequences end in 222 wins, 879 deadends and 924 live states;
only step four may terminate or reset. Reset targets are tested, but this subset
contains no resets. No extra unsafe loss is added. Full backend validation ran
343 tests: 342 passed and one skipped. CUDA preflight measured 3.99 GiB allocated,
with 68 final wins and 211 unsafe endings in its 512 closing rows; its two updates
were discarded. No closing-run completion result exists yet. See
`artifacts/world-closing-run.json` and `artifacts/world-closing-b1024-preflight.json`.

Frozen risk probes found readable signal on risk-enriched generated bank states,
but little improvement on fixed fatal histories. Their enriched sampling prevents
population-calibration claims. Separately, horizon-two latent MPC reduced compact
wins from 21 to 18 for second-replay epoch 1 and from 21 to 15 for K4 epoch 1;
extra imagined search did not help these controls. On a new targeted mechanism50
validation bank, second-replay epoch 1 won 8/50 versus query8's 1/50, but suffered
42 game overs; both failed all three- and four-goal cases. These fixed-room,
single-changing-attribute pilots have **not entered training**. Reports:
`artifacts/world-frozen-binary-risk-probe.json`,
`artifacts/world-latent-mpc-100-h2-{round2,k4}-epoch1.json`, and
`artifacts/world-mechanism-validation50-comparison.json`. Best observed official
completion remains query8's **1/7**; the seven-level goal remains unmet.

The encoder uses four shared transformer passes; the action-conditioned predictor
remains a two-block AdaLN MLP. Inference retains depth-1 learned lookahead and has
no learned halting. Extended training levels still use cost-1 short synchronous
rails and add no launchers. A separate generator pilot broadens mechanism counts,
but those new levels have not entered training.

An optional `--cell-recall` path now adds a frozen learned appearance decoder
to every frame's spatial tokens through a zero-initialized projection. It uses
public pixels only; role/goal probabilities are fallible evidence, with no
engine visibility mask or goal-matching rule at inference. The configuration
has 514,283 total parameters, including 110,166 frozen decoder parameters and
404,117 trainable parameters. Import requires the exact generated decoder bank,
its completed proof, and checkpoint provenance; inherited weights are checked
against a canonical digest. Existing checkpoints keep the option disabled.

The decoder scored all 256,254 eligible role cells and 3,110 active goal readings
correctly in 3,148 replay samples from 100 generated validation levels. Hidden
player underlays were excluded. A separate ten-level reset check also passed;
moving-goal and transient-animation appearance remain untested. The separate
global visibility model scored 3,120/3,148 gameplay masks exactly and is not used
by the controller. These are perception results, not completion results.
The two-epoch cell-recall controller continuation is in progress; its full
closing-sequence batch of 1,024 distinct generated levels passed a disposable
GPU check at 3.987 GiB allocated / 5.035 GiB reserved. See
`artifacts/world-cell-recall-run.json` and
`artifacts/world-cell-recall-closing-b1024-preflight.json` for run evidence.

### Stateless policy baseline

```bash
uv sync
uv run python -m pebby.ls20.bank --levels 5000 --split train --out data/levels-train.jsonl
uv run python -m pebby.agent.data  --bank data/levels-train.jsonl --out data/ls20-train.npz
uv run python -m pebby.agent.train --architecture looped --shards data/ls20-train.npz --checkpoint-out checkpoints/ls20-looped-policy.pt
uv run python -m pebby.agent.evaluate --checkpoint checkpoints/ls20-looped-policy.pt
uv run serve.py --port 11435 --checkpoint checkpoints/ls20-looped-policy.pt
# open http://127.0.0.1:11435/ui/index.html
```

## The game

You are a 5×5 avatar on a 12×12 lattice inside a 64×64 frame of colour indices 0–15.
You carry a **(shape, colour, rotation)** triple — 6 shapes, 4 colours, 4 rotations.

| Tile | Rule |
|---|---|
| Wall | Blocks. A blocked move still spends budget. |
| Shape / colour / rotation cycler | Entering it advances that one attribute by +1 (mod 6 or 4). |
| Goal pad | **Solid unless your triple matches it exactly.** Step on it matching and that goal clears. A rejected bump is free — it spends no budget. |
| Refill | Restores the step budget to full, and that move is free. Consumed. |
| Launcher pad | Flings you in a fixed direction to the last free cell before the next wall or goal pad. Free, and it cannot complete a level. |
| Rail-riding cycler | A cycler that walks a fixed circuit, advancing once per accepted move. |

Budget is 42 units at 1 or 2 per move. Running it out costs one of 3 lives and restarts
the level; three losses is `GAME_OVER`. Clearing every goal advances; clearing level 7 wins.
Actions are `ACTION1..4` = up/down/left/right. The whole 64×64 frame is observable —
LS20's `Camera(16, 16)` is dead config, overridden because every level declares
`grid_size=(64, 64)`. Level 7 adds fog of war, drawn into the frame.

## What is proved, and how

**The transition model is exact for the entire game.** `tools/differential.py` drives the
real engine and Pebby's model in lockstep over random action sequences and compares
cell, triple, cleared goals and remaining budget at every step:

```
L1 16918/16918   L2 8848/8848    L3 9003/9003    L4 20507/20507
L5 14399/14399   L6 22961/22961  L7 20880/20880     (12267 of these standing on a moving cycler)
```

113,516 transitions, zero mismatches. Rerun it with
`PYTHONPATH=. uv run python tools/differential.py`.

**The planner solves all seven shipped levels optimally**, and the engine confirms
each plan wins. It beats the published human median on every level:

| Level | Pebby optimal | Human baseline | Search states | Peak RAM | Mechanics |
|---|---:|---:|---:|---:|---|
| 1 | **13** | 22 | 4,763 | — | one rotation cycler |
| 2 | **45** | 123 | 7,417 | — | refills, 2 budget per move |
| 3 | **39** | 73 | 13,133 | — | colour cycler, launcher pads |
| 4 | **43** | 84 | 158,891 | 0.1 GiB | shape cycler, 8 launchers |
| 5 | **44** | 96 | 847,459 | 0.3 GiB | first rail-riding cycler |
| 6 | **72** | 192 | 12,040,252 | 4.3 GiB | two goals, all three cyclers on rails |
| 7 | **53** | 186 | 21,746,093 | 6.8 GiB | fog of war, 6 refills |

Levels 6 and 7 cost minutes and gigabytes, so nothing at request time re-derives them:
the values are cached in `pebby/ls20/shipped.py` and regenerated with
`tools/shipped_optima.py`. `Oracle`'s default limit of 600,000 states is sized for
generated levels and **truncates silently on shipped levels 6 and 7** — pass
`shipped.search_limit(index)` there.

**Every generated level is completable.** `generate_level` returns a level only after
the exact planner finds a solution *and* that solution is replayed in the real game,
which must report the level completed. A 6,600-level bank generated with a 100% success
rate, median optimal 25 actions.

## Generator

The current generation and collection fixes, proof versions, new-bank commands
and remaining coverage limits are described in
[generation-quality-v2.md](docs/generation-quality-v2.md). Existing banks and
cached arrays remain historical data until rebuilt.

`pebby/ls20/generate.py`, five difficulty tiers, deterministic in `(seed, difficulty)`:

| Tier | Attributes to change | Adds |
|---:|---:|---|
| 1 | 1 | sparse walls |
| 2 | 2 | denser walls, a decoy cycler |
| 3 | 3 | a launcher pad |
| 4 | 3 | refills, 2 budget per move |
| 5 | 3 | two goals, two launchers |

Decoy cyclers matter: without them a policy could win by touching every cycler it sees.
Levels are also required to leave at least 8 moves of budget slack, because LS20's budget
is tight enough that a level with no slack is unlearnable — one wrong move makes it
permanently unwinnable, so only a flawless policy would score at all.

Build banks in parallel (seeds are range-split so the sets cannot collide — train from 0,
validation from 1,000,000, test from 2,000,000):

```bash
uv run python -m pebby.ls20.bank --levels 5000 --split train      --out data/levels-train.jsonl
uv run python -m pebby.ls20.bank --levels 800  --split validation --out data/levels-validation.jsonl
uv run python -m pebby.ls20.bank --levels 800  --split test       --out data/levels-test.jsonl
```

## Viewer

`serve.py` plus `ui/` is a viewer for the levels the generator produces. It exists to
answer "what did the generator actually build?" by eye, on the real frame the engine
renders, rather than by reading a JSON spec.

```bash
uv run serve.py --port 11435
# open http://127.0.0.1:11435/ui/index.html
```

Every request the page makes goes through HostAI. Embedded in HostAI the page calls the
injected `window.hostai` bridge; standalone it posts the same `{model, input}` envelope to
`/hostai/infer` itself. There is no second transport, so what you see standalone is what
HostAI shows.

Type a seed and a difficulty, or step through seeds with the arrows. The page shows:

- the 64x64 frame, nearest-neighbour scaled, exactly as an ARC-AGI-3 agent receives it;
- three overlays, drawn from the level's own spec in its logical 12x12 lattice, which the
  frame does not encode in any way a reader can pick out by eye — the lattice itself,
  the features (start, goal pads, cyclers by silhouette, launchers pointing the way they
  fling, refills), and the route the stored solution walks;
- the anatomy: difficulty, seed, step budget and cost, fog, wall count, and the
  generator's own proof metadata — optimal actions, budget slack, reachable states,
  and whether the search truncated;
- a solution scrubber. Every generated level carries a solution the generator proved
  optimal and replayed to a win, so the scrubber is a proof you can watch. "Play solution"
  replays every prefix once and caches it, which is also what draws the route.

The level is also playable by hand with the arrow keys or the d-pad. The server is
stateless: the page owns the level and the action list, and each board is the server
replaying that whole list from the start, which is what makes undo exact and lets the
scrubber jump to any step.

Shipped levels load too, for comparison. They carry no stored solution here, and the page
says so rather than implying one.

## Agent

New training defaults to a **looped spatial transformer**: 239,116 parameters,
64 channels, four attention heads, and two physical blocks reused four times.
A palette-specific 5×5 stride-5 stem produces 144 cell tokens, with learned row
and column positions. A separate HUD encoder produces 16 column tokens. All
160 tokens participate in attention and the position-preserving action readout.
The HUD encoder still pools spatial detail; this is not lossless pixel storage.

Every loop reads both the evolving state and the original encoded frame.
Input recall enters the attention and feed-forward updates, and LayerNorm
normalizes each residual result. The weights and four-action readout are shared
across loops. Recurrence happens **within one decision**: the model carries no
memory between actions and does not simulate future game states. Fog and hidden
rail state remain possible information limits on shipped levels.

This is a task-specific adaptation of [input recall and outer normalization](https://arxiv.org/html/2604.15259v2),
with variable-depth training motivated by [recurrent-depth pretraining](https://arxiv.org/html/2502.05171v2).
The [September RecurTrace revision](https://arxiv.org/html/2609.03379v2) also motivates
measuring a quality-versus-depth curve before adding a learned halter. Its results
are from pretrained language models; they do not establish a benefit on LS20.
The local Tofy record contains failed controllers and later narrowly successful
binding tests. Those results neither rule out this architecture nor establish
useful extra loops here.

Training samples one depth per minibatch, uniformly from 1 through `--loops`
(default 4). The default `--loop-loss final` supervises that exit; `--loop-loss all`
uses mean action cross-entropy over every visited exit as a separate intervention.
With uniform sampled maximum depth, `all` weights early exits more heavily in
expectation. Full backpropagation through the short loop, gradient clipping,
and float32 are used. Reports separate the training objective from final-exit CE,
record depth counts/CE and clipping frequency, and validate at the fixed saved
depth. More loops cost more computation and can reduce accuracy.

```bash
# Fresh v2 labels: old v1 shards only supply one arbitrary optimal move.
uv run python -m pebby.agent.data --bank data/levels-train.jsonl --workers 2 --out data/ls20-train-v2.npz
uv run python -m pebby.agent.data --bank data/levels-validation.jsonl --workers 2 --out data/ls20-validation-v2.npz
uv run python -m pebby.agent.train --architecture looped --shards data/ls20-train-v2.npz --validation-shards data/ls20-validation-v2.npz --loops 4 --train-min-loops 1 --loop-loss final --checkpoint-out checkpoints/ls20-looped-policy.pt
# Repeat with --loops 1, 2, 4 and 8 for a paired depth comparison.
uv run python -m pebby.agent.evaluate --checkpoint checkpoints/ls20-looped-policy.pt --bank data/levels-test.jsonl --protocol strict --loops 4 --report-out checkpoints/ls20-looped-depth4.json
```

`--architecture cnn` retains the original convolutional baseline and its defaults;
`--broadcast-hud` remains a CNN-only option. CNN checkpoints retain format
`pebby.ls20-policy.v2`; looped checkpoints use `pebby.ls20-looped-policy.v1`.
Old weights load exactly, without conversion. The trainer uses a separate default
looped checkpoint filename. Serve a new checkpoint with the existing `--checkpoint`
flag; inference reads its saved loop count.

Behaviour cloning from the exact planner. The planner yields the optimal action for
*every* reachable state, not just states on the solution path, so training data covers
recovery. Off-path states come from **prefix-replay deviation**: replay k optimal
actions, take one deviating action, then follow the oracle. Naive ε-greedy does not work
here — measured on 40 levels, an oracle taking random actions with probability 0.05 still
completed 32/40, but at 0.25 only 10/40, because random detours spend budget the level
cannot spare.

### Training protocol

The official levels are **evaluation-only**. Nothing in `pebby/agent/data.py` or
`pebby/agent/train.py` can even construct a shipped-level environment;
`Ls20Env(levels=None)` appears only in `pebby/agent/evaluate.py`. Splits are separated
by seed range rather than by a flag, so they cannot overlap, and every shard records
its own provenance:

| Set | Source | Seeds | Used for |
|---|---|---|---|
| Train | generated | 0 – 4,999 | training only |
| Validation | generated | 1,000,000+ | epoch selection |
| Test | generated | 2,000,000+ | held-out completion rate |
| Shipped levels 1–7 | upstream, verbatim | — | **evaluation only** |

Verify it on any shard rather than taking this on trust:

```bash
uv run python -c "import numpy as np, json; z = np.load('data/ls20-train.0000.npz', allow_pickle=True); print(json.loads(str(z['meta'])))"
```

`bank.load()` accepts supported historical and current structural formats.
Collection and extension separately check contextual proofs; a readable bank
is not automatically current or verified training data.

### Reading the numbers

Per-step imitation accuracy is a poor headline here, for a measured reason: the oracle
often has several equally optimal moves. The earlier on-path estimate of 1.40
optimal moves, 30% ambiguous states and an 84% scalar-label ceiling was specific
to that sample; recovery-state distributions are more ambiguous. Current targets
use the complete optimal-action mask rather than one arbitrary tie. The metric
that matters is whether the oracle's distance-to-completion actually
fell by one, and above that, closed-loop completion on the held-out test bank.

Completion also compounds: it is roughly the per-step optimal-action rate raised to the
path length, and the median generated level needs 25 actions. At that length a per-step
rate of 0.90 is worth about 7% completion and 0.98 about 60%, so small per-step gains
matter far more than they look.

## Layout

```
third_party/ls20/     the verbatim upstream game, its licence and provenance
pebby/ls20/           env (drives the real game), names (de-obfuscation table),
                      layout, rails, plan (exact planner), generate, bank
pebby/agent/          model, data, train, evaluate
serve.py inference.py stateless HTTP, everything the UI asks for goes
                      through HostAI
ui/                   generated-level viewer: 64x64 frame renderer, lattice and
                      feature overlays, solution scrubber, hand play
tools/differential.py the engine-equivalence harness
```

`pebby/ls20/names.py` is worth reading first: upstream ships with randomised identifiers,
and that module is the translation table. The vendored file is never modified.

## Licence and provenance

`third_party/ls20/ls20.py` is the official `ls20-9607627b`, retrieved 2026-09-11,
SHA-256 `298c810d…`, **unmodified**. It carries an MIT header in the file itself,
© 2026 ARC Prize Foundation, reproduced in `third_party/ls20/LICENSE`. `arcengine` is
MIT from the same holder. Both notices are retained as MIT requires.

Read `third_party/ls20/PROVENANCE.md` before reusing this: the licence grant exists
**only** in the delivered file's header. The toolkit repositories, the docs site and the
PyPI packages say nothing about the licensing of downloaded game content, and
`arcprize.org/terms` is generic proprietary boilerplate with no open-source carve-out
(and names a different legal entity than the copyright holder). For commercial use,
confirm with the ARC Prize Foundation rather than relying on this note.

## Checks

```bash
uv run python -m unittest discover -s tests
PYTHONPATH=. uv run python tools/differential.py   # engine equivalence, all 7 levels
node --check ui/app.js ui/board.js
```

Browser coverage, the isolated-compositor method and screenshots are in
[docs/validation.md](docs/validation.md).

## What this is not

This is not an ARC-AGI-3 benchmark result. Pebby is trained on LS20's rules rather than
discovering them, which is the opposite of what ARC-AGI-3 measures — there is no
unknown-rule discovery, no online adaptation and no human-relative scoring here. The
planner has full knowledge of the rules and is used as a teacher, not as an agent.
Nothing here has been run against the official scorecard.
