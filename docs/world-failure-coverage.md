# World-data failure and route coverage

New collection uses `mixed_failure` in the world-data CLI and the mechanism and
extended collection wrappers. Explicit `prefix` and `mixed` calls retain their
older fixed-budget behavior. Existing archives are unchanged.

The source level must still have a complete contextual oracle and a real-engine
win with all three lives. That proof is necessary. Failure collection starts a
separate clone of that verified initial state, performs real budget-consuming
actions, and retains each pre-life-loss state across the three lives, including
the actual game-over branch. Lives and budgets are never edited. A bounded
trajectory can fail to reach death on a pathological free-action/refill trap;
`failure_stop`, action traces, and measured death/life-loss counts expose that
case. Coverage is not inferred from the source proof.

Unreachable current states carry `optimal=0`. The loader accepts that only when
every successor is unreachable or loses a life. Policy loss and policy accuracy
exclude these rows and report `policy_valid_fraction`; dynamics, terminal, win,
distance and state targets remain supervised. Surviving life-loss successors
keep their real reset-state distance and successor optimal mask. The on-policy
collector also continues through unreachable states until the action limit,
an exact attractor, a win or game over.

Expert anchors use a per-level seeded optimal route. The on-policy collector
selects at least start/middle/end for routes of three or more actions and at
least `ceil(route_length / 4)` anchors, including the final pre-win state.
Exact previously collected states may consume those anchors through deduplication.
`mixed_failure` uses that same route-dependent floor in addition to its prefix.
Its actual rows may therefore exceed `samples_per_level`, which is a base
budget; per-level sample counts and expert indices record the actual result.

On-policy archives distinguish `on_policy_rows` from `auxiliary_rows` containing
expert anchors and exhaustion samples. Merge provenance binds both index sets
to source archives. Training preserves the requested model-visited batch
fraction and, by default, replaces 25% of otherwise-base draws for eligible
levels with an auxiliary row. This probability is configurable with
`--on-policy-auxiliary-fraction`; it is not a guaranteed fraction of the entire
batch. Actual auxiliary sample counts are reported. Older archives without
auxiliary metadata retain the previous sampling behavior. Sequence training
keeps auxiliary rows as four ordinary counterfactual branches; only indexed
sequence anchors receive chronological labels. Unreachable future states are
excluded from sequences that reconstruct exact current distance from a
nonempty optimal mask; they remain ordinary transition examples.

The actual-successor value loss retains state-distance semantics. The imagined,
action-conditioned value loss additionally maps immediate life loss to its
existing final unsafe bin. This preserves checkpoint parameter shapes. The
current architecture still shares a value head and pulls imagined latents
toward actual successor latents; immediate loss and the resulting solvable
reset state can therefore compete. These changes supply missing supervision,
but they do not establish improved safety recall or gameplay without a new
training/evaluation run.

Current world-training objectives live in `world_training_objectives.py`.
`world_model.py` retains the exact historical bytes required by existing frozen
encoder checkpoints, including its historical loss API. Production trainers and
scorers import the new objective module; archived experiment drafts keep their
original behavior.

Diagnostics separate actual unreachable targets, imagined unsafe targets,
life-loss event prevalence and terminal-death prevalence. Recall/specificity
and policy metrics aggregate using their eligible counts. A zero unsafe-target
fraction means recall has no positive support. Existing default checkpoints
still saturate reachable distance at 63; `distance_overflow_fraction` exposes
longer targets rather than silently claiming exact distance prediction.

Regression coverage includes independent real-engine death/history replay,
saved-array loading and backward propagation, a real 53-action route with
middle anchors, separate actual/imagined value targets, and death rows passing
through the production curriculum/sequence batch path into the loss.

The active structured pipeline is also covered. Its workspace collectors reuse
the same real-engine collection, and both on-policy field-cache formats retain
failure rows, causal reset histories, event labels and explicit unreachable
current distances. Current distance `-1` is accepted only when all complete
counterfactual labels are unreachable or lose a life. Neither a missing action
mask alone nor a reset successor is treated as proof of unreachable state.

The structured on-policy dynamics trainer already has separate life-loss,
terminal and win event heads. Bound auxiliary rows now enter its paired
same-level treatment sampling, with `--auxiliary-fraction` defaulting to a 25%
chance per treatment replacement. Reports distinguish actual policy visits
from auxiliary branches and count their sampled events. This structured
experiment's fraction is a share of treatment replacements; the world-model
sampler above instead preserves its reserved policy quota. The workspace actor
excludes zero-optimal rows from its policy objective and reports the valid
fraction, while those rows still train the dynamics/event objective. Regression
tests build both cache formats from actual deaths and verify event-head
gradients and zero actor loss on an unreachable row. Existing large caches and
checkpoints are not rebuilt by these source changes.
