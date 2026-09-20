# Neural planning experiment

The user selected neural action selection first, followed by explicit search
using the same learned perception and dynamics. Seven-level completion is still
the target; no architecture or unit test is a completion result.

`NeuralImagination` predicts four trajectories, one for each possible first move.
A learned continuation policy chooses following moves. Each predicted field
becomes the input to the next application of the same learned dynamics. A reverse
GRU summarizes each trajectory, and a shared neural comparator scores first moves.
This follows the broad idea of learning to interpret imagined consequences in
[Imagination-Augmented Agents](https://proceedings.neurips.cc/paper/2017/hash/9e82757e9a1c12cb710ad680db11f6f1-Abstract.html).
It is an experimental adaptation, not a reproduction of that paper's results.

The continuation policy uses hard neural action choices, so imagined fields are
not convex averages of incompatible futures. Its discrete choices receive no
gradient from the final action loss. It learns separately from generated current
and actual-successor optimal-action labels. Actual successors never enter the
imagined trajectory scorer. The trajectory GRU is memory within one decision;
it does not provide persistent memory across game observations.

The first controlled experiment freezes the existing public field encoder and
`ls20-factored-local-h4-400.pt` dynamics. Its reported H4 accuracy on 128 generated
live sequences is 92.19% for player position, 70.31% for the carried symbol, and
75% for remaining steps. This is substantially better than earlier structured
dynamics. However, its sequence training contains no resets or terminal failures,
and held-out live sequences cannot measure positive ending detection. The same
dynamics family previously failed a small explicit-search gameplay pilot. It is
an initialization for an experiment, not qualified full-game dynamics.

`tools/train_neural_imagination.py` first trains one continuation policy, then
reuses its exact frozen weights for K1 and K4. Both neural selectors start with
identical weights, see identical sampled training rows, and share unchanged
dynamics. Validation selection spans every difficulty in the legacy generated
field bank. This isolates the effect of additional imagined steps with the same
parameter count, while reporting different inference computation.

The same-depth diagnostic repeats H1 trajectory evidence in place of H2–H4,
preserving action traces and recurrent computation. Useful future reliance needs
better held-out decisions/gameplay with native futures than with this intervention;
different logits alone are insufficient. Recorded imagined action traces support
subsequent replay on generated engine clones to measure prediction fidelity.

Qualification checks finite optimization, unchanged frozen dynamics, matching
initialization/continuation weights, and exact checkpoint reload. It does not
measure learning convergence. A trained candidate must then run the public-pixel
sequential evaluator with the same native lives and 300 charged actions per level
as the retained 1/7 baseline. No privileged state enters that controller.

Policy-induced trajectory collection and failure/reset supervision are now
implemented, with qualification and actual fit results tracked in the experiment
artifacts. Reliable predictions through these events, persistent observation
memory, and the later explicit-search comparison remain outstanding. The current neural module predicts fixed-length
tails even after a possible terminal event; those tails are unqualified, not
asserted to represent valid post-terminal game states.

The completed repair experiment trained dynamics before fitting fresh neural selectors.
`collect_neural_planning_sequences` records four model-selected branches from
actual public histories, keeps life-loss/reset transitions, and masks every
transition after a true terminal state. Generated teacher routes can place the
collector near winning endings; the teacher suffix never enters the model's
imagined action selection. Such teacher behavior is used for generated data only.

`train_neural_planning_dynamics` mixes those K4 sequences with optional broad H1
replay. Only the initial observed field enters a K4 rollout; later inputs are its
own predictions. Exact physical/event labels and observed future fields are
loss targets only. In addition to whole-field error, changed-cell supervision
gives small board updates meaningful weight. Its masks use consecutive actual
encoder outputs, so they are learned-appearance changes rather than exact
semantic goal/refill labels. Validation reports event supports, grouped levels,
and post-loss fidelity; positive-weighted event logits are not calibrated odds.

`train_neural_imagination --dynamics-parent ... --dynamics-sha256 ...` creates
fresh matched K1/K4 selectors over the frozen repaired dynamics. A separate
checkpoint format binds that dynamics artifact and the unchanged encoder.
Selector validation now separates optimal-versus-suboptimal ordering from
bad-versus-bad comparisons and records safe selected-action regret separately
from unreachable/life-loss selections. Actual gameplay remains the deciding
measure; offline ranking gains alone cannot qualify a controller.

## Gameplay-first testing and checkpoint selection

Actual sequential LS20 gameplay is the primary admission test. Use one game,
three native lives, strict neural argmax, GAME_OVER-only current-level RESETs,
and300 charged actions per level. Compare against the retained checkpoint under
the same protocol. A lower sequential completion count rejects the candidate;
equal progress cannot be promoted by better cached-state accuracy, loss, or
pair ordering. A gain permits further matched gameplay checks. Never combine
isolated wins into a claimed seven-level sequence.

The planning trainer runs this test automatically for the baseline and each
saved candidate. Cached-state measurements are opt-in via
`--offline-diagnostics`, run after gameplay, and cannot select a checkpoint.
Supporting generated-game comparisons use actual closed-loop engine episodes;
those are gameplay measurements, unlike scoring cached teacher states. Reused
shipped levels and generated development panels are development evidence, not
untouched generalization tests.

The basic and world-policy trainers now default to
`--select-on gameplay`; the old offline selection choices are rejected. Explicit
`last` retains an unpromoted final artifact. For world-model/perception warmups,
that explicit mode may skip gameplay and must say it was not evaluated. New
candidates are saved separately and cannot overwrite an existing checkpoint.
Training losses such as action cross-entropy remain useful optimization signals:
changing the evaluation criterion does not require discarding those losses.

A small training qualification checks data flow, finite gradients, saved weights
and evaluator wiring. It must never be reported as a trained performance gain.
The completed gameplay-first qualification reproduced the retained1/7 baseline
and rejected both deliberately one-update0/7 candidates without calculating
cached-state validation accuracy.

The fresh joint model is implemented in `pebby/agent/joint_goal_planning.py`.
Its encoder, dynamics, continuation policy and action scorer start randomly
and train together; no component is deliberately frozen. Exact visible goal
attributes, solved-goal status, physical consequences and successor pixels are
training targets. Only the public H8 observation tuple enters deployed inference.
The completed small H1 qualification learned all eight generated training games,
but scored 0/7 shipped and 0/70 on the reused generated gameplay panel. The
retained checkpoint reproduced 1/7 and 7/70. This establishes a transfer failure,
not a successful replacement or a capacity ceiling. See
`artifacts/seven-level-learning-v1/fresh-joint-results.md`.

The new `joint_goal_sequence_objective.py` adds exact H1–H4 supervision to that
fresh model. Each next predicted field feeds the following dynamics step; actual
future pixels never replace it. Physical, semantic, event, value and continuation
targets supervise valid steps, including life loss; terminal tails are excluded.
This has passed CPU chronology, masking and gradient tests. It has not yet
demonstrated a gameplay improvement.

`collect_joint_goal_replay` and `train_joint_goal_replay` extend this work to
versioned generated data spanning all seven tiers. The initial collector is a
bounded pilot with explicit source bindings and real engine verification, not a
complete replacement for the full training bank. The trainer samples TRAIN data
only and ranks snapshots by shipped sequential completion first, then separate
generated validation wins, retaining the earliest full tie. No loss or cached
accuracy can replace that selection rule. Broad collection and training results
must be checked in their completed manifests/reports before claiming coverage
or improvement; implementation and a successful backward pass alone do not
establish either.
