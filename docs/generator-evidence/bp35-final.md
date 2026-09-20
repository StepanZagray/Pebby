# BP35 causal-lattice correction handoff

Status is `ready`. Root accepted the family after independent structural
closure, complete train/validation/test whole-game collection, and native-frame
review across all nine official/generated tier pairs.

The authoritative independent review is
`.scratch/multigame-resume/full-standard/native-astra-bp35-chambers/review.md`.
It reviewed semantic generator hash
`f4f49c20173cf0b4d92dbeddebc368a1e0ebcf9a2ddde8d316d158e138bb26f7`
and test hash
`d1ca059758738f64cfc62ab7d61badd44b9419940e35c80347b1030f57308fb1`.
The final generator hash is
`996974cdd74a988dfb4c0ff29cc678643213b94788e045cd7f9acb29ffda8c72`
and the final test hash is
`7189e9a8f990064021c15b726e5775d7d095635c433ce8ee3f1ff2c8d2c01d1d`.
Changes after the reviewed semantic hash are admission hardening: exact-type
certificate comparison, native initial entity-grid official-copy denial,
evidence/readiness metadata, and the switch-accessibility comment. Generated
geometry, teacher routes and native runtime behavior are unchanged.

The former tier-6/7 layout made its final column open, so both accepted samples
won in six public actions with no reversal or bridge closure. The replacement
uses three distinct columns and actual state dependencies:

- tier 6 descends column 4, closes and crosses a two-cell bottom bridge to
  ascent column 2, reverses, crosses the ceiling to final column 8, reverses,
  then opens eight solid final gates; its route is 28 actions;
- tier 7 uses columns 6, 2 and 8, a four-cell bottom bridge and ten solid final
  gates; its route is 36 actions;
- continuous visible barriers bound both sides of the first-column chamber on
  every world row. The ascent-side barrier has one named spike-backed bridge
  door at the bottom and one ceiling aperture; the final-side barrier has only
  the ceiling aperture. No shelf underside or under-goal corridor is open;
- a complete visible up-spike cap blocks every middle-chamber path to the
  ceiling under reversed gravity. The intended ascent remains in column 2;
  open turn cells and the last first-column gate have visible down-spikes;
- premature reversal on the first side reaches visible up-spikes. Removing the
  exact trap makes the counterfactual stop losing;
- alternate switches remain in the same bottom/top turn chambers. They no
  longer form intermediate reversal shortcuts. Remote top-bank use is possible
  at spawn, and remote bottom-bank use is possible later, but premature use
  loses under the immutable cap rather than being camera-inaccessible.

Focused current native evidence on seeds 5006/5007:

- stored/greedy-reduced routes are 28/28 and 36/36;
- `LEFT, CLICK(27,33), RIGHT x4` does not win either level;
- the observed tier-7 `RIGHT x3, CLICK(51,33) x10` initial-shelf route, the
  tier-6 26-action remote-ascent route, and the later 22/28-action underside
  routes all remain unfinished without a score;
- transfer attempts toward both ascent and final columns from the initial
  state and every seven/nine lower arrival are blocked and do not win;
- an all-row certificate checks both partitions against fresh native entities,
  the exact mutable apertures, spike-backed bottom span, middle trap, initial
  player chamber and final goal chamber. A deliberate partition hole fails;
- a native state-deduplicated search still exhausts the restricted
  LEFT/RIGHT/adjacent-solid-opening subset in 23/27 states. It excludes
  reversal, remote clicks, closure and undo and is not the structural proof;
- removing both gravity-change actions or all two/four open-to-solid bridge
  actions from the certified route does not win;
- premature reversal loses, while exact trap ablation does not;
- representative strict validators return no errors.

All nine declared tier-6 and nine tier-7 support topologies replayed through the
native engine with all five historical/current exact shortcut classes, the
all-row native chamber certificate and both-direction transfer probes. Their D4
identities include train, validation and test classes.
Across 64 deterministic raw seeds, normalized relation signatures are now
`51,59,24,44,46,9,9,23,11` for tiers 1..9. The growth-tier corrections from the
prior handoff remain unchanged.

Current bounded verification used the shared interpreter, worktree-pinned
`PYTHONPATH` and one CPU thread. Eight exact-route/certificate/immutability tests
passed in 26.84 seconds. A separate 88.77-second native script certified all 18
declared tier-6/7 support topologies: every stored route won, all exact attacks
failed, every all-row certificate matched native entities, and both transfer
directions remained blocked. Representative native clicks and ACTION7 undo left
ordinary-wall and spike barrier cells unchanged. The final representative all-tier strict
certificate plus tier-6/7 three-split checks passed 3 tests in 70.92 seconds;
no whole-game run was included.

The independent closure then replayed both representative routes, all five
historical/current attacks, remote top/bottom reversals, click/undo restoration
at every certified closure/reversal, and all initial immutable cells. It found
no counterexample. Root's current-source train, validation and test games won
all nine levels in 289/297/295 actions in 57.662/63.116/57.068 seconds. Reports
are `bp35-chambers-root-{train,validation,test}.json`. Root refreshed and viewed
all nine native official/generated frame pairs in `bp35-render-comparison.png`
with hashes recorded in `bp35-render-comparison.json`; visual review passed.

The final validator additionally rejects Python numeric type aliases recursively
through metadata, event, strategy, chamber and proof certificates. It also
stores and recomputes a D4-canonical native initial entity-grid identity and
rejects every official BP35 initial identity. Focused regressions passed without
rerunning the unchanged three-split whole-game pipeline.

The diagnostic `generate.last_rejections` now resets on `limit=0`. Reference
notes use the independently measured official within-level gravity census of
3/11 for tiers 6/7 and state that early occupied-cell counts include fixed
boundaries/hazard bands. Official live-prefix recovery remains unsupported
outside pristine pinned witnesses; generated recovery is separate.

Limits remain explicit. Constructive witnesses and structural template probes
do not prove global optimality or exhaust arbitrary coordinated histories.
Generated tiers 6/7 use two causal reversals rather than the official 3/11.
Official tier 9 remains an external demonstrated-positive witness replayed on
the pinned local engine, reference-only and without an optimality claim. No
official search, broad cohort, training, dataset collection, GPU work, or
held-out evaluation ran in this correction.
