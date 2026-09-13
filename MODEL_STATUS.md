# Current model status

Verified 13 September 2026. Seven-level completion remains unachieved. The retained
checkpoint is `artifacts/spatial-recovery-v1/quality-fit/recovery.pt`, SHA256
`ff88327214b6dc2d4278167e0d61edcc37c683292b788be6927b71286331a5f8`.

Under deterministic public-history inference it completes shipped level 1 in 23
actions, then fails level 2. Separate fresh-start evaluations of shipped levels
2–7 clear no goals and exhaust all three lives. On the reused generated development
panel it wins 7/70, all in difficulty 1. The v3 control and route-readout candidates win 6/70 and 7/70;
neither is promoted. V4's disposable qualifications pass tiny memorization but
fail the predeclared added-feature reliance test; no full v4 training or candidate
promotion has occurred. These are local measurements, not an official scorecard.

The model receives the last 8 public frames. Its encoder performs 6 refinement
passes, but its action-conditioned predictor is only one action ahead. Persistent
game memory, learned voluntary RESET, and multi-step latent planning remain
unimplemented. A frozen pixel perceptor and direct actor exist as experimental
components, without demonstrated overall gameplay improvement.

V5 tested four matched training variants at two training seeds. Their seed-42
development results were 6/70, 7/70, 7/70 and 7/70, with no difficulty-2–7 wins.
Joint training with uniform action targets cleared shipped level 2 in isolation,
but lost level 1; its total remained 1/7. No candidate was promoted. Seed-43
replication covers cached decisions, not another gameplay sweep.

A subsequent basic-navigation diagnostic found only 30/128 correct first moves
toward an already-matching goal in an empty room; the semantic candidate scored
31/128. All 128 teacher paths won in the real engine. The retained model eventually
won 11/16 selected closed-loop cases, often taking long detours. These deliberately
simplified rooms differ from its complex training distribution, so this is evidence
of a basic transfer failure, not a new benchmark score or an architecture ceiling.
Position and carried-glyph decoding were correct on every case; the additional
goal perceptor also decoded every goal correctly and is wired into the candidate.
Detailed hypotheses, negative results and component checks are recorded in
`artifacts/spatial-repair-v5/REPORT.md`.

The opt-in generator version 4 adds goals that restore some initial glyph
components, disappearing goal rings, and a distinct test partition. Existing
banks and model weights have not been regenerated with these mechanics.

Runtime loading now accepts the spatial, route, and semantic checkpoint formats.
Select the retained checkpoint explicitly; `checkpoints/ls20-policy.pt` is not
automatically populated or pointed at an experimental candidate:

```bash
uv run serve.py --port 11435 \
  --checkpoint artifacts/spatial-recovery-v1/quality-fit/recovery.pt
```

The default missing-checkpoint behavior remains useful on machines without local
training artifacts. The numerical evaluator accepts the same explicit checkpoint.
Detailed local evidence is in `artifacts/spatial-repair-v3/REPORT.md` and
`artifacts/spatial-repair-v4/`; artifacts and checkpoint weights are not committed.
