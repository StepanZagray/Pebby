# Multi-game training after the audit

The implementation defects found in the September 20 audit have been repaired.
The frozen Run B weights are unchanged. Improved gameplay, clicks, history
robustness and dynamics **still need to be demonstrated by new training**.
The policy is recurrent imitation; its auxiliary predictor is not a planner.

## Start, stop, resume

From the repository root, with the existing `.venv`:

```bash
bash tools/run_multigame_training.sh start
bash tools/run_multigame_training.sh status
bash tools/run_multigame_training.sh stop
bash tools/run_multigame_training.sh resume
```

Training runs in the background on CUDA, in `artifacts/multigame-v3/run-C`.
Read its log with:

```bash
tail -f artifacts/multigame-v3/.run-C.launcher/training.log
```

The default is a **new experimental fine-tune**, initialized from Run B's weights,
with a fresh optimizer, 30 total epochs, learning rate 0.0001, whole-game updates,
canonical inputs and history dropout 0.5. Changed pixels receive 5 times the
ordinary pixel loss; positive events receive 3 times the ordinary event loss.
These are explicit experimental multipliers, not proven optimal settings.
Global auxiliary sampling removes the short-tail bias. To isolate that sampler
fix alone, invoke `tools/manage_multigame_training.py start` with the desired
trainer options and leave both new multipliers at 1.

**The default corpus is still the existing Run B corpus:** it has no learner-policy
transitions, only 65.4% stored click-region coverage, almost no undo targets, and
BP35 validation targets only on its first level. The start command prints this
limitation. It tests the new objective; it does not repair missing examples.
Use the data workflow below and override the manifests to train on repaired data.

`stop` validates the PID, process start time, boot identity and command before
sending SIGTERM. It waits up to 60 seconds without force killing. The trainer
finishes its current training game, saves its optimizer, RNG, epoch order,
cursor and accumulated statistics, then exits. During validation it stops at an
offline chunk or closed-loop game boundary and reruns that unfinished evaluation
after resume. If status says `stopping`, wait for `stopped` before resuming.

Checkpoints are written initially, every 25 completed games, before validation,
after each epoch and on graceful stop. A hard kill or power loss can lose work
since the last committed checkpoint. `latest.pt` and `best.pt` refer through one
atomic generation pointer, preventing mismatched promotion pairs. Before the
first completed selection panel, there is no selected `best.pt`; `latest.pt`
already supports resume. Preserve the whole run directory, including hidden
checkpoint generations and symlinks, when moving or backing up a run.
The initial dataset audit happens before fitting and before the first checkpoint.
Stopping during that startup phase can exit without a checkpoint; inspect status
and the log before using `resume`. The launcher checks for immediate startup
failures, but later validation errors must be read from the log.

Resume restores manifests and the full saved recipe. `EPOCHS` means the total
target, not additional epochs:

```bash
EPOCHS=60 bash tools/run_multigame_training.sh resume
RUN_DIR=artifacts/multigame-v3/run-D DEVICE=cpu EPOCHS=1 \
  bash tools/run_multigame_training.sh start
```

Use the same `RUN_DIR` for subsequent commands. A fresh start refuses a nonempty
run directory. Old Run B checkpoints deliberately cannot be exact-resumed under
the changed objective; `--initialize-from` starts a new, separately identified
experiment instead. For foreground training, call `tools/train_multigame.py`
directly; Ctrl-C uses the same graceful stop path. Saved-config resume is:

```bash
.venv/bin/python tools/train_multigame.py \
  --resume artifacts/multigame-v3/run-C/latest.pt --epochs 30
```

## Fill and verify the data gaps

Backfill legacy exact-only click labels into a **new** corpus, in bounded batches:

```bash
.venv/bin/python tools/backfill_multigame_click_regions.py \
  --manifest data/multigame-v2/merged-all/train/manifest.json \
  --output-root data/multigame-v3/region-backfill/train \
  --max-games 2 --probe-limit 64 --max-steps-per-game 4096 --game-seconds 60
.venv/bin/python tools/backfill_multigame_click_regions.py \
  --manifest data/multigame-v2/merged-all/validation/manifest.json \
  --output-root data/multigame-v3/region-backfill/validation \
  --max-games 2 --probe-limit 64 --max-steps-per-game 4096 --game-seconds 60
```

Repeat each command to continue its split. Exit code 1 means valid partial
progress, 0 means complete, and 2 means an error or a bound was exceeded. Do not
train a full experiment on partial backfill output. You may increase game/time/
step budgets when resuming; source hashes and the probe limit remain bound.
A single game is the commit unit. A timeout or interruption commits no partial
labels for that game; previously committed games survive. Existing stored
regions are copied unchanged and clearly marked as not newly reverified.
New labels require recorded frame/action/outcome replay agreement and cloned
engine probes; they are bounded one-step equivalence, not exhaustive future
equivalence. The original manifests and arrays are never overwritten.

Once both splits are complete, substitute their manifests for the old corpus
in the merge command below. This fixes label coverage; it does not create new
learner trajectories or later-level validation examples.

Collect bounded learner perturbations using the current checkpoint on generated
training families. This performs real generation/search and can be expensive;
none was launched as part of the code repair. Search and rollout time caps keep
individual attempts bounded, but do not guarantee complete family coverage.

```bash
.venv/bin/python tools/prepare_multigame_dataset.py \
  --output-root data/multigame-v3/learner-recovery \
  --train-master-seed 2026092001 --validation-master-seed 2026092002 \
  --completed-teacher-games-per-family 1 --mixed-games-per-family 2 \
  --mixed-epsilon 0.3 --perturbation learner \
  --learner-checkpoint artifacts/multigame-v2/frozen-v2arch-B.pt \
  --recovery-seconds 20 --game-seconds 300 --click-region-probe-limit 64
```

No official or held-out games enter this collection. The canonical learner now
correctly inverts any stored observation/control variants before inference and
maps its proposal back to the public controls. Its checkpoint history mode is
honored, and the checkpoint hash is recorded. Teacher and learner/random route
counts are reconstructed from the actual arrays when merging legacy records.

Merge successful new cohorts with the existing corpus into a new destination:

```bash
.venv/bin/python tools/merge_multigame_manifests.py \
  --train-manifest data/multigame-v2/merged-all/train/manifest.json \
    data/multigame-v3/learner-recovery/train-teacher/manifest.json \
    data/multigame-v3/learner-recovery/train-mixed/manifest.json \
  --validation-manifest data/multigame-v2/merged-all/validation/manifest.json \
    data/multigame-v3/learner-recovery/validation-teacher/manifest.json \
    data/multigame-v3/learner-recovery/validation-mixed/manifest.json \
  --out-root data/multigame-v3/merged
```

The merge rejects split collisions instead of silently accepting leakage. Inspect
its report and coverage before training. New collection seeds alone do not prove
new puzzle identities, and additional partial trajectories may still leave gaps.

Check coverage before claiming that collection solved a gap:

```bash
.venv/bin/python tools/audit_multigame_coverage.py \
  --manifest data/multigame-v2/merged-all/train/manifest.json \
  --min-click-region-coverage 1 --require-level-coverage \
  --min-learner-steps-per-family 1
.venv/bin/python tools/audit_multigame_coverage.py \
  --manifest data/multigame-v2/merged-all/validation/manifest.json \
  --require-level-coverage
```

Those commands intentionally fail on the old corpus. Point them at newly prepared
or merged manifests to check repairs. `--min-targets-per-level N` strengthens the
level gate, and `--min-undo-targets-per-family N` exposes missing undo supervision.
The latter is appropriate only when that family actually supports meaningful undo;
legal availability alone does not demonstrate that undo was taught. Missing undo
examples and BP35 later-level validation need actual reached, teacher-labelled
states; a zero count must not be repaired by inventing labels.

The trainer also accepts `--min-click-region-coverage FRACTION` as a startup
guard on the overall training click fraction. Its older `--require-click-regions`
flag only requires at least one stored region and is retained for compatibility.
Use the audit's per-family threshold when a global average could hide a weak family.

To use different manifests with the starter:

```bash
TRAIN_MANIFEST=data/multigame-v3/merged/train/manifest.json \
VALIDATION_MANIFEST=data/multigame-v3/merged/validation/manifest.json \
RUN_DIR=artifacts/multigame-v3/run-with-recovery \
  bash tools/run_multigame_training.sh start
```

## Evaluate the resulting model honestly

Training logs now sample auxiliary metrics across entire games instead of their
prefixes. They report changed-pixel accuracy, the copy-current-frame baseline,
exact-frame accuracy, and per-event positive precision/recall/support. Click
metrics distinguish stored regions from exact fallback, and action-ID accuracy
is separate from joint action-and-click correctness. History-free diagnostics
remain optional; ordinary evaluation honors a checkpoint trained without action
history. These diagnostics measure sampled transitions, not proven planning ability.

Generate a report tied to the exact selected weights:

```bash
.venv/bin/python tools/summarize_multigame_checkpoint.py \
  artifacts/multigame-v3/run-C/best.pt
```

The baseline checkpoint's verified results are in [MODEL_STATUS.md](../MODEL_STATUS.md).
Historical peaks can belong to different weights. Compare new selected weights
on the same development panels first; use a predeclared larger, unseen generated
confirmation panel before promotion. Repeatedly inspected validation records
cannot become an unseen confirmation set. Official and m0r0 results have not been
rerun or used to tune these changes.
