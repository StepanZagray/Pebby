#!/usr/bin/env bash
# Repeatedly invoke the bounded joint-goal replay collector until its manifest
# reports 'complete'. The collector caps every invocation at 180s by contract,
# so a multi-hundred-level dataset needs many resumed invocations.
#
# usage: drive_joint_goal_collection.sh <bank> <split> <out> <levels-per-tier> [max-invocations]
set -euo pipefail
BANK="$1"; SPLIT="$2"; OUT="$3"; LPT="$4"; MAX_INVOCATIONS="${5:-400}"
CHECKPOINT="artifacts/seven-level-learning-v1/fresh-joint-learning-long-seed42/candidate.pt"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(cd "$(dirname "$0")/.." && pwd)"
LOG="$OUT.log"; mkdir -p "$(dirname "$LOG")"

status_of() { python -c "import json,sys;print(json.load(open('$OUT/manifest.json'))['status'])" 2>/dev/null || echo missing; }
coverage_of() { python -c "
import json;m=json.load(open('$OUT/manifest.json'));c=m.get('coverage',{})
print(c.get('complete_levels','?'),'/',c.get('requested_levels','?'),'levels,',c.get('roots','?'),'roots')" 2>/dev/null || echo "no manifest yet"; }

for ((i = 1; i <= MAX_INVOCATIONS; i++)); do
  RESUME=(--resume)
  [[ -f "$OUT/manifest.json" ]] || RESUME=()
  uv run python tools/collect_joint_goal_replay.py \
    --bank "$BANK" --split "$SPLIT" --checkpoint "$CHECKPOINT" --out "$OUT" \
    --levels-per-tier "$LPT" --max-roots 128 --max-seconds 180 --level-seconds 120 \
    "${RESUME[@]}" >>"$LOG" 2>&1 || { echo "invocation $i failed; see $LOG"; tail -20 "$LOG"; exit 1; }
  echo "[$(date +%H:%M:%S)] invocation $i: $(status_of) — $(coverage_of)"
  [[ "$(status_of)" == complete ]] && { echo "COMPLETE after $i invocations"; exit 0; }
done
echo "gave up after $MAX_INVOCATIONS invocations; status=$(status_of) $(coverage_of)"
exit 1
