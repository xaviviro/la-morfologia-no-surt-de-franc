#!/usr/bin/env bash
# Finalise Part 2 after the main run: redo gemma-4-E2B (failed with adamw_bnb_8bit;
# now uses Adafactor — wired in p02_train.py), then regenerate curves + evaluation
# so the final results include all 5 models.
set -u
cd "$(dirname "$0")/.."
export UV_LINK_MODE=copy
LOG=part2/out/final.log
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "=== FINAL START ==="

# Drop any empty/half-built E2B output dirs from the earlier failure
rm -rf part2/out/google-gemma-4-e2b_A part2/out/google-gemma-4-e2b_B

for C in A B; do
  log ">>> p02 E2B $C (adafactor)"
  if uv run python part2/p02_train.py --model google/gemma-4-E2B --condition "$C" >>"$LOG" 2>&1; then
    log "p02 OK   google/gemma-4-E2B $C"
  else
    log "p02 FAIL google/gemma-4-E2B $C"
  fi
done

log ">>> p04 curves"
uv run python part2/p04_train_curves.py >>"$LOG" 2>&1 && log "p04 OK" || log "p04 FAIL"

log ">>> p03 evaluate"
uv run python part2/p03_evaluate.py >>"$LOG" 2>&1 && log "p03 OK" || log "p03 FAIL"

log "=== FINAL DONE ==="
