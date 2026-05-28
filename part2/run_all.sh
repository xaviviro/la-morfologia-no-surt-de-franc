#!/usr/bin/env bash
# Orchestrate Part 2 end-to-end: corpus prep -> 10 trainings (model x {A,B})
# -> training curves -> evaluation. Continues past per-run failures (e.g. OOM)
# and records them. Designed to run inside tmux so an SSH drop doesn't kill it.
#
#   export HF_TOKEN=hf_...      # gated downloads + PRIVATE weight upload
#   tmux new -s p2 'bash part2/run_all.sh'
set -u
cd "$(dirname "$0")/.."
export UV_LINK_MODE=copy
mkdir -p part2/out
LOG=part2/out/run_all.log
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

# Model list straight from config.py (single source of truth).
mapfile -t MODELS < <(uv run python -c "from part2.config import MODELS; print('\n'.join(MODELS))")

log "=== PART 2 START === (${#MODELS[@]} models)"
log "torch: $(uv run python -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>/dev/null)"
[ -n "${HF_TOKEN:-}" ] && log "HF_TOKEN set — weights will be pushed PRIVATE" || log "HF_TOKEN NOT set — no upload"

# Per model: prepare corpus, then train both conditions (so the first trained
# weights upload early and any failure surfaces on model 1, not after all p01).
for M in "${MODELS[@]}"; do
  log ">>> p01 corpus $M"
  if uv run python part2/p01_prepare_corpus.py --model "$M" >>"$LOG" 2>&1; then
    log "p01 OK   $M"
  else
    log "p01 FAIL $M (skipping its trainings)"
    continue
  fi
  for C in A B; do
    log ">>> p02 train $M $C"
    if uv run python part2/p02_train.py --model "$M" --condition "$C" >>"$LOG" 2>&1; then
      log "p02 OK   $M $C"
    else
      log "p02 FAIL $M $C"
    fi
  done
done

# 3) training diagnostics (LR / loss / grad-norm / perplexity curves, A vs B)
log ">>> p04 curves"
uv run python part2/p04_train_curves.py >>"$LOG" 2>&1 && log "p04 OK" || log "p04 FAIL"

# 4) evaluate base vs A vs B (each under its own segmentation) + B>A sign test
log ">>> p03 evaluate"
uv run python part2/p03_evaluate.py >>"$LOG" 2>&1 && log "p03 OK" || log "p03 FAIL"

log "=== PART 2 DONE ==="
