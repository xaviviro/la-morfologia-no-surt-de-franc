#!/usr/bin/env bash
# Combined sweep: 5 models × {A,B} × multiple training seeds (TRAIN_SEEDS in config).
# Each model completes all its seeds before moving on, so partial results appear early.
# Continues past per-run failures (e.g. OOM); designed for tmux/nohup so an SSH drop
# doesn't kill it.
set -u
cd "$(dirname "$0")/.."
export UV_LINK_MODE=copy
mkdir -p part2/out
LOG=part2/out/combined.log
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

mapfile -t MODELS < <(uv run python -c "from part2.config import MODELS; print('\n'.join(MODELS))")
mapfile -t SEEDS  < <(uv run python -c "from part2.config import TRAIN_SEEDS; print('\n'.join(str(s) for s in TRAIN_SEEDS))")
TOTAL=$(( ${#MODELS[@]} * ${#SEEDS[@]} * 2 ))

log "=== COMBINED RUN START === (${#MODELS[@]} models × ${#SEEDS[@]} seeds × 2 conds = $TOTAL runs)"
log "torch: $(uv run python -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>/dev/null)"
[ -n "${HF_TOKEN:-}" ] && log "HF_TOKEN set — weights will be pushed PRIVATE" || log "HF_TOKEN NOT set — no upload"

# Per model: prepare corpus once (shared across seeds — the data-shuffle seed in
# config is fixed), then sweep all (seed, condition) pairs. Done means corpus + 2|SEEDS|
# trained checkpoints pushed.
for M in "${MODELS[@]}"; do
  log ">>> p01 corpus $M"
  if uv run python part2/p01_prepare_corpus.py --model "$M" >>"$LOG" 2>&1; then
    log "p01 OK   $M"
  else
    log "p01 FAIL $M (skipping its trainings)"
    continue
  fi
  for SEED in "${SEEDS[@]}"; do
    for C in A B; do
      log ">>> p02 train $M $C seed=$SEED"
      if uv run python part2/p02_train.py --model "$M" --condition "$C" --seed "$SEED" >>"$LOG" 2>&1; then
        log "p02 OK   $M $C s$SEED"
      else
        log "p02 FAIL $M $C s$SEED"
      fi
    done
  done
done

log ">>> p04 curves (multi-seed)"
uv run python part2/p04_train_curves.py >>"$LOG" 2>&1 && log "p04 OK" || log "p04 FAIL"

log ">>> p03 evaluate (multi-seed)"
uv run python part2/p03_evaluate.py >>"$LOG" 2>&1 && log "p03 OK" || log "p03 FAIL"

log "=== COMBINED DONE ==="
