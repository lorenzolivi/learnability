#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
# GELR Smoke Test: single seed, AdamW only, all 5 architectures
#
# Purpose: validate the current pipeline end to end before launching
#          the next multi-seed pilot.
#
# What it runs:
#   - seed 101, AdamW, baselines (const, shared, diag)
#   - seed 101, AdamW, lstm + gru
#
# Output layout:
#   results/GELR_smoke_test/baselines/seed_101/<model>/
#   results/GELR_smoke_test/lstmgru/seed_101/<model>/
#
# Usage:
#   bash smoke_test.sh
#   bash smoke_test.sh && python validate_smoke_test.py
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

SEED=101
W_SEED=12345
OUTROOT="results/GELR_smoke_test"
LOGDIR="${OUTROOT}/logs"
mkdir -p "$LOGDIR"

BASE_RUNNER="run_learnability_baselines.py"
RECURRENT_RUNNER="run_learnability_lstm_gru.py"

for RUNNER in "$BASE_RUNNER" "$RECURRENT_RUNNER"; do
  if [[ ! -f "$RUNNER" ]]; then
    echo "[error] Missing runner: ${RUNNER}"
    exit 1
  fi
done

# ── v2 parameters (same as full sweep) ──────────────────────────
COMMON="--Nseq_train 8000 --Nseq_diag 12000 --T 2048 --D 16 --H 256 \
  --epochs 750 --batch_size 512 --lr 0.001 --weight_decay 0.0001 \
  --grad_clip 1.0 --lag_min 4 --lag_max 512 --num_lags 256 \
  --task_lags 32,64,128,192,256,384,512 \
  --task_coeffs 0.6,0.5,0.4,0.32,0.26,0.2,0.16 \
  --noise_std 0.3 --eps 0.1 \
  --N_grid 25,50,100,150,200,300,400,600,800,1200,1600,2400,3200,4800,6400,9600,12800,25600 \
  --include_first_order_diag 1 --orth_init --layernorm \
  --log_gate_stats 1 --gate_log_every 10 \
  --alpha_methods ecf,mcc --alpha_n_boot 500 --min_samples_alpha 1000 \
  --device cuda"

LSTM_EXTRA="--diag_batch_size 256 --diag_log_every 10"

T0=$SECONDS

echo "═══════════════════════════════════════════════════════════"
echo "  GELR Smoke Test — $(date)"
echo "  Seed: ${SEED}, Optimizer: AdamW, Models: all 5"
echo "  Params: T=2048, H=256, lag_max=512, num_lags=256"
echo "  Output: ${OUTROOT}"
echo "═══════════════════════════════════════════════════════════"

# ── Baselines (const, shared, diag) ─────────────────────────────
echo ""
echo "[1/2] $(date +%H:%M:%S) ── AdamW baselines seed_${SEED} ──"

python "$BASE_RUNNER" \
  --outdir "${OUTROOT}/baselines/seed_${SEED}" \
  --models const,shared,diag \
  --seed "$SEED" --w_seed "$W_SEED" \
  --optimizer adamw --momentum 0.9 --const_s 0.1 \
  $COMMON \
  2>&1 | tee "${LOGDIR}/baselines_seed_${SEED}.log"

# ── LSTM / GRU ──────────────────────────────────────────────────
echo ""
echo "[2/2] $(date +%H:%M:%S) ── AdamW lstm/gru seed_${SEED} ──"

python "$RECURRENT_RUNNER" \
  --outdir "${OUTROOT}/lstmgru/seed_${SEED}" \
  --models lstm,gru \
  --seed "$SEED" --w_seed "$W_SEED" \
  --optimizer adamw --momentum 0.9 \
  $COMMON $LSTM_EXTRA \
  2>&1 | tee "${LOGDIR}/lstmgru_seed_${SEED}.log"

# ── Done ─────────────────────────────────────────────────────────
ELAPSED=$(( SECONDS - T0 ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Smoke test done — $(date)"
echo "  Wall time: ${HOURS}h ${MINS}m"
echo "  Output: ${OUTROOT}/"
echo ""
echo "  Next: python validate_smoke_test.py --root ${OUTROOT}"
echo "═══════════════════════════════════════════════════════════"
