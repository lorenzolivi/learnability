#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
# Smoke Test: single seed, AdamW only, all 5 architectures
#
# Purpose: validate the current pipeline end to end with the same
#          task / statistical setup as launch_learnability.sh, but
#          shrunk so it finishes on a laptop CPU overnight (or on a
#          DGX GPU in minutes).
#
# Key principle: the task geometry (task_lags, task_coeffs, noise,
# layernorm / orth_init, K projections, LSTM/GRU init, α estimation
# settings) is IDENTICAL to launch_learnability.sh.  Only the scale
# knobs that dominate wall time are reduced (Nseq, T, H, epochs,
# num_lags, batch sizes, N_grid density, task lags capped at the
# smaller T).
#
# What it runs:
#   - seed 101, AdamW, baselines (const, shared, diag)
#   - seed 101, AdamW, lstm + gru
#
# Output layout:
#   results/smoke_test/baselines/seed_101/<model>/
#   results/smoke_test/lstmgru/seed_101/<model>/
#
# Device:
#   Auto-detects CUDA via torch.cuda.is_available(); otherwise falls
#   back to CPU.  MPS is *not* used (incomplete torch.func.jvp support
#   for recurrent kernels on Apple Silicon).  Override with DEVICE=…
#     DEVICE=cpu  bash smoke_test.sh
#     DEVICE=cuda bash smoke_test.sh
#
# Usage:
#   bash smoke_test.sh
#   bash smoke_test.sh && python validate_smoke_test.py --root results/smoke_test
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

SEED="${SEED:-101}"
W_SEED="${W_SEED:-12345}"
OUTROOT="${OUTROOT:-results/smoke_test}"
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

# ── Device auto-detect (CUDA if available, else CPU; never MPS) ─────
DEVICE="${DEVICE:-}"
if [[ -z "$DEVICE" ]]; then
  if python - <<'PY' 2>/dev/null
import sys, torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi
echo "[info] device: ${DEVICE}"

# ── Shared arguments ────────────────────────────────────────────────
# Structurally mirrors launch_learnability.sh COMMON but scaled down:
#   Nseq_train  8000 -> 1500
#   Nseq_diag  12000 -> 2000
#   T           1536 -> 512      (task_lags rescaled accordingly)
#   H            256 -> 128
#   epochs      1500 -> 200
#   batch_size   384 -> 128
#   num_lags     192 -> 64       (diagnostic grid density)
#   lag_max      768 -> 256
#   N_grid     54 pts ->  9 pts  (coarser sweep, same range shape)
# All task/statistical knobs (task_coeffs, noise_std, noise_tolerance,
# include_first_order_diag, orth_init, layernorm, alpha_methods,
# alpha_n_boot, num_projections=50, const_s=0.05) match the paper run.
COMMON="--Nseq_train 1500 --Nseq_diag 2000 --T 512 --D 16 --H 128 \
  --epochs 200 --batch_size 128 --lr 0.0002 --weight_decay 0.0001 \
  --grad_clip 1.0 --lag_min 4 --lag_max 256 --num_lags 64 \
  --task_lags 32,64,96,128,192,256 \
  --task_coeffs 0.60,0.50,0.40,0.30,0.22,0.16 \
  --noise_std 0.4 --noise_tolerance 0.05 \
  --N_grid 25,50,100,200,400,800,1200,1500 \
  --include_first_order_diag 1 --orth_init --layernorm \
  --log_gate_stats 1 --gate_log_every 10 \
  --alpha_methods ecf,mcc --alpha_n_boot 500 --min_samples_alpha 500 \
  --num_projections 50 \
  --device ${DEVICE}"

# diag_batch_size is much smaller on CPU; on CUDA it can be larger.
if [[ "$DEVICE" == "cuda" ]]; then
  LSTM_EXTRA="--diag_batch_size 256 --diag_log_every 10 --gru_init_update 0.05 --lstm_init_forget 0.5"
else
  LSTM_EXTRA="--diag_batch_size 64 --diag_log_every 10 --gru_init_update 0.05 --lstm_init_forget 0.5"
fi

T0=$SECONDS

echo "═══════════════════════════════════════════════════════════"
echo "  Smoke Test — $(date)"
echo "  Device: ${DEVICE}"
echo "  Seed: ${SEED} (w_seed base=${W_SEED}, 50 projections)"
echo "  Optimizer: AdamW, Models: const,shared,diag,lstm,gru"
echo "  Params: T=512, H=128, lag_max=256, num_lags=64"
echo "  Scale: Nseq_train=1500, Nseq_diag=2000, epochs=200, batch=128"
echo "  Projection aggregation: K=50 directions per w_seed base"
echo "  LSTM init: lstm_init_forget=0.5, gru_init_update=0.05"
echo "  Output: ${OUTROOT}"
echo "═══════════════════════════════════════════════════════════"

# ── Baselines (const, shared, diag) ─────────────────────────────
echo ""
echo "[1/2] $(date +%H:%M:%S) ── AdamW baselines seed_${SEED} ──"

python "$BASE_RUNNER" \
  --outdir "${OUTROOT}/baselines/seed_${SEED}" \
  --models const,shared,diag \
  --seed "$SEED" --w_seed "$W_SEED" \
  --optimizer adamw --momentum 0.9 --const_s 0.05 \
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
echo "  Device: ${DEVICE}"
echo "  Wall time: ${HOURS}h ${MINS}m"
echo "  Output: ${OUTROOT}/"
echo ""
echo "  Next: python validate_smoke_test.py --root ${OUTROOT}"
echo "═══════════════════════════════════════════════════════════"
