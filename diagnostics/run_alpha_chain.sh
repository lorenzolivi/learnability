#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEEDS_CSV="${1:-1}"
DEVICE="${2:-cuda}"
OUTROOT="${3:-diagnostics/alpha_chain_pilot/adamw}"
LAGS="${4:-4,64,256,512}"
W_SEED=12345
VAL_W_SEEDS="${5:-$W_SEED}"
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

COMMON_ARGS=(
  --Nseq_train 8000
  --Nseq_diag 12000
  --T 1536
  --D 16
  --H 256
  --epochs 1000
  --batch_size 512
  --lr 0.00025
  --weight_decay 0.0001
  --grad_clip 1.0
  --lag_min 4
  --lag_max 768
  --num_lags 192
  --task_lags 64,128,256,384,512,768
  --task_coeffs 0.60,0.50,0.40,0.30,0.22,0.16
  --noise_std 0.4
  --noise_tolerance 0.05
  --N_grid 50,100,150,200,300,400,600,800,1200,1600,2400,3200,4800,6400,9600,12800,19200,25600
  --include_first_order_diag 1
  --orth_init
  --layernorm
  --log_gate_stats 1
  --gate_log_every 10
  --alpha_methods ecf,mcc
  --alpha_n_boot 500
  --min_samples_alpha 1000
  --device "$DEVICE"
)

RECURRENT_EXTRA=(
  --diag_batch_size 256
  --diag_log_every 10
  --gru_init_update 0.05
  --lstm_init_forget 0.95
)

run_one_seed() {
  local SEED="$1"
  local BASE_OUTDIR="${OUTROOT}/baselines/seed_${SEED}"
  local RECURRENT_OUTDIR="${OUTROOT}/lstmgru/seed_${SEED}"
  local OUTDIR="${OUTROOT}/alpha_chain_diagnostics/seed_${SEED}"
  local BASE_LOG="${LOGDIR}/adamw_diag_seed_${SEED}.log"
  local RECURRENT_LOG="${LOGDIR}/adamw_gru_seed_${SEED}.log"

  mkdir -p "$BASE_OUTDIR" "$RECURRENT_OUTDIR" "$OUTDIR"

  echo ""
  echo "═══════════════════════════════════════════════════════════"
  echo "  Alpha-chain run — $(date)"
  echo "  Seed: ${SEED}"
  echo "  Device: ${DEVICE}"
  echo "  Output root: ${OUTROOT}"
  echo "  Lags: ${LAGS}"
  echo "  Training w_seed: ${W_SEED}"
  echo "  Validation w_seeds: ${VAL_W_SEEDS}"
  echo "═══════════════════════════════════════════════════════════"

  if ls "${BASE_OUTDIR}/diag/"*_final_checkpoint.pt >/dev/null 2>&1; then
    echo ""
    echo "[1/3] $(date +%H:%M:%S) ── AdamW diag ── existing checkpoint found, skipping training"
  else
    echo ""
    echo "[1/3] $(date +%H:%M:%S) ── AdamW diag ──"
    python "$BASE_RUNNER" \
      --outdir "$BASE_OUTDIR" \
      --models diag \
      --seed "$SEED" --w_seed "$W_SEED" \
      --optimizer adamw --momentum 0.9 --const_s 0.05 \
      "${COMMON_ARGS[@]}" \
      2>&1 | tee "$BASE_LOG"
  fi

  if ls "${RECURRENT_OUTDIR}/gru/"*_final_checkpoint.pt >/dev/null 2>&1; then
    echo ""
    echo "[2/3] $(date +%H:%M:%S) ── AdamW gru ── existing checkpoint found, skipping training"
  else
    echo ""
    echo "[2/3] $(date +%H:%M:%S) ── AdamW gru ──"
    python "$RECURRENT_RUNNER" \
      --outdir "$RECURRENT_OUTDIR" \
      --models gru \
      --seed "$SEED" --w_seed "$W_SEED" \
      --optimizer adamw --momentum 0.9 \
      "${COMMON_ARGS[@]}" \
      "${RECURRENT_EXTRA[@]}" \
      2>&1 | tee "$RECURRENT_LOG"
  fi

  echo ""
  echo "[3/3] $(date +%H:%M:%S) ── alpha-chain validation ──"
  echo "[alpha-chain] seed=${SEED} runroot=${OUTROOT} device=${DEVICE} lags=${LAGS} validation_w_seeds=${VAL_W_SEEDS}"
  python diagnostics/diagnose_alpha_chain.py \
    --root "$OUTROOT" \
    --seed "$SEED" \
    --models diag,gru \
    --lags "$LAGS" \
    --device "$DEVICE" \
    --projection_w_seeds "$VAL_W_SEEDS" \
    --gradient_dataset train \
    --full_grad_batch_size 32 \
    --grad_batch_size 32 \
    --grad_num_batches 4096 \
    --matched_batch_size 32 \
    --instantaneous_sample_cap 200000 \
    --alpha_min_samples 1000 \
    --alpha_n_boot 500 \
    --ecf_subsample_limit 100000 \
    --save_raw_samples 0 \
    --outdir "$OUTDIR" \
    2>&1 | tee "${OUTDIR}/alpha_chain_validation.log"
}

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
for SEED in "${SEEDS[@]}"; do
  SEED="$(echo "$SEED" | xargs)"
  [[ -z "$SEED" ]] && continue
  run_one_seed "$SEED"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Alpha-chain workflow complete — $(date)"
echo "  Root: ${OUTROOT}"
echo "  Diagnostics: ${OUTROOT}/alpha_chain_diagnostics"
echo "  Summary: generate manually once results are finalized"
echo "═══════════════════════════════════════════════════════════"
