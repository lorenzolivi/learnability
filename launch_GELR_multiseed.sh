#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
# GELR Multi-Seed v2: publication bundle launcher
#
# Publication defaults:
#   - One default call runs the full publication bundle:
#       AdamW (main text) + SGD / SGD+momentum / RMSProp (appendix)
#   - Results are grouped by optimizer under results/GELR_multiseed/
#   - Narrower modes remain available explicitly when needed
#
# Balanced publication regime:
#   - Use publication-scale hidden size and longer training.
#   - Keep the current detectability threshold fixed for the next pilot so the
#     effect of recurrent-memory fixes is easy to interpret.
#   - Start GRU/LSTM in a long-memory operating point comparable to the
#     baseline gated models.
#   - Keep the run compact enough for a single DGX Spark sweep.
#
# Output layout:
#   results/GELR_multiseed/<optimizer>/baselines/seed_<S>/<model>/
#   results/GELR_multiseed/<optimizer>/lstmgru/seed_<S>/<model>/
#
# Usage:
#   bash launch_GELR_multiseed.sh                # full publication bundle
#   bash launch_GELR_multiseed.sh 1              # single-seed publication pilot
#   bash launch_GELR_multiseed.sh 5 main         # AdamW only
#   bash launch_GELR_multiseed.sh 5 appendix     # SGD / SGD+momentum / RMSProp only
#   bash launch_GELR_multiseed.sh 5 rmsprop      # explicit single-optimizer run
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────
SEEDS=(1 2 3 4 5)
MAX_SEEDS="${1:-${#SEEDS[@]}}"
RUN_SPEC="${2:-publication}"

OUTROOT="results/GELR_multiseed"
LOGDIR="${OUTROOT}/logs"
mkdir -p "$LOGDIR"

BASE_RUNNER="run_learnability_baselines.py"
RECURRENT_RUNNER="run_learnability_lstm_gru.py"

for RUNNER in "$BASE_RUNNER" "$RECURRENT_RUNNER"; do
  if [[ ! -f "$RUNNER" ]]; then
    echo "[error] Missing runner: ${RUNNER}"
    echo "        Update the DGX workspace so launch_GELR_multiseed.sh and the runner filenames match."
    exit 1
  fi
done

# ── Master launcher log ───────────────────────────────────────────
if [[ "${GELR_LAUNCH_LOGGED:-0}" != "1" ]]; then
  MASTER_LOG_TS="$(date +%Y%m%d_%H%M%S)"
  MASTER_LOG="${LOGDIR}/launch_${MASTER_LOG_TS}.log"
  LATEST_LOG="${LOGDIR}/launch_latest.log"
  export GELR_LAUNCH_LOGGED=1
  export GELR_MASTER_LOG="$MASTER_LOG"
  exec > >(tee -a "$MASTER_LOG") 2>&1
  ln -sf "$(basename "$MASTER_LOG")" "$LATEST_LOG"
  echo "[info] master log: ${MASTER_LOG}"
  echo "[info] latest log alias: ${LATEST_LOG}"
fi

# ── Shared arguments (balanced publication setup) ────────────────
# noise_tolerance is the inverse SNR threshold used for detectability:
#   0.1  -> require SNR > 10  (moderately strict)
#   0.05 -> require SNR > 20  (strict)
#   0.025 -> require SNR > 40 (very strict)
COMMON="--Nseq_train 8000 --Nseq_diag 12000 --T 1536 --D 16 --H 256 \
  --epochs 1000 --batch_size 512 --lr 0.00025 --weight_decay 0.0001 \
  --grad_clip 0.5 --lag_min 4 --lag_max 768 --num_lags 192 \
  --task_lags 64,128,256,384,512,768 \
  --task_coeffs 0.60,0.50,0.40,0.30,0.22,0.16 \
  --noise_std 0.4 --noise_tolerance 0.05 \
  --N_grid 50,100,150,200,300,400,600,800,1200,1600,2400,3200,4800,6400,9600,12800,19200,25600 \
  --include_first_order_diag 1 --orth_init --layernorm \
  --log_gate_stats 1 --gate_log_every 10 \
  --alpha_methods ecf,mcc --alpha_n_boot 500 --min_samples_alpha 1000 \
  --device cuda"

LSTM_EXTRA="--diag_batch_size 256 --diag_log_every 10 --gru_init_update 0.05 --lstm_init_forget 0.95"

# ── w_seed: fixed across seeds (same JVP direction for comparability)
W_SEED=12345

# ── Optimizer definitions ─────────────────────────────────────────
declare -a MAIN_TEXT_OPT_NAMES=("adamw")
declare -a APPENDIX_OPT_NAMES=("sgd" "sgd_momentum" "rmsprop")
declare -a PUBLICATION_OPT_NAMES=("adamw" "sgd" "sgd_momentum" "rmsprop")
declare -a OPT_NAMES=()

declare -A OPT_BASE_EXTRA
OPT_BASE_EXTRA[adamw]="--optimizer adamw --momentum 0.9 --const_s 0.05"
OPT_BASE_EXTRA[sgd]="--optimizer sgd --const_s 0.05"
OPT_BASE_EXTRA[sgd_momentum]="--optimizer sgd_momentum --momentum 0.9 --const_s 0.05"
OPT_BASE_EXTRA[rmsprop]="--optimizer rmsprop --momentum 0.0 --rmsprop_alpha 0.99 --const_s 0.05"

declare -A OPT_LG_EXTRA
OPT_LG_EXTRA[adamw]="--optimizer adamw --momentum 0.9"
OPT_LG_EXTRA[sgd]="--optimizer sgd"
OPT_LG_EXTRA[sgd_momentum]="--optimizer sgd_momentum --momentum 0.9"
OPT_LG_EXTRA[rmsprop]="--optimizer rmsprop --momentum 0.0 --rmsprop_alpha 0.99"

# ── Resolve run mode / optimizer selection ────────────────────────
MODE_LABEL=""
case "$RUN_SPEC" in
  publication|all)
    OPT_NAMES=("${PUBLICATION_OPT_NAMES[@]}")
    MODE_LABEL="publication bundle"
    ;;
  main|main_text|adamw_only)
    OPT_NAMES=("${MAIN_TEXT_OPT_NAMES[@]}")
    MODE_LABEL="main text (AdamW)"
    ;;
  appendix|appendix_only)
    OPT_NAMES=("${APPENDIX_OPT_NAMES[@]}")
    MODE_LABEL="appendix optimizers"
    ;;
  adamw|sgd|sgd_momentum|rmsprop)
    OPT_NAMES=("$RUN_SPEC")
    MODE_LABEL="explicit optimizer"
    ;;
  *)
    echo "[error] Unknown run spec: ${RUN_SPEC}"
    echo "        Expected one of: publication, main, appendix, all,"
    echo "                         adamw, sgd, sgd_momentum, rmsprop"
    exit 1
    ;;
esac

if (( MAX_SEEDS < 1 || MAX_SEEDS > ${#SEEDS[@]} )); then
  echo "[error] MAX_SEEDS must be between 1 and ${#SEEDS[@]}, got: ${MAX_SEEDS}"
  exit 1
fi

# ── Counters ──────────────────────────────────────────────────────
TOTAL_RUNS=$(( ${#OPT_NAMES[@]} * MAX_SEEDS * 2 ))
RUN=0
T0=$SECONDS

echo "═══════════════════════════════════════════════════════════"
echo "  GELR multi-seed v2 launch — $(date)"
echo "  Mode: ${MODE_LABEL}"
echo "  Run spec: ${RUN_SPEC}"
echo "  Seeds: ${SEEDS[*]:0:$MAX_SEEDS}  (${MAX_SEEDS} of ${#SEEDS[@]})"
echo "  Optimizers: ${OPT_NAMES[*]}"
echo "  Total runs: ${TOTAL_RUNS}"
echo "  Output root: ${OUTROOT}"
echo "  Master log: ${GELR_MASTER_LOG:-n/a}"
echo "  Key changes: T=1536, H=256, lag_max=768, num_lags=192, noise_std=0.4, epochs=1000, lr=2.5e-4, grad_clip=0.5, noise_tolerance=0.05 (eps=20), gru_init_update=0.05, lstm_init_forget=0.95"
echo "═══════════════════════════════════════════════════════════"

# ════════════════════════════════════════════════════════════════════
# Main loop: optimizer → seed → {baselines, lstmgru}
# ════════════════════════════════════════════════════════════════════
for OPT in "${OPT_NAMES[@]}"; do
  for (( i=0; i<MAX_SEEDS; i++ )); do
    S=${SEEDS[$i]}
    SEED_TAG="seed_${S}"

    # ── Baselines (const, shared, diag) ──────────────────────────
    RUN=$((RUN + 1))
    OUTDIR_B="${OUTROOT}/${OPT}/baselines/${SEED_TAG}"
    LOGFILE_B="${LOGDIR}/${OPT}_baselines_${SEED_TAG}.log"
    mkdir -p "$OUTDIR_B"

    echo ""
    echo "[${RUN}/${TOTAL_RUNS}] $(date +%H:%M:%S) ── ${OPT} baselines ${SEED_TAG} ──"

    python "$BASE_RUNNER" \
      --outdir "$OUTDIR_B" \
      --models const,shared,diag \
      --seed "$S" --w_seed "$W_SEED" \
      ${OPT_BASE_EXTRA[$OPT]} \
      $COMMON \
      2>&1 | tee "$LOGFILE_B"

    # ── LSTM / GRU ───────────────────────────────────────────────
    RUN=$((RUN + 1))
    OUTDIR_L="${OUTROOT}/${OPT}/lstmgru/${SEED_TAG}"
    LOGFILE_L="${LOGDIR}/${OPT}_lstmgru_${SEED_TAG}.log"
    mkdir -p "$OUTDIR_L"

    echo ""
    echo "[${RUN}/${TOTAL_RUNS}] $(date +%H:%M:%S) ── ${OPT} lstm/gru ${SEED_TAG} ──"

    python "$RECURRENT_RUNNER" \
      --outdir "$OUTDIR_L" \
      --models lstm,gru \
      --seed "$S" --w_seed "$W_SEED" \
      ${OPT_LG_EXTRA[$OPT]} \
      $COMMON $LSTM_EXTRA \
      2>&1 | tee "$LOGFILE_L"

  done
done

# ════════════════════════════════════════════════════════════════════
ELAPSED=$(( SECONDS - T0 ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All done — $(date)"
echo "  Total wall time: ${HOURS}h ${MINS}m"
echo "  Results: ${OUTROOT}/"
echo "  Master log: ${GELR_MASTER_LOG:-n/a}"
echo "═══════════════════════════════════════════════════════════"
