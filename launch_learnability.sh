#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
# Learnability launcher: publication bundle + pilot-friendly entrypoint
#
# Default behavior:
#   - One default call runs the full publication bundle:
#       AdamW (main text) + SGD / SGD+momentum / RMSProp (appendix)
#   - Results are grouped by optimizer under results/<run_name>/
#   - Narrower modes remain available explicitly when needed
#
# Recommended pilot:
#   bash launch_learnability.sh 11 main pilot_1_seed 112
#
# Output layout:
#   results/<run_name>/<optimizer>/baselines/seed_<S>/<model>/
#   results/<run_name>/<optimizer>/lstmgru/seed_<S>/<model>/
#
# Usage:
#   bash launch_learnability.sh                                        # full publication bundle -> results/learnability
#   bash launch_learnability.sh 11                                     # single-seed publication pilot -> results/learnability
#   bash launch_learnability.sh 11 main pilot_1_seed                   # AdamW-only pilot -> results/pilot_1_seed
#   bash launch_learnability.sh 11 main pilot_1_seed 112               # AdamW-only pilot with fixed w_seed=112
#   bash launch_learnability.sh 1,2,3,4,5 main fullsimulation          # final AdamW multiseed -> results/fullsimulation
#   bash launch_learnability.sh 1,2,3,4,5 appendix appendix_run        # appendix optimizers only
#   bash launch_learnability.sh 1,2,3,4,5 rmsprop rmsprop_debug 12345  # explicit single-optimizer run
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────
SEEDS_CSV="${1:-1,2,3,4,5}"
RUN_SPEC="${2:-publication}"
RUN_NAME="${3:-learnability}"
W_SEED="${4:-12345}"

OUTROOT="results/${RUN_NAME}"
LOGDIR="${OUTROOT}/logs"
mkdir -p "$LOGDIR"

BASE_RUNNER="run_learnability_baselines.py"
RECURRENT_RUNNER="run_learnability_lstm_gru.py"

for RUNNER in "$BASE_RUNNER" "$RECURRENT_RUNNER"; do
  if [[ ! -f "$RUNNER" ]]; then
    echo "[error] Missing runner: ${RUNNER}"
    echo "        Update the workspace so launch_learnability.sh and the runner filenames match."
    exit 1
  fi
done

# ── Master launcher log ───────────────────────────────────────────
if [[ "${LEARNABILITY_LAUNCH_LOGGED:-0}" != "1" ]]; then
  MASTER_LOG_TS="$(date +%Y%m%d_%H%M%S)"
  MASTER_LOG="${LOGDIR}/launch_${MASTER_LOG_TS}.log"
  LATEST_LOG="${LOGDIR}/launch_latest.log"
  export LEARNABILITY_LAUNCH_LOGGED=1
  export LEARNABILITY_MASTER_LOG="$MASTER_LOG"
  exec > >(tee -a "$MASTER_LOG") 2>&1
  ln -sf "$(basename "$MASTER_LOG")" "$LATEST_LOG"
  echo "[info] master log: ${MASTER_LOG}"
  echo "[info] latest log alias: ${LATEST_LOG}"
fi

# ── Shared arguments (publication geometry + denser pilot N-grid) ───────────
# noise_tolerance is the inverse SNR threshold used for detectability:
#   0.1   -> require SNR > 10
#   0.05  -> require SNR > 20
#   0.025 -> require SNR > 40
COMMON="--Nseq_train 8000 --Nseq_diag 12000 --T 1536 --D 16 --H 256 \
  --epochs 1000 --batch_size 512 --lr 0.00025 --weight_decay 0.0001 \
  --grad_clip 1.0 --lag_min 4 --lag_max 768 --num_lags 192 \
  --task_lags 64,128,256,384,512,768 \
  --task_coeffs 0.60,0.50,0.40,0.30,0.22,0.16 \
  --noise_std 0.4 --noise_tolerance 0.05 \
  --N_grid 25,50,75,100,125,150,175,200,250,300,350,400,500,600,700,800,1000,1200,1400,1600,2000,2400,2800,3200,4000,4800,5600,6400,8000,9600,11200,12800,16000,19200,22400,25600 \
  --include_first_order_diag 1 --orth_init --layernorm \
  --log_gate_stats 1 --gate_log_every 10 \
  --alpha_methods ecf,mcc --alpha_n_boot 500 --min_samples_alpha 1000 \
  --device cuda"

LSTM_EXTRA="--diag_batch_size 256 --diag_log_every 10 --gru_init_update 0.05 --lstm_init_forget 0.95"

# ── One fixed projection direction per seed, shared across all architectures ─

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

IFS=',' read -r -a RAW_SEEDS <<< "$SEEDS_CSV"
SEEDS=()
for S in "${RAW_SEEDS[@]}"; do
  S="$(echo "$S" | xargs)"
  [[ -z "$S" ]] && continue
  if ! [[ "$S" =~ ^[0-9]+$ ]]; then
    echo "[error] Invalid seed value: ${S}"
    echo "        Use a comma-separated list like: 11 or 1,2,3,4,5"
    exit 1
  fi
  SEEDS+=("$S")
done

if (( ${#SEEDS[@]} == 0 )); then
  echo "[error] No valid seeds were provided."
  exit 1
fi

# ── Counters ──────────────────────────────────────────────────────
TOTAL_RUNS=$(( ${#OPT_NAMES[@]} * ${#SEEDS[@]} * 2 ))
RUN=0
T0=$SECONDS

echo "═══════════════════════════════════════════════════════════"
echo "  Learnability launch — $(date)"
echo "  Mode: ${MODE_LABEL}"
echo "  Run spec: ${RUN_SPEC}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Optimizers: ${OPT_NAMES[*]}"
echo "  Total runs: ${TOTAL_RUNS}"
echo "  Output root: ${OUTROOT}"
echo "  Master log: ${LEARNABILITY_MASTER_LOG:-n/a}"
echo "  Fixed w_seed across architectures: ${W_SEED}"
echo "  Key geometry: T=1536, H=256, lag_max=768, num_lags=192"
echo "  Key training: epochs=1000, lr=2.5e-4, grad_clip=1.0, noise_std=0.4"
echo "  Detectability: noise_tolerance=0.05, denser N_grid for pilot/sample-complexity readout"
echo "═══════════════════════════════════════════════════════════"

# ════════════════════════════════════════════════════════════════════
# Main loop: optimizer → seed → {baselines, lstmgru}
# ════════════════════════════════════════════════════════════════════
for OPT in "${OPT_NAMES[@]}"; do
  for S in "${SEEDS[@]}"; do
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

ELAPSED=$(( SECONDS - T0 ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All done — $(date)"
echo "  Total wall time: ${HOURS}h ${MINS}m"
echo "  Results: ${OUTROOT}/"
echo "  Master log: ${LEARNABILITY_MASTER_LOG:-n/a}"
echo "═══════════════════════════════════════════════════════════"
