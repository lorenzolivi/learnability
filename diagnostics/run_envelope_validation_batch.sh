#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEEDS=(42 123 271 314 999)
SCRIPT="diagnostics/diagnose_first_order_distortion.py"

COMMON_ARGS=(
  --H 64
  --D 8
  --T 500
  --B 256
  --epochs 200
  --lr 1e-3
  --const_s 0.1
  --n_diag_sequences 16
  --n_t_per_sequence 50
  --models "const,shared,diag,gru,lstm"
  --optimizers "sgd,adamw,rmsprop"
  --lags "1,2,3,5,8,13,21,34,55,89,144,245"
  --momentum 0.9
  --weight_decay 1e-4
  --rmsprop_alpha 0.99
)

for seed in "${SEEDS[@]}"; do
  outdir="diagnostics/envelope_approx_validation/seed${seed}"
  echo "[batch] seed=${seed} -> ${outdir}"
  mkdir -p "$outdir"
  python "$SCRIPT" "${COMMON_ARGS[@]}" --seed "$seed" --outdir "$outdir"
done
