# Learnability Window in Gated Recurrent Neural Networks

[![arXiv](https://img.shields.io/badge/arXiv-2512.05790-b31b1b.svg)](https://arxiv.org/abs/2512.05790)

Code accompanying the paper:

**Lorenzo Livi**
*Learnability Window in Gated Recurrent Neural Networks*
Paper: https://arxiv.org/abs/2512.05790

---

## Overview

This repository contains the code used to generate the experiments reported in the paper.
The work introduces the *learnability window* $H_N$, the set of temporal lags
at which a finite training budget $N$ can reliably learn long-range dependencies
in a gated RNN. The analysis rests on a matched-statistic decomposition that
factorises each lag's gradient signal into a memory-kernel envelope $\mu(\ell)$,
a parameter-sensitivity term (JVP), and a heavy-tailed residual whose tail index
$\hat\alpha(\ell)$ governs sample complexity. Five gated architectures are
compared: ConstGate, SharedGate, DiagGate, GRU, and LSTM.

The pipeline trains each architecture on a synthetic multi-lag regression task
($y_t = \sum_k c_k\, u^\top x_{t-\ell_k} + \text{noise}$), runs the full
diagnostic suite (memory kernel, tail-index estimation, SNR, sample complexity),
and produces all figures in the paper.

---

## Repository structure

```
.
├── run_learnability_DGX.py             # Training + diagnostics: ConstGate, SharedGate, DiagGate
├── run_learnability_lstm_gru_DGX.py    # Training + diagnostics: LSTM, GRU
├── launch_multiseed.py                 # Wrapper for multi-seed runs across architectures
├── seed_utils.py                       # Shared utilities: seed discovery, CSV loading, aggregation
├── plot_all_multiseed.py               # Master runner for all 10 plot steps
├── plot_empirical_learnability_win.py  # H_N curves (mean±std / percentile / boxplot)
├── plot_envelope.py                    # Memory kernel envelope f(ℓ)
├── plot_tau.py                         # Time-scale τ distributions
├── plot_N_vs_envelope.py               # Sample complexity scaling
├── plot_alpha_estimation.py            # α̂ distribution plots (with reliability filtering)
├── plot_noise_floor.py                 # Noise floor / detectability threshold
├── plot_learnability_learning_curves.py# Training loss curves
├── fit_master_proportionality.py       # Master proportionality law fit
├── make_appendix_optimizer_figs.py     # Appendix: optimizer comparison figures
├── EXAMPLES.txt                        # Full CLI examples and workflow documentation
├── requirements.txt
└── README.md
```

The two DGX scripts implement the full training and diagnostic pipeline.
The remaining scripts handle multi-seed orchestration and plotting with
automatic aggregation across seeds and automatic merging of baselines and
LSTM/GRU results.

---

## Requirements

```
pip install -r requirements.txt
```

The code requires Python 3.9+ and depends on PyTorch, NumPy, Matplotlib,
pandas, and (optionally) SciPy. SciPy is used for higher-accuracy McCulloch
table construction via `scipy.stats.levy_stable`; if absent, a hardcoded
fallback table is used with no loss of functionality.

---

## Hardware

The training scripts are optimised for the
[NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
system. The full paper run (5 seeds × 5 architectures, 750 epochs each)
requires a CUDA-capable GPU; a single DGX Spark run completes in roughly
one day. Plotting and analysis scripts run on CPU.

---

## Reproducing the paper results

### Full replication (5 seeds, as in the paper)

Training is launched via `launch_multiseed.py`, which orchestrates both the
baseline script (ConstGate, SharedGate, DiagGate) and the LSTM/GRU script
across all seeds. A typical invocation inside a tmux session:

```bash
tmux new -s learnability

python launch_multiseed.py \
  --seeds 212,1001,2002,3003,4004 \
  --outdir_baselines results/T1024/baselines/adamw_lagmax256 \
  --outdir_lstm_gru results/T1024/lstm_gru/adamw_lagmax256 \
  --common_args "--Nseq_train 8000 --Nseq_diag 8000 --T 1024 --D 16 --H 128 \
    --optimizer adamw --momentum 0.9 --epochs 750 --batch_size 512 --lr 0.001 \
    --weight_decay 0.0001 --grad_clip 1.0 \
    --lag_min 4 --lag_max 256 --num_lags 128 \
    --task_lags 32,64,128,192,256 --task_coeffs 0.6,0.5,0.4,0.32,0.26 \
    --noise_std 0.3 \
    --N_grid 25,50,100,150,200,300,400,600,800,1200,1600,2400,3200,4800,6400,9600,12800 \
    --eps 0.1 --orth_init --layernorm \
    --include_first_order_diag 1 --log_gate_stats 1 --gate_log_every 10 \
    --device cuda \
    --alpha_method ecf" \
  --baseline_extra "--models const,shared,diag --const_s 0.05" \
  --lstm_gru_extra "--models lstm,gru" \
  --w_seed 212 \
  --logdir logs \
  2>&1 | tee logs/launch_multiseed.log
```

### Generating all figures

Once training is complete, all figures can be generated in a single command:

```bash
python plot_all_multiseed.py \
  --inputdirs results/T1024/baselines/adamw_lagmax256 results/T1024/lstm_gru/adamw_lagmax256 \
  --outdir results/T1024/together/adamw_lagmax256
```

The plotting pipeline auto-discovers `seed_*` subdirectories, aggregates
across seeds (mean ± std with shaded bands), and merges baseline and LSTM/GRU
results into unified figures. Selective plotting is supported via `--only` and
`--skip` flags; run `python plot_all_multiseed.py --list` to see available steps.

### Running individual experiments

Each DGX script can also be run directly for a single seed:

```bash
# Baselines (ConstGate, SharedGate, DiagGate)
python run_learnability_DGX.py \
  --outdir results/T1024/baselines/adamw_lagmax256/seed_212 \
  --models const,shared,diag --const_s 0.05 \
  --seed 212 --w_seed 212 \
  --alpha_method ecf \
  --Nseq_train 8000 --Nseq_diag 8000 --T 1024 --D 16 --H 128 \
  --optimizer adamw --epochs 750 --batch_size 512 --lr 0.001 \
  --device cuda

# LSTM and GRU
python run_learnability_lstm_gru_DGX.py \
  --outdir results/T1024/lstm_gru/adamw_lagmax256/seed_212 \
  --models lstm,gru \
  --seed 212 --w_seed 212 \
  --alpha_method ecf \
  --Nseq_train 8000 --Nseq_diag 8000 --T 1024 --D 16 --H 128 \
  --optimizer adamw --epochs 750 --batch_size 512 --lr 0.001 \
  --device cuda
```

See `EXAMPLES.txt` for the complete set of CLI options, per-plot examples,
and additional workflow patterns.

### Outputs

Each training run produces per-model CSV files (summary statistics, learning
curves, per-unit envelope values, time-scale fits) and aggregate files
(learnability window $H_N$, CLI arguments for reproducibility). The plotting
scripts produce PNG figures at 300 dpi.

---

## Tail-index estimation

Two methods are available for estimating the stable tail index $\hat\alpha$:

**McCulloch (1986)** (`--alpha_method mcculloch`, default): quantile-ratio
method using four empirical quantiles. Fast but less statistically efficient.

**Koutrouvelis (1980) ECF** (`--alpha_method ecf`): empirical characteristic
function regression. For symmetric $\alpha$-stable distributions,
$\log(-\log|\hat\varphi(t)|^2) = \log(2\sigma^\alpha) + \alpha\log|t|$;
the slope gives $\hat\alpha$. More robust across the full $\alpha \in [1,2]$
range.

Both methods include reliability guards: minimum sample count, positive scale
check, non-degenerate IQR, and boundary detection. Estimates are flagged via
the `alpha_reliable` column in summary CSVs. The plotting script
(`plot_alpha_estimation.py`) filters unreliable estimates by default; use
`--show_unreliable` to overlay them.

---

## Citation

```bibtex
@article{livi_learnability,
  title={Learnability Window in Gated Recurrent Neural Networks},
  author={Livi, Lorenzo},
  journal={arXiv preprint arXiv:2512.05790},
  year={2025},
  doi={10.48550/arXiv.2512.05790},
  url={https://arxiv.org/abs/2512.05790}
}
```

## License

This code is released for academic and research use. See the paper for details.
