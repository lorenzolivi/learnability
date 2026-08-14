# Learnability Window in Gated Recurrent Neural Networks

[![arXiv](https://img.shields.io/badge/arXiv-2512.05790-b31b1b.svg)](https://arxiv.org/abs/2512.05790)

Code accompanying:

**Lorenzo Livi**

*Learnability Window in Gated Recurrent Neural Networks*

Paper: https://arxiv.org/abs/2512.05790

---

## Overview

This repository provides the code needed to reproduce the experimental
results of the paper. The pipeline trains five recurrent architectures on a
synthetic multi-lag regression task, restores the best-validation checkpoint,
runs the diagnostics, and generates the aggregate figures and tables used in
the manuscript.

The paper-scale runs are computationally heavy: they use long sequences, dense
lag grids, five random seeds, and 50 random projection directions per seed. A
smaller smoke test is included to validate the pipeline before launching the
full simulations.

---

## Repository structure

```text
.
├── launch_learnability.sh               # Main launcher for paper-scale runs
├── run_learnability_baselines.py        # ConstGate, SharedGate, DiagGate
├── run_learnability_lstm_gru.py         # LSTM and GRU
├── smoke_test.sh                        # Small end-to-end run
├── validate_smoke_test.py               # Smoke-test validator
├── plot_all_multiseed.py                # Master plotting/analysis runner
├── plot_*.py                            # Individual plotting scripts
├── compute_per_projection_alpha.py      # Per-projection alpha diagnostic
├── compute_ecf_bootstrap_ci.py          # ECF bootstrap confidence intervals
├── alpha_estimators.py                  # Standalone tail-index estimators
├── fit_master_proportionality.py        # Master proportionality fit
├── seed_utils.py                        # Shared loading/aggregation utilities
├── diagnostics/                         # Additional diagnostic scripts
├── requirements.txt
└── README.md
```

---

## Requirements

```bash
pip install -r requirements.txt
```

The code requires Python 3.9+ with PyTorch, NumPy, Matplotlib, pandas, and
optionally SciPy.

---

## Hardware

The scripts run on CPU or CUDA through the `--device` flag. The full
paper-scale configuration is intended for a high-memory CUDA machine. Plotting
and post-hoc analysis run on CPU.

Apple Silicon MPS is not used by the provided launchers: `torch.func.jvp`
support for recurrent kernels is incomplete on MPS, so the JVP-based
matched-statistic pipeline should be run on CPU or CUDA.

---

## Reproducing the paper results

The paper-scale outputs are grouped under `results/fullsim/`. The current
five-seed configuration is:

```text
2,12,31,41,51
```

Run the main-text AdamW experiment:

```bash
bash launch_learnability.sh 2,12,31,41,51 main fullsim
```

Run the optimizer-comparison experiment:

```bash
bash launch_learnability.sh 2,12,31,41,51 sgd fullsim
```

This produces:

```text
results/fullsim/adamw/
results/fullsim/sgd/
```

Running the optimizer sweeps one at a time is usually safer on shared
machines because each sweep is long and writes separate logs.

Launcher arguments:

1. seed list, e.g. `2,12,31,41,51`
2. run spec: `main`, `appendix`, `publication`, or a single optimizer name
3. run name, used as `results/<run_name>/<optimizer>/...`
4. optional `w_seed` policy; default is `auto`, which assigns disjoint blocks
   of 50 projection directions per seed

---

## Smoke test

Before a full run, validate the pipeline with:

```bash
bash smoke_test.sh
python validate_smoke_test.py --root results/smoke_test
```

Optional device override:

```bash
DEVICE=cpu  bash smoke_test.sh
DEVICE=cuda bash smoke_test.sh
```

---

## Plotting and post-hoc analysis

After training, generate aggregate figures and post-hoc alpha diagnostics with:

```bash
python plot_all_multiseed.py \
  --inputdirs results/<run_name>/<optimizer>/baselines \
              results/<run_name>/<optimizer>/lstmgru \
  --outdir    results/<run_name>/<optimizer>/together
```

For example:

```bash
python plot_all_multiseed.py \
  --inputdirs results/fullsim/sgd/baselines results/fullsim/sgd/lstmgru \
  --outdir    results/fullsim/sgd/together
```

Available plotting steps:

```bash
python plot_all_multiseed.py --list
```

---

## Outputs

Each full run writes:

- per-model CSV summaries, learning curves, envelope values, alpha estimates,
  and time-scale fits;
- selected checkpoint files and selection metadata;
- per-projection matched-statistic arrays used by post-hoc alpha diagnostics;
- aggregate plots and JSON/CSV summaries under each optimizer's `together/`
  directory.

Plotting scripts produce PNG figures at 300 dpi.

---

## Citation

```bibtex
@article{livi_learnability,
  title={Learnability Window in Gated Recurrent Neural Networks},
  author={Livi, Lorenzo},
  journal={arXiv preprint arXiv:2512.05790},
  year={2025}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
