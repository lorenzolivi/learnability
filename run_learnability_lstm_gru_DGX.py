#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empirical learnability-window pipeline — LSTM and GRU

Reference: "Learnability Window in Gated Recurrent Neural Networks"
           (see the paper for theoretical background and notation).

This script trains each model on a synthetic multi-lag regression task
(y_t = Σ_k c_k u^T x_{t-ℓ_k} + noise), then runs the full learnability
diagnostic pipeline from the paper:

  1. Train model via streaming CPU→GPU batches.
  2. For each diagnostic lag ℓ in a grid:
     a. Compute the architecture-specific diagonal Jacobian factorisation.
        GRU: leak = (1-z), rdiag captures z/r/candidate gate derivatives.
        LSTM: forget gate f, expression e = o·(1-tanh²(c)), cdiag = ∂c_t/∂h_{t-1}.
     b. Compute the JVP sensitivity v_t = (∂h_t/∂θ)[w] along a random
        unit direction w (fixed across lags for comparability).
     c. Form the matched statistic ψ_t(ℓ) = Σ_q μ^(q)_t(ℓ) δ_t^(q) v_{t-ℓ}^(q).
     d. Estimate tail index α̂(ℓ) via McCulloch quantile method.
     e. Compute empirical SNR and N_required(ℓ) at threshold ε.
  3. Aggregate into learnability window H_N.
  4. Fit exponential time-scale τ from per-unit envelope.

GPU-memory safety:
  - Datasets stay on CPU (pinned); batches streamed to GPU.
  - Per-ℓ matched-statistic samples written to temporary memmap files on
    disk (deleted as soon as quantiles are computed).

Outputs (per model, in <outdir>/<model>/):
  - <model>_learning_curve.csv       epoch-level train/val loss and R²
  - <model>_summary.csv              per-lag: μ, α̂, σ̂, N_required, SNR
  - <model>_mu_units.csv             per-lag × per-unit envelope values
  - <model>_tau_from_mu_units.csv    per-unit exponential τ fits
  - <model>_tau_from_mu_stats.json   summary statistics for τ distribution
  - <model>_envelope_fits.json       exponential and power-law fits to μ(ℓ)
  - <model>_H_N.csv                  learnability window vs training budget
  - gate_stats_<model>.csv           periodic gate activation statistics

Outputs (aggregate, in <outdir>/):
  - H_N_summary.csv                  all models' H_N side-by-side
  - cli_args.csv                     full CLI arguments for reproducibility

Code organisation (top → bottom):
  1. Small utilities (logger, seed, layernorm switch)
  2. Streaming evaluation helper
  3. JVP utilities (forward-mode AD for parameter sensitivity)
  4. Prefix-sum helpers for ℓ-step window products/sums
  5. McCulloch α-stable estimator
  6. SNR and detection-error helpers
  7. Synthetic data generation (same task as baselines script)
  8. Architecture-specific prefix builders (GRU, LSTM)
  9. Model definitions (GRUModel, LSTMModel)
 10. Training loop (streaming CPU→GPU)
 11. Fit utilities (exponential τ, envelope regimes, H_N)
 12. Diagnostics pipeline (run_for_model)
 13. CLI argument parser
 14. main()
"""

import argparse
import csv
import json
import math
import os
import time
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp


# ============================================================
# Small utilities
# ============================================================

def save_args_to_csv(args, filepath: str) -> None:
    """Dump all CLI arguments to a two-column CSV for reproducibility."""
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["argument", "value"])
        for k, v in vars(args).items():
            w.writerow([k, v])

def set_seed(seed: int) -> None:
    """Set random seeds for numpy, torch CPU, and all CUDA devices."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

def layernorm_if(enabled: bool, dim: int) -> nn.Module:
    """Return a LayerNorm module if enabled, otherwise nn.Identity (no-op)."""
    return nn.LayerNorm(dim) if enabled else nn.Identity()

def try_remove(path: str) -> None:
    """Silently remove a file if it exists (used for tmp memmap cleanup)."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def now_s() -> float:
    """Current wall-clock time in seconds (for timing diagnostics)."""
    return time.time()

def log(msg: str) -> None:
    """Print a timestamped log message (flushed immediately)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# Evaluation helper: streaming MSE and R² computation
# ============================================================

def _eval_streaming_mse_and_r2(
    model: nn.Module,
    X_cpu: torch.Tensor,
    Y_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Tuple[float, float]:
    """
    Compute MSE loss and R² in streaming batches (CPU→GPU).

    Uses return_intermediates=False to avoid materialising diagnostic
    tensors (leak, rdiag, gate activations) during evaluation.
    """
    model.eval()
    Btot = int(X_cpu.shape[0])
    bs = int(batch_size)
    n_batches = max(1, math.ceil(Btot / bs))

    # MSE numerator accumulates sum of squared error; then divide by numel
    sse = 0.0
    n_elem = 0

    # For R^2: SST = sum(y^2) - n * mean_y^2
    sum_y = 0.0
    sum_y2 = 0.0
    n_y = 0

    with torch.no_grad():
        for bi in range(n_batches):
            lo = bi * bs
            hi = min(Btot, (bi + 1) * bs)

            xb = X_cpu[lo:hi].to(device, non_blocking=True)
            yb = Y_cpu[lo:hi].to(device, non_blocking=True)

            yhat, _, _ = model.forward_with_intermediates(xb, return_intermediates=False)

            diff = (yhat - yb).reshape(-1)
            sse += float(torch.sum(diff * diff).item())
            n_elem += int(diff.numel())

            yflat = yb.reshape(-1)
            sum_y += float(torch.sum(yflat).item())
            sum_y2 += float(torch.sum(yflat * yflat).item())
            n_y += int(yflat.numel())

            del xb, yb, yhat, diff, yflat

    mse = sse / max(1, n_elem)
    mean_y = sum_y / max(1, n_y)
    sst = sum_y2 - max(1, n_y) * (mean_y * mean_y)
    r2 = 1.0 - (sse / (sst + 1e-12))
    return float(mse), float(r2)




# ============================================================
# JVP utilities
#
# Compute the Jacobian-vector product v_t = (∂h_t/∂θ)[w] where
# w is a random unit-norm direction in parameter space.  Uses
# torch.func.jvp (forward-mode AD) for memory efficiency.
# ============================================================

def _make_random_unit_w_pytree(model: nn.Module, device: torch.device, seed: int):
    """
    Build a random unit-norm tangent vector w in parameter space.

    Returns (params0, buffers, w) where w has ||w||₂ = 1 across all
    parameters jointly.  The seed is fixed across lags/batches.
    """
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    params0 = {k: v for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v for k, v in model.named_buffers()}

    if len(params0) == 0:
        return params0, buffers, {}

    w = {k: torch.randn(v.shape, generator=g, device=v.device, dtype=v.dtype) for k, v in params0.items()}

    norm2 = None
    for t in w.values():
        val = (t.detach() ** 2).sum()
        norm2 = val if norm2 is None else (norm2 + val)
    norm = torch.sqrt(norm2 + 1e-12)
    w = {k: t / norm for k, t in w.items()}

    return params0, buffers, w

def compute_vseq_jvp(model: nn.Module, X: torch.Tensor, w_seed: int) -> torch.Tensor:
    """
    Compute the hidden-state JVP sequence v_t = (∂h_t/∂θ)[w].

    Args:
        model:   trained model (GRU or LSTM).
        X:       input tensor (B, T, D), already on device.
        w_seed:  seed for the random tangent direction w.

    Returns:
        vseq: (B, T, H) per-timestep JVP values.
    """
    device = X.device
    model.eval()

    params0, buffers, w = _make_random_unit_w_pytree(model, device=device, seed=w_seed)

    if len(params0) == 0:
        with torch.no_grad():
            _, hseq0, _ = model.forward_with_intermediates(X)
        return torch.zeros_like(hseq0)

    def f_hseq(p):
        _, hseq, _ = functional_call(model, (p, buffers), (X,))
        return hseq

    _, vseq = jvp(f_hseq, (params0,), (w,))
    return vseq


# ============================================================
# McCulloch quantile estimator for symmetric α-stable (SαS)
#
# Estimates the tail index α ∈ [1, 2] from the ratio of
# inter-quantile ranges R̂ = (q95-q05)/(q75-q25).  A pre-computed
# grid mapping R → α is built at module load time using scipy's
# levy_stable (or a hardcoded fallback table if scipy is absent).
# See the baselines script for a more detailed explanation.
# ============================================================

class _StableQuantileCache:
    """Cached theoretical quantile grid for the McCulloch SαS estimator."""
    def __init__(self):
        self._have_scipy = False
        self.levy_stable = None
        try:
            from scipy.stats import levy_stable  # type: ignore
            self._have_scipy = True
            self.levy_stable = levy_stable
        except Exception:
            self._have_scipy = False
            self.levy_stable = None

        # fallback (alpha, R, IQR) for symmetric stable
        self.fallback = np.array([
            [2.00, 1.903, 1.349], [1.90, 2.020, 1.404], [1.80, 2.160, 1.472],
            [1.70, 2.330, 1.556], [1.60, 2.545, 1.662], [1.50, 2.820, 1.802],
            [1.40, 3.180, 2.000], [1.30, 3.670, 2.289], [1.20, 4.390, 2.781],
            [1.10, 5.560, 3.865], [1.00, 7.430, 6.314],
        ], dtype=float)

        self.cache_q: Dict[float, Tuple[float, float, float, float, float]] = {}
        self._grid_ready = False
        self._R_SORT = None
        self._A_SORT = None
        self._IQR_SORT = None

    def theo_quantiles(self, alpha: float) -> Tuple[float, float, float, float, float]:
        a = float(np.clip(alpha, 1.0, 2.0))
        key = round(a, 6)
        if key in self.cache_q:
            return self.cache_q[key]

        if self._have_scipy and self.levy_stable is not None:
            q = self.levy_stable.ppf([0.05, 0.25, 0.5, 0.75, 0.95], a, 0.0, loc=0.0, scale=1.0)
            out = tuple(float(x) for x in q)
        else:
            grid = self.fallback
            al = grid[:, 0]
            r = np.interp(a, al, grid[:, 1])
            iqr = np.interp(a, al, grid[:, 2])
            q25, q75 = -0.5 * iqr, 0.5 * iqr
            q95 = 0.5 * r * iqr
            q05 = -q95
            q50 = 0.0
            out = (float(q05), float(q25), float(q50), float(q75), float(q95))

        self.cache_q[key] = out
        return out

    def ensure_grid(self, n_grid: int = 201):
        if self._grid_ready:
            return
        alpha_grid = np.linspace(1.0, 2.0, int(n_grid))
        r_grid = np.empty_like(alpha_grid)
        iqr_grid = np.empty_like(alpha_grid)

        for i, a in enumerate(alpha_grid):
            q05, q25, _, q75, q95 = self.theo_quantiles(float(a))
            denom = (q75 - q25) + 1e-12
            r_grid[i] = (q95 - q05) / denom
            iqr_grid[i] = (q75 - q25)

        order = np.argsort(r_grid)
        self._R_SORT = r_grid[order]
        self._A_SORT = alpha_grid[order]
        self._IQR_SORT = iqr_grid[order]
        self._grid_ready = True

_STABLE_CACHE = _StableQuantileCache()
_STABLE_CACHE.ensure_grid(201)

def estimate_alpha_sigma_mcculloch_symmetric_from_quantiles(q05, q25, q75, q95) -> Tuple[float, float]:
    """Estimate (α̂, σ̂) for a symmetric stable distribution from empirical quantiles."""
    iqr = float(q75 - q25)
    if (not np.isfinite(iqr)) or (iqr <= 1e-12):
        return 2.0, 0.0

    r_hat = float((q95 - q05) / (iqr + 1e-12))

    R = _STABLE_CACHE._R_SORT
    A = _STABLE_CACHE._A_SORT
    IQR = _STABLE_CACHE._IQR_SORT
    assert R is not None and A is not None and IQR is not None

    r_hat_clamped = float(np.clip(r_hat, float(np.min(R)), float(np.max(R))))
    alpha_hat = float(np.interp(r_hat_clamped, R, A))
    iqr_theory = float(np.interp(r_hat_clamped, R, IQR))
    sigma_hat = float(iqr / (iqr_theory + 1e-12))
    return float(np.clip(alpha_hat, 1.0, 2.0)), float(max(0.0, sigma_hat))


# ============================================================
# ECF (Empirical Characteristic Function) estimator
# — Koutrouvelis (1980) regression for symmetric α-stable
# (see baselines script for detailed documentation)
# ============================================================

_MIN_SAMPLES_ALPHA = 500


def _ecf_at_t(samples: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Compute |φ̂(t)|² for each t in t_grid from real-valued samples."""
    n = samples.size
    if n == 0:
        return np.zeros_like(t_grid)

    chunk_size = min(n, 50000)
    total_cos = np.zeros(len(t_grid), dtype=np.float64)
    total_sin = np.zeros(len(t_grid), dtype=np.float64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x_chunk = samples[start:end]
        tx = np.outer(t_grid, x_chunk)
        total_cos += np.cos(tx).sum(axis=1)
        total_sin += np.sin(tx).sum(axis=1)

    phi2 = (total_cos / n) ** 2 + (total_sin / n) ** 2
    return phi2


def _choose_ecf_grid(samples: np.ndarray, n_points: int = 50) -> np.ndarray:
    """Choose t-grid in the informative region for ECF regression."""
    iqr = float(np.subtract(*np.percentile(samples, [75, 25])))
    if iqr <= 1e-12:
        iqr = float(np.std(samples)) * 1.349
    if iqr <= 1e-12:
        return np.linspace(0.1, 2.0, n_points)

    scale_est = iqr / 1.349
    t_lo = 0.05 / scale_est
    t_hi = 3.0 / scale_est
    return np.linspace(t_lo, t_hi, n_points)


def estimate_alpha_sigma_ecf_symmetric(samples: np.ndarray) -> Tuple[float, float]:
    """
    Estimate (α̂, σ̂) for symmetric α-stable using ECF regression
    (Koutrouvelis 1980, simplified for β=0).
    """
    n = samples.size
    if n < _MIN_SAMPLES_ALPHA:
        return 2.0, 0.0

    t_grid = _choose_ecf_grid(samples, n_points=50)
    phi2 = _ecf_at_t(samples, t_grid)

    mask = (phi2 > 0.01) & (phi2 < 0.95)
    if mask.sum() < 5:
        mask = (phi2 > 1e-4) & (phi2 < 0.999)
    if mask.sum() < 3:
        q05, q25, q75, q95 = np.quantile(samples, [0.05, 0.25, 0.75, 0.95])
        return estimate_alpha_sigma_mcculloch_symmetric_from_quantiles(q05, q25, q75, q95)

    t_use = t_grid[mask]
    phi2_use = phi2[mask]

    Y = np.log(-np.log(phi2_use))
    X = np.log(t_use)

    w = np.exp(-2.0 * (np.log(phi2_use) + 0.7) ** 2)
    w /= w.sum() + 1e-12

    Xbar = np.average(X, weights=w)
    Ybar = np.average(Y, weights=w)
    dx = X - Xbar
    dy = Y - Ybar
    alpha_hat = float(np.sum(w * dx * dy) / (np.sum(w * dx ** 2) + 1e-12))
    intercept = Ybar - alpha_hat * Xbar

    alpha_hat = float(np.clip(alpha_hat, 1.0, 2.0))
    if alpha_hat > 0:
        sigma_hat = float((np.exp(intercept) / 2.0) ** (1.0 / alpha_hat))
    else:
        sigma_hat = 0.0

    return alpha_hat, float(max(0.0, sigma_hat))


def estimate_alpha_sigma(
    samples: np.ndarray,
    method: str = "mcculloch",
    n_samples_for_ecf: int = 100000,
) -> Tuple[float, float, bool]:
    """Unified α̂ estimation with reliability checking."""
    n = samples.size

    if n < _MIN_SAMPLES_ALPHA:
        return 2.0, 0.0, False

    if method == "ecf":
        if n > n_samples_for_ecf:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, n_samples_for_ecf, replace=False)
            sub = np.asarray(samples[idx], dtype=np.float64)
        else:
            sub = np.asarray(samples, dtype=np.float64)
        alpha_hat, sigma_hat = estimate_alpha_sigma_ecf_symmetric(sub)
    else:
        q05, q25, q75, q95 = np.quantile(samples, [0.05, 0.25, 0.75, 0.95])
        alpha_hat, sigma_hat = estimate_alpha_sigma_mcculloch_symmetric_from_quantiles(
            q05, q25, q75, q95
        )

    reliable = True
    if sigma_hat <= 1e-12:
        reliable = False
    if (alpha_hat <= 1.01 or alpha_hat >= 1.99) and n < 2000:
        reliable = False
    iqr = float(np.subtract(*np.percentile(samples, [75, 25])))
    if iqr <= 1e-10:
        reliable = False

    return alpha_hat, sigma_hat, reliable


def compute_snr(alpha_hat: float, sigma_hat: float, mbar_Tmean: float, Nuse: int) -> float:
    """Compute empirical SNR(ℓ, N) = |m̄(ℓ)| · N^{1-1/α} / σ̂."""
    if sigma_hat <= 1e-12:
        return 0.0
    alpha_eff = max(1.0, float(alpha_hat))
    exp = 1.0 - 1.0 / alpha_eff
    return float(mbar_Tmean * (int(Nuse) ** exp) / float(sigma_hat))

def detection_error_on_prefix(T_seq_memmap: np.memmap, Nuse: int) -> float:
    """Coefficient of variation (std/|mean|) of the first Nuse matched-stat samples."""
    Nuse = max(1, int(Nuse))
    arr = np.asarray(T_seq_memmap[:Nuse], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    mu = float(np.mean(arr))
    sd = float(np.std(arr) + 1e-12)
    return float(sd / (abs(mu) + 1e-12))


# ============================================================
# Data generation (CPU resident)
#
# Identical task as the baselines script:
#   y_t = Σ_k c_k (u^T x_{t-ℓ_k}) + ε_t
# This ensures LSTM/GRU results are directly comparable to the
# ConstGate/SharedGate/DiagGate baselines.
# ============================================================

def make_dataset_cpu(Nseq: int, T: int, D: int,
                     task_lags: List[int],
                     task_coeffs: List[float],
                     noise_std: float,
                     u_vec: Optional[np.ndarray] = None):
    """
    Generate a synthetic multi-lag regression dataset on CPU.

    Returns (X, Y, u) where X is (Nseq, T, D), Y is (Nseq, T, 1),
    and u is the (D,) projection direction used.
    """
    if u_vec is None:
        u = np.random.randn(D).astype(np.float32)
        u = u / (np.linalg.norm(u) + 1e-12)
    else:
        u = u_vec.astype(np.float32)

    X = np.random.randn(Nseq, T, D).astype(np.float32)
    Y = np.zeros((Nseq, T, 1), dtype=np.float32)

    for k, lag in enumerate(task_lags):
        c = float(task_coeffs[k])
        if lag < T:
            proj = np.einsum("ntd,d->nt", X[:, :T - lag, :], u)
            Y[:, lag:, 0] += c * proj

    Y += noise_std * np.random.randn(Nseq, T, 1).astype(np.float32)

    Xt = torch.from_numpy(X)  # CPU
    Yt = torch.from_numpy(Y)  # CPU
    return Xt, Yt, u


# ============================================================
# Windowing via prefix sums (batch-local)
#
# For GRU and LSTM the memory kernel factorises into products and
# sums of per-step terms.  These are computed efficiently via
# cumulative sums in log-space (for products) and direct space
# (for first-order corrections), then sliced for each lag ℓ.
#
# _prefix_log:  cumulative sum of log(x), for computing
#               Π x[j] = exp(cs_log[t2] - cs_log[t1]).
# _prefix_sum:  cumulative sum of x, for computing Σ x[j].
# _win_prod_from_cs: extract ℓ-step product from prefix log.
# _win_sum_from_cs:  extract ℓ-step sum from prefix sum.
# ============================================================

def _prefix_log(x: torch.Tensor) -> torch.Tensor:
    """Cumulative sum of log(x) with a leading zero, shape (B, T+1, H)."""
    x64 = torch.clamp(x.double(), 1e-12, 1.0)
    logx = torch.log(x64)
    cs = torch.zeros(x.shape[0], x.shape[1] + 1, x.shape[2], dtype=torch.float64, device=x.device)
    cs[:, 1:, :] = torch.cumsum(logx, dim=1)
    return cs

def _prefix_sum(x: torch.Tensor) -> torch.Tensor:
    """Cumulative sum of x with a leading zero, shape (B, T+1, H)."""
    cs = torch.zeros(x.shape[0], x.shape[1] + 1, x.shape[2], dtype=torch.float64, device=x.device)
    cs[:, 1:, :] = torch.cumsum(x.double(), dim=1)
    return cs

def _win_prod_from_cs(cs_log: torch.Tensor, ell: int, out_dtype: torch.dtype) -> torch.Tensor:
    """Extract ℓ-step product from prefix log-sum: Π x[t-ℓ+1..t]."""
    B, Tp1, H = cs_log.shape
    T = Tp1 - 1
    if ell <= 0 or ell >= T:
        return torch.zeros(B, 0, H, dtype=out_dtype, device=cs_log.device)
    log_prod = cs_log[:, (ell + 1):(T + 1), :] - cs_log[:, 1:(T - ell + 1), :]
    return torch.exp(log_prod).to(out_dtype)

def _win_sum_from_cs(cs_sum: torch.Tensor, ell: int, out_dtype: torch.dtype) -> torch.Tensor:
    """Extract ℓ-step sum from prefix cumsum: Σ x[t-ℓ+1..t]."""
    B, Tp1, H = cs_sum.shape
    T = Tp1 - 1
    if ell <= 0 or ell >= T:
        return torch.zeros(B, 0, H, dtype=out_dtype, device=cs_sum.device)
    s = cs_sum[:, (ell + 1):(T + 1), :] - cs_sum[:, 1:(T - ell + 1), :]
    return s.to(out_dtype)


# ============================================================
# Models (GRU, LSTM)
#
# Both models expose forward_with_intermediates(x, return_intermediates)
# which returns (y, hseq, diagnostics_dict).
#
# GRU diagnostics dict: {"z", "r", "leak", "rdiag"}
#   - z_t:     update gate (analogous to s_t in baselines).
#   - r_t:     reset gate.
#   - leak_t:  (1 - z_t), the per-step retention coefficient.
#   - rdiag_t: diagonal approximation of ∂h_t/∂h_{t-1} minus leak.
#              Captures the combined effect of z, r, and candidate
#              gate derivatives on the recurrent Jacobian.
#
# LSTM diagnostics dict: {"forget", "expr", "cdiag"}
#   - forget_t: forget gate f_t.
#   - expr_t:   "expression" factor e_t = o_t · (1 - tanh²(c_t)),
#               the derivative of h_t w.r.t. c_t.
#   - cdiag_t:  diagonal approximation of ∂c_t/∂h_{t-1}, combining
#               forget, input, and candidate gate derivatives.
#
# When return_intermediates=False, the model only computes y (no
# stacking of diagnostic tensors), saving memory during training.
# ============================================================

class BaseSeqModel(nn.Module):
    """Base class providing orthogonal init with _skip_orth support."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return self.forward_with_intermediates(x)

    def apply_orthogonal(self):
        """Orthogonal init for all Linear layers except those flagged _skip_orth."""
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight is not None and m.weight.ndim == 2:
                if getattr(m, '_skip_orth', False):
                    continue  # preserve deliberate init on gate layers
                nn.init.orthogonal_(m.weight)

class GRUModel(BaseSeqModel):
    """
    GRU with explicit diagonal Jacobian intermediates.

    Recurrence:
      z_t = σ(Wz x_t + Uz h_{t-1})           (update gate)
      r_t = σ(Wr x_t + Ur h_{t-1})           (reset gate)
      g_t = tanh(Wh x_t + Uh (r_t ⊙ h_{t-1})) (candidate)
      h_t = (1 - z_t) h_{t-1} + z_t g_t

    The diagonal rdiag approximation is:
      rdiag_t ≈ (g - h_prev) z'·diag(Uz) + z·g'·diag(Uh)·(r + h_prev·r'·diag(Ur))
    """
    def __init__(self, D: int, H: int, ln: bool = False):
        super().__init__()
        self.D, self.H = D, H
        self.Wz, self.Uz = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.Wr, self.Ur = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.Wh, self.Uh = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.ln_h = layernorm_if(ln, H)
        self.out = nn.Linear(H, 1)
        nn.init.zeros_(self.out.bias)

    def forward_with_intermediates(self, x: torch.Tensor, return_intermediates=True):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)

        if return_intermediates:
            uz_diag = torch.diagonal(self.Uz.weight)
            ur_diag = torch.diagonal(self.Ur.weight)
            uh_diag = torch.diagonal(self.Uh.weight)

        ys = []
        if return_intermediates:
            hs = []
            z_list, r_list, leak_list, rdiag_list = [], [], [], []

        for t in range(T):
            h_prev = h
            z = torch.sigmoid(self.Wz(x[:, t]) + self.Uz(h_prev))
            r = torch.sigmoid(self.Wr(x[:, t]) + self.Ur(h_prev))
            g = torch.tanh(self.ln_h(self.Wh(x[:, t]) + self.Uh(r * h_prev)))

            h = (1 - z) * h_prev + z * g
            y = self.out(h)
            ys.append(y)

            if return_intermediates:
                zprime = z * (1 - z)
                rprime = r * (1 - r)
                gprime = 1.0 - g**2
                rdiag = (g - h_prev) * zprime * uz_diag + z * gprime * uh_diag * (r + h_prev * rprime * ur_diag)

                hs.append(h)
                z_list.append(z)
                r_list.append(r)
                leak_list.append(1 - z)
                rdiag_list.append(rdiag)

        y_out = torch.stack(ys, dim=1)
        if not return_intermediates:
            return y_out, None, None
        return (
            y_out,
            torch.stack(hs, dim=1),
            {
                "z": torch.stack(z_list, dim=1),
                "r": torch.stack(r_list, dim=1),
                "leak": torch.stack(leak_list, dim=1),
                "rdiag": torch.stack(rdiag_list, dim=1),
            },
        )

class LSTMModel(BaseSeqModel):
    """
    LSTM with explicit diagonal Jacobian intermediates.

    Recurrence:
      i_t = σ(Wi x_t + Ui h_{t-1})           (input gate)
      f_t = σ(Wf x_t + Uf h_{t-1})           (forget gate)
      o_t = σ(Wo x_t + Uo h_{t-1})           (output gate)
      g_t = tanh(Wg x_t + Ug h_{t-1})        (candidate)
      c_t = f_t c_{t-1} + i_t g_t
      h_t = o_t tanh(c_t)

    Diagnostics:
      expr_t = o_t (1 - tanh²(c_t))     — derivative ∂h_t/∂c_t.
      cdiag_t ≈ c_{t-1}·f'·diag(Uf) + i·g'·diag(Ug) + g·i'·diag(Ui)
              — diagonal approximation of ∂c_t/∂h_{t-1}.
    """
    def __init__(self, D: int, H: int, ln: bool = False):
        super().__init__()
        self.D, self.H = D, H
        self.Wi, self.Ui = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.Wf, self.Uf = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.Wo, self.Uo = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.Wg, self.Ug = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.ln_cand = layernorm_if(ln, H)
        self.out = nn.Linear(H, 1)
        nn.init.zeros_(self.out.bias)

    def forward_with_intermediates(self, x: torch.Tensor, return_intermediates=True):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)
        c = torch.zeros(B, self.H, device=x.device)

        if return_intermediates:
            uf_diag = torch.diagonal(self.Uf.weight)
            ui_diag = torch.diagonal(self.Ui.weight)
            ug_diag = torch.diagonal(self.Ug.weight)

        ys = []
        if return_intermediates:
            hs = []
            f_list, e_list, cdiag_list = [], [], []

        for t in range(T):
            h_prev, c_prev = h, c

            i = torch.sigmoid(self.Wi(x[:, t]) + self.Ui(h_prev))
            f = torch.sigmoid(self.Wf(x[:, t]) + self.Uf(h_prev))
            o = torch.sigmoid(self.Wo(x[:, t]) + self.Uo(h_prev))
            g = torch.tanh(self.ln_cand(self.Wg(x[:, t]) + self.Ug(h_prev)))

            c = f * c_prev + i * g
            tanh_c = torch.tanh(c)
            h = o * tanh_c
            y = self.out(h)
            ys.append(y)

            if return_intermediates:
                e = o * (1.0 - tanh_c**2)

                term_f = c_prev * (f * (1 - f)) * uf_diag
                term_g = i * (1 - g**2) * ug_diag
                term_i = g * (i * (1 - i)) * ui_diag
                cdiag = term_f + term_g + term_i

                hs.append(h)
                f_list.append(f)
                e_list.append(e)
                cdiag_list.append(cdiag)

        y_out = torch.stack(ys, dim=1)
        if not return_intermediates:
            return y_out, None, None
        return (
            y_out,
            torch.stack(hs, dim=1),
            {
                "forget": torch.stack(f_list, dim=1),
                "expr": torch.stack(e_list, dim=1),
                "cdiag": torch.stack(cdiag_list, dim=1),
            },
        )

def build_model(name: str, D: int, H: int, ln: bool) -> BaseSeqModel:
    """Instantiate a GRU or LSTM model by name."""
    name = name.lower().strip()
    if name == "gru":
        return GRUModel(D, H, ln=ln)
    if name == "lstm":
        return LSTMModel(D, H, ln=ln)
    raise ValueError(f"Unknown model {name}")


# ============================================================
# Training (CPU→GPU streaming) + learning curve CSV
#
# Data stays on CPU (pinned memory on CUDA); each mini-batch is
# transferred to GPU via non_blocking H2D copies.  This avoids
# OOM for large Nseq × T × D datasets.
#
# Learning curves (train loss, train R², val loss, val R²) are
# logged to a CSV each epoch.  Validation data is freshly sampled
# every epoch using the same task direction u_vec.
# ============================================================

def train_model(args, model: nn.Module,
                Xtr_cpu: torch.Tensor, Ytr_cpu: torch.Tensor,
                outdir: str, model_name: str, device: torch.device,
                u_vec: Optional[np.ndarray]) -> None:
    """
    Train a GRU/LSTM model with streaming mini-batches (CPU→GPU).

    Writes <outdir>/<model_name>_learning_curve.csv with columns:
        epoch, train_loss, train_acc (R²), val_loss, val_acc (R²).
    Optionally logs periodic gate statistics to gate_stats_<model>.csv.
    Halts early if NaN/Inf loss is detected.
    """
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd_momentum":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError("Unknown optimizer")

    Btot = int(Xtr_cpu.shape[0])
    bs = int(args.batch_size)
    n_batches = max(1, math.ceil(Btot / bs))

    if args.orth_init:
        model.apply_orthogonal()

    log(f"[train:{model_name}] start: epochs={args.epochs} bs={bs} opt={args.optimizer} lr={args.lr}")

    every = max(1, args.epochs // 5)  # ~5 checkpoints

    # --- learning curve files (per model)
    lc_csv = os.path.join(outdir, f"{model_name}_learning_curve.csv")
    with open(lc_csv, "w", newline="") as lf:
        wlc = csv.writer(lf)
        # NOTE: train_acc / val_acc store R^2 (regression-friendly)
        wlc.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    nan_halt = False
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(Btot)  # CPU indices

        loss_sum = 0.0
        n_seen = 0

        for bi in range(n_batches):
            lo = bi * bs
            hi = min(Btot, (bi + 1) * bs)
            idx = perm[lo:hi]

            xb = Xtr_cpu[idx].to(device, non_blocking=True)
            yb = Ytr_cpu[idx].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            yhat, _, _ = model.forward_with_intermediates(xb, return_intermediates=False)
            loss = F.mse_loss(yhat, yb)

            if not torch.isfinite(loss):
                log(f"[train:{model_name}] NaN/Inf loss at epoch={ep+1}, batch={bi}. Halting.")
                nan_halt = True
                del xb, yb, yhat, loss
                break

            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            loss_sum += float(loss.item()) * int(hi - lo)
            n_seen += int(hi - lo)

            del xb, yb, yhat, loss

        train_loss_epoch = loss_sum / max(1, n_seen)

        if nan_halt:
            break

        # ---- training "accuracy": R^2 on a fresh train subset (keeps overhead modest)
        n_train_eval = int(min(1024, Btot))
        idx_eval = perm[:n_train_eval]
        Xtr_eval = Xtr_cpu[idx_eval]
        Ytr_eval = Ytr_cpu[idx_eval]
        tr_mse_eval, tr_r2_eval = _eval_streaming_mse_and_r2(model, Xtr_eval, Ytr_eval, device=device, batch_size=bs)

        # ---- validation: re-sample every epoch (no fixed validation set)
        # Use same u_vec (task direction) to keep the task definition consistent.
        n_val = int(min(1024, max(bs, 256)))
        Xv_cpu, Yv_cpu, _ = make_dataset_cpu(
            n_val, args.T, args.D,
            args.task_lags, args.task_coeffs, args.noise_std,
            u_vec=u_vec
        )
        if device.type == "cuda":
            Xv_cpu = Xv_cpu.pin_memory()
            Yv_cpu = Yv_cpu.pin_memory()

        va_mse, va_r2 = _eval_streaming_mse_and_r2(model, Xv_cpu, Yv_cpu, device=device, batch_size=bs)

        with open(lc_csv, "a", newline="") as lf:
            wlc = csv.writer(lf)
            wlc.writerow([ep + 1, float(train_loss_epoch), float(tr_r2_eval), float(va_mse), float(va_r2)])

        if (ep == 0) or ((ep + 1) % every == 0) or (ep == args.epochs - 1):
            log(f"[train:{model_name}] epoch {ep+1}/{args.epochs} avg_loss={train_loss_epoch:.4g} "
                f"train_R2={tr_r2_eval:.3f} val_R2={va_r2:.3f}")

        # optional gate stats logging (kept compatible)
        if args.log_gate_stats and ((ep % args.gate_log_every) == 0):
            with torch.no_grad():
                idx0 = perm[:min(Btot, bs)]
                xb0 = Xtr_cpu[idx0].to(device, non_blocking=True)
                _, _, gdbg = model.forward_with_intermediates(xb0)
                rows = []
                if model_name == "gru":
                    rows.extend([
                        ("z_mean", float(gdbg["z"].mean().item())),
                        ("r_mean", float(gdbg["r"].mean().item())),
                        ("leak_mean", float(gdbg["leak"].mean().item())),
                        ("rdiag_mean", float(gdbg["rdiag"].mean().item())),
                    ])
                elif model_name == "lstm":
                    rows.extend([
                        ("forget_mean", float(gdbg["forget"].mean().item())),
                        ("expr_mean", float(gdbg["expr"].mean().item())),
                        ("cdiag_mean", float(gdbg["cdiag"].mean().item())),
                    ])

                gpath = os.path.join(outdir, f"gate_stats_{model_name}.csv")
                write_header = not os.path.exists(gpath)
                with open(gpath, "a", newline="") as gf:
                    w = csv.writer(gf)
                    if write_header:
                        w.writerow(["epoch", "metric", "value"])
                    for k, v in rows:
                        w.writerow([ep, k, v])

                del xb0, gdbg

    log(f"[train:{model_name}] done")


# ============================================================
# Per-batch prefix objects for μ windows (architecture-specific)
#
# The memory kernel μ^(q)(ℓ) requires ℓ-step products and sums of
# per-step diagonal terms.  We pre-compute cumulative arrays once
# per batch, then slice them for each lag ℓ in the diagnostic grid.
#
# GRU:  μ ≈ Π(1-z) + first-order correction via rdiag/(1-z)
#       We also track the reset gate r for potential future use.
# LSTM: μ ≈ Π f + first-order correction via cdiag·e_{t-1}/f_t
#       where e = o·(1-tanh²(c)) is the "expression" factor.
# ============================================================

def precompute_prefixes_gru(leak: torch.Tensor, reset: torch.Tensor, rdiag: torch.Tensor):
    """
    Build prefix arrays for GRU memory-kernel windows.

    Args:
        leak:  (B, T, H) retention coefficients (1 - z_t).
        reset: (B, T, H) reset gate r_t values.
        rdiag: (B, T, H) diagonal correction term.

    Returns:
        cs_log_leak:  prefix log-sum of leak (for ℓ-step product).
        cs_log_reset: prefix log-sum of reset (tracked for diagnostics).
        cs_log_eta:   prefix log-sum of leak*reset (composite kernel).
        cs_ratio:     prefix sum of rdiag/leak (first-order correction).
    """
    with torch.no_grad():
        leak64 = torch.clamp(leak.double(), 1e-12, 1.0)
        cs_log_leak = _prefix_log(leak)
        cs_log_reset = _prefix_log(torch.clamp(reset, 1e-12, 1.0))
        cs_log_eta = _prefix_log(torch.clamp(leak * reset, 1e-12, 1.0))

        ratio = (rdiag.double() / leak64).to(torch.float64)
        cs_ratio = _prefix_sum(ratio)

    return cs_log_leak, cs_log_reset, cs_log_eta, cs_ratio

def precompute_prefixes_lstm(forget: torch.Tensor, expr: torch.Tensor, cdiag: torch.Tensor):
    """
    Build prefix arrays for LSTM memory-kernel windows.

    For LSTM the cell-state kernel is  Π f_j  (product of forget gates),
    and the first-order correction involves  cdiag · e_{t-1} / f_t  where
    e = o·(1-tanh²(c)) maps cell-state perturbations back to hidden state.

    Args:
        forget: (B, T, H) forget gate f_t values.
        expr:   (B, T, H) expression factor e_t = o_t · (1 - tanh²(c_t)).
        cdiag:  (B, T, H) diagonal approximation of ∂c_t/∂h_{t-1}.

    Returns:
        cs_log_f: prefix log-sum of forget (for ℓ-step product).
        cs_ratio: prefix sum of cdiag·e_{t-1}/f_t (first-order correction).
    """
    with torch.no_grad():
        cs_log_f = _prefix_log(torch.clamp(forget, 1e-12, 1.0))

        # Shift expr by one timestep: e_{t-1} is needed at step t.
        e_shift = torch.zeros_like(expr)
        e_shift[:, 1:, :] = expr[:, :-1, :]
        f64 = torch.clamp(forget.double(), 1e-12, 1.0)
        ratio = (cdiag.double() * e_shift.double() / f64).to(torch.float64)
        cs_ratio = _prefix_sum(ratio)

    return cs_log_f, cs_ratio


# ============================================================
# Fit utilities
#
# These post-process the per-lag envelope and matched-statistic
# results into summary quantities used by the paper's figures:
#   - fit_exponential_tau: per-unit τ from |μ^(q)(ℓ)| decay
#   - fit_envelope_regimes: exponential vs power-law fit to f̂(ℓ)
#   - compute_H_N: learnability window H_N from N_required
# ============================================================

def fit_exponential_tau(ells, mu_vals, min_points: int = 5):
    """
    Fit exponential decay  μ(ℓ) = C · exp(-ℓ/τ)  in log-space.

    OLS on  log μ = a + b·ℓ  with τ = -1/b.
    Returns None if fewer than min_points have finite positive μ.
    """
    ells = np.asarray(ells, dtype=float)
    mu_vals = np.asarray(mu_vals, dtype=float)
    mask = np.isfinite(ells) & np.isfinite(mu_vals) & (ells > 0) & (mu_vals > 0)
    ells, mu_vals = ells[mask], mu_vals[mask]
    if ells.size < min_points:
        return None
    x, y = ells, np.log(mu_vals)
    A = np.vstack([x, np.ones_like(x)]).T
    b, a = np.linalg.lstsq(A, y, rcond=None)[0]
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - (a + b * x)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12) if ss_tot > 0 else float("nan")
    return {
        "tau": float(np.inf if b >= 0 else (-1.0 / b)),
        "C": float(np.exp(a)),
        "a": float(a),
        "b": float(b),
        "r2": float(r2),
        "num_points": int(ells.size),
    }

def fit_envelope_regimes(ells: np.ndarray, mu_vals: np.ndarray, log_mu_vals: np.ndarray) -> Dict:
    """
    Fit exponential and power-law models to the envelope f̂(ℓ).

    Exponential: log f̂(ℓ) = a + b·ℓ  →  τ_env = -1/b.
    Power-law:   log f̂(ℓ) = c + d·log(ℓ).

    Returns dict with sub-dicts "exp" and "power", each with
    fit coefficients and R² values.
    """
    mask = np.isfinite(log_mu_vals)
    ells_fit, log_mu_fit = ells[mask], log_mu_vals[mask]
    if ells_fit.size < 3:
        return {}

    ss_tot = float(np.sum((log_mu_fit - log_mu_fit.mean()) ** 2) + 1e-12)

    A_exp = np.vstack([np.ones_like(ells_fit), ells_fit]).T
    coeff_exp, _, _, _ = np.linalg.lstsq(A_exp, log_mu_fit, rcond=None)
    pred_exp = A_exp @ coeff_exp
    ss_res_exp = float(np.sum((log_mu_fit - pred_exp) ** 2))
    r2_exp = 1.0 - ss_res_exp / ss_tot
    b_exp = float(coeff_exp[1])

    log_ell = np.log(ells_fit.astype(float) + 1e-8)
    A_pow = np.vstack([np.ones_like(log_ell), log_ell]).T
    coeff_pow, _, _, _ = np.linalg.lstsq(A_pow, log_mu_fit, rcond=None)
    pred_pow = A_pow @ coeff_pow
    ss_res_pow = float(np.sum((log_mu_fit - pred_pow) ** 2))
    r2_pow = 1.0 - ss_res_pow / ss_tot

    return {
        "exp": {
            "a": float(coeff_exp[0]),
            "b": b_exp,
            "r2": float(r2_exp),
            "tau_env": float(-1.0 / b_exp) if b_exp < 0 else float("inf"),
        },
        "power": {
            "c": float(coeff_pow[0]),
            "d": float(coeff_pow[1]),
            "r2": float(r2_pow),
        },
    }

def compute_H_N(ells: List[int], Nreq_by_ell: Dict[int, int], N_values: List[int]) -> Dict[int, int]:
    """
    Compute the learnability window H_N for each training budget N.

    H_N = max{ℓ : N_required(ℓ) ≤ N}, i.e. the longest lag detectable
    with N samples.  Returns {N: H_N} dict.
    """
    H_by_N: Dict[int, int] = {}
    for N in N_values:
        valid = [ell for ell in ells if Nreq_by_ell.get(int(ell), -1) != -1 and Nreq_by_ell[int(ell)] <= N]
        H_by_N[int(N)] = int(max(valid)) if valid else 0
    return H_by_N


# ============================================================
# Diagnostics pipeline (streaming + memmap + cleanup)
#
# After training, this function runs the full learnability diagnostic
# pipeline on the diagnostic dataset (Xdg, Ydg).  The logic mirrors
# the baselines script but uses architecture-specific Jacobian terms:
#
#   GRU:  leak=(1-z), rdiag from z/r/candidate derivatives
#   LSTM: forget gate f, expression e, cdiag from f/i/g derivatives
#
# The matched statistic ψ samples are written to disk-backed memmap
# files (one per lag) and deleted as soon as quantiles are computed,
# keeping disk usage bounded.
# ============================================================

def run_for_model(args, model_name: str, mdir: str,
                  Xtr_cpu: torch.Tensor, Ytr_cpu: torch.Tensor,
                  Xdg_cpu: torch.Tensor, Ydg_cpu: torch.Tensor,
                  device: torch.device,
                  u_vec: Optional[np.ndarray]) -> Dict:
    """
    Train one model and run the full learnability diagnostic pipeline.

    Steps:
      1. Build and train the model.
      2. Stream diagnostic batches to extract Jacobian terms and JVP.
      3. For each lag ℓ, compute envelope μ and matched-stat ψ (→ memmap).
      4. Post-process: McCulloch α̂, σ̂, SNR, N_required per lag.
      5. Fit per-unit exponential τ from |μ^(q)(ℓ)|.

    Returns a dict with per-lag results: envelope values, tail indices,
    N_required, alpha/sigma estimates.
    """
    model = build_model(model_name, args.D, args.H, ln=args.layernorm).to(device)

    log(f"[run:{model_name}] train start")
    t_train0 = now_s()
    train_model(args, model, Xtr_cpu, Ytr_cpu, mdir, model_name, device=device, u_vec=u_vec)
    log(f"[run:{model_name}] train done  dt={now_s()-t_train0:.1f}s")

    model.eval()

    ells = np.linspace(args.lag_min, args.lag_max, args.num_lags, dtype=int)
    ells_list = [int(e) for e in ells]

    Bdg, Tdg, _ = Xdg_cpu.shape
    Hdim = int(args.H)

    memmap_paths: Dict[int, str] = {}
    memmaps: Dict[int, np.memmap] = {}
    write_offsets: Dict[int, int] = {}
    expected_sizes: Dict[int, int] = {}

    sum_mass: Dict[int, float] = {ell: 0.0 for ell in ells_list}
    sum_log_mass: Dict[int, float] = {ell: 0.0 for ell in ells_list}
    count_seq: Dict[int, int] = {ell: 0 for ell in ells_list}
    sum_unit: Dict[int, np.ndarray] = {ell: np.zeros(Hdim, dtype=np.float64) for ell in ells_list}

    sum_psi: Dict[int, float] = {ell: 0.0 for ell in ells_list}
    count_psi: Dict[int, int] = {ell: 0 for ell in ells_list}

    total_tmp_bytes = 0
    for ell in ells_list:
        n_samples = int(Bdg * max(0, (Tdg - ell)))
        expected_sizes[ell] = n_samples
        tmp_path = os.path.join(mdir, f"{model_name}_Tseq_ell{ell:04d}.tmp")
        memmap_paths[ell] = tmp_path
        write_offsets[ell] = 0
        memmaps[ell] = np.memmap(tmp_path, dtype=np.float64, mode="w+", shape=(n_samples,))
        total_tmp_bytes += n_samples * 8

    def cleanup_all_tmp():
        for ell in ells_list:
            try:
                if ell in memmaps:
                    memmaps[ell].flush()
            except Exception:
                pass
            try_remove(memmap_paths.get(ell, ""))

    try:
        log(f"[run:{model_name}] diag stream start: Bdg={Bdg} T={Tdg} H={Hdim} num_lags={len(ells_list)}")
        log(f"[diag:{model_name}] orient_matched_statistic_sign={int(bool(args.orient_matched_statistic_sign))}")
        log(f"[run:{model_name}] tmp memmaps created (~{total_tmp_bytes/1e9:.2f} GB on disk)")
        t_diag0 = now_s()

        Bb = min(int(args.diag_batch_size), int(Bdg))
        nb = int(math.ceil(Bdg / Bb))
        stepB = max(1, nb // 10)

        Wout = model.out.weight.detach()  # (1,H) on GPU

        for bi in range(nb):
            lo = bi * Bb
            hi = min(Bdg, (bi + 1) * Bb)

            if (bi == 0) or (bi == nb - 1) or ((bi + 1) % stepB == 0):
                log(f"[diag:{model_name}] batch {bi+1}/{nb} (seq {lo}:{hi})")

            xb = Xdg_cpu[lo:hi].to(device, non_blocking=True)
            yb = Ydg_cpu[lo:hi].to(device, non_blocking=True)

            with torch.no_grad():
                yhat, _, g = model.forward_with_intermediates(xb)

            vseq = compute_vseq_jvp(model, xb, w_seed=args.w_seed).detach()

            with torch.no_grad():
                err = (yhat[..., 0] - yb[..., 0])  # (Bb,T)
                delta = err.unsqueeze(-1) * Wout   # (Bb,T,H)

            with torch.no_grad():
                if model_name == "gru":
                    leak = g["leak"]
                    reset = g["r"]
                    rdiag = g["rdiag"]
                    cs_log_leak, cs_log_reset, cs_log_eta, cs_ratio = precompute_prefixes_gru(leak, reset, rdiag)
                else:
                    forget = g["forget"]
                    expr = g["expr"]
                    cdiag = g["cdiag"]
                    cs_log_f, cs_ratio = precompute_prefixes_lstm(forget, expr, cdiag)

            for ell in ells_list:
                if ell <= 0 or ell >= Tdg:
                    continue

                with torch.no_grad():
                    if model_name == "gru":
                        gamma0 = _win_prod_from_cs(cs_log_leak, ell, out_dtype=leak.dtype)
                        rho0   = _win_prod_from_cs(cs_log_reset, ell, out_dtype=leak.dtype)
                        eta0   = _win_prod_from_cs(cs_log_eta, ell, out_dtype=leak.dtype)
                        mu0 = gamma0 + rho0 + eta0

                        if bool(args.include_first_order_diag):
                            sum_ratio = _win_sum_from_cs(cs_ratio, ell, out_dtype=leak.dtype)
                            mu1 = gamma0 * sum_ratio
                            mu = mu0 + mu1
                        else:
                            mu = mu0
                    else:
                        prod_f = _win_prod_from_cs(cs_log_f, ell, out_dtype=forget.dtype)
                        expr_end = expr[:, ell:Tdg, :]
                        mu0 = expr_end * prod_f
                        if bool(args.include_first_order_diag):
                            sum_ratio = _win_sum_from_cs(cs_ratio, ell, out_dtype=forget.dtype)
                            mu1 = mu0 * sum_ratio
                            mu = mu0 + mu1
                        else:
                            mu = mu0

                with torch.no_grad():
                    if mu.numel() > 0:
                        abs_mu = torch.abs(mu).double()
                        mass_per_seq = abs_mu.mean(dim=2).mean(dim=1)  # (Bb,)
                        sum_mass[ell] += float(mass_per_seq.sum().item())
                        sum_log_mass[ell] += float(torch.log(mass_per_seq + 1e-30).sum().item())
                        count_seq[ell] += int(mass_per_seq.shape[0])
                        sum_unit[ell] += abs_mu.mean(dim=1).sum(dim=0).detach().cpu().numpy()

                with torch.no_grad():
                    delta_all = delta[:, ell:Tdg, :]             # (Bb,T-ell,H)
                    v_past_all = vseq[:, 0:(Tdg - ell), :]       # (Bb,T-ell,H)
                    psi = torch.sum(mu * delta_all * v_past_all, dim=2)  # (Bb,T-ell)

                    # ---------------------------------------------------------------------
                    # Matched-statistic sign orientation (global gauge; per lag)
                    #
                    # See baseline script for rationale. Summary:
                    #   - We optionally apply psi <- sgn(E[psi]) psi so the empirical mean is >= 0.
                    #   - This is a global orientation per lag (not per-neuron), used to ensure a
                    #     consistent matched-statistic convention across architectures and runs.
                    #   - Tail-index estimation is essentially reflection-invariant.
                    # ---------------------------------------------------------------------
                    if bool(args.orient_matched_statistic_sign):
                        mu_psi = psi.mean()
                        if torch.isfinite(mu_psi):
                            sgn = torch.sign(mu_psi)
                            if float(sgn.item()) == 0.0:
                                sgn = torch.tensor(1.0, device=psi.device)
                        else:
                            sgn = torch.tensor(1.0, device=psi.device)
                        psi = sgn * psi

                    arr = psi.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
                    off = write_offsets[ell]
                    memmaps[ell][off:off + arr.size] = arr
                    write_offsets[ell] = off + arr.size

                    sum_psi[ell] += float(arr.sum())
                    count_psi[ell] += int(arr.size)

            del xb, yb, yhat, g, vseq, err, delta
            if device.type == "cuda" and args.cuda_sync:
                torch.cuda.synchronize()

        for ell in ells_list:
            memmaps[ell].flush()

        # sanity check: wrote exactly expected size per ell
        for ell in ells_list:
            if write_offsets[ell] != expected_sizes[ell]:
                log(f"[warn:{model_name}] ell={ell}: write_offsets={write_offsets[ell]} expected={expected_sizes[ell]}")

        log(f"[run:{model_name}] diag stream done  dt={now_s()-t_diag0:.1f}s")
        log(f"[run:{model_name}] stats+write start")

        mu_by_ell: Dict[int, float] = {}
        log_mu_by_ell: Dict[int, float] = {}
        Nreq_by_ell: Dict[int, int] = {}
        alpha_by_ell: Dict[int, float] = {}
        sigma_by_ell: Dict[int, float] = {}
        mu_units_by_ell: Dict[int, np.ndarray] = {}

        # Update min samples threshold from CLI
        global _MIN_SAMPLES_ALPHA
        _MIN_SAMPLES_ALPHA = getattr(args, "min_samples_alpha", 500)
        alpha_method = getattr(args, "alpha_method", "mcculloch")

        summary_path = os.path.join(mdir, f"{model_name}_summary.csv")
        with open(summary_path, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow([
                "ell", "mu_l1_mean", "log_mu_l1_mean",
                "alpha_hat", "sigma_alpha_hat",
                "N_required_at_eps", "best_snr",
                "err_at_best_snr", "best_N_for_ell",
                "mbar_scalar", "alpha_reliable", "alpha_method", "n_samples",
            ])

            L = len(ells_list)
            stepL = max(1, L // 10)

            for i, ell in enumerate(ells_list):
                if (i == 0) or (i == L - 1) or ((i + 1) % stepL == 0):
                    log(f"[stats:{model_name}] ell progress {i+1}/{L} (current ell={ell})")

                if count_seq[ell] > 0:
                    mu_mean = float(sum_mass[ell] / count_seq[ell])
                    log_mu_mean = float(sum_log_mass[ell] / count_seq[ell])
                    mu_per_unit = (sum_unit[ell] / count_seq[ell]).astype(np.float64)
                else:
                    mu_mean = 0.0
                    log_mu_mean = float("-inf")
                    mu_per_unit = np.zeros(Hdim, dtype=np.float64)

                mu_by_ell[ell] = mu_mean
                log_mu_by_ell[ell] = log_mu_mean
                mu_units_by_ell[ell] = mu_per_unit

                n_samples = expected_sizes[ell]
                tmp_path = memmap_paths[ell]
                T_seq = np.memmap(tmp_path, dtype=np.float64, mode="r", shape=(n_samples,))

                if n_samples == 0:
                    alpha_hat, sigma_hat, alpha_reliable = 2.0, 0.0, False
                    mbar = 0.0
                else:
                    alpha_hat, sigma_hat, alpha_reliable = estimate_alpha_sigma(
                        np.asarray(T_seq), method=alpha_method
                    )
                    mean_raw = (sum_psi[ell] / max(1, count_psi[ell])) if count_psi[ell] > 0 else float(np.mean(T_seq))
                    mbar = float(abs(mean_raw))

                alpha_by_ell[ell] = float(alpha_hat)
                sigma_by_ell[ell] = float(sigma_hat)

                best_snr, best_err, best_N, N_required = -1e18, 1e18, None, -1
                for Nuse in args.N_grid:
                    Nuse_eff = min(int(Nuse), max(1, n_samples))
                    snr = compute_snr(alpha_hat, sigma_hat, mbar, Nuse_eff)
                    if (snr > args.eps) and (N_required == -1):
                        N_required = int(Nuse_eff)
                    if snr > best_snr:
                        best_snr = snr
                        best_err = detection_error_on_prefix(T_seq, int(Nuse_eff)) if n_samples > 0 else float("nan")
                        best_N = int(Nuse_eff)

                Nreq_by_ell[ell] = int(N_required)

                wcsv.writerow([
                    int(ell), float(mu_mean), float(log_mu_mean),
                    float(alpha_hat), float(sigma_hat),
                    int(N_required), float(best_snr),
                    float(best_err), int(best_N if best_N is not None else -1),
                    float(mbar), int(alpha_reliable), alpha_method, n_samples,
                ])

                del T_seq
                try_remove(tmp_path)

        sorted_ells = sorted(mu_units_by_ell.keys())
        mu_units_path = os.path.join(mdir, f"{model_name}_mu_units.csv")
        with open(mu_units_path, "w", newline="") as fmu:
            w = csv.writer(fmu)
            w.writerow(["ell"] + [f"mu_unit_{q}" for q in range(Hdim)])
            for e in sorted_ells:
                w.writerow([int(e)] + [float(v) for v in mu_units_by_ell[e]])

        # per-unit tau fits (always create outputs)
        tau_list = []
        tau_units_path = os.path.join(mdir, f"{model_name}_tau_from_mu_units.csv")
        with open(tau_units_path, "w", newline="") as f_tau:
            fieldnames = ["unit_id", "tau", "C", "a", "b", "r2", "num_points"]
            writer = csv.DictWriter(f_tau, fieldnames=fieldnames)
            writer.writeheader()
            for q in range(Hdim):
                mu_vals_q = np.array([mu_units_by_ell[e][q] for e in sorted_ells], dtype=float)
                fit_res = fit_exponential_tau(sorted_ells, np.abs(mu_vals_q), min_points=5)
                if fit_res is not None:
                    tau_list.append(fit_res["tau"])
                    row = {"unit_id": q, **fit_res}
                    writer.writerow({k: row.get(k) for k in fieldnames})

        # stats json
        tau_stats_path = os.path.join(mdir, f"{model_name}_tau_from_mu_stats.json")
        if len(tau_list) > 0:
            tau_arr = np.array(tau_list, dtype=float)
            with open(tau_stats_path, "w") as jf:
                json.dump({
                    "model": model_name,
                    "num_units": int(tau_arr.size),
                    "tau_min": float(np.min(tau_arr)),
                    "tau_max": float(np.max(tau_arr)),
                    "tau_mean": float(np.mean(tau_arr)),
                    "tau_std": float(np.std(tau_arr)),
                }, jf, indent=2)
        else:
            with open(tau_stats_path, "w") as jf:
                json.dump({
                    "model": model_name,
                    "num_units": 0,
                    "note": "No valid exponential fits (mu_unit too small/non-positive or insufficient points).",
                }, jf, indent=2)

        log(f"[run:{model_name}] stats+write done")

        return {
            "model": model_name,
            "ells": sorted_ells,
            "mu_by_ell": mu_by_ell,
            "log_mu_by_ell": log_mu_by_ell,
            "Nreq_by_ell": Nreq_by_ell,
            "alpha_by_ell": alpha_by_ell,
            "sigma_by_ell": sigma_by_ell,
        }

    except Exception:
        log(f"[ERROR:{model_name}] failed; cleaning temp files")
        traceback.print_exc()
        cleanup_all_tmp()
        raise
    finally:
        cleanup_all_tmp()


# ============================================================
# CLI
# ============================================================

def parse_args():
    """Parse command-line arguments for the LSTM/GRU learnability pipeline."""
    p = argparse.ArgumentParser(
        description="Learnability-window pipeline for LSTM and GRU (see paper)."
    )

    # --- Run identity -----------------------------------------------------------
    p.add_argument("--outdir", type=str, required=True,
                   help="Root output directory; per-model sub-dirs created automatically.")
    p.add_argument("--models", type=str, default="lstm,gru",
                   help="Comma-separated model names to train+diagnose.")

    p.add_argument("--seed", type=int, default=123)

    # --- Data geometry ----------------------------------------------------------
    p.add_argument("--Nseq_train", type=int, default=8000,
                   help="Number of training sequences.")
    p.add_argument("--Nseq_diag", type=int, default=8000,
                   help="Number of diagnostic sequences (separate from training).")

    p.add_argument("--T", type=int, default=1024,
                   help="Sequence length (timesteps).")
    p.add_argument("--D", type=int, default=16,
                   help="Input dimensionality.")
    p.add_argument("--H", type=int, default=64,
                   help="Hidden-state dimensionality.")

    # --- Optimizer --------------------------------------------------------------
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd", "sgd_momentum"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # --- Diagnostic lag grid ----------------------------------------------------
    p.add_argument("--lag_min", type=int, default=4)
    p.add_argument("--lag_max", type=int, default=128)
    p.add_argument("--num_lags", type=int, default=32)

    # --- Task definition (multi-lag regression) ---------------------------------
    # Same task as baselines script for direct comparability.
    p.add_argument("--task_lags", type=str, default="32,64,128,256,512",
                   help="Comma-separated task lag values ℓ_k.")
    p.add_argument("--task_coeffs", type=str, default="0.6,0.45,0.35,0.28,0.22",
                   help="Comma-separated coefficients c_k (one per task lag).")
    p.add_argument("--noise_std", type=float, default=0.35)

    # --- SNR / detectability ----------------------------------------------------
    p.add_argument("--N_grid", type=str, default="25,50,100,150,200,400,800,1600,3200,6400,12800",
                   help="Comma-separated training budgets to scan for N_required.")
    p.add_argument("--eps", type=float, default=0.1,
                   help="SNR detection threshold: lag is detectable when SNR > eps.")

    # --- Alpha estimation -------------------------------------------------------
    p.add_argument("--alpha_method", type=str, default="mcculloch",
                   choices=["mcculloch", "ecf"],
                   help=(
                       "Method for estimating the stable tail index α̂. "
                       "'mcculloch': McCulloch (1986) quantile ratio method. "
                       "'ecf': Koutrouvelis (1980) ECF regression. "
                       "Default: mcculloch."
                   ))
    p.add_argument("--min_samples_alpha", type=int, default=500,
                   help=(
                       "Minimum samples required for a reliable α̂ estimate. "
                       "Default: 500."
                   ))

    # --- JVP / matched statistic ------------------------------------------------
    p.add_argument("--w_seed", type=int, default=12345,
                   help="Seed for the random tangent direction w in JVP computation.")
    p.add_argument("--include_first_order_diag", type=int, default=1,
                   help="If 1, include first-order correction in matched-stat kernel.")

    # --- Init / normalisation switches ------------------------------------------
    p.add_argument("--orth_init", action="store_true",
                   help="Apply orthogonal init to recurrent weights (respects _skip_orth).")
    p.add_argument("--layernorm", action="store_true",
                   help="Enable LayerNorm on candidate pre-activation.")

    p.add_argument("--log_gate_stats", type=int, default=1)
    p.add_argument("--gate_log_every", type=int, default=10)

    # --- Device -----------------------------------------------------------------
    p.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "mps", "cuda"])

    # --- DGX-specific knobs (optional; safe defaults) ---------------------------
    p.add_argument("--diag_batch_size", type=int, default=128,
                   help="Diagnostic batch size for streaming.")
    p.add_argument("--diag_log_every", type=int, default=10,
                   help="Print diag progress every N diag batches.")
    p.add_argument("--cuda_sync", type=int, default=0,
                   help="If 1, synchronize CUDA each diag batch (debug).")

    # --- Matched-statistic sign orientation (see theory note in run_for_model) --
    p.add_argument(
        "--orient_matched_statistic_sign",
        type=int,
        default=1,
        help=(
            "If 1, orient matched-statistic samples per lag by sign(mean(psi)). "
            "This applies a *global gauge* flip psi <- sgn(E[psi]) psi so the empirical mean "
            "is nonnegative. Kept as a switch for ablations; reflection does not materially "
            "affect symmetric tail-index estimation."
        )
    )

    args = p.parse_args()

    args.task_lags = [int(s) for s in args.task_lags.split(",") if s.strip() != ""]
    args.task_coeffs = [float(s) for s in args.task_coeffs.split(",") if s.strip() != ""]
    args.N_grid = [int(s) for s in args.N_grid.split(",") if s.strip() != ""]

    args.cuda_sync = bool(int(args.cuda_sync))
    return args

def resolve_device(requested: str) -> torch.device:
    """Map 'auto'/'cpu'/'cuda'/'mps' to a concrete torch.device."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Main entry point
# ============================================================

def main():
    """
    Run the full learnability pipeline for LSTM/GRU models.

    Steps:
      1. Parse CLI, set seed, resolve device.
      2. Generate training and diagnostic datasets (CPU, shared u_vec).
      3. For each model in --models:
         a. Train → diagnostics → per-lag statistics.
         b. Fit envelope regime (exp vs power-law).
         c. Compute learnability window H_N.
      4. Write aggregate H_N summary CSV.
    """
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    save_args_to_csv(args, os.path.join(args.outdir, "cli_args.csv"))

    set_seed(args.seed)

    device = resolve_device(args.device)
    log(f"Running on device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(f"GPU: {props.name}")

    # CPU datasets (pinned for fast H2D transfers on CUDA).
    # Training and diagnostic sets share the same task direction u_vec
    # so the target function is identical across sets.
    Xtr_cpu, Ytr_cpu, u_vec = make_dataset_cpu(
        args.Nseq_train, args.T, args.D, args.task_lags, args.task_coeffs, args.noise_std, u_vec=None
    )
    Xdg_cpu, Ydg_cpu, _ = make_dataset_cpu(
        args.Nseq_diag, args.T, args.D, args.task_lags, args.task_coeffs, args.noise_std, u_vec=u_vec
    )

    log(f"Train set CPU: X={tuple(Xtr_cpu.shape)} Y={tuple(Ytr_cpu.shape)}")
    log(f"Diag  set CPU: X={tuple(Xdg_cpu.shape)} Y={tuple(Ydg_cpu.shape)}")

    if device.type == "cuda":
        Xtr_cpu = Xtr_cpu.pin_memory()
        Ytr_cpu = Ytr_cpu.pin_memory()
        Xdg_cpu = Xdg_cpu.pin_memory()
        Ydg_cpu = Ydg_cpu.pin_memory()

    models = [m.strip().lower() for m in args.models.split(",") if m.strip() != ""]
    results = []

    for mname in models:
        mdir = os.path.join(args.outdir, mname)
        os.makedirs(mdir, exist_ok=True)
        log(f"[main] start model={mname} -> {mdir}")

        # Train + run full diagnostic pipeline for this model
        res = run_for_model(args, mname, mdir, Xtr_cpu, Ytr_cpu, Xdg_cpu, Ydg_cpu, device=device, u_vec=u_vec)

        # Fit competing envelope decay regimes (exponential vs power-law)
        ells = np.array(res["ells"], dtype=int)
        mu_vals = np.array([res["mu_by_ell"][int(e)] for e in ells], dtype=float)
        log_mu_vals = np.array([res["log_mu_by_ell"][int(e)] for e in ells], dtype=float)

        fit_info = fit_envelope_regimes(ells, mu_vals, log_mu_vals)
        with open(os.path.join(mdir, f"{mname}_envelope_fits.json"), "w") as jf:
            json.dump(fit_info, jf, indent=2)

        # Compute learnability window H_N = max detectable lag given N samples
        H_by_N = compute_H_N(res["ells"], res["Nreq_by_ell"], args.N_grid)
        with open(os.path.join(mdir, f"{mname}_H_N.csv"), "w", newline="") as hf:
            wcsv = csv.writer(hf)
            wcsv.writerow(["N", "H_N"])
            for N, HN in sorted(H_by_N.items()):
                wcsv.writerow([int(N), int(HN)])

        res["H_by_N"] = H_by_N
        results.append(res)

        log(f"[main] done model={mname}")

    # Write aggregate H_N summary: one row per N, one column per model
    with open(os.path.join(args.outdir, "H_N_summary.csv"), "w", newline="") as hf:
        wcsv = csv.writer(hf)
        wcsv.writerow(["N"] + [f"H_N_{m}" for m in models])
        for N in args.N_grid:
            row = [int(N)]
            for res in results:
                row.append(int(res["H_by_N"].get(int(N), 0)))
            wcsv.writerow(row)

    log(f"Done. Results saved to: {args.outdir}")


if __name__ == "__main__":
    main()