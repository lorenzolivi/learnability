#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Empirical learnability-window pipeline — BASELINE gated RNNs
#
# Reference: "Learnability Window in Gated Recurrent Neural Networks"
#            (see the paper for theoretical background and notation).
#
# Target models: ConstGateRNN, SharedGateRNN, DiagGateRNN.
#
# This script trains each baseline model on a synthetic multi-lag regression
# task (y_t = Σ_k c_k u^T x_{t-ℓ_k} + noise), then runs the full
# learnability diagnostic pipeline from the paper:
#
#   1. Train model via streaming CPU→GPU batches.
#   2. For each diagnostic lag ℓ in a grid:
#      a. Compute the diagonal Jacobian factorisation:
#           leak_t = (1 - s_t),   rdiag_t = diag(∂h_t / ∂h_{t-1}) - leak_t
#         These give the "memory kernel" μ^(q)(ℓ) via prefix-sum products.
#      b. Compute the JVP sensitivity  v_t = (∂h_t/∂θ)[w]  along a random
#         unit direction w (fixed across lags for comparability).
#      c. Form the matched statistic  ψ_t(ℓ) = Σ_q μ^(q)_t(ℓ) δ_t^(q) v_{t-ℓ}^(q)
#         where δ_t = (ŷ_t - y_t) W_out  is the output-projected error.
#      d. Estimate the tail index α̂(ℓ) of {ψ_t(ℓ)} via the McCulloch
#         quantile method (symmetric stable assumption).
#      e. Compute the empirical SNR and N_required(ℓ) at a given ε threshold.
#   3. Aggregate results into the learnability window H_N.
#   4. Fit exponential time-scale τ from per-unit envelope μ^(q)(ℓ).
#
# GPU-memory safety:
#   - Datasets stay on CPU (pinned); batches streamed to GPU.
#   - Per-ℓ matched-statistic samples written to temporary memmap files on
#     disk (deleted as soon as quantiles are computed).
#
# Outputs (per model, in <outdir>/<model>/):
#   - <model>_learning_curve.csv       epoch-level train/val loss and R²
#   - <model>_summary.csv              per-lag: μ, α̂, σ̂, N_required, SNR
#   - <model>_mu_units.csv             per-lag × per-unit envelope values
#   - <model>_tau_from_mu_units.csv    per-unit exponential τ fits
#   - <model>_tau_from_mu_stats.json   summary statistics for τ distribution
#   - <model>_envelope_fits.json       exponential and power-law fits to μ(ℓ)
#   - <model>_H_N.csv                  learnability window vs training budget
#   - gate_stats_<model>.csv           periodic gate activation statistics
#
# Outputs (aggregate, in <outdir>/):
#   - H_N_summary.csv                  all models' H_N side-by-side
#   - cli_args.csv                     full CLI arguments for reproducibility
#
# Code organisation (top → bottom):
#   1. Small utilities (logger, seed, layernorm switch)
#   2. Streaming evaluation helper
#   3. JVP utilities (forward-mode AD for parameter sensitivity)
#   4. Memory-kernel prefix-sum helpers (μ computation)
#   5. McCulloch α-stable estimator
#   6. SNR and detection-error helpers
#   7. Time-scale fit (exponential τ)
#   8. Synthetic data generation
#   9. Model definitions (ConstGate, SharedGate, DiagGate)
#  10. Training loop (streaming CPU→GPU)
#  11. Diagnostics pipeline (run_for_model)
#  12. Envelope and H_N post-processing
#  13. CLI argument parser
#  14. main()
# =============================================================================

import argparse, os, math, csv, json, traceback, time
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp


# ============================================================
# Compact logger
# ============================================================

def log(msg: str):
    """Print a timestamped log message (flushed immediately)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def try_remove(path: str):
    """Silently remove a file if it exists (used for tmp memmap cleanup)."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ============================================================
# Utils
# ============================================================

def save_args_to_csv(args, filepath):
    """Dump all CLI arguments to a two-column CSV for reproducibility."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["argument", "value"])
        for k, v in vars(args).items():
            writer.writerow([k, v])


def set_seed(seed: int):
    """Set random seeds for numpy, torch CPU, and all CUDA devices."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def layernorm_if(enabled: bool, dim: int):
    """Return a LayerNorm module if enabled, otherwise nn.Identity (no-op)."""
    return nn.LayerNorm(dim) if enabled else nn.Identity()


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
    tensors (leak, rdiag, gate_s) during evaluation.

    R² = 1 - SSE/SST is the coefficient of determination; a value of 0
    means the model predicts the mean, and 1 means perfect prediction.
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
# These compute the Jacobian-vector product  v_t = (∂h_t/∂θ)[w]
# where w is a random unit-norm direction in parameter space.
# The JVP is computed via torch.func.jvp (forward-mode AD), which
# is memory-efficient: it does not materialise the full Jacobian.
# ============================================================

def _make_random_unit_w_pytree(model: nn.Module, device: torch.device, seed: int):
    """
    Build a random unit-norm tangent vector w in parameter space.

    Returns:
        params0: dict of trainable parameters (name -> tensor).
        buffers:  dict of model buffers (name -> tensor).
        w:        dict of tangent vectors, same structure as params0,
                  with ||w||₂ = 1 across all parameters jointly.
    The seed is fixed so the same direction is used across all lags/batches.
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
    Compute the hidden-state JVP sequence  v_t = (∂h_t/∂θ)[w].

    Args:
        model:   trained RNN model.
        X:       input tensor (B, T, D), already on device.
        w_seed:  seed for the random tangent direction w.

    Returns:
        vseq: tensor (B, T, H) of per-timestep JVP values.
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
# Memory-kernel window helpers (prefix-sum based)
#
# The learnability theory factorises the per-unit Jacobian as a product
# of per-step diagonal terms.  For the simple gated RNN:
#
#   ∂h_t^(q) / ∂h_{t-ℓ}^(q)  ≈  Π_{j=t-ℓ+1}^{t}  leak_j^(q)
#                                + first-order correction from rdiag
#
# where  leak_j = (1 - s_j)  is the retention coefficient and
#        rdiag_j = diag(∂h_j/∂h_{j-1}) - leak_j  captures the
#        recurrent-weight and gate-derivative contributions.
#
# We compute these ℓ-step products efficiently using cumulative sums
# in log-space (for the product) and a ratio accumulator (for the
# first-order correction).  This avoids the O(T×L) naive loop.
# ============================================================

def precompute_prefix_sums(leak: torch.Tensor, rdiag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build prefix-sum arrays for fast ℓ-step window computation.

    Args:
        leak:   (B, T, H) per-step retention coefficients (1 - s_t).
        rdiag:  (B, T, H) diagonal correction term.

    Returns:
        cs_log:   (B, T+1, H) cumulative sum of log(leak), for computing
                  products  Π leak[j]  via  exp(cs_log[t2] - cs_log[t1]).
        cs_ratio: (B, T+1, H) cumulative sum of rdiag/leak, for computing
                  the first-order correction  Σ (rdiag[j] / leak[j]).
    """
    with torch.no_grad():
        B, T, H = leak.shape
        device = leak.device

        leak64 = torch.clamp(leak.double(), 1e-12, 1.0)
        log_leak = torch.log(leak64)
        cs_log = torch.zeros(B, T + 1, H, dtype=torch.float64, device=device)
        cs_log[:, 1:, :] = torch.cumsum(log_leak, dim=1)

        ratio = (rdiag.double() / leak64).to(torch.float64)
        cs_ratio = torch.zeros(B, T + 1, H, dtype=torch.float64, device=device)
        cs_ratio[:, 1:, :] = torch.cumsum(ratio, dim=1)

        return cs_log, cs_ratio


def mu_for_matched_stat_from_prefix(cs_log: torch.Tensor, cs_ratio: torch.Tensor, ell: int, out_dtype: torch.dtype):
    """
    Extract the memory kernel μ(ℓ) for the matched-statistic computation.

    Uses a *shifted* window: product from step (t-ℓ+1) to step t, aligned
    so that mu[b, t, q] corresponds to the kernel connecting h_{t} back to
    h_{t-ℓ}.  This is the kernel used inside ψ_t(ℓ) = Σ_q μ δ v.

    Returns:
        mu0:  (B, T-ℓ, H) zero-order term (product of leaks).
        mu1:  (B, T-ℓ, H) first-order correction.
        mu:   (B, T-ℓ, H) total kernel  mu0 + mu1.
    """
    B, Tp1, H = cs_log.shape
    T = Tp1 - 1
    if ell <= 0 or ell >= T:
        z = torch.zeros(B, 0, H, dtype=out_dtype, device=cs_log.device)
        return z, z, z

    with torch.no_grad():
        log_prod = cs_log[:, (ell + 1):(T + 1), :] - cs_log[:, 1:(T - ell + 1), :]
        mu0 = torch.exp(log_prod).to(out_dtype)

        sum_ratio = cs_ratio[:, (ell + 1):(T + 1), :] - cs_ratio[:, 1:(T - ell + 1), :]
        mu1 = (mu0.double() * sum_ratio).to(out_dtype)

        return mu0, mu1, (mu0 + mu1)


def mu_for_envelope_from_prefix(cs_log: torch.Tensor, cs_ratio: torch.Tensor,
                               leak: torch.Tensor, rdiag: torch.Tensor,
                               ell: int, out_dtype: torch.dtype) -> torch.Tensor:
    """
    Extract the memory kernel μ(ℓ) for the envelope f̂(ℓ) computation.

    Uses an *unshifted* window starting from step 0, so we get the
    absolute magnitude of information retained over ℓ steps.  The
    envelope is computed as  f̂(ℓ) = mean over (batch, time, units)
    of |μ_envelope(ℓ)|.

    Returns:
        mu_env: (B, T-ℓ+1, H) total kernel (zero-order + first-order).
    """
    B, Tp1, H = cs_log.shape
    T = Tp1 - 1
    if ell <= 0 or ell > T:
        return torch.zeros(B, 0, H, dtype=out_dtype, device=cs_log.device)

    with torch.no_grad():
        if ell == 1:
            return (leak + rdiag).to(out_dtype)

        log_prod = cs_log[:, ell:(T + 1), :] - cs_log[:, 0:(T - ell + 1), :]
        mu0 = torch.exp(log_prod).to(out_dtype)

        sum_ratio = cs_ratio[:, ell:(T + 1), :] - cs_ratio[:, 0:(T - ell + 1), :]
        mu1 = (mu0.double() * sum_ratio).to(out_dtype)

        return mu0 + mu1


# ============================================================
# McCulloch quantile estimator for symmetric α-stable (SαS)
#
# The matched statistic ψ_t(ℓ) is modelled as a symmetric stable
# random variable with tail index α ∈ [1, 2].  The McCulloch method
# estimates α from the ratio of inter-quantile ranges:
#
#   R̂ = (q95 - q05) / (q75 - q25)
#
# We pre-compute a grid mapping R → α from the theoretical SαS
# quantiles (via scipy.stats.levy_stable if available, else a
# hardcoded fallback table), then invert via linear interpolation.
#
# The scale σ̂ is estimated from the IQR / theoretical IQR ratio.
# ============================================================

class _StableQuantileCache:
    """
    Cached theoretical quantile grid for the McCulloch SαS estimator.

    At module load time, builds a sorted lookup table mapping the
    quantile ratio R to tail index α.  This makes per-lag estimation
    a simple np.interp call (no scipy overhead in the hot loop).
    """
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

        # Fallback grid if SciPy is missing (won't be used on your DGX)
        self.fallback = np.array([
            [2.00, 1.903, 1.349],
            [1.90, 2.020, 1.404],
            [1.80, 2.160, 1.472],
            [1.70, 2.330, 1.556],
            [1.60, 2.545, 1.662],
            [1.50, 2.820, 1.802],
            [1.40, 3.180, 2.000],
            [1.30, 3.670, 2.289],
            [1.20, 4.390, 2.781],
            [1.10, 5.560, 3.865],
            [1.00, 7.430, 6.314],
        ], dtype=float)

        self.cache_q = {}
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
            out = (q05, q25, q50, q75, q95)

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
    """
    Estimate (α̂, σ̂) for a symmetric stable distribution from empirical quantiles.

    Args:
        q05, q25, q75, q95: sample quantiles at 5%, 25%, 75%, 95%.

    Returns:
        alpha_hat: estimated tail index in [1.0, 2.0].
        sigma_hat: estimated scale parameter (≥ 0).
    """
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
#
# For the symmetric stable (β=0, μ=0) case, the CF is:
#   φ(t) = exp(-σ^α |t|^α)
#
# Taking logs:
#   log(-log|φ̂(t)|²) = log(2σ^α) + α·log|t|
#
# This is a simple linear regression Y = c + α·X where
#   Y_k = log(-log|φ̂(t_k)|²), X_k = log|t_k|
# and the slope directly gives α̂.
#
# The grid of t-values is chosen in the "informative region"
# to avoid:
#   - t ≈ 0  where φ ≈ 1 and log(−log(·)) is numerically unstable
#   - t >> 1 where φ ≈ 0 and |φ̂|² is dominated by sampling noise
# ============================================================

# Minimum number of samples for a reliable α̂ estimate
_MIN_SAMPLES_ALPHA = 500


def _ecf_at_t(samples: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """
    Compute |φ̂(t)|² for each t in t_grid from real-valued samples.

    For real symmetric distributions:
        φ̂(t) = (1/n) Σ_j exp(i·t·x_j)
        |φ̂(t)|² = [(1/n)Σ cos(t·x)]² + [(1/n)Σ sin(t·x)]²

    We use chunked computation to avoid O(n_samples × n_grid) memory.

    Returns: 1-D array of |φ̂(t)|² values, shape (len(t_grid),).
    """
    n = samples.size
    if n == 0:
        return np.zeros_like(t_grid)

    phi2 = np.zeros(len(t_grid), dtype=np.float64)
    chunk_size = min(n, 50000)  # keep memory bounded

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x_chunk = samples[start:end]
        # outer product: (len(t_grid), chunk_size)
        tx = np.outer(t_grid, x_chunk)
        cos_sum = np.cos(tx).sum(axis=1)
        sin_sum = np.sin(tx).sum(axis=1)
        if start == 0:
            total_cos = cos_sum
            total_sin = sin_sum
        else:
            total_cos += cos_sum
            total_sin += sin_sum

    phi2 = (total_cos / n) ** 2 + (total_sin / n) ** 2
    return phi2


def _choose_ecf_grid(samples: np.ndarray, n_points: int = 50) -> np.ndarray:
    """
    Choose a grid of t-values in the informative region for ECF regression.

    Strategy: t should be in a range where |φ(t)|² is between ~0.01 and ~0.95.
    For a symmetric stable with scale σ:
        |φ(t)|² = exp(-2σ^α |t|^α)
    So |φ|² ≈ 0.95 when t ≈ (0.025/σ^α)^{1/α}
    and |φ|² ≈ 0.01 when t ≈ (2.3/σ^α)^{1/α}

    We use the IQR as a robust scale estimate to set the range.
    """
    iqr = float(np.subtract(*np.percentile(samples, [75, 25])))
    if iqr <= 1e-12:
        iqr = float(np.std(samples)) * 1.349  # Gaussian IQR from std
    if iqr <= 1e-12:
        return np.linspace(0.1, 2.0, n_points)

    # Rough scale: for Gaussian, IQR ≈ 1.349σ, so σ ≈ IQR/1.349
    scale_est = iqr / 1.349

    # t range: from ~0.05/scale to ~3/scale (covers the informative region)
    t_lo = 0.05 / scale_est
    t_hi = 3.0 / scale_est

    return np.linspace(t_lo, t_hi, n_points)


def estimate_alpha_sigma_ecf_symmetric(samples: np.ndarray) -> Tuple[float, float]:
    """
    Estimate (α̂, σ̂) for a symmetric α-stable distribution using the ECF
    regression method (Koutrouvelis 1980, simplified for β=0).

    For SαS: log(-log|φ̂(t)|²) = log(2σ^α) + α·log|t|

    The slope of the regression gives α̂; the intercept gives σ̂.

    Returns:
        alpha_hat: estimated tail index in [1.0, 2.0].
        sigma_hat: estimated scale parameter (≥ 0).
    """
    n = samples.size
    if n < _MIN_SAMPLES_ALPHA:
        return 2.0, 0.0

    t_grid = _choose_ecf_grid(samples, n_points=50)
    phi2 = _ecf_at_t(samples, t_grid)

    # Filter: keep only points where |φ̂|² is in a usable range
    # Too close to 1 → log(-log(·)) is unstable; too close to 0 → noise-dominated
    mask = (phi2 > 0.01) & (phi2 < 0.95)
    if mask.sum() < 5:
        # Relax bounds
        mask = (phi2 > 1e-4) & (phi2 < 0.999)
    if mask.sum() < 3:
        # Fall back to McCulloch
        q05, q25, q75, q95 = np.quantile(samples, [0.05, 0.25, 0.75, 0.95])
        return estimate_alpha_sigma_mcculloch_symmetric_from_quantiles(q05, q25, q75, q95)

    t_use = t_grid[mask]
    phi2_use = phi2[mask]

    # Regression: Y = log(-log(|φ̂(t)|²)),  X = log(|t|)
    Y = np.log(-np.log(phi2_use))
    X = np.log(t_use)

    # Weighted least squares: points near |φ̂|² ≈ 0.5 are most informative
    # Weight = exp(-2*(log|φ̂|² + 0.7)²)  peaks near |φ̂|² ≈ 0.5
    w = np.exp(-2.0 * (np.log(phi2_use) + 0.7) ** 2)
    w /= w.sum() + 1e-12

    # WLS: α̂ = Σw·(X-X̄)(Y-Ȳ) / Σw·(X-X̄)²
    Xbar = np.average(X, weights=w)
    Ybar = np.average(Y, weights=w)
    dx = X - Xbar
    dy = Y - Ybar
    alpha_hat = float(np.sum(w * dx * dy) / (np.sum(w * dx ** 2) + 1e-12))
    intercept = Ybar - alpha_hat * Xbar

    # σ̂ from intercept: log(2σ^α) = intercept → σ = (exp(intercept)/2)^{1/α}
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
    """
    Unified interface for α̂ estimation with reliability checking.

    Args:
        samples: 1-D array of matched-statistic values (float64).
        method: "mcculloch" or "ecf".
        n_samples_for_ecf: subsample limit for ECF (controls speed).

    Returns:
        alpha_hat: estimated tail index in [1.0, 2.0].
        sigma_hat: estimated scale parameter.
        reliable: True if the estimate passes quality checks.
    """
    n = samples.size

    # ── reliability check: too few samples ──
    if n < _MIN_SAMPLES_ALPHA:
        return 2.0, 0.0, False

    # ── compute estimate ──
    if method == "ecf":
        # Subsample if very large (ECF is O(n·K))
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

    # ── reliability checks ──
    reliable = True

    # Check 1: σ̂ should be positive
    if sigma_hat <= 1e-12:
        reliable = False

    # Check 2: α̂ at boundary is suspicious with few samples
    if (alpha_hat <= 1.01 or alpha_hat >= 1.99) and n < 2000:
        reliable = False

    # Check 3: IQR near zero → degenerate distribution
    iqr = float(np.subtract(*np.percentile(samples, [75, 25])))
    if iqr <= 1e-10:
        reliable = False

    return alpha_hat, sigma_hat, reliable


def compute_snr(alpha_hat: float, sigma_hat: float, mbar_Tmean: float, Nuse: int) -> float:
    """
    Compute the empirical signal-to-noise ratio for lag detection.

    SNR(ℓ, N) = |m̄(ℓ)| · N^{1 - 1/α} / σ̂

    where m̄ is the absolute mean of the matched statistic (signal strength),
    N is the number of samples, α is the tail index, and σ̂ is the scale.
    When SNR > ε, the lag ℓ is considered detectable with N samples.
    """
    if sigma_hat <= 1e-12:
        return 0.0
    alpha_eff = max(1.0, float(alpha_hat))
    exp = 1.0 - 1.0 / alpha_eff
    return float(mbar_Tmean * (Nuse ** exp) / float(sigma_hat))


def detection_error_on_prefix(T_seq_memmap: np.memmap, Nuse: int) -> float:
    """
    Coefficient of variation of the first Nuse matched-statistic samples.

    This is the empirical detection error: std(ψ) / |mean(ψ)|.
    A small value indicates the mean shift is large relative to noise.
    """
    Nuse = max(1, int(Nuse))
    arr = np.asarray(T_seq_memmap[:Nuse], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    mu = float(np.mean(arr))
    sd = float(np.std(arr) + 1e-12)
    return float(sd / (abs(mu) + 1e-12))


# ============================================================
# Time-scale fit from per-unit envelope μ^(q)(ℓ)
# ============================================================

def fit_exponential_tau(ells, mu_vals, min_points: int = 5):
    """
    Fit an exponential decay  μ(ℓ) = C · exp(-ℓ/τ)  in log-space.

    Performs OLS on  log μ = a + b·ℓ  and extracts τ = -1/b.
    Returns None if fewer than min_points have finite positive μ.

    Returns:
        dict with keys: tau, C, a, b, r2, num_points.
    """
    ells = np.asarray(ells, dtype=float)
    mu_vals = np.asarray(mu_vals, dtype=float)
    mask = np.isfinite(ells) & np.isfinite(mu_vals) & (ells > 0) & (mu_vals > 0)
    ells = ells[mask]
    mu_vals = mu_vals[mask]
    if ells.size < min_points:
        return None
    x = ells
    y = np.log(mu_vals)
    A = np.vstack([x, np.ones_like(x)]).T
    b, a = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = a + b * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    tau = np.inf if b >= 0 else (-1.0 / b)
    C = float(np.exp(a))
    return {"tau": float(tau), "C": float(C), "a": float(a), "b": float(b), "r2": float(r2), "num_points": int(ells.size)}


# ============================================================
# Data generation (CPU resident)
#
# The task is a multi-lag linear regression in D-dimensional
# input space:  y_t = Σ_k c_k (u^T x_{t-ℓ_k}) + ε_t
# where u is a fixed unit direction and ε_t ~ N(0, noise_std²).
# The difficulty is controlled by task_lags: lags well beyond
# the model's effective memory timescale τ are unlearnable.
# ============================================================

def make_dataset_cpu(Nseq: int, T: int, D: int,
                     task_lags: List[int],
                     task_coeffs: List[float],
                     noise_std: float,
                     u_vec: Optional[np.ndarray] = None):
    """
    Generate a synthetic multi-lag regression dataset on CPU.

    Args:
        Nseq:        number of independent sequences.
        T:           sequence length (timesteps).
        D:           input dimensionality.
        task_lags:   list of lag values [ℓ₁, ℓ₂, ...].
        task_coeffs: corresponding coefficients [c₁, c₂, ...].
        noise_std:   standard deviation of observation noise ε_t.
        u_vec:       (optional) fixed projection direction; if None,
                     a random unit vector is drawn and returned.

    Returns:
        X: (Nseq, T, D) float32 CPU tensor of i.i.d. Gaussian inputs.
        Y: (Nseq, T, 1) float32 CPU tensor of target outputs.
        u: (D,) float32 numpy array, the projection direction used.
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
# Models (ConstGate, SharedGate, DiagGate)
#
# All three share the same recurrence structure:
#   h_t = (1 - s_t) h_{t-1} + s_t tanh(Wx x_t + Wh h_{t-1})
#   y_t = W_out h_t
#
# They differ in how the gate s_t is parameterised:
#   ConstGateRNN:  s_t = s (scalar buffer, non-learnable)
#   SharedGateRNN: s_t = σ(Ws x_t + Us h_{t-1})  (scalar gate, shared across H)
#   DiagGateRNN:   s_t = σ(Ws x_t + Us h_{t-1})  (per-unit gate, H-dimensional)
#
# forward_with_intermediates returns (y, hseq, diagnostics_dict):
#   - y:     (B, T, 1) output predictions.
#   - hseq:  (B, T, H) hidden state sequence (None if return_intermediates=False).
#   - dict:  {"gate_s", "leak", "rdiag"} tensors for the learnability pipeline.
#            leak_t = 1 - s_t  is the per-step retention.
#            rdiag_t = diag(∂h_t/∂h_{t-1}) - leak_t  is the correction from
#            the recurrent weight and gate derivative.
#
# Initialization notes:
#   - Recurrent weight Wh: orthogonal init (via apply_orthogonal).
#   - Gate weights Ws, Us: zero-initialized with _skip_orth=True so
#     apply_orthogonal does not overwrite them.
#   - Gate bias: set to logit(init_s) so sigmoid(bias) ≈ init_s at t=0,
#     giving the model long initial timescales τ ≈ 1/init_s.
# ============================================================

class BaseRNN(nn.Module):
    """Base class providing orthogonal init with _skip_orth support."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, gate_rescale=None):
        return self.forward_with_intermediates(x, gate_rescale=gate_rescale)

    def apply_orthogonal(self):
        """Orthogonal init for all Linear layers except those flagged _skip_orth."""
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight is not None and m.weight.ndim == 2:
                if getattr(m, '_skip_orth', False):
                    continue  # preserve deliberate zero/bias init on gate layers
                nn.init.orthogonal_(m.weight)

    def get_const_gate_s(self):
        """Return fixed gate value s if applicable (ConstGateRNN only)."""
        return None


class ConstGateRNN(BaseRNN):
    """
    Gated RNN with a fixed (non-learnable) scalar gate s.

    The gate is a registered buffer, so it is not updated by the optimizer.
    This model serves as the theoretical baseline: its memory timescale
    τ = -1/log(1-s) is known in closed form.
    """
    def __init__(self, D: int, H: int, s: float = 0.7, ln: bool = False):
        super().__init__()
        self.D, self.H = D, H
        self.Wx = nn.Linear(D, H)
        self.Wh = nn.Linear(H, H, bias=False)
        self.ln = layernorm_if(ln, H)
        self.out = nn.Linear(H, 1)

        s = float(np.clip(s, 1e-6, 1.0 - 1e-6))
        self.register_buffer("s_const", torch.tensor(s, dtype=torch.float32))

        nn.init.zeros_(self.Wx.bias)
        nn.init.zeros_(self.out.bias)

    def get_const_gate_s(self):
        return float(self.s_const.item())

    def forward_with_intermediates(self, x: torch.Tensor, gate_rescale=None, return_intermediates=True):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)

        s = self.s_const
        if gate_rescale is not None:
            s = torch.clamp(s * gate_rescale, 0.0, 1.0)

        if return_intermediates:
            wh_diag = torch.diagonal(self.Wh.weight, 0)

        ys = []
        if return_intermediates:
            gates_s, leaks, rdiags, hs = [], [], [], []

        for t in range(T):
            h_prev = h
            pre = self.Wx(x[:, t]) + self.Wh(h_prev)
            pre = self.ln(pre)
            h_tilde = torch.tanh(pre)
            h = (1 - s) * h_prev + s * h_tilde
            y = self.out(h)
            ys.append(y)

            if return_intermediates:
                sH = s.expand(B, self.H)
                leak = 1 - sH
                tanh_prime = 1.0 - h_tilde**2
                rdiag = (sH * tanh_prime) * wh_diag.view(1, -1)

                hs.append(h)
                gates_s.append(sH)
                leaks.append(leak)
                rdiags.append(rdiag)

        y = torch.stack(ys, dim=1)
        if not return_intermediates:
            return y, None, None
        hseq = torch.stack(hs, dim=1)
        gate_s = torch.stack(gates_s, dim=1)
        leak = torch.stack(leaks, dim=1)
        rdiag = torch.stack(rdiags, dim=1)
        return y, hseq, {"gate_s": gate_s, "leak": leak, "rdiag": rdiag}


class SharedGateRNN(BaseRNN):
    """
    Gated RNN with a learnable scalar gate shared across all H units.

    Gate: s_t = σ(Ws x_t + Us h_{t-1}) ∈ ℝ¹, broadcast to all H units.
    At initialization, gate weights are zero and bias = logit(init_s),
    so the gate starts near init_s and the model has long memory.
    """
    def __init__(self, D: int, H: int, ln: bool = False, init_s: float = 0.005):
        super().__init__()
        self.D, self.H = D, H
        self.Wx = nn.Linear(D, H)
        self.Wh = nn.Linear(H, H, bias=False)
        self.ln_h = layernorm_if(ln, H)

        self.Ws = nn.Linear(D, 1, bias=True)
        self.Us = nn.Linear(H, 1, bias=False)
        self.Ws._skip_orth = True   # gate layers: preserve zero-init on weights
        self.Us._skip_orth = True

        self.out = nn.Linear(H, 1)

        nn.init.zeros_(self.Wx.bias)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.Ws.weight)
        nn.init.zeros_(self.Us.weight)

        # Gate bias -> logit(init_s) so sigmoid(bias) = init_s at t=0.
        init_s = float(np.clip(init_s, 1e-6, 1.0 - 1e-6))
        gate_bias = float(np.log(init_s / (1.0 - init_s)))
        nn.init.constant_(self.Ws.bias, gate_bias)

    def forward_with_intermediates(self, x: torch.Tensor, gate_rescale=None, return_intermediates=True):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)

        if return_intermediates:
            wh_diag = torch.diagonal(self.Wh.weight, 0)
            us_vec = self.Us.weight.view(-1)

        ys = []
        if return_intermediates:
            gates_s, leaks, rdiags, hs = [], [], [], []

        for t in range(T):
            h_prev = h
            a_s = self.Ws(x[:, t]) + self.Us(h_prev)
            s = torch.sigmoid(a_s)
            if gate_rescale is not None:
                s = torch.clamp(s * gate_rescale, 0.0, 1.0)

            pre = self.Wx(x[:, t]) + self.Wh(h_prev)
            pre = self.ln_h(pre)
            h_tilde = torch.tanh(pre)

            sH = s.expand(B, self.H)
            h = (1 - sH) * h_prev + sH * h_tilde
            y = self.out(h)
            ys.append(y)

            if return_intermediates:
                leak = 1 - sH
                tanh_prime = 1.0 - h_tilde**2
                s_prime = (s * (1 - s)).expand(B, self.H)

                rdiag_gate = (h_tilde - h_prev) * (s_prime * us_vec.view(1, -1))
                rdiag_rec  = (sH * tanh_prime) * wh_diag.view(1, -1)
                rdiag = rdiag_gate + rdiag_rec

                hs.append(h)
                gates_s.append(sH)
                leaks.append(leak)
                rdiags.append(rdiag)

        y = torch.stack(ys, dim=1)
        if not return_intermediates:
            return y, None, None
        hseq = torch.stack(hs, dim=1)
        gate_s = torch.stack(gates_s, dim=1)
        leak = torch.stack(leaks, dim=1)
        rdiag = torch.stack(rdiags, dim=1)
        return y, hseq, {"gate_s": gate_s, "leak": leak, "rdiag": rdiag}


class DiagGateRNN(BaseRNN):
    """
    Gated RNN with a learnable per-unit (diagonal) gate.

    Gate: s_t = σ(Ws x_t + Us h_{t-1}) ∈ ℝᴴ, one gate per hidden unit.
    Each unit can learn its own timescale independently.
    Initialization is identical to SharedGateRNN (zero weights, biased gate).
    """
    def __init__(self, D: int, H: int, ln: bool = False, init_s: float = 0.005):
        super().__init__()
        self.D, self.H = D, H
        self.Wx = nn.Linear(D, H)
        self.Wh = nn.Linear(H, H, bias=False)
        self.ln_h = layernorm_if(ln, H)

        self.Ws = nn.Linear(D, H, bias=True)
        self.Us = nn.Linear(H, H, bias=False)
        self.Ws._skip_orth = True   # gate layers: preserve zero-init on weights
        self.Us._skip_orth = True

        self.out = nn.Linear(H, 1)

        nn.init.zeros_(self.Wx.bias)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.Ws.weight)
        nn.init.zeros_(self.Us.weight)

        # Gate bias -> logit(init_s) so sigmoid(bias) = init_s at t=0.
        init_s = float(np.clip(init_s, 1e-6, 1.0 - 1e-6))
        gate_bias = float(np.log(init_s / (1.0 - init_s)))
        nn.init.constant_(self.Ws.bias, gate_bias)

    def forward_with_intermediates(self, x: torch.Tensor, gate_rescale=None, return_intermediates=True):
        B, T, _ = x.shape
        h = torch.zeros(B, self.H, device=x.device)

        if return_intermediates:
            wh_diag = torch.diagonal(self.Wh.weight, 0)
            us_diag = torch.diagonal(self.Us.weight, 0)

        ys = []
        if return_intermediates:
            gates_s, leaks, rdiags, hs = [], [], [], []

        for t in range(T):
            h_prev = h
            a_s = self.Ws(x[:, t]) + self.Us(h_prev)
            s = torch.sigmoid(a_s)
            if gate_rescale is not None:
                s = torch.clamp(s * gate_rescale, 0.0, 1.0)

            pre = self.Wx(x[:, t]) + self.Wh(h_prev)
            pre = self.ln_h(pre)
            h_tilde = torch.tanh(pre)

            h = (1 - s) * h_prev + s * h_tilde
            y = self.out(h)
            ys.append(y)

            if return_intermediates:
                leak = 1 - s
                tanh_prime = 1.0 - h_tilde**2
                s_prime = s * (1 - s)

                rdiag_gate = (h_tilde - h_prev) * (s_prime * us_diag.view(1, -1))
                rdiag_rec  = (s * tanh_prime) * wh_diag.view(1, -1)
                rdiag = rdiag_gate + rdiag_rec

                hs.append(h)
                gates_s.append(s)
                leaks.append(leak)
                rdiags.append(rdiag)

        y = torch.stack(ys, dim=1)
        if not return_intermediates:
            return y, None, None
        hseq = torch.stack(hs, dim=1)
        gate_s = torch.stack(gates_s, dim=1)
        leak = torch.stack(leaks, dim=1)
        rdiag = torch.stack(rdiags, dim=1)
        return y, hseq, {"gate_s": gate_s, "leak": leak, "rdiag": rdiag}


def build_model(name: str, D: int, H: int, const_s: float, ln: bool) -> BaseRNN:
    """Instantiate a baseline model by name. const_s is used as both the
    ConstGateRNN's fixed gate value and SharedGate/DiagGate's initial gate value."""
    name = name.lower()
    if name == "const":
        return ConstGateRNN(D, H, s=const_s, ln=ln)
    if name == "shared":
        return SharedGateRNN(D, H, ln=ln, init_s=const_s)
    if name in ["diag", "multigate"]:
        return DiagGateRNN(D, H, ln=ln, init_s=const_s)
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

def train_model(args, model: BaseRNN,
                Xtr_cpu: torch.Tensor, Ytr_cpu: torch.Tensor,
                outdir: str, model_name: str, device: torch.device,
                u_vec: Optional[np.ndarray] = None):
    """
    Train a baseline RNN model with streaming mini-batches.

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
        raise ValueError(f"Unknown optimizer {args.optimizer}")

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

        if args.log_gate_stats and (ep % args.gate_log_every) == 0:
            with torch.no_grad():
                idx0 = perm[:min(Btot, bs)]
                xb0 = Xtr_cpu[idx0].to(device, non_blocking=True)
                _, _, gdbg = model.forward_with_intermediates(xb0)
                rows = []
                if "gate_s" in gdbg:
                    rows.append(("gate_s_mean", float(gdbg["gate_s"].mean().item())))
                if "leak" in gdbg:
                    rows.append(("leak_mean", float(gdbg["leak"].mean().item())))
                if "rdiag" in gdbg:
                    rows.append(("rdiag_mean", float(gdbg["rdiag"].mean().item())))
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
# Per-model diagnostics (streaming + memmap)
#
# After training, this function runs the full learnability diagnostic
# pipeline on the diagnostic dataset (Xdg, Ydg):
#
#   1. Stream diagnostic batches through the model to extract:
#      - Diagonal Jacobian terms (leak, rdiag) for μ-kernel computation.
#      - JVP sensitivity v_t along a fixed random parameter direction.
#      - Prediction error δ_t projected through W_out.
#
#   2. For each lag ℓ in the diagnostic grid:
#      - Compute envelope μ(ℓ) and per-unit μ^(q)(ℓ) via prefix sums.
#      - Form the matched statistic ψ_t(ℓ) = Σ_q μ^(q) δ_t^(q) v_{t-ℓ}^(q).
#      - Write ψ samples to a temporary memmap file on disk.
#
#   3. After all batches, read back each memmap to compute:
#      - McCulloch α̂(ℓ) and σ̂(ℓ) from quantiles.
#      - SNR at each N in N_grid; find N_required where SNR > ε.
#      - Per-unit exponential τ fits from |μ^(q)(ℓ)|.
#
#   4. Delete tmp memmaps as soon as each lag is processed.
# ============================================================

def run_for_model(args, model_name: str, outdir: str,
                  Xtr_cpu: torch.Tensor, Ytr_cpu: torch.Tensor,
                  Xdg_cpu: torch.Tensor, Ydg_cpu: torch.Tensor,
                  device: torch.device,
                  u_vec: Optional[np.ndarray] = None) -> Dict:
    """
    Train one model and run the full learnability diagnostic pipeline.

    Returns a dict with per-lag results: envelope values, tail indices,
    N_required, alpha/sigma estimates, and per-unit τ fits.
    """
    model = build_model(model_name, args.D, args.H, const_s=args.const_s, ln=args.layernorm).to(device)

    train_model(args, model, Xtr_cpu, Ytr_cpu, outdir, model_name, device=device, u_vec=u_vec)

    model.eval()
    os.makedirs(outdir, exist_ok=True)

    ells = np.linspace(args.lag_min, args.lag_max, args.num_lags, dtype=int)
    ells_list = [int(e) for e in ells]

    Bdg, Tdg, _ = Xdg_cpu.shape
    H = int(args.H)

    # --- temp memmaps per ell
    memmap_paths: Dict[int, str] = {}
    memmaps: Dict[int, np.memmap] = {}
    write_offsets: Dict[int, int] = {}

    # raw mean accumulators for each ell
    sum_psi: Dict[int, float] = {ell: 0.0 for ell in ells_list}
    count_psi: Dict[int, int] = {ell: 0 for ell in ells_list}

    # envelope accumulators
    sum_mass: Dict[int, float] = {ell: 0.0 for ell in ells_list}
    sum_log_mass: Dict[int, float] = {ell: 0.0 for ell in ells_list}
    count_seq: Dict[int, int] = {ell: 0 for ell in ells_list}
    sum_unit: Dict[int, np.ndarray] = {ell: np.zeros(H, dtype=np.float64) for ell in ells_list}

    # batching for diagnostics
    Bb = min(128, int(Bdg))
    nb = int(math.ceil(Bdg / Bb))
    stepB = max(1, nb // 10)

    # create memmaps
    total_tmp_bytes = 0
    for ell in ells_list:
        n_samples = int(Bdg * max(0, (Tdg - ell)))
        tmp_path = os.path.join(outdir, f"{model_name}_Tseq_ell{ell:04d}.tmp")
        memmap_paths[ell] = tmp_path
        write_offsets[ell] = 0
        memmaps[ell] = np.memmap(tmp_path, dtype=np.float64, mode="w+", shape=(n_samples,))
        total_tmp_bytes += n_samples * 8

    log(f"[diag:{model_name}] start: Bdg={Bdg} T={Tdg} H={H} num_lags={len(ells_list)} Bb={Bb} nb={nb}")
    log(f"[diag:{model_name}] orient_matched_statistic_sign={int(bool(args.orient_matched_statistic_sign))}")
    log(f"[diag:{model_name}] tmp memmaps created (~{total_tmp_bytes/1e9:.2f} GB on disk)")

    # Ensure cleanup on any exception
    def cleanup_all_tmp():
        for ell in ells_list:
            try:
                if ell in memmaps:
                    memmaps[ell].flush()
            except Exception:
                pass
            try_remove(memmap_paths.get(ell, ""))

    try:
        Wout = model.out.weight.detach()  # (1,H) on GPU

        for bi in range(nb):
            lo = bi * Bb
            hi = min(Bdg, (bi + 1) * Bb)

            if (bi == 0) or (bi == nb - 1) or ((bi + 1) % stepB == 0):
                log(f"[diag:{model_name}] batch {bi+1}/{nb} (seq {lo}:{hi})")

            xb = Xdg_cpu[lo:hi].to(device, non_blocking=True)
            yb = Ydg_cpu[lo:hi].to(device, non_blocking=True)

            with torch.no_grad():
                yhat, hseq, g = model.forward_with_intermediates(xb)
                leak = g["leak"]
                rdiag = g["rdiag"]

            # sanity checks (cheap, catches silent mismatches)
            assert leak.shape == rdiag.shape, (leak.shape, rdiag.shape)
            assert leak.shape[:2] == yb.shape[:2], (leak.shape, yb.shape)

            vseq = compute_vseq_jvp(model, xb, w_seed=args.w_seed).detach()
            assert vseq.shape == leak.shape, (vseq.shape, leak.shape)

            with torch.no_grad():
                err = (yhat[..., 0] - yb[..., 0])
                delta = err.unsqueeze(-1) * Wout  # (Bb,T,H)

            cs_log, cs_ratio = precompute_prefix_sums(leak, rdiag)

            for ell in ells_list:
                # envelope μ (for mu_mean, log_mu_mean, per-unit)
                mu_env = mu_for_envelope_from_prefix(cs_log, cs_ratio, leak, rdiag, ell, out_dtype=leak.dtype)
                if mu_env.numel() > 0:
                    abs_mu = torch.abs(mu_env).double()
                    mass_per_seq = abs_mu.mean(dim=2).mean(dim=1)  # (Bb,)
                    sum_mass[ell] += float(mass_per_seq.sum().item())
                    sum_log_mass[ell] += float(torch.log(mass_per_seq + 1e-30).sum().item())
                    count_seq[ell] += int(mass_per_seq.shape[0])
                    sum_unit[ell] += abs_mu.mean(dim=1).sum(dim=0).detach().cpu().numpy()

                # matched-statistic μ and ψ, written to memmap
                mu0, mu1, mu_all = mu_for_matched_stat_from_prefix(cs_log, cs_ratio, ell, out_dtype=leak.dtype)
                mu_used = mu_all if bool(args.include_first_order_diag) else mu0
                if mu_used.numel() == 0:
                    continue

                delta_all = delta[:, ell:Tdg, :]
                v_past_all = vseq[:, 0:(Tdg - ell), :]
                psi_mat = torch.sum(mu_used * delta_all * v_past_all, dim=2)  # (Bb,T-ell)

                # ---------------------------------------------------------------------
                # Matched-statistic sign orientation (global gauge; per lag)
                #
                # Theory note:
                #   In the paper we define an *ideal* matched statistic that orients evidence
                #   so the expected signal contribution is nonnegative. In practice we do not
                #   know the population signs, and detectability depends on the magnitude of the
                #   mean shift. Here we apply a simple *global* orientation per lag:
                #
                #       psi <- sgn(E[psi]) * psi
                #
                #   where E[psi] is approximated by the batch mean at the current lag.
                #
                # Practical note:
                #   - This is NOT the per-neuron orientation sgn(m_q(l)) inside the sum; it is a
                #     global (aggregate) gauge choice that avoids arbitrary sign flips across lags.
                #   - A global sign flip does not change heavy-tail class; our McCulloch estimator
                #     is based on symmetric quantile spreads and is essentially reflection-invariant.
                #   - We keep it as a flag so it can be disabled for sanity checks/ablations.
                # ---------------------------------------------------------------------
                if bool(args.orient_matched_statistic_sign):
                    mu_psi = psi_mat.mean()
                    if torch.isfinite(mu_psi):
                        sgn = torch.sign(mu_psi)
                        # If the mean is exactly 0, do not flip (use +1).
                        if float(sgn.item()) == 0.0:
                            sgn = torch.tensor(1.0, device=psi_mat.device)
                    else:
                        # Non-finite mean: default to +1 to avoid contaminating samples.
                        sgn = torch.tensor(1.0, device=psi_mat.device)
                    psi_mat = sgn * psi_mat

                arr = psi_mat.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
                off = write_offsets[ell]
                memmaps[ell][off:off + arr.size] = arr
                write_offsets[ell] = off + arr.size

                sum_psi[ell] += float(arr.sum())
                count_psi[ell] += int(arr.size)

            # free large tensors ASAP
            del xb, yb, yhat, hseq, g, leak, rdiag, vseq, delta, cs_log, cs_ratio

        # flush memmaps
        for ell in ells_list:
            memmaps[ell].flush()

        log(f"[diag:{model_name}] done streaming; computing per-lag statistics")

        # --- per-ell stats + summary CSV (delete tmp ASAP)
        csv_path = os.path.join(outdir, f"{model_name}_summary.csv")

        mu_by_ell: Dict[int, float] = {}
        log_mu_by_ell: Dict[int, float] = {}
        Nreq_by_ell: Dict[int, int] = {}
        mu_units_by_ell: Dict[int, np.ndarray] = {}
        alpha_by_ell: Dict[int, float] = {}
        summary_rows = []

        L = len(ells_list)
        stepL = max(1, L // 10)

        # Update min samples threshold from CLI
        global _MIN_SAMPLES_ALPHA
        _MIN_SAMPLES_ALPHA = getattr(args, "min_samples_alpha", 500)
        alpha_method = getattr(args, "alpha_method", "mcculloch")

        with open(csv_path, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow([
                "ell", "mu_l1_mean", "log_mu_l1_mean",
                "alpha_hat", "sigma_alpha_hat",
                "N_required_at_eps", "best_snr", "err_at_best_snr", "best_N_for_ell",
                "mbar_scalar", "alpha_reliable", "alpha_method", "n_samples"
            ])

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
                    mu_per_unit = np.zeros(H, dtype=np.float64)

                mu_by_ell[ell] = mu_mean
                log_mu_by_ell[ell] = log_mu_mean
                mu_units_by_ell[ell] = mu_per_unit

                tmp_path = memmap_paths[ell]
                n_samples = int(Bdg * max(0, (Tdg - ell)))
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

                best_snr = -1e18
                best_err = 1e18
                best_N = None
                N_required = -1

                for Nuse in args.N_grid:
                    Nuse = min(int(Nuse), max(1, n_samples))
                    errv = detection_error_on_prefix(T_seq, Nuse) if n_samples > 0 else float("nan")
                    snr = compute_snr(alpha_hat, sigma_hat, mbar, Nuse)
                    if (snr > args.eps) and (N_required == -1):
                        N_required = Nuse
                    if snr > best_snr:
                        best_snr = snr
                        best_err = errv
                        best_N = Nuse

                Nreq_by_ell[ell] = N_required

                wcsv.writerow([
                    ell, mu_mean, log_mu_mean,
                    alpha_hat, sigma_hat,
                    N_required, best_snr, best_err, best_N if best_N is not None else -1,
                    mbar, int(alpha_reliable), alpha_method, n_samples
                ])

                summary_rows.append({
                    "ell": ell, "mu_l1_mean": mu_mean, "log_mu_l1_mean": log_mu_mean,
                    "alpha_hat": alpha_hat, "sigma_hat": sigma_hat,
                    "N_required": N_required, "best_N": best_N, "mbar": mbar,
                    "alpha_reliable": alpha_reliable,
                })

                # delete tmp ASAP for this ell
                del T_seq
                try_remove(tmp_path)

        log(f"[stats:{model_name}] done; tmp files removed")

        # Save per-unit μ averages
        if len(mu_units_by_ell) > 0:
            sorted_ells = sorted(mu_units_by_ell.keys())
            mu_units_csv_path = os.path.join(outdir, f"{model_name}_mu_units.csv")
            with open(mu_units_csv_path, "w", newline="") as fmu:
                writer = csv.writer(fmu)
                writer.writerow(["ell"] + [f"mu_unit_{q}" for q in range(H)])
                for e in sorted_ells:
                    writer.writerow([int(e)] + [float(v) for v in mu_units_by_ell[e]])

        # τ from μ^{(q)}(ℓ) exponential fits
        tau_q_mu = None
        tau_mu_results = []
        if len(mu_units_by_ell) > 0:
            sorted_ells = sorted(mu_units_by_ell.keys())
            ells_array = np.array(sorted_ells, dtype=float)
            tau_list = []
            for q in range(H):
                mu_vals_q = np.array([mu_units_by_ell[e][q] for e in sorted_ells], dtype=float)
                fit_res = fit_exponential_tau(ells_array, np.abs(mu_vals_q), min_points=5)
                if fit_res is None:
                    continue
                tau_list.append(fit_res["tau"])
                tau_mu_results.append({"unit_id": q, **fit_res})

            if tau_list:
                tau_q_mu = np.array(tau_list, dtype=float)
                tau_mu_csv = os.path.join(outdir, f"{model_name}_tau_from_mu_units.csv")
                with open(tau_mu_csv, "w", newline="") as f_tau_mu:
                    fieldnames = ["unit_id", "tau", "C", "a", "b", "r2", "num_points"]
                    writer = csv.DictWriter(f_tau_mu, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in tau_mu_results:
                        writer.writerow({k: row.get(k) for k in fieldnames})

                tau_mu_stats = {
                    "model": model_name,
                    "num_units": int(len(tau_q_mu)),
                    "tau_min": float(np.min(tau_q_mu)),
                    "tau_max": float(np.max(tau_q_mu)),
                    "tau_mean": float(np.mean(tau_q_mu)),
                    "tau_std": float(np.std(tau_q_mu)),
                }
                with open(os.path.join(outdir, f"{model_name}_tau_from_mu_stats.json"), "w") as jf:
                    json.dump(tau_mu_stats, jf, indent=2)

        # ConstGate closed-form τ
        tau_const = None
        s_const = model.get_const_gate_s()
        if s_const is not None:
            leak_val = np.clip(1.0 - float(s_const), 1e-6, 1.0 - 1e-6)
            tau_const = float(-1.0 / np.log(leak_val))
            with open(os.path.join(outdir, f"{model_name}_tau_const.json"), "w") as jf:
                json.dump({"s": float(s_const), "leak": float(leak_val), "tau": tau_const}, jf, indent=2)

        log(f"[run:{model_name}] finished")

        return {
            "ells": np.array(ells_list, dtype=int),
            "mu_by_ell": mu_by_ell,
            "log_mu_by_ell": log_mu_by_ell,
            "Nreq_by_ell": Nreq_by_ell,
            "summary_rows": summary_rows,
            "tau_q_mu": tau_q_mu,
            "tau_const": tau_const,
            "alpha_by_ell": alpha_by_ell,
        }

    except Exception:
        log(f"[ERROR] run_for_model({model_name}) failed; cleaning temp files.")
        traceback.print_exc()
        cleanup_all_tmp()
        raise

    finally:
        cleanup_all_tmp()


# ============================================================
# Envelope regime fits & learnability window H_N
#
# The envelope f̂(ℓ) summarises how quickly the memory kernel
# decays with lag.  We fit two competing models:
#   - Exponential: log f̂(ℓ) = a + b·ℓ  →  τ_env = -1/b
#   - Power-law:   log f̂(ℓ) = c + d·log(ℓ)
# and report R² for both to let the user judge which regime holds.
#
# The learnability window H_N is the maximum lag ℓ that can be
# detected with N training samples (i.e. N_required(ℓ) ≤ N).
# ============================================================

def fit_envelope_regimes(ells: np.ndarray, mu_vals: np.ndarray, log_mu_vals: np.ndarray) -> Dict:
    """
    Fit exponential and power-law models to the envelope f̂(ℓ).

    Returns a dict with sub-dicts "exp" and "power", each containing
    fit coefficients and R² values.
    """
    mask = np.isfinite(log_mu_vals)
    ells_fit = ells[mask]
    log_mu_fit = log_mu_vals[mask]
    if ells_fit.size < 3:
        return {}

    ss_tot = float(np.sum((log_mu_fit - log_mu_fit.mean()) ** 2) + 1e-12)

    A_exp = np.vstack([np.ones_like(ells_fit), ells_fit]).T
    coeff_exp, _, _, _ = np.linalg.lstsq(A_exp, log_mu_fit, rcond=None)
    pred_log_mu_exp = A_exp @ coeff_exp
    ss_res_exp = float(np.sum((log_mu_fit - pred_log_mu_exp) ** 2))
    r2_exp = 1.0 - ss_res_exp / ss_tot
    b_exp = float(coeff_exp[1])
    tau_env = float(-1.0 / b_exp) if b_exp < 0 else float("inf")

    log_ell = np.log(ells_fit.astype(float) + 1e-8)
    A_pow = np.vstack([np.ones_like(log_ell), log_ell]).T
    coeff_pow, _, _, _ = np.linalg.lstsq(A_pow, log_mu_fit, rcond=None)
    pred_log_mu_pow = A_pow @ coeff_pow
    ss_res_pow = float(np.sum((log_mu_fit - pred_log_mu_pow) ** 2))
    r2_pow = 1.0 - ss_res_pow / ss_tot

    return {
        "exp": {"a": float(coeff_exp[0]), "b": b_exp, "r2": float(r2_exp), "tau_env": tau_env},
        "power": {"c": float(coeff_pow[0]), "d": float(coeff_pow[1]), "r2": float(r2_pow)}
    }


def compute_H_N(ells: np.ndarray, Nreq_by_ell: Dict[int, int], N_values: List[int]) -> Dict[int, int]:
    """
    Compute the learnability window H_N for each training budget N.

    H_N = max{ℓ : N_required(ℓ) ≤ N}, i.e. the longest lag detectable
    with N samples.  Returns {N: H_N} dict.
    """
    H_by_N = {}
    for N in N_values:
        reachable = [ell for ell in ells if (Nreq_by_ell.get(int(ell), -1) != -1 and Nreq_by_ell[int(ell)] <= N)]
        H_by_N[int(N)] = int(max(reachable)) if reachable else 0
    return H_by_N


# ============================================================
# CLI
# ============================================================

def parse_args():
    """Parse command-line arguments for the baselines learnability pipeline."""
    p = argparse.ArgumentParser(
        description="Learnability-window pipeline for baseline gated RNNs (see paper)."
    )

    # --- Run identity -----------------------------------------------------------
    p.add_argument("--outdir", type=str, required=True,
                   help="Root output directory; per-model sub-dirs created automatically.")
    p.add_argument("--models", type=str, default="const,shared,diag",
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
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "sgd", "sgd_momentum"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # --- Gate initialisation ----------------------------------------------------
    # const_s sets BOTH the ConstGateRNN fixed value AND the initial sigmoid
    # operating point for SharedGate/DiagGate (via logit(const_s) bias).
    # Default 0.005 → τ ≈ 200 steps, covering task lags up to 512.
    p.add_argument("--const_s", type=float, default=0.005)

    # --- Diagnostic lag grid ----------------------------------------------------
    p.add_argument("--lag_min", type=int, default=4)
    p.add_argument("--lag_max", type=int, default=128)
    p.add_argument("--num_lags", type=int, default=32)

    # --- Task definition (multi-lag regression) ---------------------------------
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
                       "'mcculloch': McCulloch (1986) quantile ratio method (fast, uses 4 quantiles). "
                       "'ecf': Koutrouvelis (1980) ECF regression (more robust, uses full sample). "
                       "Default: mcculloch."
                   ))
    p.add_argument("--min_samples_alpha", type=int, default=500,
                   help=(
                       "Minimum number of matched-statistic samples required for a reliable "
                       "α̂ estimate. Lags with fewer samples are flagged as unreliable. "
                       "Default: 500."
                   ))

    # --- JVP / matched statistic ------------------------------------------------
    p.add_argument("--w_seed", type=int, default=12345,
                   help="Seed for the random tangent direction w in JVP computation.")
    p.add_argument("--include_first_order_diag", type=int, default=1,
                   help="If 1, include first-order rdiag correction in matched-stat kernel.")

    # --- Init / normalisation switches ------------------------------------------
    p.add_argument("--orth_init", action="store_true",
                   help="Apply orthogonal init to recurrent weights (respects _skip_orth).")
    p.add_argument("--layernorm", action="store_true",
                   help="Enable LayerNorm on pre-activation (before tanh).")
    p.add_argument("--log_gate_stats", type=int, default=1)
    p.add_argument("--gate_log_every", type=int, default=10)

    # --- Device -----------------------------------------------------------------
    p.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "mps", "cuda"])

    # --- Matched-statistic sign orientation (see theory note in run_for_model) --
    p.add_argument(
        "--orient_matched_statistic_sign",
        type=int,
        default=1,
        help=(
            "If 1, orient matched-statistic samples per lag by sign(mean(psi)). "
            "This applies a *global gauge* flip psi <- sgn(E[psi]) psi so the empirical mean "
            "is nonnegative. It makes the matched-statistic convention consistent across "
            "baselines and GRU/LSTM scripts. Reflection leaves symmetric tail-index "
            "estimation (McCulloch) essentially unchanged; detectability depends on |E[psi]|."
        )
    )

    args = p.parse_args()

    args.task_lags = [int(s) for s in args.task_lags.split(",") if s.strip()]
    args.task_coeffs = [float(s) for s in args.task_coeffs.split(",") if s.strip()]
    assert len(args.task_lags) == len(args.task_coeffs)

    args.N_grid = [int(s) for s in args.N_grid.split(",") if s.strip()]
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
    Run the full learnability pipeline for baseline models.

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

    set_seed(args.seed)

    device = resolve_device(args.device)
    log(f"Running on device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(f"GPU: {props.name}")

    save_args_to_csv(args, os.path.join(args.outdir, "cli_args.csv"))

    # CPU datasets (pinned for fast H2D transfers on CUDA).
    # Training and diagnostic sets share the same task direction u_vec
    # so the target function is identical across sets.
    Xtr_cpu, Ytr_cpu, u_vec = make_dataset_cpu(args.Nseq_train, args.T, args.D,
                                               args.task_lags, args.task_coeffs, args.noise_std, u_vec=None)
    Xdg_cpu, Ydg_cpu, _ = make_dataset_cpu(args.Nseq_diag, args.T, args.D,
                                           args.task_lags, args.task_coeffs, args.noise_std, u_vec=u_vec)

    log(f"Train set CPU: X={tuple(Xtr_cpu.shape)} Y={tuple(Ytr_cpu.shape)}")
    log(f"Diag  set CPU: X={tuple(Xdg_cpu.shape)} Y={tuple(Ydg_cpu.shape)}")

    if device.type == "cuda":
        Xtr_cpu = Xtr_cpu.pin_memory()
        Ytr_cpu = Ytr_cpu.pin_memory()
        Xdg_cpu = Xdg_cpu.pin_memory()
        Ydg_cpu = Ydg_cpu.pin_memory()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    results_by_model = {}

    for mname in models:
        mdir = os.path.join(args.outdir, mname)
        os.makedirs(mdir, exist_ok=True)
        log(f"[run] model={mname} -> {mdir}")

        # Train + run full diagnostic pipeline for this model
        res = run_for_model(args, mname, mdir, Xtr_cpu, Ytr_cpu, Xdg_cpu, Ydg_cpu, device=device, u_vec=u_vec)

        # Fit competing envelope decay regimes (exponential vs power-law)
        ells = np.array(res["ells"], dtype=int)
        mu_vals = np.array([res["mu_by_ell"][int(e)] for e in ells])
        log_mu_vals = np.array([res["log_mu_by_ell"][int(e)] for e in ells])

        fit_info = fit_envelope_regimes(ells, mu_vals, log_mu_vals)
        with open(os.path.join(mdir, f"{mname}_envelope_fits.json"), "w") as jf:
            json.dump(fit_info, jf, indent=2)

        # Compute learnability window H_N = max detectable lag given N samples
        H_by_N = compute_H_N(ells, res["Nreq_by_ell"], args.N_grid)
        res["H_by_N"] = H_by_N
        with open(os.path.join(mdir, f"{mname}_H_N.csv"), "w", newline="") as hf:
            wcsv = csv.writer(hf)
            wcsv.writerow(["N", "H_N"])
            for N, HN in sorted(H_by_N.items()):
                wcsv.writerow([N, HN])

        results_by_model[mname] = res

    # Write aggregate H_N summary: one row per N, one column per model
    H_summary_path = os.path.join(args.outdir, "H_N_summary.csv")
    with open(H_summary_path, "w", newline="") as hf:
        wcsv = csv.writer(hf)
        header = ["N"] + [f"H_N_{m}" for m in models]
        wcsv.writerow(header)
        for N in args.N_grid:
            row = [N] + [results_by_model[m]["H_by_N"].get(N, 0) for m in models]
            wcsv.writerow(row)

    log("All models done.")
    log("Done.")


if __name__ == "__main__":
    main()