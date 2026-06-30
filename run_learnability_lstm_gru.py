#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Empirical learnability-window pipeline for GRU and LSTM models."""

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

from seed_utils import bootstrap_mcculloch


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


def _state_to_cpu(obj):
    """Recursively move tensors in a nested checkpoint payload to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_state_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_state_to_cpu(v) for v in obj)
    return obj


def save_checkpoint(
    outdir: str,
    model_name: str,
    checkpoint_tag: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args,
    u_vec: Optional[np.ndarray],
    extra_payload: Optional[dict] = None,
) -> str:
    """Save a model/optimizer checkpoint needed for post-hoc diagnostics."""
    ckpt_path = os.path.join(outdir, f"{model_name}_{checkpoint_tag}_checkpoint.pt")
    payload = {
        "runner_type": "lstmgru",
        "model_name": str(model_name),
        "checkpoint_tag": str(checkpoint_tag),
        "args": dict(vars(args)),
        "u_vec": None if u_vec is None else np.asarray(u_vec, dtype=np.float32),
        "model_state_dict": _state_to_cpu(model.state_dict()),
        "optimizer_state_dict": _state_to_cpu(optimizer.state_dict()),
    }
    if extra_payload:
        payload.update(_state_to_cpu(extra_payload))
    torch.save(payload, ckpt_path)
    return ckpt_path


def save_final_checkpoint(
    outdir: str,
    model_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args,
    u_vec: Optional[np.ndarray],
    extra_payload: Optional[dict] = None,
) -> str:
    """Backward-compatible wrapper for the final checkpoint filename."""
    return save_checkpoint(outdir, model_name, "final", model, optimizer, args, u_vec, extra_payload)


def save_selection_metadata(outdir: str, model_name: str, payload: dict) -> str:
    """Save a compact JSON summary of checkpoint selection and final-epoch metrics."""
    path = os.path.join(outdir, f"{model_name}_selection.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path

def save_dense_unit_npz(
    filepath: str,
    ell_values: np.ndarray,
    value_matrix: np.ndarray,
    *,
    component: str,
    rate_scale: str,
    extra_arrays: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Save a dense per-lag/per-unit artifact as a compact NPZ bundle."""
    payload = {
        "ell": np.asarray(ell_values, dtype=np.int64),
        "values": np.asarray(value_matrix, dtype=np.float64),
        "unit_ids": np.arange(value_matrix.shape[1], dtype=np.int64),
        "component": np.array(component),
        "rate_scale": np.array(rate_scale),
    }
    if extra_arrays:
        for key, value in extra_arrays.items():
            payload[key] = np.asarray(value, dtype=np.float64)
    np.savez_compressed(filepath, **payload)

def set_seed(seed: int) -> None:
    """Set random seeds for numpy, torch CPU, and all CUDA devices."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

def layernorm_if(enabled: bool, dim: int) -> nn.Module:
    """Return LayerNorm when enabled, otherwise an identity module."""
    return nn.LayerNorm(dim) if enabled else nn.Identity()

def now_s() -> float:
    """Current wall-clock time in seconds (for timing diagnostics)."""
    return time.time()

def log(msg: str) -> None:
    """Print a timestamped log message (flushed immediately)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# Adaptive base rates (generalized effective learning rates)
# ============================================================

def compute_adaptive_base_rates(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr: float,
    eps: float = 1e-8,
    beta2: float = 0.999,
) -> np.ndarray:
    """
    Compute per-neuron adaptive base rates Λ^(q)_r via the row-mean projection.

    Under layer normalization, the Rayleigh-quotient projection
    (Eq. layernorm_simplification in the paper) simplifies to:

        Λ^(q)_r ≈ (1/H) Σ_j [Λ_r^(U•)]_{qj}

    where [Λ_r^(U•)]_{qj} = lr / (sqrt(v̂_{qj}) + ε).

    We average across all recurrent weight matrices (H×H).
    For SGD-like optimizers without second-moment state, returns
    uniform lr (i.e., Λ^(q)_r = μ for all q).

    Returns:
        Lambda_q: (H,) array of per-neuron adaptive base rates Λ^(q)_r.
    """
    H = model.H

    # Collect all recurrent weight matrices (H×H)
    recurrent_params = []
    for name, param in model.named_parameters():
        # Match recurrent weight matrices: U*.weight with shape (H, H)
        if param.shape == (H, H) and "weight" in name and "out" not in name:
            recurrent_params.append((name, param))

    if not recurrent_params:
        return np.full(H, lr, dtype=np.float64)

    # Check if optimizer has second-moment state (Adam/AdamW/RMSprop)
    state = optimizer.state
    has_v = False
    for _, p in recurrent_params:
        if p in state:
            if "exp_avg_sq" in state[p]:       # Adam/AdamW
                has_v = True
                break
            elif "square_avg" in state[p]:      # RMSprop
                has_v = True
                break

    if not has_v:
        log(f"[adaptive_base_rates] No second-moment state found; returning uniform lr={lr:.2e}")
        return np.full(H, lr, dtype=np.float64)

    # Compute row-mean projection for each recurrent weight matrix
    Lambda_per_matrix = []
    for name, param in recurrent_params:
        pstate = state.get(param, {})

        # Get second-moment estimate v
        if "exp_avg_sq" in pstate:         # Adam/AdamW
            v = pstate["exp_avg_sq"]       # (H, H)
            step = pstate.get("step", 1)
            if isinstance(step, torch.Tensor):
                step = step.item()
            step = max(int(step), 1)
            v_hat = v / (1.0 - beta2 ** step)
        elif "square_avg" in pstate:        # RMSprop
            v_hat = pstate["square_avg"]    # already the EMA, no bias correction
        else:
            continue

        # Guard: skip matrices with corrupted optimizer state (e.g. after NaN halt)
        if not torch.isfinite(v_hat).all():
            log(f"[adaptive_base_rates] {name}: non-finite v_hat entries "
                f"({(~torch.isfinite(v_hat)).sum().item()} / {v_hat.numel()}), skipping")
            continue

        # Per-parameter adaptive rates: λ_{qj} = lr / (sqrt(v̂_{qj}) + ε)
        lam = lr / (torch.sqrt(v_hat.double()) + eps)  # (H, H)

        # Row mean: Λ^(q, U•)_r = (1/H) Σ_j λ_{qj}
        row_mean = lam.mean(dim=1)  # (H,)
        Lambda_per_matrix.append(row_mean.detach().cpu().numpy())
        log(f"[adaptive_base_rates] {name}: row-mean range "
            f"[{row_mean.min().item():.4e}, {row_mean.max().item():.4e}], "
            f"mean={row_mean.mean().item():.4e}")

    if not Lambda_per_matrix:
        return np.full(H, lr, dtype=np.float64)

    # Average across all recurrent weight matrices (equal weight under layer norm)
    Lambda_q = np.mean(Lambda_per_matrix, axis=0).astype(np.float64)  # (H,)
    log(f"[adaptive_base_rates] final Λ^(q) range [{Lambda_q.min():.4e}, {Lambda_q.max():.4e}], "
        f"mean={Lambda_q.mean():.4e}, ratio max/min={Lambda_q.max()/max(Lambda_q.min(),1e-30):.2f}")
    return Lambda_q


def extract_adaptive_rate_matrix(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr: float,
    eps: float = 1e-8,
    beta2: float = 0.999,
) -> Tuple[Optional[torch.Tensor], np.ndarray]:
    """
    Extract the per-parameter adaptive rate matrix λ_{qj} from the optimizer.

    Returns:
        lambda_matrix: (H, H) tensor of per-parameter rates, averaged across
                       all recurrent weight matrices.  None if no second-moment
                       state (SGD).
        lambda_rowmean: (H,) numpy array — the LN-approximated row mean
                        (same as compute_adaptive_base_rates output).
    """
    H = model.H

    # Collect recurrent weight matrices (H×H)
    recurrent_params = []
    for name, param in model.named_parameters():
        if param.shape == (H, H) and "weight" in name and "out" not in name:
            recurrent_params.append((name, param))

    if not recurrent_params:
        return None, np.full(H, lr, dtype=np.float64)

    # Check for second-moment state
    state = optimizer.state
    has_v = False
    for _, p in recurrent_params:
        if p in state:
            if "exp_avg_sq" in state[p] or "square_avg" in state[p]:
                has_v = True
                break

    if not has_v:
        return None, np.full(H, lr, dtype=np.float64)

    # Collect per-parameter rate matrices
    lam_matrices = []
    for name, param in recurrent_params:
        pstate = state.get(param, {})
        if "exp_avg_sq" in pstate:
            v = pstate["exp_avg_sq"]
            step = pstate.get("step", 1)
            if isinstance(step, torch.Tensor):
                step = step.item()
            step = max(int(step), 1)
            v_hat = v / (1.0 - beta2 ** step)
        elif "square_avg" in pstate:
            v_hat = pstate["square_avg"]
        else:
            continue

        if not torch.isfinite(v_hat).all():
            log(f"[adaptive_rate_matrix] {name}: non-finite v_hat, skipping")
            continue

        lam = lr / (torch.sqrt(v_hat.double()) + eps)  # (H, H)
        lam_matrices.append(lam)

    if not lam_matrices:
        return None, np.full(H, lr, dtype=np.float64)

    # Average across recurrent weight matrices
    lambda_matrix = torch.stack(lam_matrices, dim=0).mean(dim=0)  # (H, H)
    lambda_rowmean = lambda_matrix.mean(dim=1).detach().cpu().numpy().astype(np.float64)

    log(f"[adaptive_rate_matrix] λ_{{qj}} matrix: "
        f"mean={lambda_matrix.mean().item():.4e}, "
        f"row-mean range [{lambda_rowmean.min():.4e}, {lambda_rowmean.max():.4e}]")

    return lambda_matrix.detach(), lambda_rowmean


def compute_lag_dependent_rates(
    lambda_matrix: torch.Tensor,
    hseq: torch.Tensor,
    T_valid: int,
    fallback_rate: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the lag-dependent Rayleigh-quotient base rates Λ^(q)_{r,ℓ}(b,t).

    For a perturbation at time step k, the pre-synaptic state is h_{k-1}.
    The Rayleigh quotient restricted to the recurrent block is:

        Λ^(q)(b,k) = Σ_j λ_{qj} h²_{k-1,j}(b) / Σ_j h²_{k-1,j}(b)

    When h_{k-1} = 0 (initial state), falls back to the row-mean rate.

    Args:
        lambda_matrix: (H, H) per-parameter adaptive rate matrix λ_{qj}.
        hseq:          (B, T, H) hidden state sequence from forward pass.
        T_valid:       number of valid time positions.
        fallback_rate: (H,) rate to use when h=0 (typically the row mean).

    Returns:
        Lambda_ell: (B, T_valid, H) per-neuron adaptive base rates.
    """
    B, T_full, H = hseq.shape
    device = hseq.device
    # Pre-synaptic hidden states at perturbation times:
    #   position k=0 → h_{-1} = 0 (initial state)
    #   position k≥1 → h_{k-1} = hseq[:, k-1, :]
    h_pre = torch.zeros(B, T_valid, H, device=device, dtype=torch.float64)
    n_copy = min(T_valid - 1, T_full)
    if n_copy > 0:
        h_pre[:, 1:1+n_copy, :] = hseq[:, :n_copy, :].double()

    h_sq = h_pre ** 2  # (B, T_valid, H)
    h_sq_sum = h_sq.sum(dim=2, keepdim=True)  # (B, T_valid, 1)

    # Rayleigh quotient: Λ[b,t,q] = Σ_j λ[q,j] h²[b,t,j] / Σ_j h²[b,t,j]
    numer = torch.matmul(h_sq, lambda_matrix.T.double())  # (B, T_valid, H)

    # Avoid division by zero: mask positions with h=0
    zero_mask = (h_sq_sum.squeeze(-1) < 1e-30)  # (B, T_valid)
    Lambda_ell = numer / (h_sq_sum + 1e-30)  # (B, T_valid, H)

    # Replace h=0 positions with fallback (row-mean) rate
    if zero_mask.any():
        Lambda_ell[zero_mask] = fallback_rate.double().unsqueeze(0)

    return Lambda_ell


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
            al = grid[::-1, 0]
            r = np.interp(a, al, grid[::-1, 1])
            iqr = np.interp(a, al, grid[::-1, 2])
            q25, q75 = -0.5 * iqr, 0.5 * iqr
            q95 = 0.5 * r * iqr
            q05 = -q95
            q50 = 0.0
            out = (float(q05), float(q25), float(q50), float(q75), float(q95))

        self.cache_q[key] = out
        return out

    def ensure_grid(self, n_grid: int = 201):  # 201 points → Δα ≈ 0.005 resolution
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


def _default_alpha_meta(method_requested: str, n_samples_total: int) -> Dict[str, object]:
    """Default metadata payload for alpha-estimation diagnostics."""
    return {
        "method_requested": method_requested,
        "method_origin": "none",
        "method_reason": "not_run",
        "reliability_reason": "not_run",
        "alpha_hat": float("nan"),
        "sigma_hat": float("nan"),
        "reliable": False,
        "n_samples_total": int(n_samples_total),
        "n_samples_used": 0,
        "used_subsample": 0,
        "boundary_hit": 0,
        "iqr": float("nan"),
        "quantile_ratio": float("nan"),
        "ecf_n_grid": 0,
        "ecf_n_points_strict": 0,
        "ecf_n_points_relaxed": 0,
        "ecf_n_points_used": 0,
        "ecf_filter_mode": "none",
    }


def estimate_alpha_sigma_ecf_symmetric_with_meta(samples: np.ndarray) -> Dict[str, object]:
    """
    Estimate (α̂, σ̂) for symmetric α-stable using ECF regression
    (Koutrouvelis 1980, simplified for β=0).
    """
    samples = np.asarray(samples, dtype=np.float64)
    n = samples.size
    meta = _default_alpha_meta("ecf", n)
    if n < _MIN_SAMPLES_ALPHA:
        meta["method_reason"] = "too_few_samples"
        meta["reliability_reason"] = "too_few_samples"
        return meta

    t_grid = _choose_ecf_grid(samples, n_points=50)
    phi2 = _ecf_at_t(samples, t_grid)
    meta["ecf_n_grid"] = int(len(t_grid))

    mask_strict = (phi2 > 0.01) & (phi2 < 0.95)
    mask_relaxed = (phi2 > 1e-4) & (phi2 < 0.999)
    n_strict = int(mask_strict.sum())
    n_relaxed = int(mask_relaxed.sum())
    meta["ecf_n_points_strict"] = n_strict
    meta["ecf_n_points_relaxed"] = n_relaxed

    if n_strict >= 5:
        mask = mask_strict
        meta["ecf_filter_mode"] = "strict"
    elif n_relaxed >= 3:
        mask = mask_relaxed
        meta["ecf_filter_mode"] = "relaxed"
    else:
        meta["method_reason"] = "too_few_informative_points"
        meta["reliability_reason"] = "too_few_informative_points"
        return meta

    t_use = t_grid[mask]
    phi2_use = phi2[mask]
    meta["ecf_n_points_used"] = int(mask.sum())

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
    sigma_hat = float((np.exp(intercept) / 2.0) ** (1.0 / alpha_hat))

    meta.update({
        "method_origin": "ecf_regression",
        "method_reason": "ok",
        "reliability_reason": "ok",
        "alpha_hat": alpha_hat,
        "sigma_hat": float(max(0.0, sigma_hat)),
    })
    return meta


def estimate_alpha_sigma_ecf_symmetric(samples: np.ndarray) -> Tuple[float, float]:
    """Backward-compatible wrapper around the ECF estimator."""
    meta = estimate_alpha_sigma_ecf_symmetric_with_meta(samples)
    return float(meta["alpha_hat"]), float(meta["sigma_hat"])


def estimate_alpha_sigma_with_meta(
    samples: np.ndarray,
    method: str = "ecf",
    n_samples_for_ecf: int = 100000,
) -> Dict[str, object]:
    """Unified alpha-estimation interface with explicit provenance metadata."""
    samples = np.asarray(samples, dtype=np.float64)
    n = samples.size
    meta = _default_alpha_meta(method, n)

    iqr = float(np.subtract(*np.percentile(samples, [75, 25]))) if n > 0 else float("nan")
    meta["iqr"] = iqr

    if n < _MIN_SAMPLES_ALPHA:
        meta["method_reason"] = "too_few_samples"
        meta["reliability_reason"] = "too_few_samples"
        return meta

    if method == "ecf":
        if n > n_samples_for_ecf:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, n_samples_for_ecf, replace=False)
            sub = np.asarray(samples[idx], dtype=np.float64)
            meta["used_subsample"] = 1
            meta["n_samples_used"] = int(sub.size)
        else:
            sub = np.asarray(samples, dtype=np.float64)
            meta["n_samples_used"] = int(sub.size)

        ecf_meta = estimate_alpha_sigma_ecf_symmetric_with_meta(sub)
        for key in [
            "method_origin",
            "method_reason",
            "alpha_hat",
            "sigma_hat",
            "ecf_n_grid",
            "ecf_n_points_strict",
            "ecf_n_points_relaxed",
            "ecf_n_points_used",
            "ecf_filter_mode",
        ]:
            meta[key] = ecf_meta[key]
    else:
        if (not np.isfinite(iqr)) or (iqr <= 1e-12):
            meta["method_origin"] = "none"
            meta["method_reason"] = "degenerate_iqr"
            meta["reliability_reason"] = "degenerate_iqr"
            meta["n_samples_used"] = int(n)
            return meta

        q05, q25, q75, q95 = np.quantile(samples, [0.05, 0.25, 0.75, 0.95])
        meta["quantile_ratio"] = float((q95 - q05) / (iqr + 1e-12))
        alpha_hat, sigma_hat = estimate_alpha_sigma_mcculloch_symmetric_from_quantiles(
            q05, q25, q75, q95
        )
        meta.update({
            "method_origin": "mcculloch",
            "method_reason": "ok",
            "alpha_hat": float(alpha_hat),
            "sigma_hat": float(sigma_hat),
            "n_samples_used": int(n),
        })

    reliability_reasons = []
    alpha_hat = float(meta["alpha_hat"])
    sigma_hat = float(meta["sigma_hat"])

    if not np.isfinite(alpha_hat) or not np.isfinite(sigma_hat):
        reliability_reasons.append(str(meta["method_reason"]))
    if np.isfinite(sigma_hat) and sigma_hat <= 1e-12:
        reliability_reasons.append("nonpositive_sigma")
    if np.isfinite(alpha_hat) and (alpha_hat <= 1.01 or alpha_hat >= 1.99) and n < 2000:
        reliability_reasons.append("boundary_with_few_samples")
    if np.isfinite(iqr) and iqr <= 1e-10:
        reliability_reasons.append("degenerate_iqr")

    meta["boundary_hit"] = int(np.isfinite(alpha_hat) and (alpha_hat <= 1.01 or alpha_hat >= 1.99))
    meta["reliable"] = len(reliability_reasons) == 0
    meta["reliability_reason"] = (
        "ok" if meta["reliable"] else ";".join(dict.fromkeys(reliability_reasons))
    )
    return meta


def estimate_alpha_sigma(
    samples: np.ndarray,
    method: str = "ecf",
    n_samples_for_ecf: int = 100000,
) -> Tuple[float, float, bool]:
    """Unified α̂ estimation with reliability checking."""
    meta = estimate_alpha_sigma_with_meta(
        samples,
        method=method,
        n_samples_for_ecf=n_samples_for_ecf,
    )
    return float(meta["alpha_hat"]), float(meta["sigma_hat"]), bool(meta["reliable"])


def compute_snr(alpha_hat: float, sigma_hat: float, mbar_Tmean: float, Nuse: int) -> float:
    """Compute empirical SNR(ℓ, N) = |m̄(ℓ)| · N^{1-1/α} / σ̂."""
    if (not np.isfinite(alpha_hat)) or (not np.isfinite(sigma_hat)) or (not np.isfinite(mbar_Tmean)):
        return 0.0
    if sigma_hat <= 1e-12:
        return 0.0
    alpha_eff = max(1.0, float(alpha_hat))
    exp = 1.0 - 1.0 / alpha_eff
    return float(mbar_Tmean * (int(Nuse) ** exp) / float(sigma_hat))


def noise_tolerance_to_eps(noise_tolerance: float) -> float:
    """
    Convert a user-facing noise tolerance into the raw SNR cutoff epsilon.

    The detection rule is unchanged: detect lag ell when SNR(ell, N) > eps.
    We expose noise_tolerance = 1 / eps because it is easier to interpret:
    noise_tolerance = 0.1 means require SNR > 10, i.e. the effective
    noise-to-signal ratio in the detection metric must be below about 10%.
    Smaller values are therefore stricter.
    """
    tol = float(noise_tolerance)
    if not (0.0 < tol <= 1.0):
        raise ValueError(f"--noise_tolerance must lie in (0, 1], got {tol}")
    return float(1.0 / tol)


def detection_error_on_prefix_arr(arr: np.ndarray, Nuse: int) -> float:
    """
    Coefficient of variation of the first Nuse per-sequence matched-statistic means.

    This is the empirical detection error: std(ψ̄) / |mean(ψ̄)|.
    A small value indicates the mean shift is large relative to noise.

    Parameters
    ----------
    arr : np.ndarray
        1-D array of per-sequence means ψ̄_n(ℓ).
    Nuse : int
        Number of samples (sequences) to use.
    """
    Nuse = max(1, min(int(Nuse), len(arr)))
    subset = np.asarray(arr[:Nuse], dtype=np.float64)
    if subset.size == 0:
        return float("nan")
    mu = float(np.mean(subset))
    sd = float(np.std(subset) + 1e-12)
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

def _env_prod_from_cs(cs_log: torch.Tensor, ell: int, out_dtype: torch.dtype) -> torch.Tensor:
    """Extract unshifted ℓ-step products for the envelope window."""
    B, Tp1, H = cs_log.shape
    T = Tp1 - 1
    if ell <= 0 or ell > T:
        return torch.zeros(B, 0, H, dtype=out_dtype, device=cs_log.device)
    log_prod = cs_log[:, ell:(T + 1), :] - cs_log[:, 0:(T - ell + 1), :]
    return torch.exp(log_prod).to(out_dtype)

def _env_sum_from_cs(cs_sum: torch.Tensor, ell: int, out_dtype: torch.dtype) -> torch.Tensor:
    """Extract unshifted ℓ-step sums for the envelope window."""
    B, Tp1, H = cs_sum.shape
    T = Tp1 - 1
    if ell <= 0 or ell > T:
        return torch.zeros(B, 0, H, dtype=out_dtype, device=cs_sum.device)
    s = cs_sum[:, ell:(T + 1), :] - cs_sum[:, 0:(T - ell + 1), :]
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
    def __init__(self, D: int, H: int, ln: bool = False, init_update: float = 0.5):
        super().__init__()
        self.D, self.H = D, H
        self.Wz, self.Uz = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.Wr, self.Ur = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.Wh, self.Uh = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.Wz._skip_orth = True
        self.Uz._skip_orth = True
        self.ln_h = layernorm_if(ln, H)
        self.out = nn.Linear(H, 1)
        nn.init.zeros_(self.Wz.weight)
        nn.init.zeros_(self.Uz.weight)
        init_update = float(np.clip(init_update, 1e-6, 1.0 - 1e-6))
        nn.init.constant_(self.Wz.bias, float(np.log(init_update / (1.0 - init_update))))
        nn.init.zeros_(self.Wr.bias)
        nn.init.zeros_(self.Wh.bias)
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
    def __init__(self, D: int, H: int, ln: bool = False, init_forget: float = 0.5):
        super().__init__()
        self.D, self.H = D, H
        self.Wi, self.Ui = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.Wf, self.Uf = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.Wo, self.Uo = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.Wg, self.Ug = nn.Linear(D, H), nn.Linear(H, H, bias=False)
        self.Wf._skip_orth = True
        self.Uf._skip_orth = True
        self.ln_cand = layernorm_if(ln, H)
        self.out = nn.Linear(H, 1)
        nn.init.zeros_(self.Wf.weight)
        nn.init.zeros_(self.Uf.weight)
        init_forget = float(np.clip(init_forget, 1e-6, 1.0 - 1e-6))
        nn.init.constant_(self.Wf.bias, float(np.log(init_forget / (1.0 - init_forget))))
        nn.init.zeros_(self.Wi.bias)
        nn.init.zeros_(self.Wo.bias)
        nn.init.zeros_(self.Wg.bias)
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

def build_model(name: str, D: int, H: int, ln: bool,
                gru_init_update: float, lstm_init_forget: float) -> BaseSeqModel:
    """Instantiate a GRU or LSTM model by name."""
    name = name.lower().strip()
    if name == "gru":
        return GRUModel(D, H, ln=ln, init_update=gru_init_update)
    if name == "lstm":
        return LSTMModel(D, H, ln=ln, init_forget=lstm_init_forget)
    raise ValueError(f"Unknown model {name}")


def make_optimizer(args, model: nn.Module) -> torch.optim.Optimizer:
    """Construct the optimizer specified by CLI args for a given model."""
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0, weight_decay=args.weight_decay)
    if args.optimizer == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optimizer == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(), lr=args.lr,
            alpha=args.rmsprop_alpha,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unknown optimizer {args.optimizer}")


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
                u_vec: Optional[np.ndarray]):
    """
    Train a GRU/LSTM model with streaming mini-batches (CPU→GPU).

    Writes <outdir>/<model_name>_learning_curve.csv with columns:
        epoch, train_loss, train_acc (R²), val_loss, val_acc (R²).
    Optionally logs periodic gate statistics to gate_stats_<model>.csv.
    Halts early if NaN/Inf loss is detected.

    Uses one fixed validation split for stable checkpoint selection and
    returns the optimizer restored to the best validation epoch.
    """
    opt = make_optimizer(args, model)

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

    # Fixed validation split for stable checkpoint selection.
    n_val = int(min(1024, max(bs, 256)))
    Xv_cpu, Yv_cpu, _ = make_dataset_cpu(
        n_val, args.T, args.D,
        args.task_lags, args.task_coeffs, args.noise_std,
        u_vec=u_vec
    )
    if device.type == "cuda":
        Xv_cpu = Xv_cpu.pin_memory()
        Yv_cpu = Yv_cpu.pin_memory()

    best_val_mse = float("inf")
    best_val_r2 = float("nan")
    best_epoch = 0
    best_model_state = None
    best_optimizer_state = None
    last_epoch = 0
    last_train_loss = float("nan")
    last_train_r2 = float("nan")
    last_val_loss = float("nan")
    last_val_r2 = float("nan")

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

        va_mse, va_r2 = _eval_streaming_mse_and_r2(model, Xv_cpu, Yv_cpu, device=device, batch_size=bs)

        if va_mse < best_val_mse:
            best_val_mse = float(va_mse)
            best_val_r2 = float(va_r2)
            best_epoch = int(ep + 1)
            best_model_state = _state_to_cpu(model.state_dict())
            best_optimizer_state = _state_to_cpu(opt.state_dict())

        with open(lc_csv, "a", newline="") as lf:
            wlc = csv.writer(lf)
            wlc.writerow([ep + 1, float(train_loss_epoch), float(tr_r2_eval), float(va_mse), float(va_r2)])

        last_epoch = int(ep + 1)
        last_train_loss = float(train_loss_epoch)
        last_train_r2 = float(tr_r2_eval)
        last_val_loss = float(va_mse)
        last_val_r2 = float(va_r2)

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

    if nan_halt:
        log(f"[train:{model_name}] WARNING: training halted due to NaN/Inf loss")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        opt.load_state_dict(best_optimizer_state)
        if best_epoch != args.epochs:
            log(
                f"[train:{model_name}] restoring best validation checkpoint "
                f"(epoch={best_epoch}, val_loss={best_val_mse:.4g}, val_R2={best_val_r2:.3f})"
            )
    log(f"[train:{model_name}] done (nan_halt={nan_halt})")
    return opt, nan_halt, {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_mse),
        "best_val_r2": float(best_val_r2),
        "val_selection": "fixed_validation_split",
        "n_val": int(n_val),
        "last_epoch": int(last_epoch),
        "last_train_loss": float(last_train_loss),
        "last_train_r2": float(last_train_r2),
        "last_val_loss": float(last_val_loss),
        "last_val_r2": float(last_val_r2),
    }


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


def mu_for_matched_stat_gru(
    cs_log_leak: torch.Tensor,
    cs_log_reset: torch.Tensor,
    cs_log_eta: torch.Tensor,
    cs_ratio: torch.Tensor,
    ell: int,
    include_first_order: bool,
    out_dtype: torch.dtype,
):
    """Return GRU matched-statistic kernel parts on the shifted window."""
    gamma0 = _win_prod_from_cs(cs_log_leak, ell, out_dtype=out_dtype)
    rho0 = _win_prod_from_cs(cs_log_reset, ell, out_dtype=out_dtype)
    eta0 = _win_prod_from_cs(cs_log_eta, ell, out_dtype=out_dtype)
    mu0 = gamma0 + rho0 + eta0
    if include_first_order and mu0.numel() > 0:
        sum_ratio = _win_sum_from_cs(cs_ratio, ell, out_dtype=out_dtype)
        mu1 = gamma0 * sum_ratio
    else:
        mu1 = torch.zeros_like(mu0)
    return mu0, mu1, mu0 + mu1


def mu_for_envelope_gru(
    cs_log_leak: torch.Tensor,
    cs_log_reset: torch.Tensor,
    cs_log_eta: torch.Tensor,
    cs_ratio: torch.Tensor,
    ell: int,
    include_first_order: bool,
    out_dtype: torch.dtype,
):
    """Return GRU envelope kernel parts on the unshifted window."""
    gamma0 = _env_prod_from_cs(cs_log_leak, ell, out_dtype=out_dtype)
    rho0 = _env_prod_from_cs(cs_log_reset, ell, out_dtype=out_dtype)
    eta0 = _env_prod_from_cs(cs_log_eta, ell, out_dtype=out_dtype)
    mu0 = gamma0 + rho0 + eta0
    if include_first_order and mu0.numel() > 0:
        sum_ratio = _env_sum_from_cs(cs_ratio, ell, out_dtype=out_dtype)
        mu1 = gamma0 * sum_ratio
    else:
        mu1 = torch.zeros_like(mu0)
    return mu0, mu1, mu0 + mu1


def mu_for_matched_stat_lstm(
    cs_log_f: torch.Tensor,
    cs_ratio: torch.Tensor,
    expr: torch.Tensor,
    ell: int,
    include_first_order: bool,
    out_dtype: torch.dtype,
):
    """Return LSTM matched-statistic kernel parts on the shifted window."""
    B, T, H = expr.shape
    if ell <= 0 or ell >= T:
        z = torch.zeros(B, 0, H, dtype=out_dtype, device=expr.device)
        return z, z, z
    prod_f = _win_prod_from_cs(cs_log_f, ell, out_dtype=out_dtype)
    mu0 = expr[:, ell:T, :].double().to(out_dtype) * prod_f
    if include_first_order and mu0.numel() > 0:
        sum_ratio = _win_sum_from_cs(cs_ratio, ell, out_dtype=out_dtype)
        mu1 = mu0 * sum_ratio
    else:
        mu1 = torch.zeros_like(mu0)
    return mu0, mu1, mu0 + mu1


def mu_for_envelope_lstm(
    cs_log_f: torch.Tensor,
    cs_ratio: torch.Tensor,
    expr: torch.Tensor,
    ell: int,
    include_first_order: bool,
    out_dtype: torch.dtype,
):
    """Return LSTM envelope kernel parts on the unshifted window."""
    B, T, H = expr.shape
    if ell <= 0 or ell > T:
        z = torch.zeros(B, 0, H, dtype=out_dtype, device=expr.device)
        return z, z, z
    prod_f = _env_prod_from_cs(cs_log_f, ell, out_dtype=out_dtype)
    mu0 = expr[:, (ell - 1):T, :].double().to(out_dtype) * prod_f
    if include_first_order and mu0.numel() > 0:
        sum_ratio = _env_sum_from_cs(cs_ratio, ell, out_dtype=out_dtype)
        mu1 = mu0 * sum_ratio
    else:
        mu1 = torch.zeros_like(mu0)
    return mu0, mu1, mu0 + mu1


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
# Diagnostics pipeline
# ============================================================

def run_for_model(args, model_name: str, mdir: str,
                  Xtr_cpu: torch.Tensor, Ytr_cpu: torch.Tensor,
                  Xdg_cpu: torch.Tensor, Ydg_cpu: torch.Tensor,
                  device: torch.device,
                  u_vec: Optional[np.ndarray]) -> Dict:
    """Train one model and run the learnability diagnostics."""
    model = build_model(
        model_name,
        args.D,
        args.H,
        ln=args.layernorm,
        gru_init_update=args.gru_init_update,
        lstm_init_forget=args.lstm_init_forget,
    ).to(device)

    log(f"[run:{model_name}] train start")
    t_train0 = now_s()
    opt, nan_halt, train_meta = train_model(
        args, model, Xtr_cpu, Ytr_cpu, mdir, model_name, device=device, u_vec=u_vec
    )
    log(f"[run:{model_name}] train done  dt={now_s()-t_train0:.1f}s")

    selected_ckpt_path = save_checkpoint(
        mdir, model_name, "selected", model, opt, args, u_vec, extra_payload=train_meta
    )
    final_ckpt_path = save_final_checkpoint(
        mdir, model_name, model, opt, args, u_vec, extra_payload=train_meta
    )
    selection_meta = dict(train_meta)
    selection_meta.update({
        "model_name": str(model_name),
        "selected_checkpoint": os.path.basename(selected_ckpt_path),
        "compat_final_checkpoint": os.path.basename(final_ckpt_path),
        "selection_gap_val_loss": (
            float(train_meta["last_val_loss"] - train_meta["best_val_loss"])
            if np.isfinite(train_meta["last_val_loss"]) and np.isfinite(train_meta["best_val_loss"])
            else float("nan")
        ),
    })
    selection_meta_path = save_selection_metadata(mdir, model_name, selection_meta)
    log(f"[diag:{model_name}] saved selected checkpoint -> {selected_ckpt_path}")
    log(f"[diag:{model_name}] saved compatibility final checkpoint -> {final_ckpt_path}")
    log(f"[diag:{model_name}] saved selection metadata -> {selection_meta_path}")

    if nan_halt:
        log(f"[diag:{model_name}] WARNING: training diverged (NaN halt). "
            f"Adaptive base rates will fall back to uniform lr={args.lr:.2e}. "
            f"Diagnostic metrics may be unreliable.")

    # ------------------------------------------------------------------
    # Generalized effective learning rates: compute per-neuron adaptive
    # base rates from optimizer second-moment state.
    #
    # Two versions:
    #   lambda_matrix (H,H): full per-parameter rate matrix λ_{qj} for
    #       lag-dependent Rayleigh-quotient projection (exact).
    #   Lambda_q_rowmean (H,): row mean of lambda_matrix — the
    #       LN-approximated, lag-independent base rate (for comparison).
    # ------------------------------------------------------------------
    lambda_matrix, Lambda_q_rowmean = extract_adaptive_rate_matrix(model, opt, lr=args.lr)
    use_lag_dependent = (lambda_matrix is not None)

    # Fallback for SGD or corrupted state: uniform lr
    Lambda_q_fallback = torch.tensor(Lambda_q_rowmean, dtype=torch.float64, device=device)  # (H,)
    if use_lag_dependent:
        lambda_matrix = lambda_matrix.to(device)
        log(f"[diag:{model_name}] using LAG-DEPENDENT Rayleigh-quotient base rates "
            f"(row-mean range [{Lambda_q_rowmean.min():.4e}, {Lambda_q_rowmean.max():.4e}])")
    else:
        log(f"[diag:{model_name}] using UNIFORM base rate lr={args.lr:.4e} (SGD or no second-moment state)")

    # Save LN-approximated (row-mean) base rates to CSV for comparison
    Lambda_q_path = os.path.join(mdir, f"{model_name}_adaptive_base_rates.csv")
    with open(Lambda_q_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["neuron_q", "Lambda_q"])
        for q_idx in range(len(Lambda_q_rowmean)):
            w.writerow([q_idx, float(Lambda_q_rowmean[q_idx])])

    del opt  # free optimizer memory

    model.eval()

    ells = np.linspace(args.lag_min, args.lag_max, args.num_lags, dtype=int)
    ells_list = [int(e) for e in ells]

    Bdg, Tdg, _ = Xdg_cpu.shape
    Hdim = int(args.H)

    # Per-sequence matched-statistic means ψ̄_n(ℓ).
    psi_seq_lists: Dict[int, list] = {ell: [] for ell in ells_list}
    # Per-projection, per-sequence matched statistic means
    #     psi_bar^{(n,k)}(ell) = (1/(T-ell)) * sum_t psi^{(n,k)}_{t,ell}
    # stored as list of (Bb, K) arrays per batch; concatenated along
    # axis 0 at the end to give a (N, K) matrix per lag.  Used only for
    # post-hoc cross-projection diagnostics (K-convergence, UQ); downstream
    # statistics still consume the K-averaged psi_seq_lists.
    psi_seq_per_proj_lists: Dict[int, list] = {ell: [] for ell in ells_list}

    sum_mass: Dict[int, float] = {ell: 0.0 for ell in ells_list}
    sum_log_mass: Dict[int, float] = {ell: 0.0 for ell in ells_list}
    count_seq: Dict[int, int] = {ell: 0 for ell in ells_list}
    sum_unit: Dict[int, np.ndarray] = {ell: np.zeros(Hdim, dtype=np.float64) for ell in ells_list}
    sum_unit_zero_order: Dict[int, np.ndarray] = {ell: np.zeros(Hdim, dtype=np.float64) for ell in ells_list}
    sum_unit_first_order: Dict[int, np.ndarray] = {ell: np.zeros(Hdim, dtype=np.float64) for ell in ells_list}

    # Envelope decomposition terms.
    sum_mass_gates: Dict[int, float] = {ell: 0.0 for ell in ells_list}
    sum_unit_gates: Dict[int, np.ndarray] = {ell: np.zeros(Hdim, dtype=np.float64) for ell in ells_list}

    # Lag-dependent base rate statistics: track per-lag Λ^(q)_{r,ℓ} distribution
    sum_lambda_mean: Dict[int, float] = {ell: 0.0 for ell in ells_list}
    sum_lambda_sq: Dict[int, float] = {ell: 0.0 for ell in ells_list}
    count_lambda: Dict[int, int] = {ell: 0 for ell in ells_list}

    try:
        log(f"[run:{model_name}] diag stream start: Bdg={Bdg} T={Tdg} H={Hdim} num_lags={len(ells_list)}")
        log(f"[diag:{model_name}] orient_matched_statistic_sign={int(bool(args.orient_matched_statistic_sign))}")
        log(f"[diag:{model_name}] num_projections K={int(max(1, int(args.num_projections)))} "
            f"(w_seed base={int(args.w_seed)})")
        log(f"[diag:{model_name}] per-sequence means: {Bdg} sequences per lag")
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
                yhat, hseq, g = model.forward_with_intermediates(xb)

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

            # ----------------------------------------------------------
            # Pass 1: envelope statistics (no projection dependence).
            # ----------------------------------------------------------
            for ell in ells_list:
                if ell <= 0 or ell >= Tdg:
                    continue

                with torch.no_grad():
                    if model_name == "gru":
                        mu_env0, mu_env1, mu_env = mu_for_envelope_gru(
                            cs_log_leak,
                            cs_log_reset,
                            cs_log_eta,
                            cs_ratio,
                            ell,
                            include_first_order=bool(args.include_first_order_diag),
                            out_dtype=torch.float64,
                        )
                    else:
                        mu_env0, mu_env1, mu_env = mu_for_envelope_lstm(
                            cs_log_f,
                            cs_ratio,
                            expr,
                            ell,
                            include_first_order=bool(args.include_first_order_diag),
                            out_dtype=torch.float64,
                        )

                    if mu_env0.numel() > 0:
                        abs_gate_only = torch.abs(mu_env0).double()
                        gates_mass = (args.lr * abs_gate_only.sum(dim=2).mean(dim=1)).sum().item()
                        sum_mass_gates[ell] += float(gates_mass)
                        sum_unit_gates[ell] += (
                            args.lr * abs_gate_only.mean(dim=1).sum(dim=0)
                        ).detach().cpu().numpy()

                    if use_lag_dependent and mu_env.numel() > 0:
                        Lambda_ell_env = compute_lag_dependent_rates(
                            lambda_matrix, hseq, mu_env.shape[1], Lambda_q_fallback
                        ).to(mu_env.dtype)
                        mu_env = mu_env * Lambda_ell_env
                        mu_env0 = mu_env0 * Lambda_ell_env
                        mu_env1 = mu_env1 * Lambda_ell_env
                        lam_flat = Lambda_ell_env.detach().double()
                        lam_mean_val = lam_flat.mean().item()
                        lam_sq_val = (lam_flat ** 2).mean().item()
                        n_lam = int(lam_flat.numel())
                        sum_lambda_mean[ell] += lam_mean_val * n_lam
                        sum_lambda_sq[ell] += lam_sq_val * n_lam
                        count_lambda[ell] += n_lam
                        del Lambda_ell_env
                    else:
                        fallback_env = Lambda_q_fallback.unsqueeze(0).unsqueeze(0)
                        mu_env = mu_env * fallback_env
                        mu_env0 = mu_env0 * fallback_env
                        mu_env1 = mu_env1 * fallback_env

                    if mu_env.numel() > 0:
                        abs_mu = torch.abs(mu_env).double()
                        abs_mu0 = torch.abs(mu_env0).double()
                        abs_mu1 = torch.abs(mu_env1).double()
                        mass_per_seq = abs_mu.sum(dim=2).mean(dim=1)  # (Bb,)
                        sum_mass[ell] += float(mass_per_seq.sum().item())
                        sum_log_mass[ell] += float(torch.log(mass_per_seq + 1e-30).sum().item())
                        count_seq[ell] += int(mass_per_seq.shape[0])
                        sum_unit[ell] += abs_mu.mean(dim=1).sum(dim=0).detach().cpu().numpy()
                        sum_unit_zero_order[ell] += abs_mu0.mean(dim=1).sum(dim=0).detach().cpu().numpy()
                        sum_unit_first_order[ell] += abs_mu1.mean(dim=1).sum(dim=0).detach().cpu().numpy()

            # ----------------------------------------------------------
            # Pass 2: matched statistic, aggregated over K random
            # projections.  OPTIMISED LOOP ORDER: precompute all K
            # JVPs (parked on CPU), then iterate ell-outer / k-inner
            # so that the lag kernel (mu, rates, mu*delta) is built
            # once per ell instead of K times.
            #
            # The per-sequence multi-projection output
            #     psi_bar^{(n)}(ell, K) = (1/K) sum_k psi_bar^{(n,k)}(ell)
            # matches the appendix definition of
            # \widetilde{\bar S}^{(n)}_{ell,K}.
            # ----------------------------------------------------------
            K_proj = int(max(1, int(args.num_projections)))

            # 2a. Precompute all K JVP sequences; store on CPU to keep
            #     GPU memory free for the per-ell tensor work.
            vseqs_cpu = []
            for k_idx in range(K_proj):
                w_seed_k = int(args.w_seed) + int(k_idx)
                vk = compute_vseq_jvp(model, xb, w_seed=w_seed_k).detach()
                vseqs_cpu.append(vk.cpu())
                del vk
            if device.type == "cuda" and args.cuda_sync:
                torch.cuda.synchronize()

            psi_seq_means_sum: Dict[int, Optional[np.ndarray]] = {
                ell: None for ell in ells_list
            }
            # Per-projection per-sequence matched-statistic means for this
            # batch: psi_bar^{(n,k)}(ell) stored in a (Bb, K) matrix per ell.
            psi_seq_means_per_proj: Dict[int, Optional[np.ndarray]] = {
                ell: None for ell in ells_list
            }

            # 2b. ell-outer loop: compute lag kernel once per ell,
            #     then sweep the K projections in the inner loop.
            for ell in ells_list:
                if ell <= 0 or ell >= Tdg:
                    continue

                with torch.no_grad():
                    if model_name == "gru":
                        mu_ms0, _, mu_ms = mu_for_matched_stat_gru(
                            cs_log_leak,
                            cs_log_reset,
                            cs_log_eta,
                            cs_ratio,
                            ell,
                            include_first_order=bool(args.include_first_order_diag),
                            out_dtype=torch.float64,
                        )
                    else:
                        mu_ms0, _, mu_ms = mu_for_matched_stat_lstm(
                            cs_log_f,
                            cs_ratio,
                            expr,
                            ell,
                            include_first_order=bool(args.include_first_order_diag),
                            out_dtype=torch.float64,
                        )

                    if use_lag_dependent and mu_ms.numel() > 0:
                        Lambda_ell_ms = compute_lag_dependent_rates(
                            lambda_matrix, hseq, mu_ms.shape[1], Lambda_q_fallback
                        ).to(mu_ms.dtype)
                        mu_used = (mu_ms if bool(args.include_first_order_diag) else mu_ms0) * Lambda_ell_ms
                        del Lambda_ell_ms
                    else:
                        fallback_ms = Lambda_q_fallback.unsqueeze(0).unsqueeze(0)
                        mu_used = (mu_ms if bool(args.include_first_order_diag) else mu_ms0) * fallback_ms

                    if mu_used.numel() == 0:
                        continue

                    # mu_delta = mu_used * delta_all depends only on ell;
                    # precompute once, then the k-loop needs a single
                    # element-wise multiply + sum.
                    delta_ell = delta[:, ell:Tdg, :]              # (Bb, T-ell, H)
                    mu_delta = mu_used * delta_ell
                    del mu_used, delta_ell

                    for k_idx in range(K_proj):
                        v_past_all = vseqs_cpu[k_idx][:, 0:(Tdg - ell), :].to(
                            device, non_blocking=True
                        )
                        psi_k = torch.sum(mu_delta * v_past_all, dim=2)  # (Bb, T-ell)
                        del v_past_all

                        # Per-projection sign orientation: compute sign
                        # from psi_k so that each projection is oriented
                        # on its own before being averaged across k.
                        if bool(args.orient_matched_statistic_sign):
                            mu_psi = psi_k.mean()
                            if torch.isfinite(mu_psi):
                                sgn = torch.sign(mu_psi)
                                if float(sgn.item()) == 0.0:
                                    sgn = torch.tensor(1.0, device=psi_k.device)
                            else:
                                sgn = torch.tensor(1.0, device=psi_k.device)
                            psi_k = sgn * psi_k

                        # Per-sequence mean for projection k: psi_bar^{(n,k)}(ell).
                        psi_seq_means_k = psi_k.mean(dim=1).detach().cpu().numpy().astype(np.float64)

                        prev = psi_seq_means_sum[ell]
                        if prev is None:
                            psi_seq_means_sum[ell] = psi_seq_means_k
                        else:
                            psi_seq_means_sum[ell] = prev + psi_seq_means_k

                        # Stash per-projection means for post-hoc diagnostics.
                        perK = psi_seq_means_per_proj[ell]
                        if perK is None:
                            perK = np.empty((psi_seq_means_k.shape[0], K_proj), dtype=np.float64)
                            psi_seq_means_per_proj[ell] = perK
                        perK[:, k_idx] = psi_seq_means_k

                    del mu_delta

            del vseqs_cpu
            if device.type == "cuda" and args.cuda_sync:
                torch.cuda.synchronize()

            # Average per-sequence means over the K projections and
            # append the aggregated psi_bar^{(n)}(ell, K).
            for ell in ells_list:
                acc = psi_seq_means_sum[ell]
                if acc is None:
                    continue
                psi_seq_lists[ell].append(acc / float(K_proj))
                perK = psi_seq_means_per_proj[ell]
                if perK is not None:
                    psi_seq_per_proj_lists[ell].append(perK)

            del xb, yb, yhat, hseq, g, err, delta
            if device.type == "cuda" and args.cuda_sync:
                torch.cuda.synchronize()

        log(f"[run:{model_name}] diag stream done  dt={now_s()-t_diag0:.1f}s")
        log(f"[run:{model_name}] stats+write start")

        mu_by_ell: Dict[int, float] = {}
        log_mu_by_ell: Dict[int, float] = {}
        Nreq_by_ell_ecf: Dict[int, int] = {}
        Nreq_by_ell_mcc: Dict[int, int] = {}
        alpha_by_ell_ecf: Dict[int, float] = {}
        alpha_by_ell_mcc: Dict[int, float] = {}
        mu_units_by_ell: Dict[int, np.ndarray] = {}
        mu_units_zero_order_by_ell: Dict[int, np.ndarray] = {}
        mu_units_first_order_by_ell: Dict[int, np.ndarray] = {}

        # Update min samples threshold from CLI
        global _MIN_SAMPLES_ALPHA
        _MIN_SAMPLES_ALPHA = getattr(args, "min_samples_alpha", 500)

        summary_path = os.path.join(mdir, f"{model_name}_summary.csv")
        with open(summary_path, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow([
                "ell", "mu_l1_mean", "log_mu_l1_mean",
                "f_gates", "f_ratio",
                "lambda_mean", "lambda_std",
                "alpha_ecf", "sigma_ecf", "alpha_ecf_reliable",
                "alpha_ecf_origin", "alpha_ecf_reliability_reason",
                "alpha_ecf_n_samples_used", "alpha_ecf_used_subsample",
                "alpha_ecf_n_points_strict", "alpha_ecf_n_points_relaxed",
                "alpha_ecf_n_points_used", "alpha_ecf_filter_mode",
                "alpha_mcc", "sigma_mcc", "alpha_mcc_reliable",
                "alpha_mcc_reliability_reason",
                "alpha_mcc_ci_lo", "alpha_mcc_ci_hi",
                "alpha_mcc_bootstrap_median", "alpha_mcc_ci_width",
                "alpha_mcc_quantile_ratio", "alpha_mcc_iqr",
                "alpha_methods_comparable", "alpha_methods_agree",
                "alpha_hat", "sigma_hat", "alpha_reliable", "alpha_method_used",
                "alpha_selection_reason",
                "N_required_ecf", "best_snr_ecf", "err_at_best_snr_ecf", "best_N_ecf",
                "N_required_mcc", "best_snr_mcc", "err_at_best_snr_mcc", "best_N_mcc",
                "mbar_scalar", "n_samples", "n_sequences",
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
                    mu_zero_order_per_unit = (sum_unit_zero_order[ell] / count_seq[ell]).astype(np.float64)
                    mu_first_order_per_unit = (sum_unit_first_order[ell] / count_seq[ell]).astype(np.float64)
                    f_gates_ell = float(sum_mass_gates[ell] / count_seq[ell])
                    f_ratio_ell = float(mu_mean / f_gates_ell) if f_gates_ell > 1e-30 else float("nan")
                    if count_lambda[ell] > 0:
                        lam_m = sum_lambda_mean[ell] / count_lambda[ell]
                        lam_sq_m = sum_lambda_sq[ell] / count_lambda[ell]
                        lam_std = float(max(0.0, lam_sq_m - lam_m ** 2) ** 0.5)
                        lam_mean_ell = float(lam_m)
                    else:
                        lam_mean_ell = float(args.lr)
                        lam_std = 0.0
                else:
                    mu_mean = 0.0
                    log_mu_mean = float("-inf")
                    mu_per_unit = np.zeros(Hdim, dtype=np.float64)
                    mu_zero_order_per_unit = np.zeros(Hdim, dtype=np.float64)
                    mu_first_order_per_unit = np.zeros(Hdim, dtype=np.float64)
                    f_gates_ell = 0.0
                    f_ratio_ell = float("nan")
                    lam_mean_ell = float(args.lr)
                    lam_std = 0.0

                mu_by_ell[ell] = mu_mean
                log_mu_by_ell[ell] = log_mu_mean
                mu_units_by_ell[ell] = mu_per_unit
                mu_units_zero_order_by_ell[ell] = mu_zero_order_per_unit
                mu_units_first_order_by_ell[ell] = mu_first_order_per_unit

                psi_seq_arr = np.concatenate(psi_seq_lists[ell]) if psi_seq_lists[ell] else np.array([], dtype=np.float64)
                n_seq = len(psi_seq_arr)
                mbar = float(abs(np.mean(psi_seq_arr))) if n_seq > 0 else 0.0

                _run_ecf = "ecf" in args.alpha_methods
                _run_mcc = "mcc" in args.alpha_methods

                if _run_ecf:
                    ecf_info = estimate_alpha_sigma_with_meta(psi_seq_arr, method="ecf")
                else:
                    ecf_info = _default_alpha_meta("ecf", n_seq)
                    ecf_info["method_origin"] = "disabled"
                    ecf_info["method_reason"] = "disabled"
                    ecf_info["reliability_reason"] = "disabled"

                if _run_mcc:
                    mcc_info = estimate_alpha_sigma_with_meta(psi_seq_arr, method="mcculloch")
                else:
                    mcc_info = _default_alpha_meta("mcculloch", n_seq)
                    mcc_info["method_origin"] = "disabled"
                    mcc_info["method_reason"] = "disabled"
                    mcc_info["reliability_reason"] = "disabled"

                alpha_ecf = float(ecf_info["alpha_hat"])
                sigma_ecf = float(ecf_info["sigma_hat"])
                rel_ecf = bool(ecf_info["reliable"])
                alpha_mcc = float(mcc_info["alpha_hat"])
                sigma_mcc = float(mcc_info["sigma_hat"])
                rel_mcc = bool(mcc_info["reliable"])

                alpha_by_ell_ecf[ell] = float(alpha_ecf)
                alpha_by_ell_mcc[ell] = float(alpha_mcc)

                # SNR and detectability.
                best_snr_ecf_val, best_err_ecf, best_N_ecf, N_req_ecf = float("nan"), float("nan"), None, -1
                if _run_ecf:
                    best_snr_ecf_val = -1e18
                    best_err_ecf = 1e18
                    for Nuse in args.N_grid:
                        Nuse = int(Nuse)
                        snr = compute_snr(alpha_ecf, sigma_ecf, mbar, Nuse)
                        if (snr > args.eps) and (N_req_ecf == -1):
                            N_req_ecf = Nuse
                        if snr > best_snr_ecf_val:
                            best_snr_ecf_val = snr
                            Nuse_capped = min(Nuse, n_seq)
                            best_err_ecf = detection_error_on_prefix_arr(psi_seq_arr, Nuse_capped) if n_seq > 0 else float("nan")
                            best_N_ecf = Nuse

                Nreq_by_ell_ecf[ell] = int(N_req_ecf)

                best_snr_mcc_val, best_err_mcc, best_N_mcc, N_req_mcc = float("nan"), float("nan"), None, -1
                if _run_mcc:
                    best_snr_mcc_val = -1e18
                    best_err_mcc = 1e18
                    for Nuse in args.N_grid:
                        Nuse = int(Nuse)
                        snr = compute_snr(alpha_mcc, sigma_mcc, mbar, Nuse)
                        if (snr > args.eps) and (N_req_mcc == -1):
                            N_req_mcc = Nuse
                        if snr > best_snr_mcc_val:
                            best_snr_mcc_val = snr
                            Nuse_capped = min(Nuse, n_seq)
                            best_err_mcc = detection_error_on_prefix_arr(psi_seq_arr, Nuse_capped) if n_seq > 0 else float("nan")
                            best_N_mcc = Nuse

                Nreq_by_ell_mcc[ell] = int(N_req_mcc)

                # Bootstrap confidence intervals for McCulloch estimate
                alpha_mcc_ci_lo = float("nan")
                alpha_mcc_ci_hi = float("nan")
                alpha_mcc_median = alpha_mcc  # default to point estimate
                if _run_mcc and n_seq >= 4 and np.isfinite(alpha_mcc):
                    alpha_mcc_median, alpha_mcc_ci_lo, alpha_mcc_ci_hi, _ = bootstrap_mcculloch(
                        psi_seq_arr,
                        estimate_alpha_sigma_mcculloch_symmetric_from_quantiles,
                        n_boot=args.alpha_n_boot,
                        ci=0.95
                    )

                mcc_ci_width = (alpha_mcc_ci_hi - alpha_mcc_ci_lo) if (
                    np.isfinite(alpha_mcc_ci_lo) and np.isfinite(alpha_mcc_ci_hi)
                ) else float("inf")

                alpha_methods_comparable = int(
                    _run_ecf and _run_mcc
                    and str(ecf_info["method_origin"]) == "ecf_regression"
                    and np.isfinite(alpha_ecf)
                    and np.isfinite(alpha_mcc_ci_lo)
                    and np.isfinite(alpha_mcc_ci_hi)
                )
                if alpha_methods_comparable:
                    alpha_methods_agree = int(alpha_mcc_ci_lo <= alpha_ecf <= alpha_mcc_ci_hi)
                else:
                    alpha_methods_agree = -1

                if _run_ecf and rel_ecf and str(ecf_info["method_origin"]) == "ecf_regression":
                    alpha_hat = alpha_ecf
                    sigma_hat_unified = sigma_ecf
                    alpha_reliable = True
                    alpha_method_used = "ecf"
                    alpha_selection_reason = "ecf_regression_reliable"
                elif _run_mcc and np.isfinite(alpha_mcc_median):
                    alpha_hat = float(alpha_mcc_median)
                    sigma_hat_unified = sigma_mcc
                    alpha_reliable = bool(mcc_ci_width < 0.3)
                    alpha_method_used = "mcculloch"
                    alpha_selection_reason = "mcculloch_available"
                else:
                    alpha_hat = 2.0
                    sigma_hat_unified = 0.0
                    alpha_reliable = False
                    alpha_method_used = "none"
                    alpha_selection_reason = "no_reliable_alpha_estimator"

                wcsv.writerow([
                    int(ell), float(mu_mean), float(log_mu_mean),
                    float(f_gates_ell), float(f_ratio_ell),
                    float(lam_mean_ell), float(lam_std),
                    float(alpha_ecf), float(sigma_ecf), int(rel_ecf),
                    ecf_info["method_origin"], ecf_info["reliability_reason"],
                    int(ecf_info["n_samples_used"]), int(ecf_info["used_subsample"]),
                    int(ecf_info["ecf_n_points_strict"]), int(ecf_info["ecf_n_points_relaxed"]),
                    int(ecf_info["ecf_n_points_used"]), ecf_info["ecf_filter_mode"],
                    float(alpha_mcc), float(sigma_mcc), int(rel_mcc),
                    mcc_info["reliability_reason"],
                    float(alpha_mcc_ci_lo), float(alpha_mcc_ci_hi),
                    float(alpha_mcc_median), float(mcc_ci_width),
                    float(mcc_info["quantile_ratio"]), float(mcc_info["iqr"]),
                    int(alpha_methods_comparable), int(alpha_methods_agree),
                    float(alpha_hat), float(sigma_hat_unified), int(alpha_reliable), alpha_method_used,
                    alpha_selection_reason,
                    int(N_req_ecf), float(best_snr_ecf_val), float(best_err_ecf), int(best_N_ecf if best_N_ecf is not None else -1),
                    int(N_req_mcc), float(best_snr_mcc_val), float(best_err_mcc), int(best_N_mcc if best_N_mcc is not None else -1),
                    float(mbar), n_seq * max(1, Tdg - ell), n_seq,
                ])

                # Free per-sequence list for this ell to release memory
                del psi_seq_lists[ell]

        sorted_ells = sorted(mu_units_by_ell.keys())
        ell_values = np.asarray(sorted_ells, dtype=np.int64)
        mu_units_matrix = np.vstack([mu_units_by_ell[e] for e in sorted_ells]).astype(np.float64)
        mu_zero_order_matrix = np.vstack(
            [mu_units_zero_order_by_ell[e] for e in sorted_ells]
        ).astype(np.float64)
        mu_first_order_matrix = np.vstack(
            [mu_units_first_order_by_ell[e] for e in sorted_ells]
        ).astype(np.float64)
        mu_units_path = os.path.join(mdir, f"{model_name}_mu_units.npz")
        save_dense_unit_npz(
            mu_units_path,
            ell_values,
            mu_units_matrix,
            component="total",
            rate_scale="adaptive",
            extra_arrays={
                "zero_order_values": mu_zero_order_matrix,
                "first_order_values": mu_first_order_matrix,
            },
        )

        # per-unit envelope decomposition: f_gates per unit
        mu_units_gates_path = os.path.join(mdir, f"{model_name}_mu_units_gates.npz")
        mu_gates_matrix = np.vstack([
            (sum_unit_gates[e] / max(1, count_seq[e])).astype(np.float64)
            for e in sorted_ells
        ])
        save_dense_unit_npz(
            mu_units_gates_path,
            ell_values,
            mu_gates_matrix,
            component="zero_order_gate_only",
            rate_scale="base_lr",
        )

        # Per-projection per-sequence matched statistic tensor:
        # psi_per_proj[l, n, k] = psi_bar^{(n,k)}(ell_l).  Enables post-hoc
        # K-convergence checks and free cross-projection UQ without affecting
        # any downstream consumer (which uses the K-averaged psi_seq_lists).
        K_proj_save = int(max(1, int(args.num_projections)))
        if any(len(psi_seq_per_proj_lists[e]) > 0 for e in sorted_ells):
            per_proj_blocks = []
            valid = True
            N_ref = None
            for e in sorted_ells:
                chunks = psi_seq_per_proj_lists[e]
                if not chunks:
                    valid = False
                    break
                arr = np.concatenate(chunks, axis=0)  # (N, K)
                if N_ref is None:
                    N_ref = arr.shape[0]
                elif arr.shape[0] != N_ref:
                    valid = False
                    break
                per_proj_blocks.append(arr)
            if valid and N_ref is not None and per_proj_blocks:
                psi_per_proj_arr = np.stack(per_proj_blocks, axis=0)  # (L, N, K)
                psi_per_proj_path = os.path.join(mdir, f"{model_name}_psi_per_proj.npz")
                np.savez_compressed(
                    psi_per_proj_path,
                    ell=ell_values,
                    psi_per_proj=psi_per_proj_arr.astype(np.float64),
                    num_projections=np.int64(K_proj_save),
                    w_seed_base=np.int64(args.w_seed),
                )
                # Across-projection summary: mean_n psi_bar^{(n,k)}(ell)
                # gives a K-vector per lag; std/|mean| of that vector
                # quantifies residual Monte-Carlo error from the projection
                # average.
                proj_means = psi_per_proj_arr.mean(axis=1)  # (L, K)
                across_mean = proj_means.mean(axis=1)
                # Sign-agnostic per-lag UQ: mean_k | mean_n psi_bar^{(n,k)} |
                # stays meaningful even when projections are not sign-oriented.
                across_mean_abs = np.mean(np.abs(proj_means), axis=1)
                if K_proj_save > 1:
                    across_std = proj_means.std(axis=1, ddof=1)
                else:
                    across_std = np.zeros_like(across_mean)
                sem = across_std / float(max(1, K_proj_save)) ** 0.5
                summary_csv = os.path.join(mdir, f"{model_name}_psi_per_proj_summary.csv")
                with open(summary_csv, "w", newline="") as fsum:
                    wsum = csv.writer(fsum)
                    wsum.writerow([
                        "ell", "K", "N",
                        "psi_mean_across_proj", "psi_mean_abs_across_proj",
                        "psi_std_across_proj", "psi_sem_across_proj",
                        "psi_rel_std_across_proj", "psi_rel_std_abs_across_proj",
                    ])
                    for i, e in enumerate(sorted_ells):
                        m = float(across_mean[i])
                        mabs = float(across_mean_abs[i])
                        s = float(across_std[i])
                        rel = float(s / abs(m)) if abs(m) > 1e-30 else float("nan")
                        rel_abs = float(s / mabs) if mabs > 1e-30 else float("nan")
                        wsum.writerow([
                            e, K_proj_save, int(N_ref),
                            m, mabs, s, float(sem[i]), rel, rel_abs,
                        ])
                log(f"[run:{model_name}] saved psi_per_proj npz "
                    f"shape=(L={len(sorted_ells)}, N={N_ref}, K={K_proj_save})")

        # per-unit tau fits (always create outputs)
        tau_list = []
        tau_mu_results = []
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
                    tau_mu_results.append({"unit_id": q, **fit_res})
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

        # Per-neuron Λ_q × τ_q correlation.
        if len(tau_mu_results) >= 5:
            tau_unit_ids = np.array([r["unit_id"] for r in tau_mu_results], dtype=int)
            tau_vals = np.array([r["tau"] for r in tau_mu_results], dtype=float)
            lam_vals = Lambda_q_rowmean[tau_unit_ids]

            max_ell = max(mu_units_by_ell.keys())
            mu_at_max_ell = mu_units_by_ell[max_ell][tau_unit_ids]

            def _spearman_r(x, y):
                n = len(x)
                if n < 3:
                    return float("nan")
                rx = np.empty(n)
                ry = np.empty(n)
                rx[np.argsort(x)] = np.arange(n, dtype=float)
                ry[np.argsort(y)] = np.arange(n, dtype=float)
                rx -= rx.mean()
                ry -= ry.mean()
                denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
                if denom < 1e-30:
                    return float("nan")
                return float((rx * ry).sum() / denom)

            rho_lam_tau = _spearman_r(lam_vals, tau_vals)
            pearson_lam_tau = float(np.corrcoef(lam_vals, tau_vals)[0, 1]) if len(lam_vals) >= 3 else float("nan")

            log(f"[lambda_tau:{model_name}] Spearman ρ(Λ_q, τ_q) = {rho_lam_tau:.3f}, "
                f"Pearson r = {pearson_lam_tau:.3f}  (n={len(tau_vals)} neurons)")

            lt_csv_path = os.path.join(mdir, f"{model_name}_lambda_tau_correlation.csv")
            with open(lt_csv_path, "w", newline="") as flt:
                lt_writer = csv.writer(flt)
                lt_writer.writerow(["neuron_q", "Lambda_q", "tau_q", "mu_at_max_ell"])
                for j in range(len(tau_unit_ids)):
                    lt_writer.writerow([
                        int(tau_unit_ids[j]),
                        float(lam_vals[j]),
                        float(tau_vals[j]),
                        float(mu_at_max_ell[j])
                    ])

            lt_stats = {
                "model": model_name,
                "n_neurons": int(len(tau_vals)),
                "spearman_rho_lambda_tau": rho_lam_tau,
                "pearson_r_lambda_tau": pearson_lam_tau,
                "lambda_mean": float(lam_vals.mean()),
                "lambda_std": float(lam_vals.std()),
                "tau_mean": float(tau_vals.mean()),
                "tau_std": float(tau_vals.std()),
                "interpretation": (
                    "slow_units_emphasized"
                    if rho_lam_tau > 0.2 else
                    "fast_units_emphasized"
                    if rho_lam_tau < -0.2 else
                    "approximately_uniform"
                ),
            }
            with open(os.path.join(mdir, f"{model_name}_lambda_tau_stats.json"), "w") as jf:
                json.dump(lt_stats, jf, indent=2)

        log(f"[run:{model_name}] stats+write done")

        return {
            "model": model_name,
            "ells": sorted_ells,
            "mu_by_ell": mu_by_ell,
            "log_mu_by_ell": log_mu_by_ell,
            "Nreq_by_ell_ecf": Nreq_by_ell_ecf,
            "Nreq_by_ell_mcc": Nreq_by_ell_mcc,
            "alpha_by_ell_ecf": alpha_by_ell_ecf,
            "alpha_by_ell_mcc": alpha_by_ell_mcc,
        }

    except Exception:
        log(f"[ERROR:{model_name}] failed.")
        traceback.print_exc()
        raise


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
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd", "sgd_momentum", "rmsprop"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--rmsprop_alpha", type=float, default=0.99,
                   help="Smoothing coefficient for RMSprop's running average of "
                        "squared gradients (torch calls this 'alpha'). Default: 0.99.")
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
                   help=(
                       "Raw SNR detection threshold: lag is detectable when SNR > eps. "
                       "Larger eps is stricter. Kept for backward compatibility; "
                       "prefer --noise_tolerance for a more interpretable interface."
                   ))
    p.add_argument("--noise_tolerance", type=float, default=None,
                   help=(
                       "User-facing inverse threshold in (0, 1]. "
                       "Defined as noise_tolerance = 1 / eps, so smaller values are stricter. "
                       "Example: --noise_tolerance 0.1 means require SNR > 10, i.e. "
                       "the effective noise-to-signal ratio in the detection metric "
                       "must be below about 10%%. Overrides --eps when provided."
                   ))

    # --- Alpha estimation -------------------------------------------------------
    p.add_argument("--alpha_methods", type=str, default="ecf,mcc",
                   help=(
                       "Comma-separated list of alpha estimation methods to run. "
                       "Choices: 'ecf' (Koutrouvelis 1980 ECF regression), "
                       "'mcc' (McCulloch 1986 quantile method). "
                       "Default: 'ecf,mcc' (both). Use 'ecf' or 'mcc' for a single method."
                   ))
    p.add_argument("--min_samples_alpha", type=int, default=500,
                   help=(
                       "Minimum samples required for a reliable α̂ estimate. "
                       "Default: 500."
                   ))
    p.add_argument("--alpha_n_boot", type=int, default=200,
                   help=(
                       "Number of bootstrap resamples for McCulloch confidence intervals. "
                       "Default: 200."
                   ))

    # --- JVP / matched statistic ------------------------------------------------
    p.add_argument("--w_seed", type=int, default=12345,
                   help="Base seed for the random tangent direction w in JVP computation.")
    p.add_argument("--num_projections", type=int, default=1,
                   help=(
                       "Number of independent random tangent directions w_1,...,w_K "
                       "to aggregate. For K>1, the K-th projection uses "
                       "w_seed + (k-1) as its seed. The per-sequence matched "
                       "statistic is averaged over the K projections. K=1 "
                       "reproduces the single-projection baseline."
                   ))
    p.add_argument("--include_first_order_diag", type=int, default=1,
                   help="If 1, include first-order correction in matched-stat kernel.")

    # --- Init / normalisation switches ------------------------------------------
    p.add_argument("--orth_init", action="store_true",
                   help="Apply orthogonal init to recurrent weights (respects _skip_orth).")
    p.add_argument("--layernorm", action="store_true",
                   help="Enable LayerNorm on candidate pre-activation.")
    p.add_argument("--gru_init_update", type=float, default=0.5,
                   help=(
                       "Initial GRU update-gate value z at t=0 when gate weights are zeroed. "
                       "Smaller is more memory-preserving because leak = 1 - z. "
                       "Example: 0.05 means initial leak is about 0.95."
                   ))
    p.add_argument("--lstm_init_forget", type=float, default=0.5,
                   help=(
                       "Initial LSTM forget-gate value f at t=0 when forget weights are zeroed. "
                       "Example: 0.95 means retain about 95%% of cell state per step initially."
                   ))

    p.add_argument("--log_gate_stats", type=int, default=1)
    p.add_argument("--gate_log_every", type=int, default=10)

    # --- Device -----------------------------------------------------------------
    p.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "mps", "cuda"])

    # --- Runtime / throughput knobs --------------------------------------------
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
        default=0,
        help=(
            "If 1, flip samples per lag by sign(mean(psi)). "
            "Default 0 keeps the raw matched statistic."
        )
    )

    args = p.parse_args()

    args.task_lags = [int(s) for s in args.task_lags.split(",") if s.strip() != ""]
    args.task_coeffs = [float(s) for s in args.task_coeffs.split(",") if s.strip() != ""]
    args.N_grid = [int(s) for s in args.N_grid.split(",") if s.strip() != ""]
    if not (0.0 < args.gru_init_update < 1.0):
        raise ValueError(f"--gru_init_update must lie in (0, 1), got {args.gru_init_update}")
    if not (0.0 < args.lstm_init_forget < 1.0):
        raise ValueError(f"--lstm_init_forget must lie in (0, 1), got {args.lstm_init_forget}")

    if args.noise_tolerance is not None:
        args.eps = noise_tolerance_to_eps(args.noise_tolerance)
    if args.eps <= 0:
        raise ValueError(f"--eps must be positive, got {args.eps}")
    args.eps = float(args.eps)
    args.eps_raw = float(args.eps)
    args.noise_tolerance = float(1.0 / args.eps)

    args.cuda_sync = bool(int(args.cuda_sync))

    # Parse alpha methods set
    _valid_alpha = {"ecf", "mcc"}
    args.alpha_methods = {s.strip().lower() for s in args.alpha_methods.split(",") if s.strip()}
    if not args.alpha_methods & _valid_alpha:
        raise ValueError(f"--alpha_methods must contain at least one of {_valid_alpha}, got {args.alpha_methods}")
    args.alpha_methods = args.alpha_methods & _valid_alpha

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
        H_by_N_ecf = compute_H_N(res["ells"], res["Nreq_by_ell_ecf"], args.N_grid)
        H_by_N_mcc = compute_H_N(res["ells"], res["Nreq_by_ell_mcc"], args.N_grid)
        res["H_by_N_ecf"] = H_by_N_ecf
        res["H_by_N_mcc"] = H_by_N_mcc

        with open(os.path.join(mdir, f"{mname}_H_N.csv"), "w", newline="") as hf:
            wcsv = csv.writer(hf)
            wcsv.writerow(["N", "H_N_ecf", "H_N_mcc"])
            for N in sorted(set(list(H_by_N_ecf.keys()) + list(H_by_N_mcc.keys()))):
                wcsv.writerow([int(N), int(H_by_N_ecf.get(N, 0)), int(H_by_N_mcc.get(N, 0))])
        results.append(res)

        log(f"[main] done model={mname}")

    # Write aggregate H_N summary: one row per N, one column per model (both ECF and McCulloch)
    with open(os.path.join(args.outdir, "H_N_summary.csv"), "w", newline="") as hf:
        wcsv = csv.writer(hf)
        header = ["N"]
        for m in models:
            header += [f"H_N_{m}_ecf", f"H_N_{m}_mcc"]
        wcsv.writerow(header)
        for N in args.N_grid:
            row = [int(N)]
            for res in results:
                row.append(int(res["H_by_N_ecf"].get(int(N), 0)))
                row.append(int(res["H_by_N_mcc"].get(int(N), 0)))
            wcsv.writerow(row)

    log(f"Done. Results saved to: {args.outdir}")


if __name__ == "__main__":
    main()
