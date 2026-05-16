"""
Standalone alpha-stable tail-index estimators (numpy only).

Extracted from run_learnability_baselines.py so that the analysis pipeline
(per-projection diagnostic, bootstrap CIs) can run without the torch
dependency. The numerical behavior is identical to the canonical estimator
used in the training/diagnostics pipeline.

Public API:
    estimate_alpha_ecf(samples)        -> float (alpha_hat in [1.0, 2.0])
    estimate_alpha_sigma_ecf(samples)  -> (alpha_hat, sigma_hat)

Both return NaN if the estimator cannot run reliably (insufficient samples
or insufficient informative ECF grid points).
"""
import numpy as np
from typing import Tuple

_MIN_SAMPLES_ALPHA = 500


def _ecf_at_t(samples: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """|phi_hat(t)|^2 for each t in t_grid, computed in chunks for memory."""
    n = samples.size
    if n == 0:
        return np.zeros_like(t_grid)
    chunk_size = min(n, 50000)
    total_cos = None
    total_sin = None
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x_chunk = samples[start:end]
        tx = np.outer(t_grid, x_chunk)
        cos_sum = np.cos(tx).sum(axis=1)
        sin_sum = np.sin(tx).sum(axis=1)
        if total_cos is None:
            total_cos = cos_sum
            total_sin = sin_sum
        else:
            total_cos += cos_sum
            total_sin += sin_sum
    return (total_cos / n) ** 2 + (total_sin / n) ** 2


def _choose_ecf_grid(samples: np.ndarray, n_points: int = 50) -> np.ndarray:
    iqr = float(np.subtract(*np.percentile(samples, [75, 25])))
    if iqr <= 1e-12:
        iqr = float(np.std(samples)) * 1.349
    if iqr <= 1e-12:
        return np.linspace(0.1, 2.0, n_points)
    scale_est = iqr / 1.349
    return np.linspace(0.05 / scale_est, 3.0 / scale_est, n_points)


def estimate_alpha_sigma_ecf(samples: np.ndarray) -> Tuple[float, float]:
    """
    Koutrouvelis (1980) ECF regression for symmetric alpha-stable, simplified
    for beta=0. Returns (alpha_hat, sigma_hat); both NaN if not reliable.

    For S_alpha_S:  log(-log|phi_hat(t)|^2) = log(2*sigma^alpha) + alpha*log|t|
    """
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    n = samples.size
    if n < _MIN_SAMPLES_ALPHA:
        return float("nan"), float("nan")

    t_grid = _choose_ecf_grid(samples, n_points=50)
    phi2 = _ecf_at_t(samples, t_grid)

    mask_strict = (phi2 > 0.01) & (phi2 < 0.95)
    mask_relaxed = (phi2 > 1e-4) & (phi2 < 0.999)
    if int(mask_strict.sum()) >= 5:
        mask = mask_strict
    elif int(mask_relaxed.sum()) >= 3:
        mask = mask_relaxed
    else:
        return float("nan"), float("nan")

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
    sigma_hat = float((np.exp(intercept) / 2.0) ** (1.0 / alpha_hat))
    return alpha_hat, max(0.0, sigma_hat)


def estimate_alpha_ecf(samples: np.ndarray) -> float:
    """Convenience wrapper returning only alpha_hat."""
    a, _ = estimate_alpha_sigma_ecf(samples)
    return a
