"""
Standalone extraction of the ECF and McCulloch alpha-stable estimators.

Copied verbatim from run_learnability_baselines.py to avoid the torch
import dependency.  Any changes to the estimators in the main pipeline
should be mirrored here.
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


# ============================================================
# McCulloch quantile estimator for symmetric α-stable (SαS)
# ============================================================

class _StableQuantileCache:
    def __init__(self):
        self._have_scipy = False
        self.levy_stable = None
        try:
            from scipy.stats import levy_stable
            self._have_scipy = True
            self.levy_stable = levy_stable
        except Exception:
            self._have_scipy = False
            self.levy_stable = None

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

        self.cache_q: dict = {}
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
            # Fallback table has α in descending order; np.interp needs ascending xp
            al = grid[::-1, 0]
            r = np.interp(a, al, grid[::-1, 1])
            iqr = np.interp(a, al, grid[::-1, 2])
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


def estimate_alpha_sigma_mcculloch_symmetric_from_quantiles(
    q05, q25, q75, q95
) -> Tuple[float, float]:
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
# ============================================================

_MIN_SAMPLES_ALPHA = 500


def _ecf_at_t(samples: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
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

    phi2 = (total_cos / n) ** 2 + (total_sin / n) ** 2
    return phi2


def _choose_ecf_grid(samples: np.ndarray, n_points: int = 50) -> np.ndarray:
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


def estimate_alpha_sigma_with_meta(
    samples: np.ndarray,
    method: str = "ecf",
    n_samples_for_ecf: int = 100000,
) -> Dict[str, object]:
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
            "method_origin", "method_reason", "alpha_hat", "sigma_hat",
            "ecf_n_grid", "ecf_n_points_strict", "ecf_n_points_relaxed",
            "ecf_n_points_used", "ecf_filter_mode",
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


# ============================================================
# Bootstrap CI for McCulloch (from seed_utils.py)
# ============================================================

def bootstrap_mcculloch(
    samples: np.ndarray,
    estimator_fn=None,
    n_boot: int = 200,
    ci: float = 0.95,
) -> Tuple[float, float, float, float]:
    if estimator_fn is None:
        estimator_fn = estimate_alpha_sigma_mcculloch_symmetric_from_quantiles

    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    if n < 4:
        return 2.0, 1.0, 2.0, 0.0

    rng = np.random.RandomState(42)
    alpha_boots = []
    sigma_boots = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_samples = samples[idx]
        q05 = float(np.quantile(boot_samples, 0.05))
        q25 = float(np.quantile(boot_samples, 0.25))
        q75 = float(np.quantile(boot_samples, 0.75))
        q95 = float(np.quantile(boot_samples, 0.95))
        alpha_hat, sigma_hat = estimator_fn(q05, q25, q75, q95)
        alpha_boots.append(float(alpha_hat))
        sigma_boots.append(float(sigma_hat))

    alpha_boots = np.array(alpha_boots, dtype=np.float64)
    sigma_boots = np.array(sigma_boots, dtype=np.float64)

    alpha_median = float(np.median(alpha_boots))
    sigma_median = float(np.median(sigma_boots))

    alpha_lower = (1.0 - ci) / 2.0
    alpha_upper = 1.0 - alpha_lower
    alpha_ci_lo = float(np.quantile(alpha_boots, alpha_lower))
    alpha_ci_hi = float(np.quantile(alpha_boots, alpha_upper))

    return alpha_median, alpha_ci_lo, alpha_ci_hi, sigma_median
