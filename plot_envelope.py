#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-seed + auto-merge version of plot_envelope.py

Replot envelope scaling f(ell) from *_summary.csv files across multiple seeds and directories.

Supports: const, shared, diag, gru, lstm
  - Loads <model>_summary.csv from each seed dir
  - Aligns curves on a common ell grid (union or intersection)
  - Uses clamping for log plots to avoid -inf and underflow artifacts
  - Optionally overlays fitted trends on log-plots (exponential, power-law, tempered power-law)
  - Aggregates via mean/std across seeds, then fits on the seed-averaged envelope
  - Saves AIC/BIC for each fit

NEW (multi-seed):
  - Discover seed_* subdirs in each inputdir
  - Load <model>_summary.csv from each seed, aggregate by ell
  - Plot mean envelope with ±1std shaded band (multi-seed) or plain line (single-seed)
  - Fits are performed on the seed-averaged envelope

Saves:
    envelope_mu_vs_ell.png
    log_envelope_vs_ell.png
    log_envelope_vs_log_ell.png
Optionally:
    envelope_fits.json   (lambda, beta, tempered params, R^2, SSR, AIC/BIC, fit ranges)

Usage:
  python plot_envelope.py --inputdirs dir1 dir2 ... --outdir results/...
  python plot_envelope.py --inputdir single_dir --outdir results/...

Optional:
  --grid_mode union|intersection
  --mask_mode per_model|common|none
  --floor_quantile 0.001
  --floor_scale 0.01
  --min_floor 1e-300
  --drop_nonpositive 1
  --show_fits 1
  --trim_frac 0.02
  --floor_exclude_factor 10.0
  --min_fit_points 8
  --save_fits 1
  --fits_filename envelope_fits.json
  --fit_tempered 1
  --k_grid 0.25,0.35,0.5,0.7,1.0,1.3,1.6
  --ellc_grid_size 40
  --ellc_grid_min_factor 0.5
  --ellc_grid_max_factor 2.0
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seed_utils


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    seed_utils.add_multiseed_args(p)
    p.add_argument("--outdir", type=str, default=".", help="Directory where figures will be saved (default: .)")
    p.add_argument(
        "--grid_mode",
        type=str,
        default="union",
        choices=["union", "intersection"],
        help="How to align ell grids across models (default: union).",
    )
    p.add_argument(
        "--mask_mode",
        type=str,
        default="per_model",
        choices=["per_model", "common", "none"],
        help="How to define the clamp epsilon for log-plots (default: per_model). "
             "per_model: eps per model; common: max eps across models; none: eps=min_floor.",
    )
    p.add_argument("--floor_quantile", type=float, default=0.001, help="Quantile used to estimate clamp epsilon.")
    p.add_argument("--floor_scale", type=float, default=0.01, help="Scale factor applied to quantile.")
    p.add_argument("--min_floor", type=float, default=1e-300, help="Minimum allowed clamp epsilon.")
    p.add_argument(
        "--drop_nonpositive",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, nonpositive f(ell) is treated as missing (NaN) (default: 1).",
    )

    # Fit overlays
    p.add_argument(
        "--show_fits",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, overlay fitted trends (black dotted lines) on log plots (default: 1).",
    )
    p.add_argument(
        "--trim_frac",
        type=float,
        default=0.02,
        help="Trim fraction from each end of the valid fit points (default: 0.02).",
    )
    p.add_argument(
        "--floor_exclude_factor",
        type=float,
        default=10.0,
        help="Exclude points with f(ell) < factor*eps from fitting (default: 10.0).",
    )
    p.add_argument(
        "--min_fit_points",
        type=int,
        default=8,
        help="Minimum number of points required to perform a fit (default: 8).",
    )
    p.add_argument(
        "--save_fits",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, save fit parameters to JSON in outdir (default: 1).",
    )
    p.add_argument(
        "--fits_filename",
        type=str,
        default="envelope_fits.json",
        help="Filename for saved fit parameters (default: envelope_fits.json).",
    )

    # Tempered fit controls
    p.add_argument(
        "--fit_tempered",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, also fit tempered power-law: A*ell^{-beta}*exp(-(ell/ell_c)^k) (default: 1).",
    )
    p.add_argument(
        "--k_grid",
        type=str,
        default="0.25,0.35,0.5,0.7,1.0,1.3,1.6",
        help="Comma-separated k values for tempered cutoff grid search (default: 0.25,0.35,0.5,0.7,1.0,1.3,1.6).",
    )
    p.add_argument(
        "--ellc_grid_size",
        type=int,
        default=40,
        help="Number of ell_c values (log-spaced) for grid search (default: 40).",
    )
    p.add_argument(
        "--ellc_grid_min_factor",
        type=float,
        default=0.5,
        help="Lower bound for ell_c grid: factor * ell_min_fit (default: 0.5).",
    )
    p.add_argument(
        "--ellc_grid_max_factor",
        type=float,
        default=2.0,
        help="Upper bound for ell_c grid: factor * ell_max_fit (default: 2.0).",
    )

    return p.parse_args()


# ── Canonical model names ──────────────────────────────────
CANDIDATE_MODELS = ["const", "shared", "diag", "gru", "lstm"]
REQUIRED_COLS = {"ell", "mu_l1_mean"}


# ── Helpers ────────────────────────────────────────────────

def robust_floor(y: np.ndarray, q: float, scale: float, min_floor: float) -> float:
    """
    Estimate a clamp epsilon for y based on a low quantile of positive values:
        eps = max(min_floor, scale * quantile(pos, q))
    """
    y = np.asarray(y, dtype=float)
    pos = y[np.isfinite(y) & (y > 0)]
    if pos.size < 10:
        return float(min_floor)
    fq = float(np.quantile(pos, q))
    return float(max(min_floor, scale * fq))


def clamp_log(y: np.ndarray, eps: float) -> np.ndarray:
    """
    Log with clamping (no masking of small positives):
      - y<=0 -> NaN
      - y>0  -> log(max(y, eps))
    """
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    mask = np.isfinite(y) & (y > 0)
    if mask.any():
        out[mask] = np.log(np.maximum(y[mask], eps))
    return out


def aggregate_model_data_across_seeds(seed_dirs: list[str], model: str) -> dict:
    """
    Load <model>_summary.csv from each seed, aggregate by ell.
    Returns: {
        "ell": array of ell values,
        "mu_mean": mean of mu_l1_mean across seeds,
        "mu_std": std of mu_l1_mean across seeds,
    }
    """
    dfs = seed_utils.load_model_summary_across_seeds(seed_dirs, model, REQUIRED_COLS)

    if not dfs:
        return None

    # Aggregate across seeds on common ell values
    agg = seed_utils.aggregate_numeric_by_key(dfs, "ell", ["mu_l1_mean"])

    if agg.empty:
        return None

    return {
        "ell": agg["ell"].to_numpy(dtype=float),
        "mu_mean": agg["mu_l1_mean_mean"].to_numpy(dtype=float),
        "mu_std": agg["mu_l1_mean_std"].to_numpy(dtype=float),
        "count": agg["mu_l1_mean_count"].to_numpy(dtype=int) if "mu_l1_mean_count" in agg.columns else np.ones(len(agg), dtype=int),
    }


def _linear_fit_with_stats(x: np.ndarray, y: np.ndarray):
    """Fit y = a + b x, return dict with: a, b, yhat, ssr, r2"""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    A = np.vstack([np.ones_like(x), x]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = a + b * x
    ssr = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - ssr / sst) if sst > 0 else np.nan
    return {"a": float(a), "b": float(b), "yhat": yhat, "ssr": ssr, "r2": r2}


def _aic_bic_from_ssr(ssr: float, n: int, p: int, tiny: float = 1e-300):
    """Gaussian LS information criteria (up to additive constants)."""
    n = int(n)
    p = int(p)
    if n <= 0 or p <= 0 or not np.isfinite(ssr):
        return {"aic": np.nan, "bic": np.nan, "p": p}
    ssr_eff = max(float(ssr), float(tiny))
    val = n * np.log(ssr_eff / n)
    aic = float(val + 2.0 * p)
    bic = float(val + p * np.log(n))
    return {"aic": aic, "bic": bic, "p": p}


def _select_fit_window(x: np.ndarray, y_pos: np.ndarray, eps: float, trim_frac: float,
                       floor_exclude_factor: float, min_fit_points: int):
    """Select indices for fitting based on quality criteria."""
    x = np.asarray(x, dtype=float)
    y_pos = np.asarray(y_pos, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y_pos) & (y_pos > 0) & (x > 0)
    if floor_exclude_factor is not None and float(floor_exclude_factor) > 0:
        mask = mask & (y_pos >= float(floor_exclude_factor) * float(eps))

    idx = np.where(mask)[0]
    if idx.size < int(min_fit_points):
        return np.array([], dtype=int)

    idx = idx[np.argsort(x[idx])]

    L = idx.size
    lo = int(np.floor(float(trim_frac) * L))
    hi = int(np.ceil((1.0 - float(trim_frac)) * L))
    hi = max(hi, lo + 2)
    idx2 = idx[lo:hi]

    if idx2.size < int(min_fit_points):
        return np.array([], dtype=int)

    return idx2


def fit_exponential_lambda(ells: np.ndarray, f_pos: np.ndarray, eps: float,
                           trim_frac: float, floor_exclude_factor: float, min_fit_points: int):
    """Semi-log fit: log f(ell) = a + b * ell, with lambda = -b."""
    idx = _select_fit_window(ells, f_pos, eps, trim_frac, floor_exclude_factor, min_fit_points)
    if idx.size == 0:
        return None

    x = ells[idx]
    y = np.log(f_pos[idx])
    st = _linear_fit_with_stats(x, y)
    lam = -st["b"]
    ic = _aic_bic_from_ssr(st["ssr"], n=int(x.size), p=2)

    return {
        "a": st["a"],
        "b": st["b"],
        "lambda": float(lam),
        "r2": float(st["r2"]),
        "ssr": float(st["ssr"]),
        **ic,
        "ell_min": float(np.min(x)),
        "ell_max": float(np.max(x)),
        "n": int(x.size),
    }


def fit_powerlaw_beta(ells: np.ndarray, f_pos: np.ndarray, eps: float,
                      trim_frac: float, floor_exclude_factor: float, min_fit_points: int):
    """Log-log fit: log f(ell) = a + b * log ell, with beta = -b."""
    idx = _select_fit_window(ells, f_pos, eps, trim_frac, floor_exclude_factor, min_fit_points)
    if idx.size == 0:
        return None

    x = np.log(ells[idx])
    y = np.log(f_pos[idx])
    st = _linear_fit_with_stats(x, y)
    beta = -st["b"]
    ic = _aic_bic_from_ssr(st["ssr"], n=int(idx.size), p=2)

    return {
        "a": st["a"],
        "b": st["b"],
        "beta": float(beta),
        "r2": float(st["r2"]),
        "ssr": float(st["ssr"]),
        **ic,
        "ell_min": float(np.min(ells[idx])),
        "ell_max": float(np.max(ells[idx])),
        "n": int(idx.size),
    }


def fit_tempered_powerlaw(ells: np.ndarray, f_pos: np.ndarray, eps: float,
                          trim_frac: float, floor_exclude_factor: float, min_fit_points: int,
                          k_grid: list[float], ellc_grid_size: int,
                          ellc_grid_min_factor: float, ellc_grid_max_factor: float):
    """Fit tempered power law: f(ell) = A * ell^{-beta} * exp(-(ell/ell_c)^k)"""
    idx = _select_fit_window(ells, f_pos, eps, trim_frac, floor_exclude_factor, min_fit_points)
    if idx.size == 0:
        return None

    ell = np.asarray(ells[idx], dtype=float)
    y = np.log(np.asarray(f_pos[idx], dtype=float))
    logell = np.log(ell)

    ell_min = float(np.min(ell))
    ell_max = float(np.max(ell))

    # Build ell_c grid
    lo = max(1e-12, float(ellc_grid_min_factor) * ell_min)
    hi = max(lo * 1.001, float(ellc_grid_max_factor) * ell_max)
    ellc_grid = np.logspace(np.log10(lo), np.log10(hi), int(max(5, ellc_grid_size)))

    best = None
    best_ssr = np.inf
    ones = np.ones_like(logell)

    for k in k_grid:
        k = float(k)
        if not np.isfinite(k) or k <= 0:
            continue
        for ell_c in ellc_grid:
            ell_c = float(ell_c)
            t = (ell / ell_c) ** k
            X = np.vstack([ones, logell, t]).T
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            c, b1, b2 = [float(v) for v in coeffs]
            yhat = X @ coeffs
            ssr = float(np.sum((y - yhat) ** 2))
            if ssr < best_ssr:
                best_ssr = ssr
                best = (c, b1, b2, ell_c, k, yhat)

    if best is None:
        return None

    c, b1, b2, ell_c, k, yhat = best
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - best_ssr / sst) if sst > 0 else np.nan

    logA = float(c)
    A = float(np.exp(logA))
    beta = float(-b1)
    cutoff_coeff = float(-b2)

    n = int(len(ell))
    ic = _aic_bic_from_ssr(best_ssr, n=n, p=5)

    return {
        "logA": float(logA),
        "A": float(A),
        "beta": float(beta),
        "ell_c": float(ell_c),
        "k": float(k),
        "cutoff_coeff": float(cutoff_coeff),
        "r2": float(r2),
        "ssr": float(best_ssr),
        **ic,
        "ell_min": float(ell_min),
        "ell_max": float(ell_max),
        "n": int(n),
        "grid": {
            "k_grid": [float(v) for v in k_grid],
            "ellc_grid_size": int(ellc_grid_size),
            "ellc_grid_min_factor": float(ellc_grid_min_factor),
            "ellc_grid_max_factor": float(ellc_grid_max_factor),
        },
    }


def plot_curves(x: np.ndarray,
                series_dict: dict,
                xlabel: str,
                ylabel: str,
                title: str,
                outpath: str,
                fit_overlays: dict = None,
                fit_x_mode: str = "linear",
                show_fits: bool = False) -> None:
    """Plot curves with optional fit overlays."""
    plt.figure(figsize=(6, 4))
    plotted = False

    for label, data in series_dict.items():
        if data is None:
            continue

        x_ = np.asarray(x, dtype=float)
        y_mean = np.asarray(data.get("y_mean"), dtype=float)
        y_std = np.asarray(data.get("y_std"), dtype=float)

        mask = np.isfinite(x_) & np.isfinite(y_mean)
        if mask.sum() == 0:
            print(f"[warn] {label}: no valid points for {os.path.basename(outpath)}")
            continue

        # Check if single-seed (std all zero or NaN)
        is_single_seed = np.all((np.isnan(y_std) | (y_std == 0)))

        if is_single_seed:
            plt.plot(x_[mask], y_mean[mask], "-", label=label)
        else:
            seed_utils.shade_between(plt.gca(), x_[mask], y_mean[mask], y_std[mask], label=label)

        plotted = True

        # Plot fit overlay
        if show_fits and fit_overlays is not None and label in fit_overlays and fit_overlays[label] is not None:
            fd = fit_overlays[label]
            ell_min = float(fd["ell_min"])
            ell_max = float(fd["ell_max"])

            if fit_x_mode == "linear":
                x_fit = x_[mask]
                x_fit = x_fit[(x_fit >= ell_min) & (x_fit <= ell_max)]
                if x_fit.size >= 2:
                    y_fit = float(fd["a"]) + float(fd["b"]) * x_fit
                    plt.plot(x_fit, y_fit, ":", color="black", linewidth=1.5)
            elif fit_x_mode == "log":
                lo = np.log(ell_min)
                hi = np.log(ell_max)
                x_fit = x_[mask]
                x_fit = x_fit[(x_fit >= lo) & (x_fit <= hi)]
                if x_fit.size >= 2:
                    y_fit = float(fd["a"]) + float(fd["b"]) * x_fit
                    plt.plot(x_fit, y_fit, ":", color="black", linewidth=1.5)

    if not plotted:
        print(f"[warn] no curves plotted for {os.path.basename(outpath)}")
        plt.close()
        return

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[ok] saved: {outpath}")


# ── Main ───────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve input directories and discover seed dirs
    inputdirs = seed_utils.resolve_inputdirs(args)
    seed_dirs = seed_utils.discover_from_multiple_inputdirs(inputdirs)

    if not seed_dirs:
        raise ValueError("No seed directories found in: " + ", ".join(inputdirs))

    outdir = args.outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    seed_utils.print_seed_info(seed_dirs, inputdirs)
    print(f"[info] saving figures to: {os.path.abspath(outdir)}")

    # Load and aggregate model data across seeds
    print("[info] loading and aggregating model summaries across seeds...")
    model_data = {}
    for model in CANDIDATE_MODELS:
        data = aggregate_model_data_across_seeds(seed_dirs, model)
        if data is not None:
            model_data[model] = data
            n_seeds = int(data["count"].max())
            print(f"  - {model}: {n_seeds} seed(s)")

    if not model_data:
        raise ValueError(
            f"No model '*_summary.csv' files found in {len(seed_dirs)} seed dir(s)\n"
            "Expected one or more of: " + ", ".join([f"{m}_summary.csv" for m in CANDIDATE_MODELS])
        )

    print(f"[info] found {len(model_data)} model(s): {', '.join(model_data.keys())}")

    # Build ell grid
    ell_sets = []
    for model, data in model_data.items():
        e = data["ell"]
        e = e[np.isfinite(e)]
        e = e[e > 0]
        ell_sets.append(set(e.tolist()))

    if args.grid_mode == "intersection":
        grid_ells = sorted(set.intersection(*ell_sets)) if ell_sets else []
    else:
        grid_ells = sorted(set.union(*ell_sets)) if ell_sets else []

    if len(grid_ells) == 0:
        raise ValueError("No usable 'ell' values found across detected models.")

    ells = np.array(grid_ells, dtype=float)
    log_ells = np.log(ells)

    # Interpolate model data to common ell grid
    f = {}
    f_std = {}
    for model, data in model_data.items():
        # Interpolate mean
        mu = np.interp(ells, data["ell"], data["mu_mean"], left=np.nan, right=np.nan)
        f[model] = mu

        # Interpolate std (or zeros if single-seed)
        is_single_seed = np.all((np.isnan(data["mu_std"]) | (data["mu_std"] == 0)))
        if is_single_seed:
            f_std[model] = np.zeros_like(mu)
        else:
            mu_std = np.interp(ells, data["ell"], data["mu_std"], left=np.nan, right=np.nan)
            f_std[model] = mu_std

    # Optionally drop nonpositive
    if args.drop_nonpositive == 1:
        for model in f:
            y = f[model].astype(float, copy=False)
            y[~np.isfinite(y)] = np.nan
            y[y <= 0] = np.nan
            f[model] = y

    # Define clamp epsilons
    eps = {
        model: robust_floor(
            y,
            q=float(args.floor_quantile),
            scale=float(args.floor_scale),
            min_floor=float(args.min_floor),
        )
        for model, y in f.items()
    }

    if args.mask_mode == "common":
        common_eps = max(eps.values()) if len(eps) > 0 else float(args.min_floor)
        print(f"[info] clamp mode: common eps = {common_eps:.3e}")
        eps = {model: common_eps for model in eps}
    elif args.mask_mode == "per_model":
        print("[info] clamp mode: per-model eps")
        for model, e in eps.items():
            print(f"  - {model}: eps ~ {e:.3e}")
    else:
        print("[info] clamp mode: none (eps=min_floor)")
        eps = {model: float(args.min_floor) for model in eps}

    # Build clamped log f(ell)
    log_f = {model: clamp_log(y, eps=eps[model]) for model, y in f.items()}

    # Fits
    show_fits = bool(int(args.show_fits) == 1)
    do_tempered = bool(int(args.fit_tempered) == 1)

    fits_out = {"meta": {
        "trim_frac": float(args.trim_frac),
        "floor_exclude_factor": float(args.floor_exclude_factor),
        "min_fit_points": int(args.min_fit_points),
        "mask_mode": str(args.mask_mode),
        "floor_quantile": float(args.floor_quantile),
        "floor_scale": float(args.floor_scale),
        "min_floor": float(args.min_floor),
        "fit_tempered": int(args.fit_tempered),
        "k_grid": str(args.k_grid),
        "ellc_grid_size": int(args.ellc_grid_size),
        "ellc_grid_min_factor": float(args.ellc_grid_min_factor),
        "ellc_grid_max_factor": float(args.ellc_grid_max_factor),
        "ic_formulas": {
            "AIC": "n*log(SSR/n) + 2p",
            "BIC": "n*log(SSR/n) + p*log(n)",
            "p_exp": 2,
            "p_power": 2,
            "p_tempered": 5,
        },
    }, "models": {}}

    exp_fits = {}
    pow_fits = {}
    tmp_fits = {}

    # Parse k_grid
    try:
        k_grid = [float(s.strip()) for s in str(args.k_grid).split(",") if s.strip() != ""]
    except Exception:
        k_grid = [0.25, 0.35, 0.5, 0.7, 1.0, 1.3, 1.6]

    if show_fits:
        for model in f.keys():
            y = f[model]
            e = float(eps[model])

            fe = fit_exponential_lambda(
                ells, y, eps=e,
                trim_frac=float(args.trim_frac),
                floor_exclude_factor=float(args.floor_exclude_factor),
                min_fit_points=int(args.min_fit_points),
            )
            fp = fit_powerlaw_beta(
                ells, y, eps=e,
                trim_frac=float(args.trim_frac),
                floor_exclude_factor=float(args.floor_exclude_factor),
                min_fit_points=int(args.min_fit_points),
            )
            ft = None
            if do_tempered:
                ft = fit_tempered_powerlaw(
                    ells, y, eps=e,
                    trim_frac=float(args.trim_frac),
                    floor_exclude_factor=float(args.floor_exclude_factor),
                    min_fit_points=int(args.min_fit_points),
                    k_grid=k_grid,
                    ellc_grid_size=int(args.ellc_grid_size),
                    ellc_grid_min_factor=float(args.ellc_grid_min_factor),
                    ellc_grid_max_factor=float(args.ellc_grid_max_factor),
                )

            exp_fits[model] = fe
            pow_fits[model] = fp
            tmp_fits[model] = ft

            fits_out["models"][model] = {
                "eps": e,
                "exp": fe,
                "power": fp,
                "tempered": ft,
            }

        # Console summary
        print("[info] fit summary:")
        for model in f.keys():
            fe, fp, ft = exp_fits.get(model), pow_fits.get(model), tmp_fits.get(model)
            s = f"  - {model}: "
            if fe is not None:
                s += f"exp: lambda={fe['lambda']:.4g}, R2={fe['r2']:.4g}, AIC={fe['aic']:.3g}, BIC={fe['bic']:.3g} | "
            else:
                s += "exp: NA | "
            if fp is not None:
                s += f"power: beta={fp['beta']:.4g}, R2={fp['r2']:.4g}, AIC={fp['aic']:.3g}, BIC={fp['bic']:.3g} | "
            else:
                s += "power: NA | "
            if do_tempered:
                if ft is not None:
                    s += (f"tempered: beta={ft['beta']:.4g}, ell_c={ft['ell_c']:.4g}, "
                          f"k={ft['k']:.4g}, R2={ft['r2']:.4g}, "
                          f"AIC={ft['aic']:.3g}, BIC={ft['bic']:.3g}, coeff={ft['cutoff_coeff']:.3g}")
                else:
                    s += "tempered: NA"
            print(s)

        if int(args.save_fits) == 1:
            fits_path = os.path.join(outdir, args.fits_filename)
            with open(fits_path, "w") as fjson:
                json.dump(fits_out, fjson, indent=2)
            print(f"[ok] saved fits: {fits_path}")

    # Plot 1: Linear-scale envelope f(ell)
    series_lin = {model: {"y_mean": f[model], "y_std": f_std[model]} for model in f}
    plot_curves(
        ells,
        series_lin,
        xlabel=r"lag $\ell$",
        ylabel=r"$\hat{f}(\ell)$",
        title=r"Envelope scaling $\hat{f}(\ell)$",
        outpath=os.path.join(outdir, "envelope_mu_vs_ell.png"),
        show_fits=False,
    )

    # Plot 2: Semi-log: log f(ell) vs ell (+ exp fit overlay)
    series_semilog = {model: {"y_mean": log_f[model], "y_std": f_std[model] / np.maximum(np.abs(f[model]), 1e-10)} for model in f}
    plot_curves(
        ells,
        series_semilog,
        xlabel=r"lag $\ell$",
        ylabel=r"$\log \hat{f}(\ell)$",
        title=r"Envelope scaling $\log \hat{f}(\ell)$",
        outpath=os.path.join(outdir, "log_envelope_vs_ell.png"),
        fit_overlays=exp_fits if show_fits else None,
        fit_x_mode="linear",
        show_fits=show_fits,
    )

    # Plot 3: Log-log: log f(ell) vs log ell (+ power fit overlay)
    series_loglog = {model: {"y_mean": log_f[model], "y_std": f_std[model] / np.maximum(np.abs(f[model]), 1e-10)} for model in f}
    plot_curves(
        log_ells,
        series_loglog,
        xlabel=r"$\log \ell$",
        ylabel=r"$\log \hat{f}(\ell)$",
        title=r"Envelope scaling $\log \hat{f}(\ell)$ vs $\log \ell$",
        outpath=os.path.join(outdir, "log_envelope_vs_log_ell.png"),
        fit_overlays=pow_fits if show_fits else None,
        fit_x_mode="log",
        show_fits=show_fits,
    )


if __name__ == "__main__":
    main()
