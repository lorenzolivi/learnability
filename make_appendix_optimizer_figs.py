#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_appendix_optimizer_figs.py (multi-seed version)

Appendix-only plots for extra optimizers, styled to match the main-text
plotting utilities (replot_envelopes.py, replot_tau.py).

Reads (per architecture, across seeds, if present):
  - <arch>_summary.csv     (needs columns: ell/lag and mu_l1_mean or mu_mean)
  - <arch>_mu_units.csv    (needs first column ell/lag + per-unit mu^{(q)}(ell))

Aggregates mu_l1_mean across seeds on ell grid, optionally with ±1std band.
Pools tau values across seeds for histograms.

Log-plots use *clamping* to avoid -inf / underflow artifacts:
  log(max(f(ell), eps))
No small positive points are removed from the log plots (only nonpositive are NaN).

Optionally overlays fitted trends on log-plots:
  - semi-log:  log f(ell) = a - lambda * ell
  - log-log:   log f(ell) = c - beta * log ell
Fits are performed on the *unclamped* positive values, inside a reduced window
(default: keep middle 80% of usable points, and exclude f < floor_exclude_factor*eps).

Produces (in --outdir, default: figs/appendix):
  - <optimizer>_log_envelope_vs_ell.png
  - <optimizer>_log_envelope_vs_log_ell.png
  - <optimizer>_tau_spectra.png

Optionally also saves a combined 1x2 panel for backward compatibility:
  - <optimizer>_envelope_scaling.png

Optionally saves fit parameters:
  - <optimizer>_envelope_fits.json

Usage:
  python make_appendix_optimizer_figs.py --inputdirs momentum --outdir figs/appendix
  python make_appendix_optimizer_figs.py --inputdirs sgd      --outdir figs/appendix
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seed_utils

# ── CLI ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    seed_utils.add_multiseed_args(p)
    p.add_argument(
        "--outdir",
        type=str,
        default="figs/appendix",
        help="Directory to save figures."
    )
    p.add_argument(
        "--grid_mode",
        type=str,
        default="union",
        choices=["union", "intersection"],
        help="Align ell grids across models by union or intersection (default: union).",
    )
    p.add_argument(
        "--mask_mode",
        type=str,
        default="per_model",
        choices=["per_model", "common", "none"],
        help="How to define the clamp epsilon for log-plots (default: per_model). "
        "per_model: eps per model; common: max eps across models; none: eps=min_floor.",
    )
    p.add_argument(
        "--floor_quantile",
        type=float,
        default=0.001,
        help="Quantile for epsilon estimation."
    )
    p.add_argument(
        "--floor_scale",
        type=float,
        default=0.01,
        help="Scale factor applied to quantile."
    )
    p.add_argument(
        "--min_floor",
        type=float,
        default=1e-300,
        help="Minimum allowed epsilon."
    )
    p.add_argument(
        "--save_combined_panel",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, also save a combined semi-log/log-log 1x2 panel (default: 1).",
    )

    # Fit overlays + export
    p.add_argument(
        "--show_fits",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, overlay fitted trends (black dotted) on log plots (default: 1)."
    )
    p.add_argument(
        "--trim_frac",
        type=float,
        default=0.10,
        help="Trim fraction from each end of usable fit points (default: 0.10)."
    )
    p.add_argument(
        "--floor_exclude_factor",
        type=float,
        default=10.0,
        help="Exclude points with f(ell) < factor*eps from fitting (default: 10.0)."
    )
    p.add_argument(
        "--min_fit_points",
        type=int,
        default=8,
        help="Minimum points required to perform a fit (default: 8)."
    )
    p.add_argument(
        "--save_fits",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, save fit parameters JSON (default: 1)."
    )

    return p.parse_args()

# ── I/O helpers ────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates):
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

def read_summary(path: str):
    """
    Returns:
      ell: (L,)
      mu_mean: (L,)
    Accepts ell/lag/l columns, and mu_l1_mean or mu_mean columns.
    """
    df = pd.read_csv(path)

    ell_col = _find_col(df, ["ell", "lag", "l"])
    mu_col = _find_col(df, ["mu_l1_mean", "mu_mean", "mu_l1", "mu"])

    if ell_col is None or mu_col is None:
        raise ValueError(f"Missing required columns in {path}. Found: {list(df.columns)}")

    ell = pd.to_numeric(df[ell_col], errors="coerce").to_numpy(dtype=float)
    mu = pd.to_numeric(df[mu_col], errors="coerce").to_numpy(dtype=float)
    return ell, mu

def read_mu_units(path: str):
    """
    Reads per-unit mu^{(q)}(ell) matrix from <arch>_mu_units.csv.
    Assumes first column is ell/lag/l (or falls back to first col).
    Returns:
      ell: (L,)
      units: (L, H)
    """
    df = pd.read_csv(path)

    ell_col = _find_col(df, ["ell", "lag", "l"])
    if ell_col is None:
        ell_col = df.columns[0]

    ell = pd.to_numeric(df[ell_col], errors="coerce").to_numpy(dtype=float)
    units = df.drop(columns=[ell_col]).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return ell, units

# ── Clamp utilities ────────────────────────────────────────────────────

def robust_floor(y: np.ndarray, q: float, scale: float, min_floor: float) -> float:
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

def extract_on_grid(ell: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Average duplicates, then reindex on the provided grid.
    """
    df = pd.DataFrame({"ell": ell, "y": y})
    df["ell"] = pd.to_numeric(df["ell"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ell", "y"])
    df = df.groupby("ell", as_index=False)["y"].mean().set_index("ell")
    return df.reindex(grid)["y"].to_numpy(dtype=float)

# ── Fit utilities ──────────────────────────────────────────────────────

def _linear_fit_r2(x: np.ndarray, y: np.ndarray):
    """
    Fit y = a + b x, return (a, b, r2). Requires len(x)>=2.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    A = np.vstack([np.ones_like(x), x]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = a + b * x
    ssr = np.sum((y - yhat) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ssr / sst if sst > 0 else np.nan
    return float(a), float(b), float(r2)

def _select_fit_window(
    ells: np.ndarray,
    f_pos: np.ndarray,
    eps: float,
    trim_frac: float,
    floor_exclude_factor: float,
    min_fit_points: int
):
    ells = np.asarray(ells, dtype=float)
    f_pos = np.asarray(f_pos, dtype=float)

    mask = np.isfinite(ells) & np.isfinite(f_pos) & (ells > 0) & (f_pos > 0)
    if float(floor_exclude_factor) > 0:
        mask &= (f_pos >= float(floor_exclude_factor) * float(eps))

    idx = np.where(mask)[0]
    if idx.size < int(min_fit_points):
        return np.array([], dtype=int)

    idx = idx[np.argsort(ells[idx])]
    L = idx.size
    lo = int(np.floor(float(trim_frac) * L))
    hi = int(np.ceil((1.0 - float(trim_frac)) * L))
    hi = max(hi, lo + 2)
    idx2 = idx[lo:hi]

    if idx2.size < int(min_fit_points):
        return np.array([], dtype=int)
    return idx2

def fit_exponential_lambda(
    ells: np.ndarray,
    f_pos: np.ndarray,
    eps: float,
    trim_frac: float,
    floor_exclude_factor: float,
    min_fit_points: int
):
    """
    Semi-log fit: log f(ell) = a + b*ell, lambda = -b.
    """
    idx = _select_fit_window(ells, f_pos, eps, trim_frac, floor_exclude_factor, min_fit_points)
    if idx.size == 0:
        return None
    x = ells[idx]
    y = np.log(f_pos[idx])
    a, b, r2 = _linear_fit_r2(x, y)
    lam = -b
    return {
        "a": a,
        "b": b,
        "lambda": float(lam),
        "r2": float(r2),
        "ell_min": float(np.min(x)),
        "ell_max": float(np.max(x)),
        "n": int(x.size),
    }

def fit_powerlaw_beta(
    ells: np.ndarray,
    f_pos: np.ndarray,
    eps: float,
    trim_frac: float,
    floor_exclude_factor: float,
    min_fit_points: int
):
    """
    Log-log fit: log f(ell) = a + b*log ell, beta = -b.
    """
    idx = _select_fit_window(ells, f_pos, eps, trim_frac, floor_exclude_factor, min_fit_points)
    if idx.size == 0:
        return None
    x = np.log(ells[idx])
    y = np.log(f_pos[idx])
    a, b, r2 = _linear_fit_r2(x, y)
    beta = -b
    return {
        "a": a,
        "b": b,
        "beta": float(beta),
        "r2": float(r2),
        "ell_min": float(np.min(ells[idx])),
        "ell_max": float(np.max(ells[idx])),
        "n": int(idx.size),
    }

def plot_curves(
    x: np.ndarray,
    series_dict: dict,
    xlabel: str,
    ylabel: str,
    title: str,
    outpath: str,
    fit_overlays: dict | None = None,
    fit_x_mode: str = "linear",
    show_fits: bool = False,
    band_series_dict: dict | None = None,
) -> None:
    """
    Plot curves with optional ±std band.
    band_series_dict: dict mapping label -> std array (same shape as series values)
    """
    plt.figure(figsize=(6, 4))
    plotted = False

    for label, y in series_dict.items():
        x_ = np.asarray(x, dtype=float)
        y_ = np.asarray(y, dtype=float)
        mask = np.isfinite(x_) & np.isfinite(y_)
        if mask.sum() == 0:
            print(f"[warn] {label}: no valid points for {os.path.basename(outpath)}")
            continue

        plt.plot(x_[mask], y_[mask], "-", label=label, linewidth=2)
        plotted = True

        # Plot band if available
        if band_series_dict is not None and label in band_series_dict:
            y_std = np.asarray(band_series_dict[label], dtype=float)
            y_std = y_std[mask]
            seed_utils.shade_between(
                plt.gca(),
                x_[mask],
                y_[mask],
                y_std,
                alpha=0.2
            )

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
            else:
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

    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[ok] saved: {outpath}")

# ── Tau estimation and plotting ────────────────────────────────────────

def safe_log(x, floor=1e-300):
    return np.log(np.maximum(x, floor))

def fit_semilog(ell: np.ndarray, mu: np.ndarray, use_mask: np.ndarray):
    x = ell[use_mask]
    y = safe_log(mu[use_mask])
    if x.size < 2:
        return np.nan, np.nan, np.nan, np.nan
    A = np.vstack([np.ones_like(x), x]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = a + b * x
    ssr = np.sum((y - yhat) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ssr / sst if sst > 0 else np.nan
    tau = -1.0 / b if b < 0 else np.inf
    return r2, tau, a, b

def estimate_unit_taus(ell: np.ndarray, units: np.ndarray, min_points: int = 6, floor: float = 1e-300):
    if units.ndim != 2:
        return np.array([], dtype=float)
    ell = np.asarray(ell, dtype=float).reshape(-1)
    units = np.asarray(units, dtype=float)
    _, H = units.shape
    taus = np.full(H, np.nan, dtype=float)
    for q in range(H):
        mu = np.abs(units[:, q])
        mask = np.isfinite(ell) & np.isfinite(mu) & (ell > 0) & (mu > floor)
        if int(mask.sum()) < int(min_points):
            continue
        _, tau, _, _ = fit_semilog(ell, mu, use_mask=mask)
        taus[q] = tau
    return taus

def _clean_tau(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size:
        cap = np.nanpercentile(arr, 99.5)
        arr = arr[arr <= max(cap, 1e-12)]
    return arr

def plot_tau_histogram(tau_by_arch: dict, title: str, outpath: str):
    plt.figure(figsize=(6.0, 4.0))
    plotted = False
    for label, taus in tau_by_arch.items():
        taus = _clean_tau(taus)
        if taus.size < 2:
            print(f"[warn] {label}: not enough τ values for histogram (n={taus.size}); skipping.")
            continue
        bins = min(30, int(taus.size))
        plt.hist(taus, bins=bins, density=True, alpha=0.5, label=label)
        plotted = True
    if not plotted:
        print(f"[warn] no τ histograms plotted for {os.path.basename(outpath)}")
        plt.close()
        return
    plt.xlabel(r"$\tau_q$", fontsize=11)
    plt.ylabel("density", fontsize=11)
    plt.title(title, fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[ok] saved: {outpath}")

# ── Architecture detection ────────────────────────────────────────────

def detect_archs_in_seed_dirs(seed_dirs: list[str], arch_map: dict, require_mu_units: bool = True):
    """Detect archs that have required files in ANY seed dir."""
    found = set()
    for a in arch_map.keys():
        for sd in seed_dirs:
            summ = os.path.join(sd, f"{a}_summary.csv")
            mu_u = os.path.join(sd, f"{a}_mu_units.csv")
            ok = os.path.exists(summ) and (os.path.exists(mu_u) if require_mu_units else True)
            if ok:
                found.add(a)
                break
    return sorted([a for a in arch_map.keys() if a in found])

# ── Main ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve input directories
    inputdirs = seed_utils.resolve_inputdirs(args)
    seed_dirs = seed_utils.discover_from_multiple_inputdirs(inputdirs)

    if not seed_dirs:
        raise ValueError(f"No seed directories found in {inputdirs}")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    optimizer_tag = os.path.basename(os.path.abspath(inputdirs[0]))

    print(f"[info] discovered {len(seed_dirs)} seed/data dir(s)")
    for sd in seed_dirs:
        print(f"  - {os.path.abspath(sd)}")
    print(f"[info] optimizer_tag: {optimizer_tag}")
    print(f"[info] outdir: {os.path.abspath(outdir)}")

    # LOWERCASE LABELS ONLY (for legends/JSON consistency)
    arch_map = {
        "const": "const",
        "shared": "shared",
        "diag": "diag",
        "gru": "gru",
        "lstm": "lstm",
    }

    archs = detect_archs_in_seed_dirs(seed_dirs, arch_map, require_mu_units=True)
    if not archs:
        raise FileNotFoundError(
            f"No usable architectures found in seed dirs. "
            "Need at least one pair: <arch>_summary.csv and <arch>_mu_units.csv."
        )

    print(f"[info] detected architectures: {archs}")

    # Load summaries and mu_units across seeds, aggregate by ell
    ell_sets = []
    summary_agg = {}  # label -> (ell, mu_mean, mu_std)
    units_by_arch = {}  # label -> list of (ell, units) per seed

    for a in archs:
        label = arch_map[a]

        # Load summaries across seeds
        ell_sets_a = []
        summary_dfs = []
        for sd in seed_dirs:
            summ_path = os.path.join(sd, f"{a}_summary.csv")
            if not os.path.exists(summ_path):
                continue
            try:
                ell_s, mu_mean = read_summary(summ_path)
                ell_sets_a.append(set(np.asarray(ell_s, dtype=float)[np.isfinite(ell_s) & (ell_s > 0)].tolist()))
                summary_dfs.append((ell_s, mu_mean))
            except Exception as e:
                print(f"[warn] failed reading {summ_path}: {e}")
                continue

        if not summary_dfs:
            print(f"[info] skipping '{a}' (no summary CSVs in seed dirs)")
            continue

        # Aggregate summaries on ell
        if args.grid_mode == "intersection":
            grid_a = sorted(set.intersection(*ell_sets_a)) if ell_sets_a else []
        else:
            grid_a = sorted(set.union(*ell_sets_a)) if ell_sets_a else []

        if not grid_a:
            print(f"[info] skipping '{a}' (no common ell values)")
            continue

        grid_a = np.array(grid_a, dtype=float)

        # Extract all on grid and aggregate
        mu_on_grid = []
        for ell_s, mu_mean in summary_dfs:
            y = extract_on_grid(ell_s, mu_mean, grid_a)
            y[~np.isfinite(y)] = np.nan
            y[y <= 0] = np.nan
            mu_on_grid.append(y)

        # Compute mean and std across seeds
        mu_stack = np.array(mu_on_grid)
        mu_mean_agg = np.nanmean(mu_stack, axis=0)
        mu_std_agg = np.nanstd(mu_stack, axis=0)

        summary_agg[label] = (grid_a, mu_mean_agg, mu_std_agg)
        ell_sets.append(set(grid_a.tolist()))

        # Load mu_units across seeds
        units_list = []
        for sd in seed_dirs:
            mu_u_path = os.path.join(sd, f"{a}_mu_units.csv")
            if not os.path.exists(mu_u_path):
                continue
            try:
                ell_u, units = read_mu_units(mu_u_path)
                units_list.append((ell_u, units))
            except Exception as e:
                print(f"[warn] failed reading {mu_u_path}: {e}")
                continue

        if units_list:
            units_by_arch[label] = units_list

    if not summary_agg:
        raise ValueError("No architectures with valid summaries after aggregation")

    # Build aligned ell grid across all architectures
    if args.grid_mode == "intersection":
        grid_ells = sorted(set.intersection(*ell_sets)) if ell_sets else []
    else:
        grid_ells = sorted(set.union(*ell_sets)) if ell_sets else []

    if len(grid_ells) == 0:
        raise ValueError("No usable 'ell/lag' values found across detected summaries.")

    grid_ells = np.array(grid_ells, dtype=float)
    log_grid_ells = np.log(grid_ells)

    # Re-extract aggregated summaries on global grid
    mu_mean_grid = {}
    mu_std_grid = {}
    for label, (ell_a, mu_mean_a, mu_std_a) in summary_agg.items():
        y_mean = extract_on_grid(ell_a, mu_mean_a, grid_ells)
        y_std = extract_on_grid(ell_a, mu_std_a, grid_ells)
        y_mean[~np.isfinite(y_mean)] = np.nan
        y_mean[y_mean <= 0] = np.nan
        y_std[~np.isfinite(y_std)] = 0.0
        mu_mean_grid[label] = y_mean
        mu_std_grid[label] = y_std

    # Define clamp epsilons for log plots
    eps = {
        label: robust_floor(
            y,
            q=float(args.floor_quantile),
            scale=float(args.floor_scale),
            min_floor=float(args.min_floor),
        )
        for label, y in mu_mean_grid.items()
    }

    if args.mask_mode == "common":
        common_eps = max(eps.values()) if len(eps) > 0 else float(args.min_floor)
        print(f"[info] clamp mode: common eps = {common_eps:.3e}")
        eps = {label: common_eps for label in eps}
    elif args.mask_mode == "per_model":
        print("[info] clamp mode: per-model eps")
        for label, e in eps.items():
            print(f"  - {label}: eps ~ {e:.3e}")
    else:
        print("[info] clamp mode: none (eps=min_floor)")
        eps = {label: float(args.min_floor) for label in eps}

    # Clamped logs for plotting
    log_mu_mean_grid = {label: clamp_log(y, eps=eps[label]) for label, y in mu_mean_grid.items()}
    log_mu_std_grid = {label: mu_std_grid[label] for label in mu_mean_grid}  # not clamped, used for band

    # Fits (on unclamped positive data)
    show_fits = bool(int(args.show_fits) == 1)
    exp_fits = {}
    pow_fits = {}
    fits_out = {
        "meta": {
            "optimizer": optimizer_tag,
            "trim_frac": float(args.trim_frac),
            "floor_exclude_factor": float(args.floor_exclude_factor),
            "min_fit_points": int(args.min_fit_points),
            "mask_mode": str(args.mask_mode),
            "floor_quantile": float(args.floor_quantile),
            "floor_scale": float(args.floor_scale),
            "min_floor": float(args.min_floor),
            "n_seeds": len(seed_dirs),
        },
        "models": {},
    }

    if show_fits:
        for label, y in mu_mean_grid.items():
            e = float(eps[label])
            fe = fit_exponential_lambda(
                grid_ells,
                y,
                eps=e,
                trim_frac=float(args.trim_frac),
                floor_exclude_factor=float(args.floor_exclude_factor),
                min_fit_points=int(args.min_fit_points),
            )
            fp = fit_powerlaw_beta(
                grid_ells,
                y,
                eps=e,
                trim_frac=float(args.trim_frac),
                floor_exclude_factor=float(args.floor_exclude_factor),
                min_fit_points=int(args.min_fit_points),
            )
            exp_fits[label] = fe
            pow_fits[label] = fp
            fits_out["models"][label] = {"eps": e, "exp": fe, "power": fp}

        print("[info] fit summary (exp: lambda, power: beta):")
        for label in mu_mean_grid.keys():
            fe, fp = exp_fits.get(label), pow_fits.get(label)
            s = f"  - {label}: "
            if fe is not None:
                s += f"lambda={fe['lambda']:.4g}, R2={fe['r2']:.4g} | "
            else:
                s += "lambda=NA | "
            if fp is not None:
                s += f"beta={fp['beta']:.4g}, R2={fp['r2']:.4g}"
            else:
                s += "beta=NA"
            print(s)

        if int(args.save_fits) == 1:
            fits_path = os.path.join(outdir, f"{optimizer_tag}_envelope_fits.json")
            with open(fits_path, "w") as fjson:
                json.dump(fits_out, fjson, indent=2)
            print(f"[ok] saved fits: {fits_path}")

    # --- Save envelope plots (lines with optional bands + dotted black fits)
    plot_curves(
        grid_ells,
        log_mu_mean_grid,
        xlabel=r"lag $\ell$",
        ylabel=r"$\log \hat{f}(\ell)$",
        title=rf"Envelope scaling $\log \hat{{f}}(\ell)$ — {optimizer_tag.upper()}",
        outpath=os.path.join(outdir, f"{optimizer_tag}_log_envelope_vs_ell.png"),
        fit_overlays=exp_fits if show_fits else None,
        fit_x_mode="linear",
        show_fits=show_fits,
        band_series_dict=log_mu_std_grid if len(seed_dirs) > 1 else None,
    )

    plot_curves(
        log_grid_ells,
        log_mu_mean_grid,
        xlabel=r"$\log \ell$",
        ylabel=r"$\log \hat{f}(\ell)$",
        title=rf"Envelope scaling $\log \hat{{f}}(\ell)$ vs $\log \ell$ — {optimizer_tag.upper()}",
        outpath=os.path.join(outdir, f"{optimizer_tag}_log_envelope_vs_log_ell.png"),
        fit_overlays=pow_fits if show_fits else None,
        fit_x_mode="log",
        show_fits=show_fits,
        band_series_dict=log_mu_std_grid if len(seed_dirs) > 1 else None,
    )

    # Optional combined 1x2 panel
    if int(args.save_combined_panel) == 1:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
        plotted0 = False
        plotted1 = False

        for label, ylog in log_mu_mean_grid.items():
            m0 = np.isfinite(grid_ells) & np.isfinite(ylog)
            if m0.sum() > 0:
                axs[0].plot(grid_ells[m0], ylog[m0], "-", label=label, linewidth=2)
                plotted0 = True

                # Add band if multi-seed
                if len(seed_dirs) > 1:
                    y_std = log_mu_std_grid[label]
                    seed_utils.shade_between(
                        axs[0],
                        grid_ells[m0],
                        ylog[m0],
                        y_std[m0],
                        alpha=0.2
                    )

                if show_fits and exp_fits.get(label) is not None:
                    fd = exp_fits[label]
                    x_fit = grid_ells[m0]
                    x_fit = x_fit[(x_fit >= fd["ell_min"]) & (x_fit <= fd["ell_max"])]
                    if x_fit.size >= 2:
                        y_fit = float(fd["a"]) + float(fd["b"]) * x_fit
                        axs[0].plot(x_fit, y_fit, ":", color="black", linewidth=1.5)

            m1 = np.isfinite(log_grid_ells) & np.isfinite(ylog)
            if m1.sum() > 0:
                axs[1].plot(log_grid_ells[m1], ylog[m1], "-", label=label, linewidth=2)
                plotted1 = True

                # Add band if multi-seed
                if len(seed_dirs) > 1:
                    y_std = log_mu_std_grid[label]
                    seed_utils.shade_between(
                        axs[1],
                        log_grid_ells[m1],
                        ylog[m1],
                        y_std[m1],
                        alpha=0.2
                    )

                if show_fits and pow_fits.get(label) is not None:
                    fd = pow_fits[label]
                    lo = np.log(float(fd["ell_min"]))
                    hi = np.log(float(fd["ell_max"]))
                    x_fit = log_grid_ells[m1]
                    x_fit = x_fit[(x_fit >= lo) & (x_fit <= hi)]
                    if x_fit.size >= 2:
                        y_fit = float(fd["a"]) + float(fd["b"]) * x_fit
                        axs[1].plot(x_fit, y_fit, ":", color="black", linewidth=1.5)

        if plotted0 or plotted1:
            axs[0].set_xlabel(r"$\ell$", fontsize=11)
            axs[0].set_ylabel(r"$\log \hat{f}(\ell)$", fontsize=11)
            axs[0].set_title("Semi-log", fontsize=11)
            axs[0].legend()
            axs[1].set_xlabel(r"$\log \ell$", fontsize=11)
            axs[1].set_ylabel(r"$\log \hat{f}(\ell)$", fontsize=11)
            axs[1].set_title("Log-log", fontsize=11)
            axs[1].legend()
            fig.suptitle(f"Envelope scaling ({optimizer_tag.upper()})", fontsize=12)
            fig.tight_layout()
            panel_path = os.path.join(outdir, f"{optimizer_tag}_envelope_scaling.png")
            fig.savefig(panel_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[ok] saved: {panel_path}")
        else:
            plt.close(fig)
            print(
                f"[warn] no valid curves for combined panel {optimizer_tag}_envelope_scaling.png"
            )

    # --- Tau spectra (pooled across seeds)
    tau_by_arch = {}
    for label in units_by_arch.keys():
        taus_pooled = []
        for ell_u, units in units_by_arch[label]:
            taus = estimate_unit_taus(
                ell_u, units, min_points=6, floor=float(args.min_floor)
            )
            taus_pooled.extend(taus[np.isfinite(taus)])
        if taus_pooled:
            tau_by_arch[label] = np.array(taus_pooled)
        else:
            tau_by_arch[label] = np.array([])

    plot_tau_histogram(
        tau_by_arch=tau_by_arch,
        title=rf"Time-scale distribution $\tau_q$ — {optimizer_tag.upper()}",
        outpath=os.path.join(outdir, f"{optimizer_tag}_tau_spectra.png"),
    )

if __name__ == "__main__":
    main()
