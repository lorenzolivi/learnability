#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_N_vs_envelope.py (multi-seed + auto-merge version)

Multi-seed variant of the original plot_N_vs_envelope.py that:
  - Accepts --inputdirs / --inputdir arguments
  - Auto-discovers seed_* subdirectories within each inputdir
  - For each model, loads <model>_summary.csv from each seed dir
  - Aggregates: computes mean of mu_l1_mean and N_required_at_eps across seeds per lag ell
  - Plots scatter of seed-averaged points with linear fit and R²
  - If multi-seed: also scatters individual seed points in lighter color
  - Preserves linear fit computation and R² metric
  - Outputs per-model figures: N_vs_mu_<model>.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seed_utils

# Subfolder for organizing output (relative to outdir)
SUBFOLDER = "sample_complexity"

CANDIDATE_MODELS = ["const", "shared", "diag", "gru", "lstm"]


# ── Helper functions for dual-alpha format ──
def _alpha_col(method):
    """Return column name for alpha given method."""
    return f"alpha_{method}"

def _nreq_col(method):
    """Return column name for N_required given method."""
    if method == "hat":
        return "N_required_at_eps"  # old single-method format
    return f"N_required_{method}"


SEED_MARKERS = ["o", "s", "^", "D", "v"]  # distinct markers per seed


def parse_args():
    p = argparse.ArgumentParser()

    # Multi-seed arguments
    seed_utils.add_multiseed_args(p)
    seed_utils.add_view_arg(p)
    p.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Directory where figures will be saved (default: current directory)"
    )
    return p.parse_args()


def load_summaries_across_seeds(seed_dirs, model, method=None):
    """
    Load <model>_summary.csv from each seed dir.
    Returns list of DataFrames (one per seed with the file).
    """
    required_cols = {"ell", "mu_l1_mean"}

    if method is None:
        return seed_utils.load_model_summary_across_seeds(
            seed_dirs,
            model,
            required_cols=required_cols,
        )

    nreq_col = _nreq_col(method)
    dfs = seed_utils.load_model_summary_across_seeds(
        seed_dirs,
        model,
        required_cols=required_cols | {nreq_col},
    )
    if dfs:
        return dfs

    if nreq_col != "N_required_at_eps":
        return seed_utils.load_model_summary_across_seeds(
            seed_dirs,
            model,
            required_cols=required_cols | {"N_required_at_eps"},
        )

    return []


def aggregate_summaries(dfs, method):
    """
    Aggregate list of summary DataFrames across seeds.
    For each lag ell, compute mean and std of mu_l1_mean and the
    method-specific sample-complexity column.

    Returns a DataFrame with columns:
      ell, mu_l1_mean_mean, mu_l1_mean_std, <N_required_col>_mean, <N_required_col>_std
    """
    if not dfs:
        return pd.DataFrame()

    nreq_col = _nreq_col(method)
    nreq_col_to_use = nreq_col if nreq_col in dfs[0].columns else "N_required_at_eps"

    return seed_utils.aggregate_numeric_by_key(
        dfs,
        key_col="ell",
        value_cols=["mu_l1_mean", nreq_col_to_use],
    )


def plot_model_with_individual_seeds(model, dfs, agg_df, outfile, method: str = "ecf"):
    """
    Plot N vs mu for a model, showing:
      - Individual seed points in light gray
      - Averaged points with linear fit in bold
    """
    nreq_col = _nreq_col(method)
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot individual seed data points (light gray)
    if len(dfs) > 1:
        for df in dfs:
            mu_vals = df["mu_l1_mean"].to_numpy(dtype=float)
            # Try method-specific column first, fall back to old format
            col_to_use = nreq_col if nreq_col in df.columns else "N_required_at_eps"
            Nreq_vals = df[col_to_use].to_numpy(dtype=float)

            mask = (
                np.isfinite(mu_vals) & (mu_vals > 0) &
                np.isfinite(Nreq_vals) & (Nreq_vals > 0)
            )

            if not np.any(mask):
                continue

            x = -np.log(mu_vals[mask] + 1e-20)
            y = np.log(Nreq_vals[mask])
            ax.scatter(x, y, alpha=0.3, s=30, color="gray", zorder=1)

    # Plot aggregated (mean) data points
    mu_vals = agg_df["mu_l1_mean_mean"].to_numpy(dtype=float)
    nreq_col_mean = f"{nreq_col}_mean" if f"{nreq_col}_mean" in agg_df.columns else "N_required_at_eps_mean"
    if nreq_col_mean not in agg_df.columns:
        nreq_col_mean = "N_required_at_eps"
    Nreq_vals = agg_df[nreq_col_mean].to_numpy(dtype=float)

    mask = (
        np.isfinite(mu_vals) & (mu_vals > 0) &
        np.isfinite(Nreq_vals) & (Nreq_vals > 0)
    )

    if not np.any(mask):
        print(f"[warn] {model}: no valid aggregated points to plot.")
        plt.close(fig)
        return

    x = -np.log(mu_vals[mask] + 1e-20)
    y = np.log(Nreq_vals[mask])

    ax.scatter(x, y, label="data (averaged)", zorder=2)

    coeff = None
    y_pred = None
    r2 = None

    if x.size >= 2:
        A = np.vstack([np.ones_like(x), x]).T
        coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        y_pred = A @ coeff

        # Compute R^2
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    if y_pred is not None:
        order = np.argsort(x)
        label_fit = f"linear fit (R² = {r2:.4f})" if r2 is not None else "linear fit"
        ax.plot(x[order], y_pred[order], linestyle="--", label=label_fit, zorder=3)

    ax.set_xlabel(r"$-\log \hat{f}(\ell)$")
    ax.set_ylabel(r"$\log \widehat{N}(\ell)$")
    ax.set_title(rf"Scaling of $\log \widehat{{N}}(\ell)$ vs $-\log \hat{{f}}(\ell)$ ({model}, {method})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(outfile, dpi=300)
    plt.close(fig)

    if coeff is not None:
        print(
            f"[ok] {model} ({method}): saved {outfile} | "
            f"fit: log N ≈ {coeff[0]:.3f} + {coeff[1]:.3f} * (-log f) | "
            f"R² = {r2:.6f}"
        )
    else:
        print(f"[ok] {model} ({method}): saved {outfile} | fit skipped (need >=2 points)")


def plot_model_per_seed(model, seed_traces, outfile, nreq_col="N_required_at_eps"):
    """Plot per-seed scatter with color-coded seeds and per-seed fits."""
    fig, ax = plt.subplots(figsize=(6, 4))
    color = seed_utils.get_model_color(model)

    for i, (seed_label, df) in enumerate(seed_traces):
        mu_vals = df["mu_l1_mean"].to_numpy(dtype=float)
        # Use method-specific column, fall back to old format
        col = nreq_col if nreq_col in df.columns else "N_required_at_eps"
        Nreq_vals = df[col].to_numpy(dtype=float)
        mask = np.isfinite(mu_vals) & (mu_vals > 0) & np.isfinite(Nreq_vals) & (Nreq_vals > 0)
        if not np.any(mask):
            continue

        x = -np.log(mu_vals[mask] + 1e-20)
        y = np.log(Nreq_vals[mask])
        alpha = seed_utils.SEED_ALPHAS[i] if i < len(seed_utils.SEED_ALPHAS) else 0.3
        marker = SEED_MARKERS[i] if i < len(SEED_MARKERS) else "o"
        ax.scatter(x, y, alpha=alpha, s=20, color=color, marker=marker,
                   label=seed_label, zorder=2)

    ax.set_xlabel(r"$-\log \hat{f}(\ell)$")
    ax.set_ylabel(r"$\log \widehat{N}(\ell)$")
    ax.set_title(rf"$\log \widehat{{N}}(\ell)$ vs $-\log \hat{{f}}(\ell)$ ({model}) [per seed]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"[ok] saved: {outfile}")


def main():
    args = parse_args()
    view = args.view

    inputdirs = seed_utils.resolve_inputdirs(args)
    seed_dirs = seed_utils.discover_from_multiple_inputdirs(inputdirs)

    if not seed_dirs:
        raise RuntimeError(f"No seed directories or input directories found")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Create subfolder for this script's outputs
    plot_outdir = os.path.join(outdir, SUBFOLDER)
    os.makedirs(plot_outdir, exist_ok=True)

    print(f"[info] saving figures to: {os.path.abspath(plot_outdir)}")
    print(f"[info] view mode: {view}")

    seed_utils.print_seed_info(seed_dirs, inputdirs)

    models = seed_utils.detect_models_in_dirs(seed_dirs)
    if not models:
        raise FileNotFoundError("No model '*_summary.csv' files found")

    print(f"[info] found {len(models)} model(s): {', '.join(models)}")

    # Detect which alpha methods are available
    alpha_methods = ["ecf", "mcc"]  # Try both; fall back to old format if needed
    sample_df = None
    for mname in models:
        dfs = load_summaries_across_seeds(seed_dirs, mname)
        if dfs:
            sample_df = dfs[0]
            break

    if sample_df is not None:
        if "alpha_ecf" in sample_df.columns:
            alpha_methods = ["ecf", "mcc"]
        elif "alpha_mcc" in sample_df.columns:
            alpha_methods = ["mcc"]
        elif "alpha_hat" in sample_df.columns:
            alpha_methods = ["hat"]
        else:
            alpha_methods = ["hat"]
    else:
        alpha_methods = ["hat"]

    print(f"[info] detected alpha methods: {alpha_methods}")

    tag = "agg_" if view == "both" else ""
    ps_tag = "ps_" if view == "both" else ""

    for method in alpha_methods:
        print(f"\n[processing] alpha method: {method}")
        method_tag = f"_{method}" if len(alpha_methods) > 1 else ""

        for mname in models:
            print(f"  [model] {mname}")

            dfs = load_summaries_across_seeds(seed_dirs, mname, method=method)
            if not dfs:
                print(f"[warn] {mname}: no summary files found.")
                continue

            print(f"    [info] {len(dfs)} seed(s)")

            # ── AGGREGATED VIEW ──
            if view in ("aggregated", "both"):
                agg_df = aggregate_summaries(dfs, method)
                if not agg_df.empty:
                    outpath = os.path.join(plot_outdir, f"{tag}N_vs_mu_{mname}{method_tag}.png")
                    plot_model_with_individual_seeds(mname, dfs, agg_df, outpath, method=method)

            # ── PER-SEED VIEW ──
            if view in ("per_seed", "both"):
                required = {"ell", "mu_l1_mean", _nreq_col(method) if method != "hat" else "N_required_at_eps"}
                seed_traces = seed_utils.load_model_data_per_seed(seed_dirs, mname, required)
                if seed_traces:
                    outpath = os.path.join(plot_outdir, f"{ps_tag}N_vs_mu_{mname}{method_tag}.png")
                    nreq = _nreq_col(method) if method != "hat" else "N_required_at_eps"
                    plot_model_per_seed(mname, seed_traces, outpath, nreq_col=nreq)

    print("\n[done] scaling plot complete")


if __name__ == "__main__":
    main()
