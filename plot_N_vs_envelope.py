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

CANDIDATE_MODELS = ["const", "shared", "diag", "gru", "lstm"]


def parse_args():
    p = argparse.ArgumentParser()

    # Multi-seed arguments
    seed_utils.add_multiseed_args(p)
    p.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Directory where figures will be saved (default: current directory)"
    )
    return p.parse_args()


def load_summaries_across_seeds(seed_dirs, model):
    """
    Load <model>_summary.csv from each seed dir.
    Returns list of DataFrames (one per seed with the file).
    """
    return seed_utils.load_model_summary_across_seeds(
        seed_dirs,
        model,
        required_cols={"ell", "mu_l1_mean", "N_required_at_eps"}
    )


def aggregate_summaries(dfs):
    """
    Aggregate list of summary DataFrames across seeds.
    For each lag ell, compute mean and std of mu_l1_mean and N_required_at_eps.

    Returns a DataFrame with columns:
      ell, mu_l1_mean_mean, mu_l1_mean_std, N_required_at_eps_mean, N_required_at_eps_std
    """
    if not dfs:
        return pd.DataFrame()

    return seed_utils.aggregate_numeric_by_key(
        dfs,
        key_col="ell",
        value_cols=["mu_l1_mean", "N_required_at_eps"]
    )


def plot_model_with_individual_seeds(model, dfs, agg_df, outfile):
    """
    Plot N vs mu for a model, showing:
      - Individual seed points in light gray
      - Averaged points with linear fit in bold
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot individual seed data points (light gray)
    if len(dfs) > 1:
        for df in dfs:
            mu_vals = df["mu_l1_mean"].to_numpy(dtype=float)
            Nreq_vals = df["N_required_at_eps"].to_numpy(dtype=float)

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
    Nreq_vals = agg_df["N_required_at_eps_mean"].to_numpy(dtype=float)

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
    ax.set_title(rf"Scaling of $\log \widehat{{N}}(\ell)$ vs $-\log \hat{{f}}(\ell)$ ({model})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(outfile, dpi=300)
    plt.close(fig)

    if coeff is not None:
        print(
            f"[ok] {model}: saved {outfile} | "
            f"fit: log N ≈ {coeff[0]:.3f} + {coeff[1]:.3f} * (-log f) | "
            f"R² = {r2:.6f}"
        )
    else:
        print(f"[ok] {model}: saved {outfile} | fit skipped (need >=2 points)")


def main():
    args = parse_args()

    # Resolve input directories and discover seed directories
    inputdirs = seed_utils.resolve_inputdirs(args)
    seed_dirs = seed_utils.discover_from_multiple_inputdirs(inputdirs)

    if not seed_dirs:
        raise RuntimeError(f"No seed directories or input directories found")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print(f"[info] loading CSVs from: {[os.path.abspath(d) for d in seed_dirs]}")
    print(f"[info] saving figures to: {os.path.abspath(outdir)}")

    # Print seed information
    seed_utils.print_seed_info(seed_dirs, inputdirs)

    # Detect which models are present across all seed dirs
    models = seed_utils.detect_models_in_dirs(seed_dirs)

    if not models:
        raise FileNotFoundError(
            "No model '*_summary.csv' files found in:\n"
            f"  {[os.path.abspath(d) for d in seed_dirs]}\n"
            f"Expected one or more of: const, shared, diag, gru, lstm"
        )

    print(f"[info] found {len(models)} model(s): {', '.join(models)}")

    for mname in models:
        print(f"\n[processing] {mname}")

        # Load summaries from each seed
        dfs = load_summaries_across_seeds(seed_dirs, mname)

        if not dfs:
            print(f"[warn] {mname}: no summary files found across seeds.")
            continue

        print(f"  [info] {len(dfs)} seed(s) with {mname}_summary.csv")

        # Aggregate across seeds
        agg_df = aggregate_summaries(dfs)

        if agg_df.empty:
            print(f"[warn] {mname}: aggregation produced empty DataFrame.")
            continue

        # Plot
        outpath = os.path.join(outdir, f"N_vs_mu_{mname}.png")
        plot_model_with_individual_seeds(mname, dfs, agg_df, outpath)

    print("\n[done] scaling plot complete")


if __name__ == "__main__":
    main()
