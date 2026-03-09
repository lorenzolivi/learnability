#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-seed version of plot_alpha_estimation.py

Accepts --inputdirs or --inputdir, auto-discovers seed_* subdirs,
loads alpha_hat values across all seeds, pools them as a distribution,
and plots KDE/histogram.

Reliability filtering:
  - If summary CSVs contain 'alpha_reliable' column (new format),
    only estimates flagged as reliable (alpha_reliable == 1) are used.
  - If 'alpha_reliable' is absent (old format), falls back to basic
    filtering: discards alpha_hat <= 1.0 or non-finite values.

Default: KDE only
Optional:
  --hist   -> histogram only
  --both   -> histogram + KDE
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seed_utils


def parse_args():
    p = argparse.ArgumentParser()
    seed_utils.add_multiseed_args(p)
    p.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Directory where figures will be saved (default: current directory)"
    )
    p.add_argument(
        "--hist",
        action="store_true",
        help="Plot histograms instead of KDE (default: KDE)"
    )
    p.add_argument(
        "--both",
        action="store_true",
        help="Plot both histogram and KDE"
    )
    p.add_argument(
        "--show_unreliable",
        action="store_true",
        help="Also show unreliable estimates (in gray, dashed) for comparison"
    )
    return p.parse_args()


def load_alpha_from_df(df: pd.DataFrame, reliable_only: bool = True):
    """
    Load and filter alpha_hat from a summary DataFrame.

    Returns:
        reliable: array of reliable alpha_hat values
        unreliable: array of unreliable alpha_hat values (for optional display)
        n_discarded: number of estimates discarded
    """
    if "alpha_hat" not in df.columns:
        return np.array([]), np.array([]), 0

    a = df["alpha_hat"].to_numpy(dtype=float)

    # Basic quality filter: finite, positive, within [1, 2]
    basic_mask = np.isfinite(a) & (a >= 1.0) & (a <= 2.0)

    # Also require finite positive sigma if available
    if "sigma_alpha_hat" in df.columns:
        s = df["sigma_alpha_hat"].to_numpy(dtype=float)
        basic_mask &= np.isfinite(s) & (s > 0)

    # Reliability column (new format from robust estimation)
    if "alpha_reliable" in df.columns:
        rel_col = df["alpha_reliable"].to_numpy()
        # Handle mixed types: could be int (0/1), bool, or string
        try:
            rel_flags = rel_col.astype(int)
        except (ValueError, TypeError):
            rel_flags = np.ones(len(rel_col), dtype=int)  # assume reliable if unparseable

        reliable_mask = basic_mask & (rel_flags == 1)
        unreliable_mask = basic_mask & (rel_flags == 0)
    else:
        # Old format: no reliability column. Use heuristic:
        # alpha exactly at boundaries (<=1.01 or >=1.99) with no reliability flag → suspect
        reliable_mask = basic_mask
        unreliable_mask = np.zeros_like(basic_mask)

    reliable = a[reliable_mask]
    unreliable = a[unreliable_mask]
    n_discarded = int(basic_mask.sum() - reliable_mask.sum())

    return reliable, unreliable, n_discarded


def kde_1d(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Compute 1D KDE using Gaussian kernel."""
    x = x.astype(float)
    n = x.size
    if n < 2:
        return np.zeros_like(grid)

    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(std, iqr / 1.349) if (std > 0 and iqr > 0) else (std if std > 0 else 1.0)
    h = max(0.9 * sigma * n ** (-1 / 5), 1e-3)

    z = (grid[:, None] - x[None, :]) / h
    dens = np.mean(np.exp(-0.5 * z ** 2), axis=1) / (h * np.sqrt(2 * np.pi))
    return dens


def main():
    args = parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Resolve input directories
    inputdirs = seed_utils.resolve_inputdirs(args)
    seed_dirs = seed_utils.discover_from_multiple_inputdirs(inputdirs)

    if not seed_dirs:
        raise ValueError("No seed directories found (or inputdir not specified)")

    seed_utils.print_seed_info(seed_dirs, inputdirs)

    outpath = os.path.join(outdir, "alpha_hat_distributions.png")

    # Load alpha_hat across all models and all seeds
    alphas_reliable = {}
    alphas_unreliable = {}
    total_discarded = 0

    for model in seed_utils.CANDIDATE_MODELS:
        dfs = seed_utils.load_model_summary_across_seeds(seed_dirs, model)
        if not dfs:
            continue

        all_reliable = []
        all_unreliable = []
        for df in dfs:
            rel, unrel, n_disc = load_alpha_from_df(df)
            if rel.size > 0:
                all_reliable.append(rel)
            if unrel.size > 0:
                all_unreliable.append(unrel)
            total_discarded += n_disc

        if all_reliable:
            alphas_reliable[model] = np.concatenate(all_reliable)
        if all_unreliable:
            alphas_unreliable[model] = np.concatenate(all_unreliable)

    if not alphas_reliable:
        raise FileNotFoundError("No reliable alpha_hat data found in any seed directory")

    if total_discarded > 0:
        print(f"[info] Discarded {total_discarded} unreliable α̂ estimates "
              f"(flagged by reliability checks)")

    sizes = {k: v.size for k, v in alphas_reliable.items()}
    print(f"[info] Reliable α̂ counts: {sizes}")

    # Shared axis limits (from reliable data only)
    all_alpha = np.concatenate([a for a in alphas_reliable.values() if a.size > 0])
    xmin = max(0.8, float(np.min(all_alpha)) - 0.05)
    xmax = min(2.05, float(np.max(all_alpha)) + 0.05)
    bins = np.linspace(xmin, xmax, 30)

    # Plot
    plt.figure(figsize=(6.5, 4.2))
    grid = np.linspace(xmin, xmax, 400)

    # Reliable estimates (solid lines, full color)
    for name, a in alphas_reliable.items():
        if a.size == 0:
            continue

        if args.hist or args.both:
            plt.hist(
                a,
                bins=bins,
                density=True,
                alpha=0.3,
                label=f"{name} (n={a.size})"
            )

        if (not args.hist) or args.both:
            plt.plot(
                grid,
                kde_1d(a, grid),
                linewidth=2,
                label=f"{name} (n={a.size})"
            )

    # Unreliable estimates (dashed, gray) — only if --show_unreliable
    if args.show_unreliable:
        for name, a in alphas_unreliable.items():
            if a.size == 0:
                continue
            plt.plot(
                grid,
                kde_1d(a, grid),
                linewidth=1,
                linestyle="--",
                color="gray",
                alpha=0.5,
                label=f"{name} unreliable (n={a.size})"
            )

    plt.axvline(2.0, linestyle="--", linewidth=1.5, color="black", alpha=0.4)
    plt.text(2.0, plt.ylim()[1] * 0.92, r"$\alpha = 2$ (Gaussian)", ha="right",
             va="top", fontsize=8, alpha=0.6)

    plt.xlabel(r"Estimated tail index $\hat\alpha(\ell)$")
    plt.ylabel("Density")
    plt.title(r"Distributions of $\hat\alpha(\ell)$ across diagnostic lags"
              + (" (reliable only)" if total_discarded > 0 else ""))
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"[ok] saved: {outpath}")


if __name__ == "__main__":
    main()
