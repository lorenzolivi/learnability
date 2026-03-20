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


# ── Helper functions for dual-alpha format ──
def _alpha_col(method):
    """Return column name for alpha given method ('ecf', 'mcc', or single method)."""
    return f"alpha_{method}"

def _sigma_col(method):
    """Return column name for sigma_alpha given method."""
    return f"sigma_{method}"

def _reliable_col(method):
    """Return column name for reliability flag given method."""
    return f"alpha_{method}_reliable"


def parse_args():
    p = argparse.ArgumentParser()
    seed_utils.add_multiseed_args(p)
    seed_utils.add_view_arg(p)
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


def load_alpha_from_df(df: pd.DataFrame, method: str = "ecf", reliable_only: bool = True):
    """
    Load and filter alpha_hat from a summary DataFrame for a given method.

    Args:
        df: DataFrame to load from
        method: 'ecf', 'mcc', or 'hat' (for backward compat with old format)
        reliable_only: whether to filter on reliability

    Returns:
        reliable: array of reliable alpha_hat values
        unreliable: array of unreliable alpha_hat values (for optional display)
        n_discarded: number of estimates discarded
    """
    # Determine which columns to use based on available format
    alpha_col = _alpha_col(method)
    sigma_col = _sigma_col(method)
    reliable_col = _reliable_col(method)

    # Try new format first, fall back to old format
    if alpha_col not in df.columns:
        if method != "hat" and "alpha_hat" in df.columns:
            # Fall back to old single-alpha format
            alpha_col = "alpha_hat"
            sigma_col = "sigma_alpha_hat"
            reliable_col = "alpha_reliable"
        else:
            return np.array([]), np.array([]), 0

    a = df[alpha_col].to_numpy(dtype=float)

    # Basic quality filter: finite, positive, within [1, 2]
    basic_mask = np.isfinite(a) & (a >= 1.0) & (a <= 2.0)

    # Also require finite positive sigma if available
    if sigma_col in df.columns:
        s = df[sigma_col].to_numpy(dtype=float)
        basic_mask &= np.isfinite(s) & (s > 0)

    # Reliability column (new format from robust estimation)
    if reliable_col in df.columns:
        rel_col = df[reliable_col].to_numpy()
        # Handle mixed types: could be int (0/1), bool, or string
        try:
            rel_flags = rel_col.astype(int)
        except (ValueError, TypeError):
            rel_flags = np.ones(len(rel_col), dtype=int)  # assume reliable if unparseable

        reliable_mask = basic_mask & (rel_flags == 1)
        unreliable_mask = basic_mask & (rel_flags == 0)
    else:
        # Old format: no reliability column. Use heuristic
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
    view = args.view
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Resolve input directories
    inputdirs = seed_utils.resolve_inputdirs(args)
    seed_dirs = seed_utils.discover_from_multiple_inputdirs(inputdirs)

    if not seed_dirs:
        raise ValueError("No seed directories found (or inputdir not specified)")

    seed_utils.print_seed_info(seed_dirs, inputdirs)
    print(f"[info] view mode: {view}")

    # Detect which alpha methods are available in the data
    alpha_methods = []
    sample_df = None
    for model in seed_utils.CANDIDATE_MODELS:
        per_seed_traces = seed_utils.load_model_data_per_seed(seed_dirs, model)
        if per_seed_traces:
            sample_df = per_seed_traces[0][1]
            break

    if sample_df is not None:
        # Check which columns exist
        if "alpha_ecf" in sample_df.columns:
            alpha_methods = ["ecf", "mcc"]
        elif "alpha_mcc" in sample_df.columns:
            alpha_methods = ["mcc"]
        elif "alpha_hat" in sample_df.columns:
            alpha_methods = ["hat"]  # Old format, use single method
        else:
            raise ValueError("Could not find alpha columns (alpha_ecf, alpha_mcc, or alpha_hat)")
    else:
        raise ValueError("No model data found in seed directories")

    print(f"[info] detected alpha methods: {alpha_methods}")

    # Load alpha across all models and all seeds, per method
    all_results = {}  # method -> {model -> (reliable_array, unreliable_array, per_seed_list)}

    for method in alpha_methods:
        alphas_reliable = {}       # model -> pooled array
        alphas_unreliable = {}     # model -> pooled array
        alphas_per_seed = {}       # model -> [(seed_label, array), ...]
        total_discarded = 0

        for model in seed_utils.CANDIDATE_MODELS:
            per_seed_traces = seed_utils.load_model_data_per_seed(seed_dirs, model)
            if not per_seed_traces:
                continue

            all_reliable = []
            all_unreliable = []
            seed_traces = []
            for seed_label, df in per_seed_traces:
                rel, unrel, n_disc = load_alpha_from_df(df, method=method)
                if rel.size > 0:
                    all_reliable.append(rel)
                    seed_traces.append((seed_label, rel))
                if unrel.size > 0:
                    all_unreliable.append(unrel)
                total_discarded += n_disc

            if all_reliable:
                alphas_reliable[model] = np.concatenate(all_reliable)
                alphas_per_seed[model] = seed_traces
            if all_unreliable:
                alphas_unreliable[model] = np.concatenate(all_unreliable)

        if not alphas_reliable:
            raise FileNotFoundError(f"No reliable alpha data found for method {method}")

        if total_discarded > 0:
            print(f"[info] Method {method}: discarded {total_discarded} unreliable α̂ estimates")

        sizes = {k: v.size for k, v in alphas_reliable.items()}
        print(f"[info] Method {method}: reliable α̂ counts: {sizes}")

        all_results[method] = (alphas_reliable, alphas_unreliable, alphas_per_seed)

    def _add_gaussian_line(ax):
        ax.axvline(2.0, linestyle="--", linewidth=1.5, color="black", alpha=0.4)
        ylim = ax.get_ylim()
        ax.text(2.0, ylim[1] * 0.92, r"$\alpha = 2$ (Gaussian)", ha="right",
                va="top", fontsize=8, alpha=0.6)

    # Plot separately for each method
    for method in alpha_methods:
        alphas_reliable, alphas_unreliable, alphas_per_seed = all_results[method]

        # Shared axis limits for this method
        all_alpha = np.concatenate([a for a in alphas_reliable.values() if a.size > 0])
        xmin = max(0.8, float(np.min(all_alpha)) - 0.05)
        xmax = min(2.05, float(np.max(all_alpha)) + 0.05)
        bins = np.linspace(xmin, xmax, 30)
        grid = np.linspace(xmin, xmax, 400)

        method_tag = f"_{method}" if len(alpha_methods) > 1 else ""

        # ── AGGREGATED VIEW: pooled KDE/hist ──
        if view in ("aggregated", "both"):
            tag = "agg_" if view == "both" else ""
            plt.figure(figsize=(6.5, 4.2))

            for name, a in alphas_reliable.items():
                if a.size == 0:
                    continue
                color = seed_utils.get_model_color(name)
                if args.hist or args.both:
                    plt.hist(a, bins=bins, density=True, alpha=0.3,
                             label=f"{name} (n={a.size})", color=color)
                if (not args.hist) or args.both:
                    plt.plot(grid, kde_1d(a, grid), linewidth=2,
                             label=f"{name} (n={a.size})", color=color)

            if args.show_unreliable:
                for name, a in alphas_unreliable.items():
                    if a.size == 0:
                        continue
                    plt.plot(grid, kde_1d(a, grid), linewidth=1, linestyle="--",
                             color="gray", alpha=0.5,
                             label=f"{name} unreliable (n={a.size})")

            _add_gaussian_line(plt.gca())
            plt.xlabel(r"Estimated tail index $\hat\alpha(\ell)$")
            plt.ylabel("Density")
            plt.title(rf"Distributions of $\hat\alpha(\ell)$ [aggregated, {method}]")
            plt.legend(fontsize=8)
            plt.tight_layout()
            outpath = os.path.join(outdir, f"{tag}alpha_hat_distributions{method_tag}.png")
            plt.savefig(outpath, dpi=300)
            plt.close()
            print(f"[ok] saved: {outpath}")

        # ── PER-SEED VIEW: individual KDEs overlaid ──
        if view in ("per_seed", "both"):
            tag = "ps_" if view == "both" else ""
            plt.figure(figsize=(6.5, 4.2))
            legend_handles = {}

            for model, seed_traces in alphas_per_seed.items():
                color = seed_utils.get_model_color(model)
                for i, (seed_label, a) in enumerate(seed_traces):
                    if a.size < 2:
                        continue
                    alpha = seed_utils.SEED_ALPHAS[i] if i < len(seed_utils.SEED_ALPHAS) else 0.3
                    line, = plt.plot(grid, kde_1d(a, grid), linewidth=1.0,
                                     color=color, alpha=alpha)
                    if model not in legend_handles:
                        legend_handles[model] = line

            _add_gaussian_line(plt.gca())
            plt.xlabel(r"Estimated tail index $\hat\alpha(\ell)$")
            plt.ylabel("Density")
            plt.title(rf"Distributions of $\hat\alpha(\ell)$ [per seed, {method}]")
            if legend_handles:
                plt.legend(legend_handles.values(), legend_handles.keys(), fontsize=8)
            plt.tight_layout()
            outpath = os.path.join(outdir, f"{tag}alpha_hat_distributions{method_tag}.png")
            plt.savefig(outpath, dpi=300)
            plt.close()
            print(f"[ok] saved: {outpath}")


if __name__ == "__main__":
    main()
