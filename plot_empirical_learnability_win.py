#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-seed + auto-merge version of plot_empirical_learnability_win.py

Replot empirical learnability window Ĥ_N vs N across multiple seeds and input directories.
Auto-discover which model columns are present:
  const, shared, diag, gru, lstm

Features:
  - Accept --inputdirs (multiple dirs) and --inputdir (single, backward compatible)
  - Auto-discover seed_* subdirs in each inputdir using seed_utils
  - Load H_N_summary.csv from each seed dir
  - Aggregate: compute mean and std for each (N, model) pair across seeds
  - Plot: mean line with ±1std shaded band (multi-seed) or plain line (single-seed)
  - Preserve original aesthetics: log x-axis, same labels, 6x4 figure, 300 dpi
  - Auto-merge results across baselines and lstm_gru directories
  - Optional: percentile bands (25th-75th) or box plots across seeds (--hn_view)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seed_utils


def parse_args():
    p = argparse.ArgumentParser()
    seed_utils.add_multiseed_args(p)
    p.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Directory where the figure will be saved (default: current directory)"
    )
    p.add_argument(
        "--hn_view",
        type=str,
        default="mean_std",
        choices=["mean_std", "percentile", "boxplot"],
        help=(
            "How to visualise H_N across seeds: "
            "'mean_std' = mean line with ±1std band (default), "
            "'percentile' = median line with 25th–75th shaded band, "
            "'boxplot' = box plot at each N value."
        ),
    )
    p.add_argument(
        "--pct_lo",
        type=float,
        default=25.0,
        help="Lower percentile for 'percentile' view (default: 25)",
    )
    p.add_argument(
        "--pct_hi",
        type=float,
        default=75.0,
        help="Upper percentile for 'percentile' view (default: 75)",
    )
    return p.parse_args()


# ── Colours for consistent model styling ────────────────────────────────
MODEL_COLORS = {
    "const": "C0",
    "shared": "C1",
    "diag": "C2",
    "gru": "C3",
    "lstm": "C4",
}


def plot_mean_std(ax, agg, present, N_grid):
    """Original view: mean ± 1 std shaded band."""
    for label, (col, mean_col) in present.items():
        std_col = f"{col}_std"
        y_mean = pd.to_numeric(agg[mean_col], errors="coerce").to_numpy(dtype=float)
        y_std = (
            pd.to_numeric(agg[std_col], errors="coerce").to_numpy(dtype=float)
            if std_col in agg.columns
            else np.zeros_like(y_mean)
        )

        is_single_seed = np.all((np.isnan(y_std) | (y_std == 0)))
        mask = pd.notna(N_grid) & pd.notna(y_mean) & (N_grid > 0)
        if mask.sum() == 0:
            print(f"[warn] {label}: no valid points. Skipping.")
            continue

        N_plot = N_grid[mask]
        y_plot = y_mean[mask]
        y_std_plot = y_std[mask] if not is_single_seed else np.zeros_like(y_plot)

        c = MODEL_COLORS.get(label)
        if is_single_seed:
            ax.plot(N_plot, y_plot, "o-", label=label, color=c)
        else:
            seed_utils.shade_between(ax, N_plot, y_plot, y_std_plot, label=label, color=c)


def plot_percentile(ax, seed_dirs, present, pct_lo, pct_hi):
    """Median line with percentile band."""
    for label, (col, _) in present.items():
        N_grid, matrix = seed_utils.collect_H_N_matrix(seed_dirs, col)
        if matrix.size == 0:
            continue

        # Compute statistics ignoring NaN
        median = np.nanmedian(matrix, axis=0)
        lo = np.nanpercentile(matrix, pct_lo, axis=0)
        hi = np.nanpercentile(matrix, pct_hi, axis=0)

        valid = np.isfinite(median) & (N_grid > 0)
        if valid.sum() == 0:
            continue

        c = MODEL_COLORS.get(label)
        ax.plot(N_grid[valid], median[valid], "o-", label=label, color=c, markersize=4)
        ax.fill_between(
            N_grid[valid], lo[valid], hi[valid],
            alpha=0.2, color=c,
        )


def plot_boxplot(ax, seed_dirs, present):
    """Box plot of H_N at each N, one colour per model."""
    # Collect all N values first
    all_N = set()
    model_matrices = {}
    for label, (col, _) in present.items():
        N_grid, matrix = seed_utils.collect_H_N_matrix(seed_dirs, col)
        if matrix.size == 0:
            continue
        model_matrices[label] = (N_grid, matrix)
        all_N.update(N_grid.tolist())

    if not model_matrices:
        return

    N_sorted = np.sort(list(all_N))
    n_models = len(model_matrices)

    # Width of each box group in log-space
    positions_map = {n: i for i, n in enumerate(N_sorted)}

    for m_idx, (label, (N_grid, matrix)) in enumerate(model_matrices.items()):
        c = MODEL_COLORS.get(label, f"C{m_idx}")
        n_map = {n: j for j, n in enumerate(N_grid)}

        box_data = []
        box_positions = []
        for n_val in N_sorted:
            if n_val not in n_map:
                continue
            col_idx = n_map[n_val]
            vals = matrix[:, col_idx]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            box_data.append(vals)
            # Offset within group
            offset = (m_idx - n_models / 2 + 0.5) * 0.15
            box_positions.append(positions_map[n_val] + offset)

        if not box_data:
            continue

        bp = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.12,
            patch_artist=True,
            manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color(c)
        for flier in bp["fliers"]:
            flier.set(marker=".", markerfacecolor=c, markersize=3, alpha=0.5)

        # Invisible line for legend
        ax.plot([], [], color=c, label=label, linewidth=2)

    # Set x-ticks to actual N values
    ax.set_xticks(range(len(N_sorted)))
    ax.set_xticklabels([f"{int(n)}" for n in N_sorted], rotation=45, fontsize=7)


def main():
    args = parse_args()

    # Resolve input directories
    inputdirs = seed_utils.resolve_inputdirs(args)

    # Discover seed directories
    seed_dirs = seed_utils.discover_from_multiple_inputdirs(inputdirs)

    if not seed_dirs:
        raise ValueError("No seed directories found in: " + ", ".join(inputdirs))

    outdir = args.outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    seed_utils.print_seed_info(seed_dirs, inputdirs)
    print(f"[info] saving figure to: {os.path.abspath(outdir)}")
    print(f"[info] H_N view: {args.hn_view}")

    # Load H_N_summary.csv from each seed
    print("[info] loading H_N_summary.csv from each seed...")
    dfs = seed_utils.load_csv_across_seeds(seed_dirs, "H_N_summary.csv", {"N"})

    if not dfs:
        raise FileNotFoundError(
            f"No H_N_summary.csv found in any of {len(seed_dirs)} seed dir(s)"
        )

    print(f"[info] loaded {len(dfs)} seed files")

    # Aggregate across seeds (needed for mean_std and for model detection)
    print("[info] aggregating H_N values across seeds...")
    agg = seed_utils.aggregate_H_N_across_seeds(seed_dirs)

    if agg.empty:
        raise ValueError("Failed to aggregate H_N data from seeds")

    print("[info] columns in aggregated data:", list(agg.columns))

    # Extract model columns
    MODEL_COLS = {
        "const": "H_N_const",
        "shared": "H_N_shared",
        "diag": "H_N_diag",
        "gru": "H_N_gru",
        "lstm": "H_N_lstm",
    }

    # Find which models are present and have mean data
    present = {}
    for label, col in MODEL_COLS.items():
        mean_col = f"{col}_mean"
        if mean_col in agg.columns:
            present[label] = (col, mean_col)

    if not present:
        raise ValueError(
            "Aggregated data does not contain any recognized H_N_*_mean columns.\n"
            "Expected at least one of:\n  "
            + "\n  ".join([f"{col}_mean" for col in MODEL_COLS.values()])
        )

    print(f"[info] plotting {len(present)} model curve(s): {', '.join(present.keys())}")

    # Print seed counts for each model
    for label, (col, mean_col) in present.items():
        count_col = f"{col}_count"
        if count_col in agg.columns:
            n_seeds = int(agg[count_col].max())
            print(f"  - {label}: {n_seeds} seed(s)")

    N_grid = agg["N"].to_numpy(dtype=float)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(6, 4))

    if args.hn_view == "mean_std":
        plot_mean_std(ax, agg, present, N_grid)
        ax.set_xscale("log")

    elif args.hn_view == "percentile":
        plot_percentile(ax, seed_dirs, present, args.pct_lo, args.pct_hi)
        ax.set_xscale("log")
        pct_label = f"{int(args.pct_lo)}th–{int(args.pct_hi)}th pctl"
        ax.set_title(
            r"Empirical learnability window $\widehat{\mathcal{H}}_N$"
            f" (median + {pct_label})"
        )

    elif args.hn_view == "boxplot":
        plot_boxplot(ax, seed_dirs, present)
        # x-axis is categorical for boxplot; don't use log scale

    ax.set_xlabel(r"Training budget $N$")
    ax.set_ylabel(r"$\widehat{\mathcal{H}}_N$")
    if args.hn_view != "percentile":
        ax.set_title(r"Empirical learnability window $\widehat{\mathcal{H}}_N$")
    ax.legend()
    fig.tight_layout()

    suffix = {"mean_std": "", "percentile": "_percentile", "boxplot": "_boxplot"}
    outpath = os.path.join(outdir, f"H_N_curves{suffix[args.hn_view]}.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    print(f"[ok] saved: {outpath}")


if __name__ == "__main__":
    main()
