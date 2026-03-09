#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-seed version of plot_learnability_learning_curves.py

Accepts --inputdirs or --inputdir (replaces --indir), auto-discovers
seed_* subdirs, finds *_learning_curve.csv recursively in each seed dir,
groups learning curves by model name across seeds, and plots mean + ±1std.

Produces:
  <outdir>/learning_curves_loss.png    — overlay: train+val loss, all models
  <outdir>/learning_curves_r2.png      — overlay: train+val R², all models
  <outdir>/per_model/<model>_learning_curve.png — 2-panel per model (loss + R²)
"""

import os
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seed_utils


# ───────────────────────────────────────────────────────────────
# Recursive discovery
# ───────────────────────────────────────────────────────────────

def find_learning_curve_csvs(seed_dirs: list[str]):
    """
    Walk each seed dir recursively and find *_learning_curve.csv files.
    Returns dict mapping model_name -> list of {csv: path, seed_dir: path, model: name}
    """
    model_data = {}

    for seed_dir in seed_dirs:
        for root, dirs, files in os.walk(seed_dir):
            folder = os.path.basename(root)
            exact = f"{folder}_learning_curve.csv"

            # Prefer exact match
            if exact in files:
                csv_path = os.path.join(root, exact)
                model_name = folder
                if model_name not in model_data:
                    model_data[model_name] = []
                model_data[model_name].append({
                    "csv": csv_path,
                    "seed_dir": seed_dir,
                    "model": model_name
                })
                continue

            # Fallback: any *_learning_curve.csv
            lc_files = sorted(f for f in files if f.endswith("_learning_curve.csv"))
            if lc_files:
                csv_path = os.path.join(root, lc_files[0])
                name = lc_files[0].replace("_learning_curve.csv", "")
                model_name = name or folder
                if model_name not in model_data:
                    model_data[model_name] = []
                model_data[model_name].append({
                    "csv": csv_path,
                    "seed_dir": seed_dir,
                    "model": model_name
                })

    return model_data


def read_lc(path: str):
    """
    Returns dict with arrays: epoch, train_loss, train_acc, val_loss, val_acc.
    Returns None if file is empty or malformed.
    """
    try:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    except Exception:
        return None

    if getattr(data, "size", 0) == 0:
        return None
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)

    required = {"epoch", "train_loss", "val_loss"}
    if not required.issubset(set(data.dtype.names or [])):
        return None

    d = {}
    d["epoch"] = np.array(data["epoch"], dtype=int)
    d["train_loss"] = np.array(data["train_loss"], dtype=float)
    d["val_loss"] = np.array(data["val_loss"], dtype=float)
    d["train_acc"] = (
        np.array(data["train_acc"], dtype=float)
        if "train_acc" in (data.dtype.names or [])
        else None
    )
    d["val_acc"] = (
        np.array(data["val_acc"], dtype=float)
        if "val_acc" in (data.dtype.names or [])
        else None
    )

    # mask out NaN epochs
    mask = np.isfinite(d["epoch"]) & np.isfinite(d["train_loss"])
    for k in d:
        if d[k] is not None:
            d[k] = d[k][mask]
    return d


# ───────────────────────────────────────────────────────────────
# Aggregation across seeds
# ───────────────────────────────────────────────────────────────

def aggregate_learning_curves(entries_for_model: list) -> dict:
    """
    Given list of entries (one per seed) for a single model,
    aggregate by epoch: compute mean/std for train_loss, val_loss, etc.

    Returns {
        "epoch": array,
        "train_loss_mean": array,
        "train_loss_std": array,
        "val_loss_mean": array,
        "val_loss_std": array,
        "train_acc_mean": array or None,
        "train_acc_std": array or None,
        "val_acc_mean": array or None,
        "val_acc_std": array or None,
        "n_seeds": int
    }
    """
    # Load all CSVs
    lcs = []
    for e in entries_for_model:
        lc = read_lc(e["csv"])
        if lc is not None:
            lcs.append(lc)

    if not lcs:
        return None

    # Get common epochs across all seeds
    epoch_sets = [set(lc["epoch"]) for lc in lcs]
    common_epochs = sorted(set.intersection(*epoch_sets))

    if not common_epochs:
        # If no common epochs, use union and forward-fill
        common_epochs = sorted(set.union(*epoch_sets))

    common_epochs = np.array(common_epochs, dtype=int)

    # Align all to common epochs
    aligned = {}
    for key in ["train_loss", "val_loss", "train_acc", "val_acc"]:
        aligned[key] = []

    for lc in lcs:
        for key in ["train_loss", "val_loss", "train_acc", "val_acc"]:
            if lc[key] is None:
                continue
            # Create mapping from epoch to value
            epoch_to_val = {e: v for e, v in zip(lc["epoch"], lc[key])}
            vals = np.array([epoch_to_val.get(ep, np.nan) for ep in common_epochs])
            aligned[key].append(vals)

    # Compute mean/std
    result = {"epoch": common_epochs}
    for key in ["train_loss", "val_loss", "train_acc", "val_acc"]:
        if aligned[key]:
            stack = np.array(aligned[key])
            result[f"{key}_mean"] = np.nanmean(stack, axis=0)
            result[f"{key}_std"] = np.nanstd(stack, axis=0)
        else:
            result[f"{key}_mean"] = None
            result[f"{key}_std"] = None

    result["n_seeds"] = len(lcs)
    return result


# ───────────────────────────────────────────────────────────────
# Style helpers
# ───────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "const":  "#1f77b4",
    "shared": "#ff7f0e",
    "diag":   "#2ca02c",
    "lstm":   "#d62728",
    "gru":    "#9467bd",
}


def color_for(label: str):
    """Get color for a model label."""
    key = label.lower().strip()
    return MODEL_COLORS.get(key, None)


# ───────────────────────────────────────────────────────────────
# Plotting
# ───────────────────────────────────────────────────────────────

def plot_overlay_loss(model_data_agg, outdir, dpi, ylog):
    """All models, train (solid) + val (dashed) loss with ±1std band if multi-seed."""
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    any_data = False

    for model_name in sorted(model_data_agg.keys()):
        agg = model_data_agg[model_name]
        if agg is None or agg["epoch"].size == 0:
            continue
        any_data = True

        c = color_for(model_name)

        # Train loss
        seed_utils.shade_between(
            ax,
            agg["epoch"],
            agg["train_loss_mean"],
            agg["train_loss_std"],
            color=c,
            alpha=0.2,
            linewidth=1.8,
            label=f'{model_name} (train)',
        )

        # Val loss
        seed_utils.shade_between(
            ax,
            agg["epoch"],
            agg["val_loss_mean"],
            agg["val_loss_std"],
            color=c,
            alpha=0.2,
            linewidth=1.2,
            linestyle="--",
            label=f'{model_name} (val)',
        )

    if not any_data:
        plt.close(fig)
        return

    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Learning curves — loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "learning_curves_loss.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_r2(model_data_agg, outdir, dpi):
    """All models, train (solid) + val (dashed) R² with ±1std band if multi-seed."""
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    any_data = False

    for model_name in sorted(model_data_agg.keys()):
        agg = model_data_agg[model_name]
        if agg is None or agg["epoch"].size == 0:
            continue
        if agg["train_acc_mean"] is None or agg["val_acc_mean"] is None:
            continue
        any_data = True

        c = color_for(model_name)

        # Train R²
        seed_utils.shade_between(
            ax,
            agg["epoch"],
            agg["train_acc_mean"],
            agg["train_acc_std"],
            color=c,
            alpha=0.2,
            linewidth=1.8,
            label=f'{model_name} (train)',
        )

        # Val R²
        seed_utils.shade_between(
            ax,
            agg["epoch"],
            agg["val_acc_mean"],
            agg["val_acc_std"],
            color=c,
            alpha=0.2,
            linewidth=1.2,
            linestyle="--",
            label=f'{model_name} (val)',
        )

    if not any_data:
        plt.close(fig)
        return

    ax.set_xlabel("epoch")
    ax.set_ylabel(r"R$^2$")
    ax.set_title(r"Learning curves — R$^2$")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "learning_curves_r2.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_per_model(model_data_agg, outdir, dpi, ylog):
    """Per-model 2-panel: top = loss, bottom = R² with ±1std band if multi-seed."""
    perdir = os.path.join(outdir, "per_model")
    os.makedirs(perdir, exist_ok=True)

    for model_name in sorted(model_data_agg.keys()):
        agg = model_data_agg[model_name]
        if agg is None or agg["epoch"].size == 0:
            continue

        has_r2 = (agg["train_acc_mean"] is not None) and (agg["val_acc_mean"] is not None)
        nrows = 2 if has_r2 else 1
        fig, axes = plt.subplots(nrows, 1, figsize=(7.4, 3.4 * nrows), sharex=True)
        if nrows == 1:
            axes = [axes]

        # Loss panel
        ax = axes[0]
        seed_utils.shade_between(
            ax,
            agg["epoch"],
            agg["train_loss_mean"],
            agg["train_loss_std"],
            alpha=0.2,
            linewidth=1.8,
            label="train"
        )
        seed_utils.shade_between(
            ax,
            agg["epoch"],
            agg["val_loss_mean"],
            agg["val_loss_std"],
            alpha=0.2,
            linewidth=1.2,
            linestyle="--",
            label="val"
        )
        if ylog:
            ax.set_yscale("log")
        ax.set_ylabel("MSE loss")
        ax.set_title(f'Learning curve — {model_name}')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

        # R² panel
        if has_r2:
            ax2 = axes[1]
            seed_utils.shade_between(
                ax2,
                agg["epoch"],
                agg["train_acc_mean"],
                agg["train_acc_std"],
                alpha=0.2,
                linewidth=1.8,
                label="train"
            )
            seed_utils.shade_between(
                ax2,
                agg["epoch"],
                agg["val_acc_mean"],
                agg["val_acc_std"],
                alpha=0.2,
                linewidth=1.2,
                linestyle="--",
                label="val"
            )
            ax2.set_ylabel(r"R$^2$")
            ax2.grid(True, alpha=0.25)
            ax2.legend(fontsize=8)

        axes[-1].set_xlabel("epoch")
        fig.tight_layout()

        safe = model_name.replace("/", "__").replace("\\", "__")
        fig.savefig(os.path.join(perdir, f"{safe}_learning_curve.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)


# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Plot multi-seed learnability learning curves.")
    seed_utils.add_multiseed_args(ap)
    ap.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Where to save plots (default: based on inputdir)"
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=300
    )
    ap.add_argument(
        "--ylog",
        type=int,
        default=0,
        help="If 1, use log-scale on loss y-axis"
    )
    args = ap.parse_args()

    # Resolve input directories
    inputdirs = seed_utils.resolve_inputdirs(args)
    seed_dirs = seed_utils.discover_from_multiple_inputdirs(inputdirs)

    if not seed_dirs:
        raise ValueError("No seed directories found (or inputdir not specified)")

    seed_utils.print_seed_info(seed_dirs, inputdirs)

    # Set output directory
    if args.outdir is None:
        # Use first inputdir as base
        args.outdir = os.path.join(inputdirs[0], "plots_learning_curves")
    os.makedirs(args.outdir, exist_ok=True)

    # Find learning curve CSVs
    model_data = find_learning_curve_csvs(seed_dirs)

    if not model_data:
        raise RuntimeError(f"No *_learning_curve.csv found in seed directories")

    print(f"Found {len(model_data)} model(s):")
    for model_name in sorted(model_data.keys()):
        entries = model_data[model_name]
        print(f"  {model_name:12s}  ({len(entries)} seed(s))")
        for e in entries:
            print(f"    - {e['csv']}")

    # Aggregate each model across seeds
    model_data_agg = {}
    for model_name, entries in model_data.items():
        agg = aggregate_learning_curves(entries)
        if agg is not None:
            model_data_agg[model_name] = agg

    if not model_data_agg:
        raise RuntimeError("No valid learning curve data after aggregation")

    # Plot
    plot_overlay_loss(model_data_agg, args.outdir, args.dpi, ylog=bool(args.ylog))
    plot_overlay_r2(model_data_agg, args.outdir, args.dpi)
    plot_per_model(model_data_agg, args.outdir, args.dpi, ylog=bool(args.ylog))

    print(f"[OK] Saved learning-curve plots to: {args.outdir}")


if __name__ == "__main__":
    main()
