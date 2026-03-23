#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot f(ell), f_gates(ell), and the ratio R(ell) = f(ell) / f_gates(ell)."""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seed_utils

# Subfolder for organizing output (relative to outdir)
SUBFOLDER = "envelope"

# ── CLI ───────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Plot envelope decomposition: f_gates and shape correction R(ℓ)")
    seed_utils.add_multiseed_args(p)
    seed_utils.add_view_arg(p)
    p.add_argument("--outdir", type=str, default=".", help="Directory where figures will be saved")
    p.add_argument(
        "--grid_mode", type=str, default="union", choices=["union", "intersection"],
        help="How to align ℓ grids across models (default: union).",
    )
    p.add_argument(
        "--drop_nonpositive", type=int, default=1, choices=[0, 1],
        help="If 1, nonpositive values are treated as NaN (default: 1).",
    )
    p.add_argument(
        "--floor_quantile", type=float, default=0.001,
        help="Quantile used to estimate clamp epsilon for log plots.",
    )
    p.add_argument("--floor_scale", type=float, default=0.01)
    p.add_argument("--min_floor", type=float, default=1e-300)
    p.add_argument(
        "--mask_mode", type=str, default="per_model",
        choices=["per_model", "common", "none"],
    )
    return p.parse_args()


# ── Canonical model names ─────────────────────────────────
CANDIDATE_MODELS = ["const", "shared", "diag", "gru", "lstm"]
REQUIRED_COLS = {"ell", "mu_l1_mean", "f_gates", "f_ratio"}


# ── Helpers ───────────────────────────────────────────────

def robust_floor(y: np.ndarray, q: float, scale: float, min_floor: float) -> float:
    y = np.asarray(y, dtype=float)
    pos = y[np.isfinite(y) & (y > 0)]
    if pos.size < 10:
        return float(min_floor)
    fq = float(np.quantile(pos, q))
    return float(max(min_floor, scale * fq))


def clamp_log(y: np.ndarray, eps: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    mask = np.isfinite(y) & (y > 0)
    if mask.any():
        out[mask] = np.log(np.maximum(y[mask], eps))
    return out


def aggregate_decomp_across_seeds(seed_dirs: list, model: str) -> dict | None:
    """
    Load <model>_summary.csv from each seed, aggregate f_gates and f_ratio by ℓ.

    Returns dict with ell, and for each component: mean, std arrays.
    Returns None if model not found or required columns missing.
    """
    dfs = seed_utils.load_model_summary_across_seeds(seed_dirs, model, REQUIRED_COLS)
    if not dfs:
        return None

    agg = seed_utils.aggregate_numeric_by_key(dfs, "ell", ["f_gates", "f_ratio", "mu_l1_mean"])
    if agg.empty:
        return None

    result = {"ell": agg["ell"].to_numpy(dtype=float)}
    for col in ["f_gates", "f_ratio", "mu_l1_mean"]:
        result[f"{col}_mean"] = agg[f"{col}_mean"].to_numpy(dtype=float)
        std_col = f"{col}_std"
        if std_col in agg.columns:
            result[std_col] = agg[std_col].to_numpy(dtype=float)
        else:
            result[std_col] = np.zeros_like(result[f"{col}_mean"])
        cnt_col = f"{col}_count"
        if cnt_col in agg.columns:
            result[f"{col}_count"] = agg[cnt_col].to_numpy(dtype=int)
        else:
            result[f"{col}_count"] = np.ones(len(agg), dtype=int)

    return result


# ── Plotting: envelope components ─────────────────────────

def plot_envelope_components(ells: np.ndarray,
                              model_data: dict,
                              x_transform=None,
                              y_transform=None,
                              xlabel: str = "",
                              ylabel: str = "",
                              title: str = "",
                              outpath: str = "decomp.png",
                              eps_dict: dict | None = None):
    """
    Plot f(ℓ) (solid) and f_gates(ℓ) (dashed) for all architectures.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    legend_handles = []
    legend_labels = []

    components = {
        "mu_l1_mean": {"suffix": r"$f(\ell)$", "ls": "-"},
        "f_gates": {"suffix": r"$f_{\mathrm{gates}}(\ell)$", "ls": "--"},
    }

    for model, data in model_data.items():
        color = seed_utils.get_model_color(model)
        ell_m = data["ell"]

        for comp, cfg in components.items():
            y_mean = np.interp(ells, ell_m, data[f"{comp}_mean"], left=np.nan, right=np.nan)
            y_std = np.interp(ells, ell_m, data.get(f"{comp}_std", np.zeros_like(ell_m)),
                              left=np.nan, right=np.nan)

            x = ells if x_transform is None else x_transform(ells)

            if y_transform is not None:
                e = eps_dict.get(model, 1e-300) if eps_dict else 1e-300
                y_mean_t = y_transform(y_mean, e)
                y_std_t = np.where(
                    np.isfinite(y_mean) & (y_mean > 0),
                    y_std / np.maximum(np.abs(y_mean), 1e-30),
                    np.nan,
                )
            else:
                y_mean_t = y_mean
                y_std_t = y_std

            mask = np.isfinite(x) & np.isfinite(y_mean_t)
            if mask.sum() == 0:
                continue

            is_single_seed = np.all(np.isnan(y_std) | (y_std == 0))
            label = f"{model} {cfg['suffix']}"

            if is_single_seed:
                line, = ax.plot(x[mask], y_mean_t[mask], cfg["ls"],
                                color=color, label=label, linewidth=1.5)
            else:
                lo = y_mean_t[mask] - y_std_t[mask]
                hi = y_mean_t[mask] + y_std_t[mask]
                line, = ax.plot(x[mask], y_mean_t[mask], cfg["ls"],
                                color=color, linewidth=1.5)
                ax.fill_between(x[mask], lo, hi, color=color, alpha=0.15)

            legend_handles.append(line)
            legend_labels.append(label)
            plotted = True

    if not plotted:
        print(f"[warn] no curves plotted for {os.path.basename(outpath)}")
        plt.close()
        return

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(legend_handles, legend_labels, fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[ok] saved: {outpath}")


def plot_shape_correction(ells: np.ndarray,
                          model_data: dict,
                          outpath: str = "decomp_ratio.png"):
    """
    Plot the shape correction ratio R(ℓ) = f(ℓ)/f_gates(ℓ) vs ℓ.

    If R(ℓ) is flat, the optimizer is a uniform scalar.
    If R(ℓ) increases with ℓ, the optimizer extends the memory tail.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False

    for model, data in model_data.items():
        color = seed_utils.get_model_color(model)
        ell_m = data["ell"]

        r_mean = np.interp(ells, ell_m, data["f_ratio_mean"], left=np.nan, right=np.nan)
        r_std = np.interp(ells, ell_m, data.get("f_ratio_std", np.zeros_like(ell_m)),
                          left=np.nan, right=np.nan)

        mask = np.isfinite(r_mean)
        if mask.sum() == 0:
            continue

        is_single_seed = np.all(np.isnan(r_std) | (r_std == 0))

        if is_single_seed:
            ax.plot(ells[mask], r_mean[mask], "-", color=color,
                    label=model, linewidth=1.5)
        else:
            ax.plot(ells[mask], r_mean[mask], "-", color=color,
                    label=model, linewidth=1.5)
            lo = r_mean[mask] - r_std[mask]
            hi = r_mean[mask] + r_std[mask]
            ax.fill_between(ells[mask], lo, hi, color=color, alpha=0.15)

        plotted = True

    if not plotted:
        print(f"[warn] no curves plotted for {os.path.basename(outpath)}")
        plt.close()
        return

    ax.axhline(y=1.0, color="gray", ls=":", alpha=0.5, label=r"$R = 1$ (no correction)")
    ax.set_xlabel(r"lag $\ell$")
    ax.set_ylabel(r"$R(\ell) = f(\ell) / f_{\mathrm{gates}}(\ell)$")
    ax.set_title(r"Shape correction ratio $R(\ell)$: optimizer effect on envelope")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"[ok] saved: {outpath}")


# ── Main ──────────────────────────────────────────────────

def main():
    args = parse_args()

    inputdirs = seed_utils.resolve_inputdirs(args)
    seed_dirs = seed_utils.discover_from_multiple_inputdirs(inputdirs)

    if not seed_dirs:
        raise ValueError("No seed directories found in: " + ", ".join(inputdirs))

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Create subfolder for this script's outputs
    plot_outdir = os.path.join(outdir, SUBFOLDER)
    os.makedirs(plot_outdir, exist_ok=True)

    seed_utils.print_seed_info(seed_dirs, inputdirs)
    print(f"[info] saving figures to: {os.path.abspath(plot_outdir)}")

    # ── Load data ─────────────────────────────────────────
    print("[info] loading decomposition data...")
    model_data = {}
    for model in CANDIDATE_MODELS:
        data = aggregate_decomp_across_seeds(seed_dirs, model)
        if data is not None:
            fg = data["f_gates_mean"]
            fr = data["f_ratio_mean"]
            if np.all(np.isnan(fg)):
                print(f"  - {model}: SKIPPED (all NaN — likely training diverged)")
                continue
            model_data[model] = data
            n_seeds = int(data.get("f_gates_count", np.ones(1)).max())
            r_finite = fr[np.isfinite(fr)]
            print(f"  - {model}: {n_seeds} seed(s), "
                  f"f_gates range [{np.nanmin(fg):.3e}, {np.nanmax(fg):.3e}], "
                  f"R(ℓ) range [{np.nanmin(r_finite):.3f}, {np.nanmax(r_finite):.3f}]"
                  if r_finite.size > 0 else
                  f"  - {model}: {n_seeds} seed(s), f_gates range [{np.nanmin(fg):.3e}, {np.nanmax(fg):.3e}]")

    if not model_data:
        print(
            "[warn] No models with valid decomposition data found. "
            "This plot requires summary CSVs with columns "
            f"{sorted(REQUIRED_COLS)}. Older runs do not contain the "
            "new decomposition outputs and must be regenerated with the "
            "updated diagnostic pipeline."
        )
        return

    print(f"[info] plotting {len(model_data)} model(s): {', '.join(model_data.keys())}")

    # ── Build common ℓ grid ───────────────────────────────
    ell_sets = []
    for data in model_data.values():
        e = data["ell"]
        e = e[np.isfinite(e) & (e > 0)]
        ell_sets.append(set(e.tolist()))

    if args.grid_mode == "intersection":
        grid_ells = sorted(set.intersection(*ell_sets)) if ell_sets else []
    else:
        grid_ells = sorted(set.union(*ell_sets)) if ell_sets else []

    ells = np.array(grid_ells, dtype=float)
    if len(ells) == 0:
        print("[warn] No usable ℓ values found after filtering; nothing to plot.")
        return

    # Optionally drop nonpositive
    if args.drop_nonpositive:
        for data in model_data.values():
            for col in ["f_gates_mean", "mu_l1_mean_mean"]:
                if col in data:
                    y = data[col].astype(float, copy=False)
                    y[~np.isfinite(y)] = np.nan
                    y[y <= 0] = np.nan
                    data[col] = y

    # ── Compute clamp epsilons (for log plots) ────────────
    eps_dict = {}
    for model, data in model_data.items():
        all_vals = np.concatenate([
            data["f_gates_mean"][np.isfinite(data["f_gates_mean"])],
            data["mu_l1_mean_mean"][np.isfinite(data["mu_l1_mean_mean"])],
        ])
        eps_dict[model] = robust_floor(
            all_vals,
            q=float(args.floor_quantile),
            scale=float(args.floor_scale),
            min_floor=float(args.min_floor),
        )

    if args.mask_mode == "common":
        common_eps = max(eps_dict.values()) if eps_dict else float(args.min_floor)
        eps_dict = {m: common_eps for m in eps_dict}

    # ── Plot 1: log-lin (log f vs ℓ) ─────────────────────
    plot_envelope_components(
        ells, model_data,
        y_transform=clamp_log,
        xlabel=r"lag $\ell$",
        ylabel=r"$\log f(\ell)$",
        title=r"Envelope: $f(\ell)$ (solid) vs $f_{\mathrm{gates}}(\ell)$ (dashed)",
        outpath=os.path.join(plot_outdir, "decomp_log_envelope_vs_ell.png"),
        eps_dict=eps_dict,
    )

    # ── Plot 2: log-log (log f vs log ℓ) ─────────────────
    plot_envelope_components(
        ells, model_data,
        x_transform=np.log,
        y_transform=clamp_log,
        xlabel=r"$\log \ell$",
        ylabel=r"$\log f(\ell)$",
        title=r"Envelope: $\log f$ vs $\log \ell$",
        outpath=os.path.join(plot_outdir, "decomp_log_envelope_vs_log_ell.png"),
        eps_dict=eps_dict,
    )

    # ── Plot 3: shape correction ratio R(ℓ) ──────────────
    plot_shape_correction(
        ells, model_data,
        outpath=os.path.join(plot_outdir, "decomp_ratio_vs_ell.png"),
    )

    print("[done]")


if __name__ == "__main__":
    main()
