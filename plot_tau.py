#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_tau.py (multi-seed + auto-merge version)

Multi-seed variant of the original plot_tau.py that:
  - Accepts --inputdirs / --inputdir arguments
  - Auto-discovers seed_* subdirectories within each inputdir
  - Pools tau values across all seeds for each model
  - Preserves all original plotting features: --hist, --both, --ccdf, --separate,
    --logx, --logy, --xlow_percentile, --tau_min, --print_counts, etc.
  - Reports seed counts when printing statistics
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
    p.add_argument("--outdir", type=str, default="figures", help="Output folder (default: figures)")

    p.add_argument("--min_points", type=int, default=3,
                   help="Min positive points per unit to fit τ from μ units (default: 3)")
    p.add_argument("--mu_floor", type=float, default=1e-300,
                   help="Floor for |μ| in log fit (default: 1e-300)")

    p.add_argument("--cap_percentile", type=float, default=99.5,
                   help="Upper cap percentile for τ cleaning (default: 99.5)")

    p.add_argument("--max_bins", type=int, default=30, help="Max histogram bins (default: 30)")
    p.add_argument("--kde_grid", type=int, default=400, help="Grid points for KDE curve (default: 400)")
    p.add_argument("--kde_min_n", type=int, default=5,
                   help="Minimum n for KDE; below this uses rug or hist (default: 5)")
    p.add_argument("--rug_height_frac", type=float, default=0.06,
                   help="Rug tick height as fraction of y-range (default: 0.06)")

    p.add_argument("--hist", action="store_true", help="Histogram only (default: KDE only)")
    p.add_argument("--both", action="store_true", help="Histogram + KDE")
    p.add_argument("--ccdf", action="store_true", help="Plot CCDF (overlay only)")
    p.add_argument("--separate", action="store_true", help="Plot per-model figures instead of overlay")

    p.add_argument("--logx", action="store_true",
                   help="Log-scale x-axis (overlay only; ignored in --separate)")
    p.add_argument("--logy", action="store_true",
                   help="Log-scale y-axis (useful with --ccdf; overlay only)")

    p.add_argument("--xlow_percentile", type=float, default=0.5,
                   help=("Overlay-only: pooled lower x cutoff percentile. "
                         "Values below are not shown. Default: 0.5"))
    p.add_argument("--tau_min", type=float, default=0.0,
                   help=("Overlay-only: absolute lower τ cutoff for visualization. "
                         "Default 0 (disabled). Suggested: 1e-3"))

    p.add_argument("--print_counts", action="store_true",
                   help="Print per-model τ counts (cleaned and after xmin cut)")

    return p.parse_args()


def _safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [warn] failed reading {os.path.basename(path)}: {e}")
        return None


def _clean_tau(arr, cap_percentile=99.5):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size >= 2:
        cap = np.nanpercentile(arr, float(cap_percentile))
        arr = arr[arr <= max(cap, 1e-12)]
    return arr


def kde_1d(x, grid):
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    if n < 2:
        return np.zeros_like(grid)

    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(std, iqr / 1.349) if (std > 0 and iqr > 0) else (std if std > 0 else 1.0)
    h = max(0.9 * sigma * n ** (-1 / 5), 1e-6)

    z = (grid[:, None] - x[None, :]) / h
    dens = np.mean(np.exp(-0.5 * z ** 2), axis=1) / (h * np.sqrt(2 * np.pi))
    return dens


def _maybe_add_legend(ax):
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)


def _vlines_rug(ax, x, base_y, height, label=None):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return
    ax.vlines(x, base_y, base_y + height, linewidth=1.2, label=label)


def estimate_tau_from_mu_units(path, min_points=3, mu_floor=1e-300):
    df = _safe_read_csv(path)
    if df is None or df.shape[1] < 2:
        return np.array([], dtype=float)

    ell_col = df.columns[0]
    ell = pd.to_numeric(df[ell_col], errors="coerce").to_numpy(dtype=float)
    units = df.drop(columns=[ell_col]).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    taus = []
    for q in range(units.shape[1]):
        mu_abs = np.abs(units[:, q])
        mask = np.isfinite(ell) & np.isfinite(mu_abs) & (ell > 0) & (mu_abs > float(mu_floor))
        if int(mask.sum()) < int(min_points):
            continue

        x = ell[mask]
        y = np.log(np.maximum(mu_abs[mask], float(mu_floor)))
        A = np.vstack([np.ones_like(x), x]).T
        _, b = np.linalg.lstsq(A, y, rcond=None)[0]
        if np.isfinite(b) and b < 0:
            taus.append(-1.0 / b)

    return np.asarray(taus, dtype=float)


def load_tau_mu(seed_dirs, model, args):
    """Load tau from mu units across all seed dirs, pooling results."""
    all_taus = []
    for indir in seed_dirs:
        tau_file = seed_utils.find_file_in_seed_dir(indir, f"{model}_tau_from_mu_units.csv", model)
        if tau_file is not None:
            df = _safe_read_csv(tau_file)
            if df is not None:
                if "tau" in df.columns:
                    tau_vals = df["tau"].to_numpy(dtype=float)
                    if tau_vals.size > 0:
                        all_taus.append(tau_vals)
                    continue
                if "tau_mu" in df.columns:
                    tau_vals = df["tau_mu"].to_numpy(dtype=float)
                    if tau_vals.size > 0:
                        all_taus.append(tau_vals)
                    continue

        mu_units = seed_utils.find_file_in_seed_dir(indir, f"{model}_mu_units.csv", model)
        if mu_units is not None:
            tau_vals = estimate_tau_from_mu_units(mu_units, min_points=args.min_points, mu_floor=args.mu_floor)
            if tau_vals.size > 0:
                all_taus.append(tau_vals)

    if not all_taus:
        return None

    return np.concatenate(all_taus)


def load_tau_gate(seed_dirs, model):
    """Load tau from gate units across all seed dirs, pooling results."""
    candidates = [
        f"{model}_tau_from_gate_units.csv",
        f"{model}_tau_from_gate.csv",
        f"{model}_tau_units.csv",
    ]

    all_taus = []
    for indir in seed_dirs:
        for fname in candidates:
            path = seed_utils.find_file_in_seed_dir(indir, fname, model)
            if path is None:
                continue
            df = _safe_read_csv(path)
            if df is None:
                continue
            for col in ["tau", "tau_q", "tau_gate"]:
                if col in df.columns:
                    tau_vals = df[col].to_numpy(dtype=float)
                    if tau_vals.size > 0:
                        all_taus.append(tau_vals)
                    break

    if not all_taus:
        return None

    return np.concatenate(all_taus)


def detect_models(seed_dirs):
    """Detect which models have tau files in any seed dir (flat or nested)."""
    present = set()
    for indir in seed_dirs:
        for m in CANDIDATE_MODELS:
            if (
                seed_utils.find_file_in_seed_dir(indir, f"{m}_mu_units.csv", m) is not None
                or seed_utils.find_file_in_seed_dir(indir, f"{m}_tau_from_mu_units.csv", m) is not None
                or seed_utils.find_file_in_seed_dir(indir, f"{m}_tau_from_gate.csv", m) is not None
                or seed_utils.find_file_in_seed_dir(indir, f"{m}_tau_from_gate_units.csv", m) is not None
                or seed_utils.find_file_in_seed_dir(indir, f"{m}_tau_units.csv", m) is not None
            ):
                present.add(m)
    return [m for m in CANDIDATE_MODELS if m in present]


def _overlay_xmin_cut(all_taus, args):
    pooled = np.concatenate([np.asarray(v) for v in all_taus.values()
                             if v is not None and np.asarray(v).size > 0])
    pooled = _clean_tau(pooled, cap_percentile=float(args.cap_percentile))
    if pooled.size == 0:
        return None
    p_low = float(np.nanpercentile(pooled, float(args.xlow_percentile)))
    xmin_cut = max(float(args.tau_min), p_low)
    if args.logx:
        xmin_cut = max(xmin_cut, 1e-6)
    return xmin_cut


def _overlay_grid_from_pooled(all_taus, xmin_cut, args):
    pooled = np.concatenate([np.asarray(v) for v in all_taus.values()
                             if v is not None and np.asarray(v).size > 0])
    pooled = _clean_tau(pooled, cap_percentile=float(args.cap_percentile))
    pooled = pooled[pooled >= float(xmin_cut)]
    if pooled.size < 2:
        return None, None, None

    xmin, xmax = float(np.min(pooled)), float(np.max(pooled))
    pad = 0.02 * (xmax - xmin) if xmax > xmin else 0.0
    xmin_plot = max(0.0, xmin - pad)
    xmax_plot = xmax + pad
    if args.logx:
        xmin_plot = max(xmin_plot, 1e-6)

    grid = np.linspace(xmin_plot, xmax_plot, int(args.kde_grid))
    return xmin_plot, xmax_plot, grid


def plot_overlay_pdf(all_taus, title, outfile, args, n_seeds=None):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    xmin_cut = _overlay_xmin_cut(all_taus, args)
    if xmin_cut is None:
        print(f"[warn] Not enough τ values to plot: {outfile}")
        plt.close(fig)
        return

    xmin_plot, xmax_plot, grid = _overlay_grid_from_pooled(all_taus, xmin_cut, args)
    if grid is None:
        print(f"[warn] Not enough τ values after xmin_cut to plot: {outfile}")
        plt.close(fig)
        return

    if args.print_counts:
        if n_seeds is not None:
            print(f"  [info] overlay xmin_cut (visual): {xmin_cut:.3e} ({n_seeds} seed(s))")
        else:
            print(f"  [info] overlay xmin_cut (visual): {xmin_cut:.3e}")

    rug_pending = []

    for model, tau_raw in all_taus.items():
        tau_clean = _clean_tau(tau_raw, cap_percentile=float(args.cap_percentile))
        n_clean = int(tau_clean.size)
        tau = tau_clean[tau_clean >= float(xmin_cut)]
        n_vis = int(tau.size)

        if args.print_counts:
            print(f"  [counts] {model}: n_clean={n_clean}, n_after_xmin_cut={n_vis}")

        if n_vis == 0:
            continue

        label = f"{model}"

        if n_vis == 1:
            ax.axvline(float(tau[0]), linewidth=2, label=label)
            continue

        if n_vis < int(args.kde_min_n):
            if args.hist or args.both:
                bins = min(int(args.max_bins), n_vis)
                ax.hist(tau, bins=bins, density=True, alpha=0.8, label=label, color="navy", edgecolor="black", linewidth=0.5)
            else:
                rug_pending.append((tau, label))
            continue

        if args.hist or args.both:
            bins = min(int(args.max_bins), n_vis)
            hist_label = label if args.hist else None
            ax.hist(tau, bins=bins, density=True, alpha=0.8, label=hist_label, color="navy", edgecolor="black", linewidth=0.5)

        if (not args.hist) or args.both:
            dens = kde_1d(tau, grid)
            ax.plot(grid, dens, linewidth=2, label=label)

    if rug_pending:
        y0, y1 = ax.get_ylim()
        base_y = y0 + 0.02 * (y1 - y0)
        height = float(args.rug_height_frac) * (y1 - y0)
        for tau, label in rug_pending:
            _vlines_rug(ax, tau, base_y=base_y, height=height, label=label)

    if args.logx:
        ax.set_xscale("log")

    ax.set_xlim(xmin_plot, xmax_plot)
    ax.set_xlabel(r"$\tau_q$")
    ax.set_ylabel("density")
    ax.set_title(title + " (PDF)")
    _maybe_add_legend(ax)

    try:
        fig.tight_layout()
    except Exception:
        pass

    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"[ok] saved: {outfile}")


def plot_overlay_ccdf(all_taus, title, outfile, args, n_seeds=None):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    xmin_cut = _overlay_xmin_cut(all_taus, args)
    if xmin_cut is None:
        print(f"[warn] Not enough τ values to plot CCDF: {outfile}")
        plt.close(fig)
        return

    if args.print_counts:
        if n_seeds is not None:
            print(f"  [info] overlay xmin_cut (visual): {xmin_cut:.3e} ({n_seeds} seed(s))")
        else:
            print(f"  [info] overlay xmin_cut (visual): {xmin_cut:.3e}")

    for model, tau_raw in all_taus.items():
        tau_clean = _clean_tau(tau_raw, cap_percentile=float(args.cap_percentile))
        n_clean = int(tau_clean.size)
        tau = tau_clean[tau_clean >= float(xmin_cut)]
        n_vis = int(tau.size)

        if args.print_counts:
            print(f"  [counts] {model}: n_clean={n_clean}, n_after_xmin_cut={n_vis}")

        if n_vis < 2:
            continue

        tau_sorted = np.sort(tau)
        ccdf = 1.0 - np.arange(1, n_vis + 1) / n_vis
        if args.logy:
            ccdf = np.maximum(ccdf, 1.0 / (n_vis + 1))

        ax.plot(tau_sorted, ccdf, linewidth=2, label=model)

    if args.logx:
        ax.set_xscale("log")
    if args.logy:
        ax.set_yscale("log")

    ax.set_xlabel(r"$\tau_q$")
    ax.set_ylabel("P(T ≥ τ)")
    ax.set_title(title + " (CCDF)")
    _maybe_add_legend(ax)

    try:
        fig.tight_layout()
    except Exception:
        pass

    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"[ok] saved: {outfile}")


def plot_single_model_pdf(tau_raw, title, outfile, args):
    fig, ax = plt.subplots(figsize=(6.6, 4.3))

    tau = _clean_tau(tau_raw, cap_percentile=float(args.cap_percentile))
    n = int(tau.size)
    if n == 0:
        print(f"[warn] no τ values for: {outfile}")
        plt.close(fig)
        return

    if n == 1:
        ax.axvline(float(tau[0]), linewidth=2, label="n=1")
        ax.set_xlabel(r"$\tau_q$")
        ax.set_ylabel(" ")
        ax.set_title(title + " (single unit)")
        _maybe_add_legend(ax)
        fig.tight_layout()
        fig.savefig(outfile, dpi=300)
        plt.close(fig)
        print(f"[ok] saved: {outfile}")
        return

    if args.hist or args.both:
        bins = min(int(args.max_bins), n)
        ax.hist(tau, bins=bins, density=True, alpha=0.8, label=("hist" if args.hist else None), color="navy", edgecolor="black", linewidth=0.5)

    if (not args.hist) or args.both:
        xmin, xmax = float(np.min(tau)), float(np.max(tau))
        pad = 0.02 * (xmax - xmin) if xmax > xmin else 0.0
        grid = np.linspace(max(0.0, xmin - pad), xmax + pad, int(args.kde_grid))
        dens = kde_1d(tau, grid)
        ax.plot(grid, dens, linewidth=2, label=("KDE" if args.hist else None))

    ax.set_xlabel(r"$\tau_q$")
    ax.set_ylabel("density")
    ax.set_title(title + " (PDF)")
    _maybe_add_legend(ax)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"[ok] saved: {outfile}")


def main():
    args = parse_args()

    # Resolve input directories and discover seed directories
    inputdirs = seed_utils.resolve_inputdirs(args)
    seed_dirs = seed_utils.discover_from_multiple_inputdirs(inputdirs)

    if not seed_dirs:
        raise RuntimeError(f"No seed directories or input directories found")

    os.makedirs(args.outdir, exist_ok=True)

    if args.hist and args.both:
        print("[warn] both --hist and --both set; using --both.")
        args.hist = False

    if args.separate:
        if args.logx:
            print("[info] --logx ignored in --separate mode.")
            args.logx = False
        if args.logy:
            print("[info] --logy ignored in --separate mode.")
            args.logy = False
        if args.ccdf:
            print("[info] --ccdf ignored in --separate mode (separate uses PDF only).")
            args.ccdf = False

    # Print seed information
    seed_utils.print_seed_info(seed_dirs, inputdirs)

    models = detect_models(seed_dirs)
    if not models:
        raise RuntimeError(f"No model tau files found in: {[os.path.abspath(d) for d in seed_dirs]}")

    print(f"[info] outdir:   {os.path.abspath(args.outdir)}")
    print(f"[info] detected models: {models}")

    tau_mu_all = {}
    tau_gate_all = {}

    for m in models:
        tau_mu = load_tau_mu(seed_dirs, m, args)
        tau_gate = load_tau_gate(seed_dirs, m)

        if tau_mu is not None and np.asarray(tau_mu).size:
            tau_mu_all[m] = np.asarray(tau_mu, dtype=float)
        if tau_gate is not None and np.asarray(tau_gate).size:
            tau_gate_all[m] = np.asarray(tau_gate, dtype=float)

    n_seeds = len(seed_dirs)

    if not args.separate:
        if tau_gate_all:
            if args.ccdf:
                plot_overlay_ccdf(
                    tau_gate_all,
                    title="Time-scale distribution (gate-derived)",
                    outfile=os.path.join(args.outdir, "tau_ccdf_gate_all.png"),
                    args=args,
                    n_seeds=n_seeds,
                )
            else:
                plot_overlay_pdf(
                    tau_gate_all,
                    title="Time-scale distribution (gate-derived)",
                    outfile=os.path.join(args.outdir, "tau_pdf_gate_all.png"),
                    args=args,
                    n_seeds=n_seeds,
                )
        else:
            print("[info] no gate-derived τ distributions found.")

        if tau_mu_all:
            if args.ccdf:
                plot_overlay_ccdf(
                    tau_mu_all,
                    title=r"Time-scale distribution $\tau_q$",
                    outfile=os.path.join(args.outdir, "tau_ccdf_all.png"),
                    args=args,
                    n_seeds=n_seeds,
                )
            else:
                plot_overlay_pdf(
                    tau_mu_all,
                    title=r"Time-scale distribution $\tau_q$",
                    outfile=os.path.join(args.outdir, "tau_pdf_all.png"),
                    args=args,
                    n_seeds=n_seeds,
                )
        else:
            print("[info] no μ-fit τ distributions found.")
    else:
        if not tau_gate_all and not tau_mu_all:
            print("[warn] no τ distributions found to plot.")
            return

        for m, tau in tau_gate_all.items():
            plot_single_model_pdf(
                tau,
                title=f"Time-scale distribution — {m}",
                outfile=os.path.join(args.outdir, f"tau_pdf_gate_{m}.png"),
                args=args,
            )

        for m, tau in tau_mu_all.items():
            plot_single_model_pdf(
                tau,
                title=rf"Time-scale distribution $\tau_q$ — {m}",
                outfile=os.path.join(args.outdir, f"tau_pdf_{m}.png"),
                args=args,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
