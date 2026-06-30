#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-seed version of plot_noise_floor.py

Accepts --inputdirs or --inputdir, auto-discovers seed_* subdirs,
loads summary CSVs across seeds, aggregates by averaging on ell,
and plots envelope + eps_th curves with optional shaded bands.

Produces:
  - envelope_with_eps_th.png (with shaded ±1std band if multi-seed)
  - sigma_alpha_hat_vs_ell.png (with shaded band if multi-seed)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seed_utils

# Subfolder for organizing output (relative to outdir)
SUBFOLDER = "noise_floor"

# ── Helper functions for dual-alpha format ──
def _alpha_col(method):
    """Return column name for alpha given method."""
    return f"alpha_{method}"

def _sigma_col(method):
    """Return column name for sigma_alpha given method."""
    if method == "hat":
        return "sigma_alpha_hat"  # old single-method format
    return f"sigma_{method}"

def _nreq_col(method):
    """Return column name for N_required given method."""
    if method == "hat":
        return "N_required_at_eps"  # old single-method format
    return f"N_required_{method}"


CANDIDATE_FILES = {
    "const": "const_summary.csv",
    "shared": "shared_summary.csv",
    "diag": "diag_summary.csv",
    "gru": "gru_summary.csv",
    "lstm": "lstm_summary.csv",
}


def parse_args():
    p = argparse.ArgumentParser()
    seed_utils.add_multiseed_args(p)
    seed_utils.add_view_arg(p)
    p.add_argument(
        "--outdir",
        type=str,
        default="figures",
        help="Directory where figures will be saved (default: figures)"
    )
    p.add_argument(
        "--N_budgets",
        type=str,
        default="500,8000",
        help="Comma-separated list of two training budgets N (default: 500,8000)"
    )
    return p.parse_args()


def safe_alpha(a: np.ndarray) -> np.ndarray:
    """
    Numerical safeguard:
    If alpha <= 1, set alpha = 2 (Gaussian limit).
    Also clip to (0, 2].
    """
    a = a.astype(float)
    a = np.where(a <= 1.0, 2.0, a)
    a = np.clip(a, 1e-6, 2.0)
    return a


def kappa_from_alpha(a: np.ndarray) -> np.ndarray:
    """
    kappa_alpha = alpha / (alpha - 1) for alpha > 1.
    With the safeguard alpha<=1 -> 2, this is well-defined.
    """
    a = safe_alpha(a)
    return a / (a - 1.0)


def load_summary_from_df(df: pd.DataFrame, method: str = "ecf") -> pd.DataFrame:
    """Extract method-specific columns from a summary DataFrame."""
    alpha_col = _alpha_col(method)
    sigma_col = _sigma_col(method)
    nreq_col = _nreq_col(method)

    # Try new format first
    if alpha_col in df.columns:
        result = df[["ell", "mu_l1_mean", alpha_col, sigma_col, nreq_col]].copy()
        result.columns = ["ell", "mu_l1_mean", "alpha_hat", "sigma_alpha_hat", "N_required_at_eps"]
    else:
        # Old format
        result = df[["ell", "mu_l1_mean", "alpha_hat", "sigma_alpha_hat", "N_required_at_eps"]].copy()

    return result

def load_summary(path: str, method: str = "ecf") -> pd.DataFrame:
    """Load and validate summary CSV. Auto-detects old or new format."""
    df = pd.read_csv(path)

    # Determine which columns to expect based on format
    alpha_col = _alpha_col(method)
    sigma_col = _sigma_col(method)
    nreq_col = _nreq_col(method)

    # Try new format first
    needed_new = ["ell", "mu_l1_mean", alpha_col, sigma_col, nreq_col]
    cols_new = [c for c in needed_new if c in df.columns]

    # Fall back to old format
    if len(cols_new) < len(needed_new):
        needed_old = ["ell", "mu_l1_mean", "alpha_hat", "sigma_alpha_hat", "N_required_at_eps"]
        missing = [c for c in needed_old if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}. Found: {list(df.columns)}")
        agg_cols = ["mu_l1_mean", "alpha_hat", "sigma_alpha_hat", "N_required_at_eps"]
    else:
        agg_cols = ["mu_l1_mean", alpha_col, sigma_col, nreq_col]

    df = df.copy()
    df["ell"] = pd.to_numeric(df["ell"], errors="coerce")
    df = df.dropna(subset=["ell"])
    df["ell"] = df["ell"].astype(int)

    # Average duplicate lag rows before plotting.
    agg_cols = [c for c in agg_cols if c in df.columns]
    for c in agg_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.groupby("ell", as_index=False)[agg_cols].mean()

    return df.sort_values("ell").reset_index(drop=True)


def implied_eps_th(
    f_hat: np.ndarray,
    Nreq: np.ndarray,
    alpha_hat: np.ndarray,
    N_budget: int
) -> np.ndarray:
    """
    Construct an implied detectability threshold eps_th(ell; N_budget) in "envelope units",
    using the local scaling relation N(ell) ∝ f(ell)^(-kappa_alpha).

    Given observed (f_hat(ell), Nreq(ell)), define:
        eps_th = f_hat * (Nreq / N_budget)^(1/kappa).

    Then:
        f_hat >= eps_th   <=>   N_budget >= Nreq
    """
    f_hat = f_hat.astype(float)
    Nreq = Nreq.astype(float)
    kappa = kappa_from_alpha(alpha_hat)

    eps_th = np.full_like(f_hat, np.nan, dtype=float)
    mask = (
        np.isfinite(f_hat) & (f_hat > 0) &
        np.isfinite(Nreq) & (Nreq > 0) &
        np.isfinite(kappa) & (kappa > 0)
    )
    ratio = Nreq[mask] / float(N_budget)
    eps_th[mask] = f_hat[mask] * np.exp((1.0 / kappa[mask]) * np.log(ratio))
    return eps_th


def restrict_to_common_ell(df: pd.DataFrame, common_ells: np.ndarray) -> pd.DataFrame:
    """Restrict DataFrame to common ell values."""
    return df.set_index("ell").reindex(common_ells).reset_index()


def aggregate_summaries_by_ell(
    dfs: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Given list of summaries (one per seed), aggregate by ell using mean/std.
    Returns DataFrame with columns: ell, <col>_mean, <col>_std
    """
    if not dfs:
        return pd.DataFrame()

    # Concatenate all
    combined = pd.concat(dfs, ignore_index=True)

    # Group by ell and aggregate
    agg_cols = ["mu_l1_mean", "alpha_hat", "sigma_alpha_hat", "N_required_at_eps"]
    agg_dict = {c: ["mean", "std"] for c in agg_cols if c in combined.columns}

    if not agg_dict:
        return pd.DataFrame()

    grouped = combined.groupby("ell").agg(agg_dict)

    # Flatten column names
    result = pd.DataFrame({"ell": grouped.index})
    for col in agg_cols:
        if col in combined.columns:
            result[f"{col}_mean"] = grouped[(col, "mean")].values
            result[f"{col}_std"] = grouped[(col, "std")].values

    return result.reset_index(drop=True).sort_values("ell")


def _plot_envelope_eps_aggregated(summaries, ells, N_budgets, outpath, is_multiseed, method: str = "ecf"):
    """Plot aggregated envelope + eps_th (mean ± std)."""
    alpha_col = _alpha_col(method)
    nreq_col = _nreq_col(method)

    plt.figure(figsize=(7.4, 4.8))
    for name, df in summaries.items():
        f_hat = df["mu_l1_mean_mean"].to_numpy(dtype=float) if "mu_l1_mean_mean" in df.columns else df["mu_l1_mean"].to_numpy(dtype=float)
        f_std = df["mu_l1_mean_std"].to_numpy(dtype=float) if "mu_l1_mean_std" in df.columns else None
        color = seed_utils.get_model_color(name)
        mask = np.isfinite(f_hat) & (f_hat > 0)
        if np.any(mask):
            line = plt.plot(ells[mask], f_hat[mask], marker="o", linewidth=2,
                            label=rf"{name}: $\hat f(\ell)$", color=color)
            if is_multiseed and f_std is not None:
                plt.fill_between(ells[mask], f_hat[mask] - f_std[mask],
                                 f_hat[mask] + f_std[mask], alpha=0.2, color=color)

    for N in N_budgets:
        for name, df in summaries.items():
            f_hat = df["mu_l1_mean_mean"].to_numpy(dtype=float) if "mu_l1_mean_mean" in df.columns else df["mu_l1_mean"].to_numpy(dtype=float)
            # Try method-specific columns first, fall back to old format
            a_hat_col = f"{alpha_col}_mean" if f"{alpha_col}_mean" in df.columns else "alpha_hat_mean"
            if a_hat_col not in df.columns:
                a_hat_col = "alpha_hat"
            nreq_col_with_mean = f"{nreq_col}_mean" if f"{nreq_col}_mean" in df.columns else "N_required_at_eps_mean"
            if nreq_col_with_mean not in df.columns:
                nreq_col_with_mean = "N_required_at_eps"

            a_hat = df[a_hat_col].to_numpy(dtype=float) if a_hat_col in df.columns else df["alpha_hat"].to_numpy(dtype=float)
            Nreq = df[nreq_col_with_mean].to_numpy(dtype=float) if nreq_col_with_mean in df.columns else df["N_required_at_eps"].to_numpy(dtype=float)
            eps_th_vals = implied_eps_th(f_hat, Nreq, a_hat, N_budget=N)
            mask = np.isfinite(eps_th_vals) & (eps_th_vals > 0)
            if np.any(mask):
                plt.plot(ells[mask], eps_th_vals[mask], linestyle="--", linewidth=1.6,
                         label=rf"{name}: $\varepsilon_{{\mathrm{{th}}}}(\ell; N={N})$")

    plt.yscale("log")
    plt.xlabel(r"lag $\ell$")
    plt.ylabel(r"Envelope / threshold level (log scale)")
    plt.title(rf"Envelope $\hat f(\ell)$ and $\varepsilon_{{\mathrm{{th}}}}(\ell;N)$ [{method.upper()}]")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[ok] saved: {outpath}")


def _plot_sigma_aggregated(summaries, ells, outpath, is_multiseed, method: str = "ecf"):
    """Plot aggregated sigma_alpha vs ell."""
    sigma_col = _sigma_col(method)
    plt.figure(figsize=(7.0, 4.2))
    for name, df in summaries.items():
        sig_col_with_mean = f"{sigma_col}_mean" if f"{sigma_col}_mean" in df.columns else "sigma_alpha_hat_mean"
        if sig_col_with_mean not in df.columns:
            sig_col_with_mean = "sigma_alpha_hat"

        sig = df[sig_col_with_mean].to_numpy(dtype=float) if sig_col_with_mean in df.columns else df["sigma_alpha_hat"].to_numpy(dtype=float)

        sig_std_col = f"{sigma_col}_std" if f"{sigma_col}_std" in df.columns else "sigma_alpha_hat_std"
        sig_std = df[sig_std_col].to_numpy(dtype=float) if sig_std_col in df.columns else None

        color = seed_utils.get_model_color(name)
        mask = np.isfinite(sig) & (sig > 0)
        if np.any(mask):
            line = plt.plot(ells[mask], sig[mask], linewidth=2, label=name, color=color)
            if is_multiseed and sig_std is not None:
                plt.fill_between(ells[mask], np.maximum(sig[mask] - sig_std[mask], 1e-8),
                                 sig[mask] + sig_std[mask], alpha=0.2, color=color)

    plt.yscale("log")
    plt.xlabel(r"lag $\ell$")
    plt.ylabel(r"Estimated noise scale $\hat\sigma_\alpha(\ell)$ (log scale)")
    plt.title(rf"Noise scale $\hat\sigma_\alpha(\ell)$ [{method.upper()}]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[ok] saved: {outpath}")


def _plot_envelope_per_seed(seed_dirs, N_budgets, outpath):
    """Plot per-seed envelope traces overlaid."""
    plt.figure(figsize=(7.4, 4.8))
    legend_handles = {}

    for model in seed_utils.CANDIDATE_MODELS:
        traces = seed_utils.load_model_data_per_seed(
            seed_dirs, model, {"ell", "mu_l1_mean"})
        if not traces:
            continue
        color = seed_utils.get_model_color(model)
        for i, (seed_label, df) in enumerate(traces):
            ell = df["ell"].to_numpy(dtype=float)
            mu = df["mu_l1_mean"].to_numpy(dtype=float)
            mask = np.isfinite(ell) & np.isfinite(mu) & (mu > 0)
            if mask.sum() == 0:
                continue
            alpha = seed_utils.SEED_ALPHAS[i] if i < len(seed_utils.SEED_ALPHAS) else 0.3
            line, = plt.plot(ell[mask], mu[mask], "-", color=color, alpha=alpha,
                             linewidth=0.8)
            if model not in legend_handles:
                legend_handles[model] = line

    plt.yscale("log")
    plt.xlabel(r"lag $\ell$")
    plt.ylabel(r"Envelope (log scale)")
    plt.title(r"Envelope $\hat f(\ell)$ [per seed]")
    if legend_handles:
        plt.legend(legend_handles.values(), legend_handles.keys(), fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[ok] saved: {outpath}")


def _plot_sigma_per_seed(seed_dirs, outpath, method="ecf"):
    """Plot per-seed sigma_alpha traces overlaid."""
    plt.figure(figsize=(7.0, 4.2))
    legend_handles = {}
    sigma_col = _sigma_col(method)  # e.g. "sigma_ecf"

    for model in seed_utils.CANDIDATE_MODELS:
        # Try method-specific column first, fall back to old format
        traces = seed_utils.load_model_data_per_seed(
            seed_dirs, model, {"ell", sigma_col})
        if not traces:
            traces = seed_utils.load_model_data_per_seed(
                seed_dirs, model, {"ell", "sigma_alpha_hat"})
            sigma_col_use = "sigma_alpha_hat"
        else:
            sigma_col_use = sigma_col
        if not traces:
            continue
        color = seed_utils.get_model_color(model)
        for i, (seed_label, df) in enumerate(traces):
            ell = df["ell"].to_numpy(dtype=float)
            sig = df[sigma_col_use].to_numpy(dtype=float)
            mask = np.isfinite(ell) & np.isfinite(sig) & (sig > 0)
            if mask.sum() == 0:
                continue
            alpha = seed_utils.SEED_ALPHAS[i] if i < len(seed_utils.SEED_ALPHAS) else 0.3
            line, = plt.plot(ell[mask], sig[mask], "-", color=color, alpha=alpha,
                             linewidth=0.8)
            if model not in legend_handles:
                legend_handles[model] = line

    plt.yscale("log")
    plt.xlabel(r"lag $\ell$")
    plt.ylabel(r"$\hat\sigma_\alpha(\ell)$ (log scale)")
    plt.title(r"Noise scale $\hat\sigma_\alpha(\ell)$ [per seed]")
    if legend_handles:
        plt.legend(legend_handles.values(), legend_handles.keys(), fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[ok] saved: {outpath}")


def main():
    args = parse_args()
    view = args.view
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Create subfolder for this script's outputs
    plot_outdir = os.path.join(outdir, SUBFOLDER)
    os.makedirs(plot_outdir, exist_ok=True)

    inputdirs = seed_utils.resolve_inputdirs(args)
    seed_dirs = seed_utils.discover_from_multiple_inputdirs(inputdirs)

    if not seed_dirs:
        raise ValueError("No seed directories found (or inputdir not specified)")

    seed_utils.print_seed_info(seed_dirs, inputdirs)
    print(f"[info] view mode: {view}")

    # Parse N budgets
    try:
        N_budgets = [int(s) for s in args.N_budgets.split(",") if s.strip()]
    except Exception as e:
        raise ValueError(f"Could not parse --N_budgets='{args.N_budgets}'.") from e
    if len(N_budgets) != 2:
        raise ValueError(f"--N_budgets must contain exactly two integers, got: {N_budgets}")

    print(f"[info] loading CSVs from: {', '.join(inputdirs)}")
    print(f"[info] saving figures to: {os.path.abspath(plot_outdir)}")
    print(f"[info] N budgets: {N_budgets}")

    # Detect which alpha methods are available by probing a sample CSV
    alpha_methods = ["hat"]  # default to old format
    for model in seed_utils.CANDIDATE_MODELS:
        per_seed = seed_utils.load_model_data_per_seed(seed_dirs, model)
        if per_seed:
            sample_df = per_seed[0][1]
            if "alpha_ecf" in sample_df.columns:
                alpha_methods = ["ecf", "mcc"]
            elif "alpha_mcc" in sample_df.columns:
                alpha_methods = ["mcc"]
            break

    # Load and aggregate summaries for each model, detecting format
    summaries_by_method = {}
    for method in alpha_methods:
        summaries = {}
        for model in seed_utils.CANDIDATE_MODELS:
            dfs = seed_utils.load_model_summary_across_seeds(seed_dirs, model)
            if dfs:
                # Reload with method parameter
                dfs_method = []
                for df in dfs:
                    try:
                        df_method = load_summary_from_df(df, method)
                        dfs_method.append(df_method)
                    except:
                        pass

                if dfs_method:
                    agg = aggregate_summaries_by_ell(dfs_method)
                    if not agg.empty:
                        summaries[model] = agg

        if summaries:
            summaries_by_method[method] = summaries

    if not summaries_by_method:
        raise FileNotFoundError("No model summary CSVs found in seed directories")

    # Process each method separately
    for method, summaries in summaries_by_method.items():
        print(f"\n[info] processing alpha method: {method}")
        print(f"[info] found {len(summaries)} model(s): {', '.join(summaries.keys())}")

        # Build common ell grid
        ell_sets = [set(df["ell"].to_list()) for df in summaries.values()]
        common_ells = sorted(set.intersection(*ell_sets))
        if not common_ells:
            raise ValueError("No common 'ell' values across detected models.")
        common_ells = np.array(common_ells, dtype=int)

        for name in list(summaries.keys()):
            summaries[name] = restrict_to_common_ell(summaries[name], common_ells)

        ells = common_ells
        is_multiseed = any(
            f"{col}_std" in summaries[name].columns
            for name in summaries
            for col in ["mu_l1_mean", "sigma_alpha_hat"]
        )

        method_tag = f"_{method}" if len(summaries_by_method) > 1 else ""
        tag = "agg_" if view == "both" else ""
        ps_tag = "ps_" if view == "both" else ""

        # ── AGGREGATED VIEW ──
        if view in ("aggregated", "both"):
            _plot_envelope_eps_aggregated(summaries, ells, N_budgets,
                                          os.path.join(plot_outdir, f"{tag}envelope_with_eps_th{method_tag}.png"),
                                          is_multiseed, method=method)
            _plot_sigma_aggregated(summaries, ells,
                                   os.path.join(plot_outdir, f"{tag}sigma_alpha_hat_vs_ell{method_tag}.png"),
                                   is_multiseed, method=method)

        # ── PER-SEED VIEW ──
        if view in ("per_seed", "both"):
            _plot_envelope_per_seed(seed_dirs, N_budgets,
                                    os.path.join(plot_outdir, f"{ps_tag}envelope_with_eps_th{method_tag}.png"))
            _plot_sigma_per_seed(seed_dirs,
                                 os.path.join(plot_outdir, f"{ps_tag}sigma_alpha_hat_vs_ell{method_tag}.png"),
                                 method=method)

        # Summary printout
        for name, df in summaries.items():
            a = safe_alpha(
                df["alpha_hat_mean"].to_numpy(dtype=float) if "alpha_hat_mean" in df.columns else df["alpha_hat"].to_numpy(dtype=float)
            )
            sig = df["sigma_alpha_hat_mean"].to_numpy(dtype=float) if "sigma_alpha_hat_mean" in df.columns else df["sigma_alpha_hat"].to_numpy(dtype=float)
            sig = sig[np.isfinite(sig) & (sig > 0)]
            if sig.size:
                print(f"{name}: alpha_mean={np.mean(a):.3f}, alpha_median={np.median(a):.3f}, "
                      f"sigma_median={np.median(sig):.3e}")
            else:
                print(f"{name}: alpha_mean={np.mean(a):.3f}, alpha_median={np.median(a):.3f}, sigma_median=NA")


if __name__ == "__main__":
    main()
