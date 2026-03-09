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


def load_summary(path: str) -> pd.DataFrame:
    """Load and validate summary CSV."""
    df = pd.read_csv(path)
    needed = ["ell", "mu_l1_mean", "alpha_hat", "sigma_alpha_hat", "N_required_at_eps"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}. Found: {list(df.columns)}")

    df = df.copy()
    df["ell"] = pd.to_numeric(df["ell"], errors="coerce")
    df = df.dropna(subset=["ell"])
    df["ell"] = df["ell"].astype(int)

    # If duplicates exist, average them (robustness)
    agg_cols = ["mu_l1_mean", "alpha_hat", "sigma_alpha_hat", "N_required_at_eps"]
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

    # Parse N budgets
    try:
        N_budgets = [int(s) for s in args.N_budgets.split(",") if s.strip()]
    except Exception as e:
        raise ValueError(f"Could not parse --N_budgets='{args.N_budgets}'. Use e.g. '500,8000'.") from e
    if len(N_budgets) != 2:
        raise ValueError(f"--N_budgets must contain exactly two integers, got: {N_budgets}")

    OUT_ENVELOPE_EPS = os.path.join(outdir, "envelope_with_eps_th.png")
    OUT_SIGMA = os.path.join(outdir, "sigma_alpha_hat_vs_ell.png")

    print(f"[info] loading CSVs from: {', '.join(inputdirs)}")
    print(f"[info] saving figures to: {os.path.abspath(outdir)}")
    print(f"[info] N budgets: {N_budgets}")

    # Load and aggregate summaries for each model
    summaries = {}
    for model in seed_utils.CANDIDATE_MODELS:
        dfs = seed_utils.load_model_summary_across_seeds(seed_dirs, model)
        if not dfs:
            continue

        # Load as summaries
        loaded_dfs = []
        for df in dfs:
            loaded_dfs.append(df)

        if loaded_dfs:
            # Aggregate by ell
            agg = aggregate_summaries_by_ell(loaded_dfs)
            if not agg.empty:
                summaries[model] = agg

    if not summaries:
        raise FileNotFoundError("No model summary CSVs found in seed directories")

    print(f"[info] found {len(summaries)} model(s): {', '.join(summaries.keys())}")

    # Build common ell grid (intersection) for aligned overlay plots
    ell_sets = []
    for name, df in summaries.items():
        ell_sets.append(set(df["ell"].to_list()))
    common_ells = sorted(set.intersection(*ell_sets))
    if len(common_ells) == 0:
        raise ValueError("No common 'ell' values across detected models (cannot align overlay plots).")
    common_ells = np.array(common_ells, dtype=int)

    # Restrict all summaries to common ell grid
    for name in list(summaries.keys()):
        summaries[name] = restrict_to_common_ell(summaries[name], common_ells)

    ells = common_ells

    # Check if we have multi-seed data (look for std columns)
    is_multiseed = any(
        f"{col}_std" in summaries[name].columns
        for name in summaries
        for col in ["mu_l1_mean", "sigma_alpha_hat"]
    )

    # ────────────────────────────────────────────────────────────────
    # 1) Envelope + eps_th(ell; N) plot
    # ────────────────────────────────────────────────────────────────
    plt.figure(figsize=(7.4, 4.8))

    # Envelope curves
    for name, df in summaries.items():
        f_hat = df["mu_l1_mean_mean"].to_numpy(dtype=float) if "mu_l1_mean_mean" in df.columns else df["mu_l1_mean"].to_numpy(dtype=float)
        f_std = df["mu_l1_mean_std"].to_numpy(dtype=float) if "mu_l1_mean_std" in df.columns else None

        mask = np.isfinite(f_hat) & (f_hat > 0)
        if np.any(mask):
            line = plt.plot(ells[mask], f_hat[mask], marker="o", linewidth=2, label=rf"{name}: $\hat f(\ell)$")
            if is_multiseed and f_std is not None:
                c = line[0].get_color()
                f_std_masked = f_std[mask]
                plt.fill_between(
                    ells[mask],
                    f_hat[mask] - f_std_masked,
                    f_hat[mask] + f_std_masked,
                    alpha=0.2,
                    color=c
                )
        else:
            print(f"[warn] {name}: no finite positive mu_l1_mean values to plot for envelope.")

    # Threshold curves eps_th(ell; N)
    for N in N_budgets:
        for name, df in summaries.items():
            f_hat = df["mu_l1_mean_mean"].to_numpy(dtype=float) if "mu_l1_mean_mean" in df.columns else df["mu_l1_mean"].to_numpy(dtype=float)
            a_hat = df["alpha_hat_mean"].to_numpy(dtype=float) if "alpha_hat_mean" in df.columns else df["alpha_hat"].to_numpy(dtype=float)
            Nreq = df["N_required_at_eps_mean"].to_numpy(dtype=float) if "N_required_at_eps_mean" in df.columns else df["N_required_at_eps"].to_numpy(dtype=float)
            a_std = df["alpha_hat_std"].to_numpy(dtype=float) if "alpha_hat_std" in df.columns else None

            eps_th = implied_eps_th(f_hat, Nreq, a_hat, N_budget=N)
            mask = np.isfinite(eps_th) & (eps_th > 0)
            if np.any(mask):
                line = plt.plot(
                    ells[mask],
                    eps_th[mask],
                    linestyle="--",
                    linewidth=1.6,
                    label=rf"{name}: $\varepsilon_{{\mathrm{{th}}}}(\ell; N={N})$",
                )
                if is_multiseed and a_std is not None:
                    # Propagate uncertainty through eps_th formula
                    c = line[0].get_color()
                    # Simple approximation: use alpha uncertainty
                    eps_th_plus = implied_eps_th(f_hat, Nreq, a_hat + a_std, N_budget=N)
                    eps_th_minus = implied_eps_th(f_hat, Nreq, a_hat - a_std, N_budget=N)
                    plt.fill_between(
                        ells[mask],
                        eps_th_minus[mask],
                        eps_th_plus[mask],
                        alpha=0.15,
                        color=c
                    )
            else:
                print(f"[warn] {name}: no valid eps_th values for N={N} (skipping threshold curve).")

    plt.yscale("log")
    plt.xlabel(r"lag $\ell$")
    plt.ylabel(r"Envelope / threshold level (log scale)")
    plt.title(r"Envelope $\hat f(\ell)$ and detectability threshold $\varepsilon_{\mathrm{th}}(\ell;N)$")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_ENVELOPE_EPS, dpi=300)
    plt.close()

    # ────────────────────────────────────────────────────────────────
    # 2) sigma_alpha_hat(ell) vs ell plot
    # ────────────────────────────────────────────────────────────────
    plt.figure(figsize=(7.0, 4.2))

    for name, df in summaries.items():
        sig = df["sigma_alpha_hat_mean"].to_numpy(dtype=float) if "sigma_alpha_hat_mean" in df.columns else df["sigma_alpha_hat"].to_numpy(dtype=float)
        sig_std = df["sigma_alpha_hat_std"].to_numpy(dtype=float) if "sigma_alpha_hat_std" in df.columns else None

        mask = np.isfinite(sig) & (sig > 0)
        if np.any(mask):
            line = plt.plot(ells[mask], sig[mask], linewidth=2, label=name)
            if is_multiseed and sig_std is not None:
                c = line[0].get_color()
                sig_std_masked = sig_std[mask]
                plt.fill_between(
                    ells[mask],
                    np.maximum(sig[mask] - sig_std_masked, 1e-8),
                    sig[mask] + sig_std_masked,
                    alpha=0.2,
                    color=c
                )
        else:
            print(f"[warn] {name}: no finite positive sigma_alpha_hat values to plot.")

    plt.yscale("log")
    plt.xlabel(r"lag $\ell$")
    plt.ylabel(r"Estimated noise scale $\hat\sigma_\alpha(\ell)$ (log scale)")
    plt.title(r"Estimated noise scale $\hat\sigma_\alpha(\ell)$ across diagnostic lags")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_SIGMA, dpi=300)
    plt.close()

    print(f"[ok] Saved: {OUT_ENVELOPE_EPS}")
    print(f"[ok] Saved: {OUT_SIGMA}")

    # Optional summary printout
    for name, df in summaries.items():
        a = safe_alpha(
            df["alpha_hat_mean"].to_numpy(dtype=float) if "alpha_hat_mean" in df.columns else df["alpha_hat"].to_numpy(dtype=float)
        )
        sig = df["sigma_alpha_hat_mean"].to_numpy(dtype=float) if "sigma_alpha_hat_mean" in df.columns else df["sigma_alpha_hat"].to_numpy(dtype=float)
        sig = sig[np.isfinite(sig) & (sig > 0)]
        if sig.size:
            print(
                f"{name}: alpha_mean={np.mean(a):.3f}, alpha_median={np.median(a):.3f}, "
                f"sigma_median={np.median(sig):.3e}"
            )
        else:
            print(f"{name}: alpha_mean={np.mean(a):.3f}, alpha_median={np.median(a):.3f}, sigma_median=NA")


if __name__ == "__main__":
    main()
