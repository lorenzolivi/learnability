#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_master_proportionality.py (multi-seed version)

Fit:
    log N(ell) ~ a + b * (-kappa_alpha(ell) * log f(ell))
with kappa_alpha = alpha/(alpha-1).

IMPORTANT:
- Use N_required_at_eps as the empirical minimal sample complexity.
- Drop N_required_at_eps <= 0 (e.g., -1 = not achievable in grid).
- Skip fits if N has too few unique values (degenerate regression).

Inputs (across seed dirs, or in --inputdir):
- envelope_fits.json (from first seed dir or parent inputdir)
- *_summary.csv (one per model, averaged across seeds on ell)

Outputs (to --outdir):
- fit_master_<model>.png (seed-averaged data, with individual seed points if multi-seed)
- fit_master_points_<model>.csv
- fit_master_summary.csv
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seed_utils

CANON_MODELS = ["const", "shared", "diag", "gru", "lstm"]
JSON_MODELS = {
    "const": "ConstGate",
    "shared": "SharedGate",
    "diag": "DiagGate",
    "gru": "GRU",
    "lstm": "LSTM"
}

def parse_args():
    p = argparse.ArgumentParser()
    seed_utils.add_multiseed_args(p)
    p.add_argument("--outdir", type=str, default=".")
    p.add_argument("--models", type=str, default="")
    p.add_argument("--json_name", type=str, default="envelope_fits.json")
    p.add_argument("--min_points", type=int, default=10)
    p.add_argument(
        "--min_unique_N",
        type=int,
        default=3,
        help="Require at least this many distinct N values"
    )
    p.add_argument("--alpha_floor", type=float, default=1.01)
    p.add_argument(
        "--f_col",
        type=str,
        default="mu_l1_mean",
        help="Envelope column in summary"
    )
    p.add_argument(
        "--N_col",
        type=str,
        default="N_required_at_eps",
        help="Sample complexity column in summary"
    )
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def get_power_window(fits_json, canon_model):
    # Try old combined format: {"models": {"ConstGate": {"power": {...}}}}
    jm = JSON_MODELS[canon_model]
    block = fits_json.get("models", {}).get(jm, {})
    power = block.get("power", None)

    # Try per-model format: {"const": {"power": {...}}} or direct {"power": {...}}
    if power is None:
        block = fits_json.get(canon_model, {})
        power = block.get("power", None)
    if power is None:
        power = fits_json.get("power", None)

    if power is None:
        return None
    # Some formats have ell_min/ell_max, others don't
    return (
        float(power.get("ell_min", 0)),
        float(power.get("ell_max", 1e6)),
        float(power.get("beta", power.get("d", np.nan))),
        float(power.get("r2", np.nan))
    )

def linfit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = np.vstack([np.ones_like(x), x]).T
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(a), float(b), float(r2), yhat

def find_envelope_fits_json(seed_dirs, inputdirs, json_name):
    """
    Find envelope_fits.json, supporting:
      1. Combined file: seed_dir/envelope_fits.json (old format)
      2. Per-model files: seed_dir/<model>/<model>_envelope_fits.json (DGX nested format)
      3. Same searches in inputdirs as fallback
    Returns a dict (possibly reconstructed from per-model files).
    """
    import seed_utils

    # 1. Try combined file (flat or nested)
    for sd in seed_dirs:
        path = seed_utils.find_file_in_seed_dir(sd, json_name)
        if path is not None:
            return load_json(path)

    for d in inputdirs:
        path = os.path.join(d, json_name)
        if os.path.exists(path):
            return load_json(path)

    # 2. Try per-model files and reconstruct combined dict
    combined = {}
    for sd in seed_dirs:
        for model in CANON_MODELS:
            per_model_name = f"{model}_envelope_fits.json"
            path = seed_utils.find_file_in_seed_dir(sd, per_model_name, model)
            if path is not None:
                try:
                    combined[model] = load_json(path)
                except Exception:
                    pass
        if combined:
            # Use first seed that has any per-model files
            return combined

    if not combined:
        raise FileNotFoundError(
            f"Could not find {json_name} or per-model *_envelope_fits.json "
            f"in any seed dir or input dir"
        )
    return combined

def load_and_aggregate_summaries(seed_dirs, model, f_col, N_col):
    """
    Load <model>_summary.csv from each seed and aggregate on ell.
    Returns aggregated DataFrame with columns: ell, f_hat_mean, alpha_hat_mean, N_req_mean, etc.
    Also returns list of per-seed DataFrames for plotting individual points (if multi-seed).
    """
    dfs = seed_utils.load_model_summary_across_seeds(
        seed_dirs,
        model,
        required_cols={f_col, "alpha_hat", N_col, "ell"}
    )

    if not dfs:
        return None, []

    # Store per-seed data before aggregation (for individual point plotting)
    seed_dfs_subset = []
    for df in dfs:
        d = df[["ell", f_col, "alpha_hat", N_col]].copy()
        d.columns = ["ell", "f_hat", "alpha_hat", "N_req"]
        d["ell"] = pd.to_numeric(d["ell"], errors="coerce")
        d["f_hat"] = pd.to_numeric(d["f_hat"], errors="coerce")
        d["alpha_hat"] = pd.to_numeric(d["alpha_hat"], errors="coerce")
        d["N_req"] = pd.to_numeric(d["N_req"], errors="coerce")
        d = d.dropna()
        seed_dfs_subset.append(d)

    # Aggregate across seeds on 'ell'
    agg_data = seed_utils.aggregate_numeric_by_key(
        dfs,
        key_col="ell",
        value_cols=[f_col, "alpha_hat", N_col]
    )

    if agg_data.empty:
        return None, seed_dfs_subset

    # aggregate_numeric_by_key produces: ell, <col>_mean, <col>_std, <col>_count
    result = pd.DataFrame({
        "ell": agg_data["ell"],
        "f_hat_mean": agg_data[f"{f_col}_mean"],
        "f_hat_std": agg_data[f"{f_col}_std"],
        "alpha_hat_mean": agg_data["alpha_hat_mean"],
        "alpha_hat_std": agg_data["alpha_hat_std"],
        "N_req_mean": agg_data[f"{N_col}_mean"],
        "N_req_std": agg_data[f"{N_col}_std"],
    })

    return result, seed_dfs_subset

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Resolve input directories
    inputdirs = seed_utils.resolve_inputdirs(args)
    seed_dirs = seed_utils.discover_from_multiple_inputdirs(inputdirs)

    print(f"[info] discovered {len(seed_dirs)} seed/data dir(s)")
    for sd in seed_dirs:
        print(f"  - {os.path.abspath(sd)}")

    # Load envelope_fits.json
    fits_json = find_envelope_fits_json(seed_dirs, inputdirs, args.json_name)

    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models.strip()
        else CANON_MODELS[:]
    )

    is_multiseed = len(seed_dirs) > 1

    summary_rows = []

    for m in models:
        win = get_power_window(fits_json, m)
        if win is None:
            if args.verbose:
                print(f"[warn] {m}: no power window in {args.json_name}; skipping")
            continue
        ell_min, ell_max, beta, beta_r2 = win

        # Load and aggregate summaries across seeds
        agg_data, seed_dfs = load_and_aggregate_summaries(
            seed_dirs, m, args.f_col, args.N_col
        )

        if agg_data is None or agg_data.empty:
            if args.verbose:
                print(f"[warn] {m}: could not load/aggregate summary CSVs; skipping")
            continue

        if args.verbose:
            print(f"\n[model {m}]  power-window=[{ell_min},{ell_max}]  beta={beta:.3g} r2={beta_r2:.3g}")

        d = agg_data[["ell", "f_hat_mean", "alpha_hat_mean", "N_req_mean"]].copy()
        d.columns = ["ell", "f_hat", "alpha_hat", "N_req"]

        d["ell"] = pd.to_numeric(d["ell"], errors="coerce")
        d["f_hat"] = pd.to_numeric(d["f_hat"], errors="coerce")
        d["alpha_hat"] = pd.to_numeric(d["alpha_hat"], errors="coerce")
        d["N_req"] = pd.to_numeric(d["N_req"], errors="coerce")

        # Window + validity filters
        d = d.dropna()
        d = d[(d["ell"] >= ell_min) & (d["ell"] <= ell_max)]
        d = d[(d["f_hat"] > 0) & (d["alpha_hat"] > args.alpha_floor) & (d["N_req"] > 0)]

        if len(d) < args.min_points:
            if args.verbose:
                print(f"[warn] {m}: only {len(d)} points after filtering; skipping fit")
            continue

        n_unique = d["N_req"].nunique()
        if n_unique < args.min_unique_N:
            if args.verbose:
                print(
                    f"[warn] {m}: N_req has only {n_unique} unique values in window (degenerate). Skipping fit."
                )
            continue

        d["kappa_alpha"] = d["alpha_hat"] / (d["alpha_hat"] - 1.0)
        d["x"] = -d["kappa_alpha"] * np.log(d["f_hat"].values)
        d["y"] = np.log(d["N_req"].values)

        a, b, r2, yhat = linfit(d["x"].values, d["y"].values)
        d["y_hat"] = yhat

        pts_path = os.path.join(args.outdir, f"fit_master_points_{m}.csv")
        d.to_csv(pts_path, index=False)

        # Plot
        plt.figure(figsize=(7, 5))

        # Plot seed-averaged data
        plt.scatter(d["x"].values, d["y"].values, s=80, alpha=0.7, zorder=3)

        # If multi-seed, also plot individual seed points (lighter)
        if is_multiseed and seed_dfs:
            for seed_df in seed_dfs:
                # Apply same filters as aggregated data
                seed_df = seed_df.dropna()
                seed_df = seed_df[
                    (seed_df["ell"] >= ell_min) & (seed_df["ell"] <= ell_max)
                ]
                seed_df = seed_df[
                    (seed_df["f_hat"] > 0)
                    & (seed_df["alpha_hat"] > args.alpha_floor)
                    & (seed_df["N_req"] > 0)
                ]

                if len(seed_df) > 0:
                    seed_df["kappa_alpha"] = seed_df["alpha_hat"] / (
                        seed_df["alpha_hat"] - 1.0
                    )
                    seed_df["x"] = -seed_df["kappa_alpha"] * np.log(
                        seed_df["f_hat"].values
                    )
                    seed_df["y"] = np.log(seed_df["N_req"].values)
                    plt.scatter(
                        seed_df["x"].values,
                        seed_df["y"].values,
                        s=20,
                        alpha=0.15,
                        color="C0",
                        zorder=1
                    )

        # Fit line
        xline = np.linspace(d["x"].min(), d["x"].max(), 200)
        plt.plot(xline, a + b * xline, "r-", linewidth=2, zorder=2)

        plt.xlabel(r"$-\kappa_{\alpha}(\ell)\,\log \hat f(\ell)$", fontsize=11)
        plt.ylabel(r"$\log \hat N(\ell)$", fontsize=11)
        plt.title(
            f"{m}: slope={b:.3f}, intercept={a:.3f}, $R^2$={r2:.3f}, n={len(d)}",
            fontsize=11
        )
        plt.tight_layout()

        fig_path = os.path.join(args.outdir, f"fit_master_{m}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

        summary_rows.append({
            "model": m,
            "ell_min_power": ell_min,
            "ell_max_power": ell_max,
            "beta_power": beta,
            "beta_r2": beta_r2,
            "n_points_fit": int(len(d)),
            "unique_N": int(n_unique),
            "slope_b": float(b),
            "intercept_a": float(a),
            "r2": float(r2),
            "f_col_used": args.f_col,
            "N_col_used": args.N_col,
            "n_seeds": len(seed_dirs),
        })

        if args.verbose:
            print(
                f"[done] {m}: slope={b:.3f} R2={r2:.3f} unique_N={n_unique}"
            )
            print(
                f"       wrote {os.path.basename(fig_path)} and {os.path.basename(pts_path)}"
            )

    if summary_rows:
        summ = pd.DataFrame(summary_rows)
        summ_path = os.path.join(args.outdir, "fit_master_summary.csv")
        summ.to_csv(summ_path, index=False)
        if args.verbose:
            print(f"\n[info] wrote summary: {summ_path}")
    else:
        if args.verbose:
            print(
                "\n[warn] no non-degenerate fits produced. "
                "(Often means N_required_at_eps saturates or is mostly invalid in that lag window.)"
            )

if __name__ == "__main__":
    main()
