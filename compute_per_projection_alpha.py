#!/usr/bin/env python3
"""
Per-projection ECF tail-index diagnostic for the matched statistic.

For each (seed, architecture) we load the saved psi_per_proj.npz array of
shape (n_lags, n_seq, K) and, for each (lag, projection direction k), fit the
ECF tail-index estimator on the n_seq sequence-level scalars from that
projection alone. This produces K=50 alpha estimates per (seed, lag), which we
then summarize by:
    median, IQR, lower decile, frac(alpha < 1.9), frac(alpha < 1.85)
The cross-(seed, lag) median of each summary becomes the per-architecture row
of the table reported in app:projection_alpha_diagnostic.

Important: these per-projection alphas are NOT pooled with the
projection-averaged statistic that defines the alpha entering the empirical
learnability window. They are a directional diagnostic.

USAGE
-----
Standalone:
    python compute_per_projection_alpha.py \
        --inputdirs results/fullsim/adamw/baselines \
                    results/fullsim/adamw/lstmgru \
        --outdir    results/fullsim/adamw/together

The script writes outputs to <outdir>/alpha_estimation/:
    agg_per_projection_alpha_summary.json : per-architecture aggregate dict
    agg_per_projection_alpha_raw.npz      : raw alphas, one (n_lags, K)
                                            array per (model, seed)
    agg_per_projection_alpha_table.tex    : LaTeX snippet ready to drop into
                                            app:projection_alpha_diagnostic

Pipeline integration:
    The plot_all_multiseed.py launcher invokes this script automatically with
    the same --inputdirs/--outdir/--view interface as the other plot scripts.

Memory note:
    Each worker may hold a (192, 12000, 50) float64 array (~0.9 GB). With 5
    workers (default) this needs ~4.5 GB of free RAM at peak; reduce to
    --workers 2 or 3 if memory is tight.
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np

# Local import (this file lives next to alpha_estimators.py in the repo root)
from alpha_estimators import estimate_alpha_ecf

BASELINES = ("const", "shared", "diag")
LSTMGRU = ("gru", "lstm")
ALL_MODELS = BASELINES + LSTMGRU

# Stable model index for RNG seeding (kept in sync with compute_ecf_bootstrap_ci)
MODEL_INDEX = {m: i for i, m in enumerate(ALL_MODELS)}


def discover_jobs(inputdirs):
    """For each inputdir, discover (model, seed, model_dir) triples by
    examining the seed_*/ subfolders and the architecture folders inside."""
    jobs = []
    for d in inputdirs:
        if not os.path.isdir(d):
            print(f"[warn] inputdir does not exist, skipping: {d}", file=sys.stderr)
            continue
        for entry in sorted(os.listdir(d)):
            if not entry.startswith("seed_"):
                continue
            try:
                seed = int(entry.split("_", 1)[1])
            except ValueError:
                continue
            seed_dir = os.path.join(d, entry)
            for model in ALL_MODELS:
                model_dir = os.path.join(seed_dir, model)
                psi_file = os.path.join(model_dir, f"{model}_psi_per_proj.npz")
                if os.path.exists(psi_file):
                    jobs.append((model, seed, psi_file))
    return jobs


def per_seed_arch_alphas(arg):
    """Compute per-(lag, projection) ECF alphas for one (seed, arch).
    Returns (model, seed, alphas[n_lags, K], ell_array[n_lags])."""
    model, seed, psi_file = arg
    arr = np.load(psi_file)
    psi = arr["psi_per_proj"]  # (n_lags, n_seq, K)
    ell = np.asarray(arr["ell"]).astype(np.int64)
    n_lags, n_seq, K = psi.shape
    out = np.full((n_lags, K), np.nan, dtype=np.float64)
    for i in range(n_lags):
        for k in range(K):
            out[i, k] = estimate_alpha_ecf(psi[i, :, k])
    return (model, seed, out, ell)


def summarize_per_seed_lag(alphas_K: np.ndarray):
    a = alphas_K[np.isfinite(alphas_K)]
    if a.size == 0:
        return None
    q = np.quantile(a, [0.10, 0.25, 0.50, 0.75])
    return {
        "median": float(q[2]),
        "iqr": float(q[3] - q[1]),
        "lower_decile": float(q[0]),
        "frac_below_19": float((a < 1.9).mean()),
        "frac_below_185": float((a < 1.85).mean()),
        "n_finite": int(a.size),
        "K": int(alphas_K.size),
    }


def aggregate(by_arch_seed, models):
    out = {}
    for m in models:
        per_sl_summaries = []
        for seed, alphas in by_arch_seed[m].items():
            for i in range(alphas.shape[0]):
                summ = summarize_per_seed_lag(alphas[i])
                if summ is not None:
                    per_sl_summaries.append(summ)
        if not per_sl_summaries:
            continue
        keys = ["median", "iqr", "lower_decile", "frac_below_19", "frac_below_185"]
        agg = {}
        for k in keys:
            vals = np.array([d[k] for d in per_sl_summaries])
            agg[k + "_cross_sl_median"] = float(np.median(vals))
            agg[k + "_cross_sl_p25"] = float(np.quantile(vals, 0.25))
            agg[k + "_cross_sl_p75"] = float(np.quantile(vals, 0.75))
        agg["n_seed_lag_pairs"] = len(per_sl_summaries)
        agg["n_seeds"] = len(by_arch_seed[m])
        out[m] = agg
    return out


def write_latex_table(arch_summaries, out_path):
    rows = []
    for m in ALL_MODELS:
        if m not in arch_summaries:
            rows.append(f"{m:6s} & --- & --- & --- & --- & --- \\\\")
            continue
        a = arch_summaries[m]
        rows.append(
            f"{m:6s} & "
            f"${a['median_cross_sl_median']:.3f}$ & "
            f"${a['iqr_cross_sl_median']:.3f}$ & "
            f"${a['lower_decile_cross_sl_median']:.3f}$ & "
            f"${a['frac_below_19_cross_sl_median']:.3f}$ & "
            f"${a['frac_below_185_cross_sl_median']:.3f}$ \\\\"
        )
    content = (
        "% Produced by compute_per_projection_alpha.py\n"
        "% Insert the rows between % BEGIN and % END into the table body\n"
        "% in app:projection_alpha_diagnostic (Table tab:projection_alpha).\n"
        "% BEGIN-PROJECTION-ALPHA-ROWS\n"
        + "\n".join(rows) + "\n"
        "% END-PROJECTION-ALPHA-ROWS\n"
    )
    with open(out_path, "w") as f:
        f.write(content)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputdirs", nargs="+", required=True,
                   help="parent dirs containing seed_*/<arch>/ "
                        "(typically baselines/ and lstmgru/ for one optimizer)")
    p.add_argument("--outdir", required=True,
                   help="aggregate output directory (e.g. results/.../together); "
                        "this script writes to <outdir>/alpha_estimation/")
    p.add_argument("--view", default="both",
                   help="present for compatibility with plot_all_multiseed; "
                        "this script always produces aggregate outputs only")
    p.add_argument("--workers", type=int, default=5,
                   help="number of parallel processes (default 5; reduce if "
                        "memory is tight, see docstring)")
    args = p.parse_args()

    out_subdir = os.path.join(args.outdir, "alpha_estimation")
    os.makedirs(out_subdir, exist_ok=True)

    jobs = discover_jobs(args.inputdirs)
    if not jobs:
        print(f"[error] No psi_per_proj.npz files found under {args.inputdirs}",
              file=sys.stderr)
        sys.exit(1)
    seeds_seen = sorted({s for _, s, _ in jobs})
    models_seen = sorted({m for m, _, _ in jobs}, key=lambda x: MODEL_INDEX[x])
    print(f"[info] Inputdirs: {args.inputdirs}")
    print(f"[info] Outdir: {out_subdir}")
    print(f"[info] Discovered {len(jobs)} (model, seed) pairs across "
          f"models={models_seen} seeds={seeds_seen}")
    print(f"[info] Workers: {args.workers}")

    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        results = pool.map(per_seed_arch_alphas, jobs)
    print(f"[info] All ECF fits done in {time.time()-t0:.1f}s")

    by_arch_seed = {m: {} for m in ALL_MODELS}
    ell_grid = None
    for model, seed, alphas, ell in results:
        by_arch_seed[model][seed] = alphas
        if ell_grid is None:
            ell_grid = ell

    arch_summaries = aggregate(by_arch_seed, ALL_MODELS)

    json_path = os.path.join(out_subdir, "agg_per_projection_alpha_summary.json")
    with open(json_path, "w") as f:
        json.dump({
            "inputdirs": args.inputdirs,
            "seeds": seeds_seen,
            "models": [m for m in ALL_MODELS if by_arch_seed[m]],
            "ell_grid": [int(x) for x in ell_grid] if ell_grid is not None else None,
            "arch_summaries": arch_summaries,
        }, f, indent=2)

    npz_path = os.path.join(out_subdir, "agg_per_projection_alpha_raw.npz")
    raw_dict = {f"{m}_seed{s}": by_arch_seed[m][s]
                for m in ALL_MODELS for s in by_arch_seed[m]}
    if raw_dict:
        raw_dict["ell_grid"] = ell_grid
        np.savez_compressed(npz_path, **raw_dict)

    tex_path = os.path.join(out_subdir, "agg_per_projection_alpha_table.tex")
    write_latex_table(arch_summaries, tex_path)

    print()
    print("=== Cross-(seed, lag) median of per-(seed, lag) summaries ===")
    print(f"{'model':6s} | {'med a':>7s} {'IQR':>6s} {'p10':>7s} "
          f"{'P(a<1.9)':>9s} {'P(a<1.85)':>10s}")
    for m in ALL_MODELS:
        if m not in arch_summaries:
            continue
        a = arch_summaries[m]
        print(
            f"{m:6s} | {a['median_cross_sl_median']:7.3f} "
            f"{a['iqr_cross_sl_median']:6.3f} "
            f"{a['lower_decile_cross_sl_median']:7.3f} "
            f"{a['frac_below_19_cross_sl_median']:9.3f} "
            f"{a['frac_below_185_cross_sl_median']:10.3f}"
        )
    print()
    print(f"[wrote] {json_path}")
    print(f"[wrote] {npz_path}")
    print(f"[wrote] {tex_path}")


if __name__ == "__main__":
    main()
