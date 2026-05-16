#!/usr/bin/env python3
"""
ECF bootstrap confidence intervals for the projection-averaged matched
statistic's tail index.

For each (seed, lag) we average the saved psi_per_proj.npz array across the
K projection directions to obtain n_seq sequence-level scalars (the same scalar
that enters the empirical learnability window). We then bootstrap-resample
those n_seq values B times, refit ECF on each resample, and report the
[2.5%, 97.5%] quantile range as the 95% CI on alpha_hat.

This is the missing-uncertainty piece for the ECF column referenced in the
noise-statistics paragraph of the main text. McCulloch already has its own
bootstrap CIs from the training pipeline; this script closes the symmetry.

USAGE
-----
Standalone:
    python compute_ecf_bootstrap_ci.py \
        --inputdirs results/fullsim/adamw/baselines \
                    results/fullsim/adamw/lstmgru \
        --outdir    results/fullsim/adamw/together \
        --n_boot 200

The script writes outputs to <outdir>/alpha_estimation/:
    agg_ecf_bootstrap_ci_per_seed_lag.json : per-(model, seed, lag) point
                                              estimate + 95% CI (low, high),
                                              with the lag grid included as
                                              top-level metadata
    agg_ecf_bootstrap_ci_per_arch.json     : per-architecture summary across
                                              seeds, including the CI for the
                                              minimum-alpha (seed, lag) per
                                              architecture

Pipeline integration:
    The plot_all_multiseed.py launcher invokes this script automatically with
    the same --inputdirs/--outdir/--view interface as the other plot scripts.

Cost note:
    B=200 bootstraps x 5 seeds x 5 archs x 192 lags = 960k ECF fits; with
    workers=5 and ~9 ms per fit, total wall time on a 5-core Mac is roughly
    25-30 minutes. Pass --n_boot 100 for a faster ~12-min pass.

Memory note:
    Same as compute_per_projection_alpha.py: ~0.9 GB per worker for a (192,
    12000, 50) array. Reduce --workers if memory is tight.

RNG seeding:
    The bootstrap RNG is seeded per (model, seed) pair so different
    architectures with the same training seed do not share the same bootstrap
    index stream. This makes the bootstrap reproducible and not artificially
    correlated across architectures.
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np

from alpha_estimators import estimate_alpha_ecf

BASELINES = ("const", "shared", "diag")
LSTMGRU = ("gru", "lstm")
ALL_MODELS = BASELINES + LSTMGRU

MODEL_INDEX = {m: i for i, m in enumerate(ALL_MODELS)}


def discover_jobs(inputdirs):
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


def per_seed_arch_bootstrap(arg):
    """Per-lag bootstrap CI for one (seed, arch). Returns
    (model, seed, alpha_point[n_lags], ci_lo[n_lags], ci_hi[n_lags], ell_grid)."""
    model, seed, psi_file, n_boot, conf = arg
    arr = np.load(psi_file)
    psi = arr["psi_per_proj"]  # (n_lags, n_seq, K)
    ell_grid = np.asarray(arr["ell"]).astype(np.int64)
    n_lags, n_seq, K = psi.shape

    seq_scalars = psi.mean(axis=2)  # (n_lags, n_seq)

    alpha_point = np.full(n_lags, np.nan, dtype=np.float64)
    ci_lo = np.full(n_lags, np.nan, dtype=np.float64)
    ci_hi = np.full(n_lags, np.nan, dtype=np.float64)

    # Seed RNG per (model, seed) so different architectures with the same
    # training seed get independent bootstrap streams.
    rng_seed = (seed * 100 + MODEL_INDEX[model]) & 0xFFFFFFFF
    rng = np.random.default_rng(seed=rng_seed)

    p_lo = (1.0 - conf) / 2.0
    p_hi = 1.0 - p_lo

    for i in range(n_lags):
        x = seq_scalars[i]
        x = x[np.isfinite(x)]
        if x.size < 500:
            continue
        alpha_point[i] = estimate_alpha_ecf(x)
        boot = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            idx = rng.integers(0, x.size, size=x.size)
            boot[b] = estimate_alpha_ecf(x[idx])
        valid = boot[np.isfinite(boot)]
        if valid.size >= max(20, n_boot // 4):
            ci_lo[i] = float(np.quantile(valid, p_lo))
            ci_hi[i] = float(np.quantile(valid, p_hi))

    return (model, seed, alpha_point, ci_lo, ci_hi, ell_grid)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputdirs", nargs="+", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--view", default="both",
                   help="present for compatibility with plot_all_multiseed")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--n_boot", type=int, default=200)
    p.add_argument("--conf_level", type=float, default=0.95)
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
    print(f"[info] Workers={args.workers}, n_boot={args.n_boot}, "
          f"conf_level={args.conf_level}")

    args_list = [(m, s, f, args.n_boot, args.conf_level) for m, s, f in jobs]
    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        results = pool.map(per_seed_arch_bootstrap, args_list)
    print(f"[info] All bootstrap CIs done in {time.time()-t0:.1f}s")

    per_record = []
    ell_grid = None
    arch_records = {m: [] for m in ALL_MODELS}

    for model, seed, alpha_pt, ci_lo, ci_hi, ell in results:
        if ell_grid is None and ell is not None:
            ell_grid = ell
        per_record.append({
            "model": model, "seed": seed,
            "alpha_point": [None if not np.isfinite(x) else float(x) for x in alpha_pt],
            "ci_lo": [None if not np.isfinite(x) else float(x) for x in ci_lo],
            "ci_hi": [None if not np.isfinite(x) else float(x) for x in ci_hi],
        })
        valid_mask = np.isfinite(alpha_pt) & np.isfinite(ci_lo) & np.isfinite(ci_hi)
        if valid_mask.any():
            valid_idx = np.flatnonzero(valid_mask)
            for j in valid_idx:
                arch_records[model].append({
                    "seed": int(seed),
                    "ell": int(ell_grid[j]) if ell_grid is not None else int(j),
                    "alpha": float(alpha_pt[j]),
                    "ci_lo": float(ci_lo[j]),
                    "ci_hi": float(ci_hi[j]),
                })

    # Per-architecture summary, including the CI for the minimum-alpha entry.
    arch_summary = {}
    for m in ALL_MODELS:
        recs = arch_records[m]
        if not recs:
            continue
        alphas = np.array([r["alpha"] for r in recs])
        widths = np.array([r["ci_hi"] - r["ci_lo"] for r in recs])
        i_min = int(np.argmin(alphas))
        min_rec = recs[i_min]
        arch_summary[m] = {
            "alpha_point_median_across_seed_lag": float(np.median(alphas)),
            "alpha_point_min_across_seed_lag": float(alphas[i_min]),
            "min_record": {
                "seed": min_rec["seed"],
                "ell": min_rec["ell"],
                "alpha_point": min_rec["alpha"],
                "ci_lo": min_rec["ci_lo"],
                "ci_hi": min_rec["ci_hi"],
                "ci_width": float(min_rec["ci_hi"] - min_rec["ci_lo"]),
            },
            "median_ci_width_across_seed_lag": float(np.median(widths)),
            "p25_ci_width_across_seed_lag": float(np.quantile(widths, 0.25)),
            "p75_ci_width_across_seed_lag": float(np.quantile(widths, 0.75)),
            "n_seed_lag_pairs": len(recs),
            "n_seeds": int(len({r["seed"] for r in recs})),
        }

    json_per = os.path.join(out_subdir, "agg_ecf_bootstrap_ci_per_seed_lag.json")
    with open(json_per, "w") as f:
        json.dump({
            "inputdirs": args.inputdirs,
            "seeds": seeds_seen,
            "models": [m for m in ALL_MODELS if arch_records[m]],
            "ell_grid": [int(x) for x in ell_grid] if ell_grid is not None else None,
            "n_boot": args.n_boot,
            "conf_level": args.conf_level,
            "records": per_record,
        }, f, indent=2)

    json_arch = os.path.join(out_subdir, "agg_ecf_bootstrap_ci_per_arch.json")
    with open(json_arch, "w") as f:
        json.dump({
            "inputdirs": args.inputdirs,
            "seeds": seeds_seen,
            "n_boot": args.n_boot,
            "conf_level": args.conf_level,
            "arch_summary": arch_summary,
        }, f, indent=2)

    print()
    print("=== Per-architecture summary ===")
    print(f"{'model':6s} | {'med α':>7s} {'min α':>7s}  "
          f"{'min CI':>23s}  {'med CI w':>9s}")
    for m in ALL_MODELS:
        if m not in arch_summary:
            continue
        a = arch_summary[m]
        mr = a["min_record"]
        ci_str = f"[{mr['ci_lo']:.3f}, {mr['ci_hi']:.3f}]"
        print(f"{m:6s} | "
              f"{a['alpha_point_median_across_seed_lag']:7.3f} "
              f"{a['alpha_point_min_across_seed_lag']:7.3f}  "
              f"{ci_str:>23s}  {a['median_ci_width_across_seed_lag']:9.4f}")
    print()
    print(f"[wrote] {json_per}")
    print(f"[wrote] {json_arch}")


if __name__ == "__main__":
    main()
