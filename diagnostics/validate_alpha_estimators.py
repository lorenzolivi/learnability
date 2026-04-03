#!/usr/bin/env python3
"""
Validation of ECF and McCulloch tail-index estimators on synthetic SαS data.

Generates large iid samples from SαS(α, σ=1) for α ∈ {1.1, 1.2, ..., 2.0}
using the Chambers–Mallows–Stuck (CMS) algorithm, then estimates α̂ with both
the ECF regression and McCulloch quantile-ratio methods from the pipeline.

Produces a Markdown report with:
  - Point estimates and absolute errors for each (α, method) pair
  - Mean absolute error (MAE) and max absolute error across α values
  - Bootstrap 95% CI for the McCulloch estimator
  - Per-α reliability flags from the pipeline

Usage:
    python diagnostics/validate_alpha_estimators.py [--n_samples 200000] [--n_reps 10]
"""

import sys, os, argparse, json, time
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Import estimators from standalone extraction (avoids torch dependency)
# ---------------------------------------------------------------------------
DIAG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIAG_DIR))

from alpha_estimators_standalone import (
    estimate_alpha_sigma_with_meta,
    estimate_alpha_sigma_mcculloch_symmetric_from_quantiles,
    bootstrap_mcculloch,
)


# ---------------------------------------------------------------------------
# Chambers–Mallows–Stuck sampler for symmetric α-stable (β = 0)
# ---------------------------------------------------------------------------

def sas_symmetric_rvs(
    alpha: float,
    scale: float = 1.0,
    size: int = 1,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """
    Draw iid samples from SαS(0, scale) using the CMS algorithm.

    For α = 2 this reduces to N(0, 2·scale²), matching the SαS(2, σ)
    parameterisation where σ is the scale parameter of the characteristic
    function  φ(t) = exp(−σ^α |t|^α).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    if abs(alpha - 2.0) < 1e-10:
        # SαS with α=2 is Gaussian with variance 2σ²
        return scale * np.sqrt(2.0) * rng.standard_normal(size=size)

    U = rng.uniform(-np.pi / 2, np.pi / 2, size=size)
    W = rng.exponential(1.0, size=size)

    num = np.sin(alpha * U)
    den = np.cos(U) ** (1.0 / alpha)
    fac = (np.cos((1.0 - alpha) * U) / W) ** ((1.0 - alpha) / alpha)

    X = (num / den) * fac
    return scale * X


# ---------------------------------------------------------------------------
# Main validation routine
# ---------------------------------------------------------------------------

def run_validation(
    n_samples: int = 200_000,
    n_reps: int = 10,
    n_boot: int = 500,
    seed_base: int = 2026,
) -> dict:
    """
    For each true α, draw n_reps independent datasets of size n_samples,
    estimate α̂ with both methods, and collect statistics.
    """
    alphas_true = np.round(np.arange(1.1, 2.01, 0.1), 1)
    scale_true = 1.0

    results = []

    for alpha in alphas_true:
        ecf_estimates = []
        mcc_estimates = []
        ecf_sigma_estimates = []
        mcc_sigma_estimates = []
        ecf_reliable_flags = []
        mcc_reliable_flags = []
        mcc_ci_records = []

        for rep in range(n_reps):
            rng = np.random.RandomState(seed_base + rep)
            samples = sas_symmetric_rvs(
                alpha, scale=scale_true, size=n_samples, rng=rng
            )

            # --- ECF ---
            meta_ecf = estimate_alpha_sigma_with_meta(
                samples, method="ecf", n_samples_for_ecf=100_000
            )
            ecf_estimates.append(float(meta_ecf["alpha_hat"]))
            ecf_sigma_estimates.append(float(meta_ecf["sigma_hat"]))
            ecf_reliable_flags.append(bool(meta_ecf["reliable"]))

            # --- McCulloch ---
            meta_mcc = estimate_alpha_sigma_with_meta(
                samples, method="mcculloch"
            )
            mcc_estimates.append(float(meta_mcc["alpha_hat"]))
            mcc_sigma_estimates.append(float(meta_mcc["sigma_hat"]))
            mcc_reliable_flags.append(bool(meta_mcc["reliable"]))

            # --- Bootstrap CI for McCulloch (on first rep only) ---
            if rep == 0:
                med, ci_lo, ci_hi, sig_med = bootstrap_mcculloch(
                    samples,
                    estimator_fn=estimate_alpha_sigma_mcculloch_symmetric_from_quantiles,
                    n_boot=n_boot,
                    ci=0.95,
                )
                mcc_ci_records.append({
                    "median": med, "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "covers_true": ci_lo <= alpha <= ci_hi,
                })

        ecf_arr = np.array(ecf_estimates)
        mcc_arr = np.array(mcc_estimates)
        ecf_sig = np.array(ecf_sigma_estimates)
        mcc_sig = np.array(mcc_sigma_estimates)

        results.append({
            "alpha_true": float(alpha),
            "scale_true": float(scale_true),
            "n_samples": n_samples,
            "n_reps": n_reps,
            # ECF
            "ecf_mean": float(np.mean(ecf_arr)),
            "ecf_std": float(np.std(ecf_arr)),
            "ecf_bias": float(np.mean(ecf_arr) - alpha),
            "ecf_mae": float(np.mean(np.abs(ecf_arr - alpha))),
            "ecf_max_err": float(np.max(np.abs(ecf_arr - alpha))),
            "ecf_sigma_mean": float(np.mean(ecf_sig)),
            "ecf_sigma_bias": float(np.mean(ecf_sig) - scale_true),
            "ecf_reliable_frac": float(np.mean(ecf_reliable_flags)),
            # McCulloch
            "mcc_mean": float(np.mean(mcc_arr)),
            "mcc_std": float(np.std(mcc_arr)),
            "mcc_bias": float(np.mean(mcc_arr) - alpha),
            "mcc_mae": float(np.mean(np.abs(mcc_arr - alpha))),
            "mcc_max_err": float(np.max(np.abs(mcc_arr - alpha))),
            "mcc_sigma_mean": float(np.mean(mcc_sig)),
            "mcc_sigma_bias": float(np.mean(mcc_sig) - scale_true),
            "mcc_reliable_frac": float(np.mean(mcc_reliable_flags)),
            # McCulloch bootstrap CI (first rep)
            "mcc_boot": mcc_ci_records[0] if mcc_ci_records else None,
        })

    return {
        "config": {
            "n_samples": n_samples,
            "n_reps": n_reps,
            "n_boot": n_boot,
            "seed_base": seed_base,
            "scale_true": scale_true,
            "alphas_true": [float(a) for a in alphas_true],
        },
        "results": results,
    }


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def generate_report(data: dict, output_path: Path) -> None:
    cfg = data["config"]
    results = data["results"]

    lines = [
        "# Validation of ECF and McCulloch tail-index estimators",
        "",
        "Synthetic SαS data generated via the Chambers–Mallows–Stuck algorithm.",
        "",
        "## Configuration",
        "",
        f"- Samples per dataset: **{cfg['n_samples']:,}**",
        f"- Independent repetitions: **{cfg['n_reps']}**",
        f"- Bootstrap resamples (McCulloch CI): **{cfg['n_boot']}**",
        f"- True scale σ: **{cfg['scale_true']}**",
        f"- Seed base: {cfg['seed_base']}",
        "",
        "## Results: α estimation",
        "",
        "| α_true | ECF α̂ (mean±std) | ECF bias | ECF MAE | MCC α̂ (mean±std) | MCC bias | MCC MAE | MCC 95% CI | Covers? |",
        "|-------:|------------------:|---------:|--------:|------------------:|---------:|--------:|:----------:|:-------:|",
    ]

    for r in results:
        boot = r["mcc_boot"]
        ci_str = f"[{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}]" if boot else "—"
        covers = "yes" if (boot and boot["covers_true"]) else "no"

        lines.append(
            f"| {r['alpha_true']:.1f} "
            f"| {r['ecf_mean']:.4f}±{r['ecf_std']:.4f} "
            f"| {r['ecf_bias']:+.4f} "
            f"| {r['ecf_mae']:.4f} "
            f"| {r['mcc_mean']:.4f}±{r['mcc_std']:.4f} "
            f"| {r['mcc_bias']:+.4f} "
            f"| {r['mcc_mae']:.4f} "
            f"| {ci_str} "
            f"| {covers} |"
        )

    # Summary statistics
    ecf_biases = [r["ecf_bias"] for r in results]
    mcc_biases = [r["mcc_bias"] for r in results]
    ecf_maes = [r["ecf_mae"] for r in results]
    mcc_maes = [r["mcc_mae"] for r in results]

    lines += [
        "",
        "## Results: σ estimation",
        "",
        "| α_true | ECF σ̂ (mean) | ECF σ bias | MCC σ̂ (mean) | MCC σ bias |",
        "|-------:|-------------:|-----------:|-------------:|-----------:|",
    ]
    for r in results:
        lines.append(
            f"| {r['alpha_true']:.1f} "
            f"| {r['ecf_sigma_mean']:.4f} "
            f"| {r['ecf_sigma_bias']:+.4f} "
            f"| {r['mcc_sigma_mean']:.4f} "
            f"| {r['mcc_sigma_bias']:+.4f} |"
        )

    lines += [
        "",
        "## Reliability flags",
        "",
        "| α_true | ECF reliable (frac) | MCC reliable (frac) |",
        "|-------:|--------------------:|--------------------:|",
    ]
    for r in results:
        lines.append(
            f"| {r['alpha_true']:.1f} "
            f"| {r['ecf_reliable_frac']:.2f} "
            f"| {r['mcc_reliable_frac']:.2f} |"
        )

    n_covers = sum(1 for r in results if r["mcc_boot"] and r["mcc_boot"]["covers_true"])
    n_total = sum(1 for r in results if r["mcc_boot"] is not None)

    lines += [
        "",
        "## Summary",
        "",
        f"- **ECF** — mean absolute bias: {np.mean(np.abs(ecf_biases)):.4f}, "
        f"mean MAE: {np.mean(ecf_maes):.4f}, "
        f"max MAE: {np.max(ecf_maes):.4f}",
        f"- **McCulloch** — mean absolute bias: {np.mean(np.abs(mcc_biases)):.4f}, "
        f"mean MAE: {np.mean(mcc_maes):.4f}, "
        f"max MAE: {np.max(mcc_maes):.4f}",
        f"- **McCulloch 95% CI coverage**: {n_covers}/{n_total} "
        f"({100*n_covers/n_total:.0f}%)" if n_total > 0 else "",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate ECF and McCulloch alpha estimators on synthetic SαS data."
    )
    parser.add_argument("--n_samples", type=int, default=200_000,
                        help="Number of iid samples per dataset (default: 200000)")
    parser.add_argument("--n_reps", type=int, default=10,
                        help="Independent repetitions per α value (default: 10)")
    parser.add_argument("--n_boot", type=int, default=500,
                        help="Bootstrap resamples for McCulloch CI (default: 500)")
    parser.add_argument("--seed_base", type=int, default=2026,
                        help="Base random seed (default: 2026)")
    args = parser.parse_args()

    print(f"Running alpha-estimator validation: "
          f"n_samples={args.n_samples}, n_reps={args.n_reps}, n_boot={args.n_boot}")

    t0 = time.time()
    data = run_validation(
        n_samples=args.n_samples,
        n_reps=args.n_reps,
        n_boot=args.n_boot,
        seed_base=args.seed_base,
    )
    elapsed = time.time() - t0
    print(f"Validation completed in {elapsed:.1f}s")

    out_dir = Path(__file__).resolve().parent / "alpha_estimator_validation"
    out_dir.mkdir(exist_ok=True)

    # Save raw JSON (convert numpy types)
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    json_path = out_dir / "validation_results.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
    print(f"Raw results saved to {json_path}")

    # Generate Markdown report
    md_path = out_dir / "validation_report.md"
    generate_report(data, md_path)


if __name__ == "__main__":
    main()
