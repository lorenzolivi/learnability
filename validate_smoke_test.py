#!/usr/bin/env python3
"""
Post-smoke-test validator: checks that all expected outputs exist,
have the correct schema, and contain sensible values.

Usage:
    python validate_smoke_test.py
    python validate_smoke_test.py --root results/GELR_smoke_test
"""

import argparse
import csv
import json
import os
import sys
import numpy as np

# ── Expected outputs per model ──────────────────────────────────

# Files every model should produce
REQUIRED_FILES = [
    "{model}_summary.csv",
    "{model}_mu_units.npz",
    "{model}_mu_units_gates.npz",
    "{model}_adaptive_base_rates.csv",
    "{model}_tau_from_mu_units.csv",
    "{model}_tau_from_mu_stats.json",
]

# Files produced when tau fits succeed + optimizer has second moments
CONDITIONAL_FILES = [
    "{model}_lambda_tau_correlation.csv",
    "{model}_lambda_tau_stats.json",
]

# Expected columns in summary CSV (after the f_adapt → f_ratio change)
SUMMARY_EXPECTED_COLS = {
    "ell", "mu_l1_mean", "log_mu_l1_mean",
    "f_gates", "f_ratio",
    "lambda_mean", "lambda_std",
    "alpha_ecf", "sigma_ecf", "alpha_ecf_reliable",
    "alpha_mcc", "sigma_mcc", "alpha_mcc_reliable",
    "alpha_mcc_ci_lo", "alpha_mcc_ci_hi",
    "alpha_methods_agree",
    "alpha_hat", "sigma_hat", "alpha_reliable", "alpha_method_used",
    "N_required_ecf", "best_snr_ecf", "err_at_best_snr_ecf", "best_N_ecf",
    "N_required_mcc", "best_snr_mcc", "err_at_best_snr_mcc", "best_N_mcc",
    "mbar_scalar", "n_samples", "n_sequences",
}

# Old column that should NOT be present
BANNED_COLS = {"f_adapt"}

LAMBDA_TAU_EXPECTED_COLS = {"neuron_q", "Lambda_q", "tau_q", "mu_at_max_ell"}

LAMBDA_TAU_STATS_KEYS = {
    "model", "n_neurons", "spearman_rho_lambda_tau", "pearson_r_lambda_tau",
    "lambda_mean", "lambda_std", "tau_mean", "tau_std", "interpretation",
}


def check_csv_schema(path, expected_cols, banned_cols=None):
    """Check CSV has expected columns and no banned columns. Returns list of issues."""
    issues = []
    try:
        with open(path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
    except Exception as e:
        return [f"Cannot read CSV: {e}"]

    header_set = set(header)
    missing = expected_cols - header_set
    if missing:
        issues.append(f"Missing columns: {missing}")

    if banned_cols:
        found_banned = banned_cols & header_set
        if found_banned:
            issues.append(f"BANNED columns still present: {found_banned}")

    # Check at least one data row
    try:
        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            row = next(reader, None)
            if row is None:
                issues.append("CSV has header but no data rows")
    except Exception:
        pass

    return issues


def check_csv_values(path, col_name, check_fn, description):
    """Check that values in a column satisfy check_fn. Returns list of issues."""
    issues = []
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            values = []
            for row in reader:
                val_str = row.get(col_name, "")
                if val_str in ("", "nan", "inf", "-inf"):
                    continue
                try:
                    values.append(float(val_str))
                except ValueError:
                    continue

            if not values:
                issues.append(f"No finite values in column '{col_name}'")
            elif not check_fn(values):
                issues.append(f"Column '{col_name}' failed check: {description}")
    except Exception as e:
        issues.append(f"Error reading {col_name}: {e}")
    return issues


def check_json_keys(path, expected_keys):
    """Check JSON file has expected top-level keys."""
    issues = []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        missing = expected_keys - set(data.keys())
        if missing:
            issues.append(f"Missing JSON keys: {missing}")
    except Exception as e:
        issues.append(f"Cannot read JSON: {e}")
    return issues


def check_dense_npz(path, required_keys):
    """Check NPZ has required arrays and consistent dense-matrix shape."""
    issues = []
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as e:
        return [f"Cannot read NPZ: {e}"]

    missing = set(required_keys) - set(data.files)
    if missing:
        return [f"Missing NPZ arrays: {missing}"]

    ell = np.asarray(data["ell"])
    values = np.asarray(data["values"])
    if ell.ndim != 1:
        issues.append("'ell' must be 1-D")
    if values.ndim != 2:
        issues.append("'values' must be 2-D")
    elif ell.ndim == 1 and values.shape[0] != ell.shape[0]:
        issues.append(f"Shape mismatch: len(ell)={ell.shape[0]} but values.shape[0]={values.shape[0]}")
    if values.size == 0:
        issues.append("Dense matrix is empty")
    return issues


def validate_model(model_dir, model_name):
    """Validate all outputs for a single model. Returns (pass_count, fail_count, messages)."""
    passes = 0
    fails = 0
    messages = []

    # 1) Check required files exist
    for template in REQUIRED_FILES:
        fname = template.format(model=model_name)
        fpath = os.path.join(model_dir, fname)
        if os.path.isfile(fpath):
            passes += 1
        else:
            fails += 1
            messages.append(f"  MISSING: {fname}")

    # 2) Check summary CSV schema
    summary_path = os.path.join(model_dir, f"{model_name}_summary.csv")
    mu_npz_path = os.path.join(model_dir, f"{model_name}_mu_units.npz")
    gates_npz_path = os.path.join(model_dir, f"{model_name}_mu_units_gates.npz")

    if os.path.isfile(mu_npz_path):
        issues = check_dense_npz(mu_npz_path, {"ell", "values", "zero_order_values", "first_order_values"})
        if issues:
            fails += 1
            for iss in issues:
                messages.append(f"  NPZ {model_name}_mu_units.npz: {iss}")
        else:
            passes += 1

    if os.path.isfile(gates_npz_path):
        issues = check_dense_npz(gates_npz_path, {"ell", "values"})
        if issues:
            fails += 1
            for iss in issues:
                messages.append(f"  NPZ {model_name}_mu_units_gates.npz: {iss}")
        else:
            passes += 1

    if os.path.isfile(summary_path):
        issues = check_csv_schema(summary_path, SUMMARY_EXPECTED_COLS, BANNED_COLS)
        if issues:
            fails += 1
            for iss in issues:
                messages.append(f"  SCHEMA {model_name}_summary.csv: {iss}")
        else:
            passes += 1

        # 3) Check f_ratio values are positive and finite
        issues = check_csv_values(
            summary_path, "f_ratio",
            lambda vals: all(v > 0 for v in vals),
            "all f_ratio values should be > 0"
        )
        if issues:
            fails += 1
            for iss in issues:
                messages.append(f"  VALUES {model_name}_summary.csv: {iss}")
        else:
            passes += 1

        # 4) Check f_gates values are positive
        issues = check_csv_values(
            summary_path, "f_gates",
            lambda vals: all(v > 0 for v in vals),
            "all f_gates values should be > 0"
        )
        if issues:
            fails += 1
            for iss in issues:
                messages.append(f"  VALUES {model_name}_summary.csv: {iss}")
        else:
            passes += 1

        # 5) Check alpha_hat in [0.5, 2.0]
        issues = check_csv_values(
            summary_path, "alpha_hat",
            lambda vals: all(0.5 <= v <= 2.01 for v in vals),
            "alpha_hat should be in [0.5, 2.0]"
        )
        if issues:
            fails += 1
            for iss in issues:
                messages.append(f"  VALUES {model_name}_summary.csv: {iss}")
        else:
            passes += 1

    # 6) Check lambda_tau_correlation files (conditional on AdamW)
    lt_csv = os.path.join(model_dir, f"{model_name}_lambda_tau_correlation.csv")
    lt_json = os.path.join(model_dir, f"{model_name}_lambda_tau_stats.json")

    if os.path.isfile(lt_csv):
        passes += 1
        issues = check_csv_schema(lt_csv, LAMBDA_TAU_EXPECTED_COLS)
        if issues:
            fails += 1
            for iss in issues:
                messages.append(f"  SCHEMA {model_name}_lambda_tau_correlation.csv: {iss}")
        else:
            passes += 1

        # Check Lambda_q values are positive
        issues = check_csv_values(
            lt_csv, "Lambda_q",
            lambda vals: all(v > 0 for v in vals),
            "Lambda_q should be > 0"
        )
        if issues:
            fails += 1
            for iss in issues:
                messages.append(f"  VALUES {model_name}_lambda_tau_correlation.csv: {iss}")
        else:
            passes += 1

        # Check tau_q values are positive
        issues = check_csv_values(
            lt_csv, "tau_q",
            lambda vals: all(v > 0 for v in vals),
            "tau_q should be > 0"
        )
        if issues:
            fails += 1
            for iss in issues:
                messages.append(f"  VALUES {model_name}_lambda_tau_correlation.csv: {iss}")
        else:
            passes += 1
    else:
        messages.append(f"  NOTE: {model_name}_lambda_tau_correlation.csv not found (may be OK if <5 valid tau fits)")

    if os.path.isfile(lt_json):
        passes += 1
        issues = check_json_keys(lt_json, LAMBDA_TAU_STATS_KEYS)
        if issues:
            fails += 1
            for iss in issues:
                messages.append(f"  SCHEMA {model_name}_lambda_tau_stats.json: {iss}")
        else:
            passes += 1

        # Check rho in [-1, 1]
        try:
            with open(lt_json, "r") as f:
                data = json.load(f)
            rho = data.get("spearman_rho_lambda_tau", None)
            if rho is not None and not (-1.0 <= rho <= 1.0):
                fails += 1
                messages.append(f"  VALUES {model_name}_lambda_tau_stats.json: rho={rho} out of [-1,1]")
            else:
                passes += 1
                messages.append(f"  INFO: {model_name} Spearman ρ(Λ_q, τ_q) = {rho:.3f}")
        except Exception:
            pass

    return passes, fails, messages


def main():
    parser = argparse.ArgumentParser(description="Validate smoke test outputs")
    parser.add_argument("--root", type=str, default="results/GELR_smoke_test",
                        help="Root directory of smoke test output")
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"ERROR: output root not found: {root}")
        sys.exit(1)

    # Discover model directories
    model_dirs = {}

    # Baselines
    baselines_seed = os.path.join(root, "baselines", "seed_101")
    if os.path.isdir(baselines_seed):
        for model in ["const", "shared", "diag"]:
            mdir = os.path.join(baselines_seed, model)
            if os.path.isdir(mdir):
                model_dirs[model] = mdir

    # LSTM/GRU
    lstmgru_seed = os.path.join(root, "lstmgru", "seed_101")
    if os.path.isdir(lstmgru_seed):
        for model in ["lstm", "gru"]:
            mdir = os.path.join(lstmgru_seed, model)
            if os.path.isdir(mdir):
                model_dirs[model] = mdir

    if not model_dirs:
        print(f"ERROR: no model directories found under {root}")
        sys.exit(1)

    print("=" * 60)
    print(f"  Smoke Test Validation — {len(model_dirs)} models found")
    print("=" * 60)

    total_pass = 0
    total_fail = 0

    for model, mdir in sorted(model_dirs.items()):
        print(f"\n── {model} ({mdir}) ──")
        p, f, msgs = validate_model(mdir, model)
        total_pass += p
        total_fail += f
        for m in msgs:
            print(m)
        status = "PASS" if f == 0 else "FAIL"
        print(f"  Result: {status}  ({p} checks passed, {f} failed)")

    print("\n" + "=" * 60)
    if total_fail == 0:
        print(f"  ALL CHECKS PASSED ({total_pass} total)")
        print("  Safe to launch full sweep: bash launch_GELR_multiseed.sh")
    else:
        print(f"  {total_fail} CHECK(S) FAILED ({total_pass} passed)")
        print("  Fix issues before launching full sweep.")
    print("=" * 60)

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
