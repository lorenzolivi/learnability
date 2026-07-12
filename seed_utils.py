#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seed_utils.py — Shared utilities for multi-seed aggregation and auto-merging
of baselines + lstm/gru results.

Core design:
  - Discover seed_* subdirectories inside any inputdir.
  - If none found, treat the inputdir itself as a single-seed run.
  - Load per-model CSVs across seeds, aggregate with mean/std.
  - Merge results from multiple inputdirs (e.g., baselines/ + lstm_gru/).
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Optional

# ── Canonical model names ──────────────────────────────────────────────
CANDIDATE_MODELS = ["const", "shared", "diag", "gru", "lstm"]

# ── H_N column mappings by alpha method ────────────────────────────────
MODEL_COLS_HN_ECF = {
    "const": "H_N_const_ecf",
    "shared": "H_N_shared_ecf",
    "diag": "H_N_diag_ecf",
    "gru": "H_N_gru_ecf",
    "lstm": "H_N_lstm_ecf",
}

MODEL_COLS_HN_MCC = {
    "const": "H_N_const_mcc",
    "shared": "H_N_shared_mcc",
    "diag": "H_N_diag_mcc",
    "gru": "H_N_gru_mcc",
    "lstm": "H_N_lstm_mcc",
}

# Keep old dict for backward compat (defaults to ECF)
MODEL_COLS_HN = MODEL_COLS_HN_ECF

CANDIDATE_SUMMARY_FILES = {
    m: f"{m}_summary.csv" for m in CANDIDATE_MODELS
}


# ── H_N column helper ──────────────────────────────────────────────────

def get_model_hn_col(model: str, method: str = "ecf") -> str:
    """
    Return the H_N column name for a given model and alpha method.

    Args:
        model: model name (e.g., "const", "diag", "lstm")
        method: alpha method, "ecf" or "mcc" (default: "ecf")

    Returns:
        column name like "H_N_const_ecf" or "H_N_diag_mcc"
    """
    return f"H_N_{model}_{method}"


# ── Seed directory discovery ───────────────────────────────────────────

def discover_seed_dirs(inputdir: str) -> list[str]:
    """
    Return a sorted list of seed_* subdirectory paths inside inputdir.
    If none are found, return [inputdir] (treat as single-seed).
    """
    if not os.path.isdir(inputdir):
        return []

    seed_dirs = []
    for name in sorted(os.listdir(inputdir)):
        full = os.path.join(inputdir, name)
        if os.path.isdir(full) and re.match(r"^seed_\d+$", name):
            seed_dirs.append(full)

    if not seed_dirs:
        # Fallback: treat inputdir itself as a single run
        return [inputdir]

    return seed_dirs


def discover_from_multiple_inputdirs(inputdirs: list[str]) -> list[str]:
    """
    Given a list of inputdirs, discover seed dirs from each and return
    a flat list. Deduplicates by absolute path.
    """
    seen = set()
    result = []
    for d in inputdirs:
        for sd in discover_seed_dirs(d):
            absd = os.path.abspath(sd)
            if absd not in seen:
                seen.add(absd)
                result.append(sd)
    return result


# ── File finding (flat or nested layout) ──────────────────────────────

def find_file_in_seed_dir(seed_dir: str, filename: str, model: str = None) -> Optional[str]:
    """
    Find a file inside a seed directory, supporting both flat and nested layouts.

    Search order:
      1. seed_dir/filename                          (flat layout)
      2. seed_dir/<model>/filename                  (nested layout, if model given)
      3. seed_dir/<any_subdir>/filename             (nested layout, auto-detect)

    Returns the first path found, or None.
    """
    # 1. Flat: directly in seed dir
    flat = os.path.join(seed_dir, filename)
    if os.path.exists(flat):
        return flat

    # 2. Nested: in model subdir (if model name given)
    if model:
        nested = os.path.join(seed_dir, model, filename)
        if os.path.exists(nested):
            return nested

    # 3. Nested: search all immediate subdirectories
    if os.path.isdir(seed_dir):
        for name in sorted(os.listdir(seed_dir)):
            subdir = os.path.join(seed_dir, name)
            if os.path.isdir(subdir):
                candidate = os.path.join(subdir, filename)
                if os.path.exists(candidate):
                    return candidate

    return None


def find_json_in_seed_dir(seed_dir: str, json_name: str, model: str = None) -> Optional[str]:
    """Convenience alias for find_file_in_seed_dir for JSON files."""
    return find_file_in_seed_dir(seed_dir, json_name, model)


def find_dense_artifact_in_seed_dir(seed_dir: str, stem: str, model: str = None) -> Optional[str]:
    """
    Find a dense artifact stored as NPZ.

    Args:
        seed_dir: Seed directory to search.
        stem: File stem without extension, e.g. "lstm_mu_units".
        model: Optional model subdirectory hint for nested layouts.

    Returns:
        Path to <stem>.npz if present, else None.
    """
    return find_file_in_seed_dir(seed_dir, f"{stem}.npz", model)


def load_dense_unit_artifact(
    path: str,
    value_keys: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a dense per-lag/per-unit artifact from NPZ.

    NPZ files must contain:
      - ell:    (L,) lag grid
      - one value matrix, typically stored under "values"
    """
    if value_keys is None:
        value_keys = [
            "values",
            "mu_total",
            "mu_units",
            "mu_gates",
            "zero_order_values",
            "first_order_values",
        ]

    if not path.endswith(".npz"):
        raise ValueError(f"Dense artifacts must be NPZ, got: {path}")

    with np.load(path, allow_pickle=False) as data:
        if "ell" not in data:
            raise ValueError(f"NPZ artifact missing 'ell' array: {path}")
        value_key = next((k for k in value_keys if k in data), None)
        if value_key is None:
            raise ValueError(
                f"NPZ artifact missing a value matrix. Tried keys {value_keys} in {path}"
            )
        ell = np.asarray(data["ell"], dtype=np.float64)
        values = np.asarray(data[value_key], dtype=np.float64)
    return ell, values


# ── CSV loading across seeds ───────────────────────────────────────────

def load_csv_across_seeds(
    seed_dirs: list[str],
    filename: str,
    required_cols: Optional[set] = None,
    model: str = None,
) -> list[pd.DataFrame]:
    """
    Load filename from each seed_dir where it exists.
    Supports both flat (seed_dir/file.csv) and nested (seed_dir/model/file.csv) layouts.
    Optionally validate required columns.
    Returns list of DataFrames (one per seed that had the file).
    """
    dfs = []
    for sd in seed_dirs:
        path = find_file_in_seed_dir(sd, filename, model)
        if path is None:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  [warn] failed reading {path}: {e}")
            continue
        if required_cols is not None:
            missing = required_cols - set(df.columns)
            if missing:
                print(f"  [warn] {path} missing columns: {sorted(missing)}, skipping")
                continue
        dfs.append(df)
    return dfs


def load_model_summary_across_seeds(
    seed_dirs: list[str],
    model: str,
    required_cols: Optional[set] = None,
) -> list[pd.DataFrame]:
    """Load <model>_summary.csv from each seed dir (flat or nested)."""
    return load_csv_across_seeds(
        seed_dirs, f"{model}_summary.csv", required_cols, model=model
    )


# ── Aggregation ────────────────────────────────────────────────────────

def aggregate_numeric_by_key(
    dfs: list[pd.DataFrame],
    key_col: str,
    value_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Given a list of DataFrames (one per seed), merge on key_col and compute
    mean/std for each numeric column.

    Returns a DataFrame with columns:
      key_col, <col>_mean, <col>_std, <col>_count  for each value_col
    """
    if not dfs:
        return pd.DataFrame()

    # Coerce key and value cols to numeric
    processed = []
    for df in dfs:
        df = df.copy()
        df[key_col] = pd.to_numeric(df[key_col], errors="coerce")
        df = df.dropna(subset=[key_col])
        if value_cols is None:
            value_cols = [c for c in df.columns if c != key_col]
        for c in value_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        processed.append(df)

    # Concatenate all seeds
    combined = pd.concat(processed, ignore_index=True)

    # Group by key and aggregate
    agg_dict = {}
    for c in value_cols:
        if c not in combined.columns:
            continue
        agg_dict[c] = ["mean", "std", "count"]

    if not agg_dict:
        return pd.DataFrame()

    grouped = combined.groupby(key_col).agg(agg_dict)

    # Flatten multi-level column names
    result = pd.DataFrame({key_col: grouped.index})
    for col in agg_dict:
        result[f"{col}_mean"] = grouped[(col, "mean")].values
        result[f"{col}_std"] = grouped[(col, "std")].values
        result[f"{col}_count"] = grouped[(col, "count")].values.astype(int)

    return result.reset_index(drop=True)


def aggregate_H_N_across_seeds(
    seed_dirs: list[str],
) -> pd.DataFrame:
    """
    Load H_N_summary.csv from each seed dir, merge, and compute mean/std
    for each model's H_N column.

    Also looks for individual <model>_H_N.csv files as fallback.

    Returns DataFrame with columns:
      N, H_N_<model>_mean, H_N_<model>_std, H_N_<model>_count
    """
    # First try H_N_summary.csv
    summary_dfs = load_csv_across_seeds(seed_dirs, "H_N_summary.csv", {"N"})

    if summary_dfs:
        # Combine from summary CSVs
        model_cols = []
        for df in summary_dfs:
            for col in df.columns:
                if col.startswith("H_N_") and col not in model_cols:
                    model_cols.append(col)

        return aggregate_numeric_by_key(summary_dfs, "N", model_cols)

    # Fallback: load per-model H_N files
    all_rows = []
    for sd in seed_dirs:
        row = {"_seed_dir": sd}
        for model in CANDIDATE_MODELS:
            hn_path = os.path.join(sd, model, f"{model}_H_N.csv")
            if not os.path.exists(hn_path):
                hn_path = os.path.join(sd, f"{model}_H_N.csv")
            if not os.path.exists(hn_path):
                continue
            try:
                df = pd.read_csv(hn_path)
                if "N" in df.columns and "H_N" in df.columns:
                    col_name = f"H_N_{model}"
                    for _, r in df.iterrows():
                        row.setdefault("_N_values", set()).add(r["N"])
                        row[f"{col_name}_{r['N']}"] = r["H_N"]
            except Exception:
                continue
        all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    # Per-model reconstruction is handled by higher-level callers.
    return pd.DataFrame()


def collect_H_N_per_seed(
    seed_dirs: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Load H_N_summary.csv from each seed dir and return per-seed DataFrames
    keyed by seed directory name.

    Returns dict: { seed_label -> DataFrame with columns [N, H_N_const, ...] }
    Each DataFrame has one row per N value for that seed.
    """
    result = {}
    for sd in seed_dirs:
        path = find_file_in_seed_dir(sd, "H_N_summary.csv")
        if path is None:
            continue
        try:
            df = pd.read_csv(path)
            if "N" not in df.columns:
                continue
            label = os.path.basename(sd)
            # If same seed label seen from different inputdirs, merge columns
            if label in result:
                existing = result[label]
                for col in df.columns:
                    if col != "N" and col not in existing.columns:
                        existing = existing.merge(df[["N", col]], on="N", how="outer")
                result[label] = existing
            else:
                result[label] = df
        except Exception:
            continue
    return result


def collect_H_N_matrix(
    seed_dirs: list[str],
    model_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect a (n_seeds x n_N) matrix of H_N values for a given model column.

    Args:
        seed_dirs: list of seed directories
        model_col: column name in H_N_summary.csv (e.g., "H_N_diag")

    Returns:
        N_grid: 1-D array of N values (from the union of all seeds)
        matrix: 2-D array (n_seeds x len(N_grid)), NaN where missing
    """
    per_seed = collect_H_N_per_seed(seed_dirs)
    if not per_seed:
        return np.array([]), np.array([]).reshape(0, 0)

    # Collect all N values across seeds
    all_N = set()
    for df in per_seed.values():
        if model_col in df.columns:
            all_N.update(df["N"].dropna().astype(float).tolist())

    if not all_N:
        return np.array([]), np.array([]).reshape(0, 0)

    N_grid = np.sort(list(all_N))
    n_map = {n: i for i, n in enumerate(N_grid)}

    matrix = np.full((len(per_seed), len(N_grid)), np.nan)
    for row_idx, (label, df) in enumerate(per_seed.items()):
        if model_col not in df.columns:
            continue
        for _, r in df.iterrows():
            n_val = float(r["N"])
            if n_val in n_map:
                matrix[row_idx, n_map[n_val]] = float(r[model_col])

    return N_grid, matrix


# ── Model detection across multiple inputdirs ─────────────────────────

def detect_models_in_dirs(dirs: list[str]) -> list[str]:
    """
    Detect which models have summary CSVs in any of the given directories.
    Supports both flat and nested (model-subdir) layouts.
    """
    found = set()
    for d in dirs:
        for model in CANDIDATE_MODELS:
            if find_file_in_seed_dir(d, f"{model}_summary.csv", model) is not None:
                found.add(model)
    # Return in canonical order
    return [m for m in CANDIDATE_MODELS if m in found]


def detect_models_with_mu_units(dirs: list[str]) -> list[str]:
    """Detect models that have mu_units artifacts or tau CSVs (flat or nested)."""
    found = set()
    for d in dirs:
        for m in CANDIDATE_MODELS:
            if (find_dense_artifact_in_seed_dir(d, f"{m}_mu_units", m) is not None or
                find_file_in_seed_dir(d, f"{m}_tau_from_mu_units.csv", m) is not None):
                found.add(m)
    return [m for m in CANDIDATE_MODELS if m in found]


# ── Common CLI argument helpers ────────────────────────────────────────

def add_multiseed_args(parser):
    """Add common multi-seed / multi-dir CLI arguments to an ArgumentParser."""
    parser.add_argument(
        "--inputdirs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more input directories (e.g., baselines/adamw lstm_gru/adamw). "
            "Each is scanned for seed_* subdirectories. Replaces --inputdir."
        ),
    )
    parser.add_argument(
        "--inputdir",
        type=str,
        default=None,
        help="Single input directory (backward compatible). Use --inputdirs for multi-dir.",
    )
    return parser


def resolve_inputdirs(args) -> list[str]:
    """
    Resolve --inputdirs / --inputdir into a list of directory paths.
    Raises ValueError if nothing is provided.
    """
    if args.inputdirs:
        return args.inputdirs
    elif args.inputdir:
        return [args.inputdir]
    else:
        return ["."]


def resolve_seed_dirs(args) -> list[str]:
    """
    Full resolution: inputdirs -> seed discovery -> flat list of data dirs.
    """
    inputdirs = resolve_inputdirs(args)
    return discover_from_multiple_inputdirs(inputdirs)


# ── Plotting helpers ───────────────────────────────────────────────────

def shade_between(ax, x, y_mean, y_std, color=None, alpha=0.2, **kwargs):
    """Plot mean line with ±1std shaded band."""
    x = np.asarray(x, dtype=float)
    y_mean = np.asarray(y_mean, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y_mean)
    x, y_mean, y_std = x[mask], y_mean[mask], y_std[mask]
    y_std = np.where(np.isfinite(y_std), y_std, 0.0)

    plot_kwargs = dict(kwargs)
    if color is not None and "color" not in plot_kwargs:
        plot_kwargs["color"] = color
    line = ax.plot(x, y_mean, **plot_kwargs)
    c = color or line[0].get_color()
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=alpha, color=c)
    return line


def shade_minmax(ax, x, y_min, y_max, color=None, alpha=0.15, **kwargs):
    """Plot min-max shaded band (no mean line) with median."""
    x = np.asarray(x, dtype=float)
    y_min = np.asarray(y_min, dtype=float)
    y_max = np.asarray(y_max, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y_min) & np.isfinite(y_max)
    x, y_min, y_max = x[mask], y_min[mask], y_max[mask]

    line = ax.plot(x, 0.5 * (y_min + y_max), linewidth=1.0, **kwargs)
    c = color or line[0].get_color()
    ax.fill_between(x, y_min, y_max, alpha=alpha, color=c)
    return line


# ── Per-seed data loading (no aggregation) ────────────────────────────

def load_model_data_per_seed(
    seed_dirs: list[str],
    model: str,
    required_cols: Optional[set] = None,
) -> list[tuple[str, pd.DataFrame]]:
    """
    Load <model>_summary.csv from each seed WITHOUT aggregating.

    Returns: list of (seed_label, DataFrame) pairs.
    """
    result = []
    for sd in seed_dirs:
        path = find_file_in_seed_dir(sd, f"{model}_summary.csv", model)
        if path is None:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if required_cols is not None:
            missing = required_cols - set(df.columns)
            if missing:
                continue
        label = os.path.basename(sd)
        result.append((label, df))
    return result


def get_seed_label(seed_dir: str) -> str:
    """Extract seed label (e.g., 'seed_1') from a seed directory path."""
    return os.path.basename(seed_dir)


# ── View mode CLI helpers ─────────────────────────────────────────────

def add_view_arg(parser):
    """Add --view argument: per_seed, aggregated, or both."""
    parser.add_argument(
        "--view",
        type=str,
        default="both",
        choices=["per_seed", "aggregated", "both"],
        help="Output mode: per_seed (individual seed traces), aggregated "
             "(spread summary), or both (default: both).",
    )
    return parser


# ── Standard model colors ─────────────────────────────────────────────

MODEL_COLORS = {
    "const": "#1f77b4",   # blue
    "shared": "#ff7f0e",  # orange
    "diag": "#2ca02c",    # green
    "gru": "#d62728",     # red
    "lstm": "#9467bd",    # purple
}

SEED_ALPHAS = [1.0, 0.7, 0.5, 0.4, 0.3]  # visual weight: seed 1 darkest


def get_model_color(model: str) -> str:
    """Return the canonical color for a model."""
    return MODEL_COLORS.get(model, "#333333")


def print_seed_info(seed_dirs: list[str], inputdirs: list[str]):
    """Print info about discovered seeds."""
    n_seeds = len(seed_dirs)
    n_dirs = len(inputdirs)
    is_multi = any(
        len(discover_seed_dirs(d)) > 1 or
        (len(discover_seed_dirs(d)) == 1 and discover_seed_dirs(d)[0] != d)
        for d in inputdirs
    )
    print(f"[info] {n_dirs} input dir(s), {n_seeds} data dir(s) "
          f"({'multi-seed' if is_multi else 'single-seed'})")
    for sd in seed_dirs:
        print(f"  - {os.path.abspath(sd)}")


# ── Bootstrap confidence intervals for McCulloch estimator ──────────────────

def bootstrap_mcculloch(
    samples: np.ndarray,
    estimator_fn,
    n_boot: int = 200,
    ci: float = 0.95,
) -> tuple[float, float, float, float]:
    """
    Compute bootstrap confidence intervals for McCulloch α and σ estimates.

    Resamples the input samples with replacement n_boot times, applies the
    McCulloch quantile-ratio estimator to each resample, and returns the
    median point estimates plus confidence interval bounds for alpha.

    Args:
        samples: 1-D array of matched-statistic values (float64).
        estimator_fn: A callable with signature:
            estimator_fn(q05, q25, q75, q95) -> (alpha_hat, sigma_hat)
            Typically estimate_alpha_sigma_mcculloch_symmetric_from_quantiles
            from the main training script. This function is passed rather than imported
            to avoid circular imports.
        n_boot: Number of bootstrap resamples (default: 200).
        ci: Confidence interval level (default: 0.95 → 2.5th to 97.5th percentile).

    Returns:
        (alpha_median, alpha_ci_lo, alpha_ci_hi, sigma_median):
            alpha_median: median α̂ across bootstrap samples
            alpha_ci_lo: lower confidence bound (e.g., 2.5th percentile)
            alpha_ci_hi: upper confidence bound (e.g., 97.5th percentile)
            sigma_median: median σ̂ across bootstrap samples
    """
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    if n < 4:
        # Insufficient data; return defaults
        return 2.0, 1.0, 2.0, 0.0

    rng = np.random.RandomState(42)  # Deterministic for reproducibility
    alpha_boots = []
    sigma_boots = []

    for _ in range(n_boot):
        # Resample with replacement
        idx = rng.choice(n, size=n, replace=True)
        boot_samples = samples[idx]

        # Compute quantiles
        q05 = float(np.quantile(boot_samples, 0.05))
        q25 = float(np.quantile(boot_samples, 0.25))
        q75 = float(np.quantile(boot_samples, 0.75))
        q95 = float(np.quantile(boot_samples, 0.95))

        # Apply McCulloch estimator
        alpha_hat, sigma_hat = estimator_fn(q05, q25, q75, q95)
        alpha_boots.append(float(alpha_hat))
        sigma_boots.append(float(sigma_hat))

    # Compute statistics
    alpha_boots = np.array(alpha_boots, dtype=np.float64)
    sigma_boots = np.array(sigma_boots, dtype=np.float64)

    alpha_median = float(np.median(alpha_boots))
    sigma_median = float(np.median(sigma_boots))

    # Confidence interval bounds
    alpha_lower = (1.0 - ci) / 2.0  # e.g., 0.025 for 95% CI
    alpha_upper = 1.0 - alpha_lower   # e.g., 0.975 for 95% CI
    alpha_ci_lo = float(np.quantile(alpha_boots, alpha_lower))
    alpha_ci_hi = float(np.quantile(alpha_boots, alpha_upper))

    return alpha_median, alpha_ci_lo, alpha_ci_hi, sigma_median
