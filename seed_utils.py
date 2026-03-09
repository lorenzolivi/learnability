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

MODEL_COLS_HN = {
    "const": "H_N_const",
    "shared": "H_N_shared",
    "diag": "H_N_diag",
    "gru": "H_N_gru",
    "lstm": "H_N_lstm",
}

CANDIDATE_SUMMARY_FILES = {
    m: f"{m}_summary.csv" for m in CANDIDATE_MODELS
}


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

    # Reconstruct from per-model H_N files
    # This is more complex; for now return empty and let callers handle it
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
    """Detect models that have mu_units or tau CSVs (flat or nested)."""
    found = set()
    for d in dirs:
        for m in CANDIDATE_MODELS:
            if (find_file_in_seed_dir(d, f"{m}_mu_units.csv", m) is not None or
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

    line = ax.plot(x, y_mean, **kwargs)
    c = color or line[0].get_color()
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=alpha, color=c)
    return line


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
