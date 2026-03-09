#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
launch_multiseed.py — Wrapper to run baselines + lstm/gru DGX scripts
across multiple random seeds.

Does NOT modify the DGX scripts. Simply calls them with different --seed
values, writing outputs to seed-tagged subdirectories:
    <outdir_baselines>/seed_<S>/
    <outdir_lstm_gru>/seed_<S>/

Usage:
    python launch_multiseed.py \
        --seeds 9212,4567,7890 \
        --outdir_baselines results/T1024/baselines/adamw \
        --outdir_lstm_gru results/T1024/lstm_gru/adamw \
        --common_args "--Nseq_train 8000 --Nseq_diag 8000 --T 1024 --D 16 --H 128 --optimizer adamw --grad_clip 1 --epochs 500 --batch_size 512 --lr 1e-3 --use_paper_lag_grid 0 --lag_min 4 --lag_max 256 --num_lags 128 --task_lags 32,64,128,192,256 --task_coeffs 0.6,0.5,0.4,0.32,0.26 --noise_std 0.3 --N_grid 25,50,100,150,200,300,400,600,800,1200,1600,2400,3200,4800,6400,9600,12800 --eps 0.1 --orth_init --layernorm --device cuda" \
        --baseline_extra "--models const,shared,diag --const_s 0.5" \
        --lstm_gru_extra "--models lstm,gru" \
        --w_seed 12 \
        --logdir logs

Each seed spawns two runs (baselines + lstm_gru) sequentially.
Seeds are run sequentially by default, or in parallel with --parallel.
"""

import os
import sys
import argparse
import subprocess
import time
import datetime


def parse_args():
    p = argparse.ArgumentParser(
        description="Run learnability experiments across multiple seeds."
    )
    p.add_argument(
        "--seeds",
        type=str,
        required=True,
        help="Comma-separated list of seed values (e.g., 9212,4567,7890)",
    )
    p.add_argument(
        "--outdir_baselines",
        type=str,
        required=True,
        help="Base output dir for baselines (seed_<S>/ subdirs will be created inside)",
    )
    p.add_argument(
        "--outdir_lstm_gru",
        type=str,
        required=True,
        help="Base output dir for lstm/gru (seed_<S>/ subdirs will be created inside)",
    )
    p.add_argument(
        "--common_args",
        type=str,
        default="",
        help="Common CLI arguments for both scripts (quoted string)",
    )
    p.add_argument(
        "--baseline_extra",
        type=str,
        default="--models const,shared,diag --const_s 0.5",
        help="Extra args for baseline script only",
    )
    p.add_argument(
        "--lstm_gru_extra",
        type=str,
        default="--models lstm,gru",
        help="Extra args for lstm/gru script only",
    )
    p.add_argument(
        "--w_seed",
        type=int,
        default=12,
        help="w_seed (JVP direction); fixed across all runs (default: 12)",
    )
    p.add_argument(
        "--data_seed",
        type=int,
        default=None,
        help=(
            "If set, pass this fixed seed for data generation to the DGX scripts "
            "via --data_seed, so that every run uses the SAME synthetic dataset "
            "while varying model initialisation through --seed. "
            "Requires a small patch to the DGX scripts (see EXAMPLES.txt). "
            "If not set, each --seed controls both data and model randomness (default)."
        ),
    )
    p.add_argument(
        "--alpha_method",
        type=str,
        default=None,
        choices=["mcculloch", "ecf"],
        help=(
            "Method for estimating the stable tail index α̂. "
            "Passed through to DGX scripts via --alpha_method. "
            "If not set, the DGX scripts use their default (mcculloch)."
        ),
    )
    p.add_argument(
        "--min_samples_alpha",
        type=int,
        default=None,
        help=(
            "Minimum samples for reliable α̂ estimate. "
            "Passed through to DGX scripts via --min_samples_alpha. "
            "If not set, DGX scripts use their default (500)."
        ),
    )
    p.add_argument(
        "--baseline_script",
        type=str,
        default="run_learnability_DGX.py",
        help="Path to baseline DGX script",
    )
    p.add_argument(
        "--lstm_gru_script",
        type=str,
        default="run_learnability_lstm_gru_DGX.py",
        help="Path to lstm/gru DGX script",
    )
    p.add_argument(
        "--logdir",
        type=str,
        default="logs",
        help="Directory for log files (default: logs)",
    )
    p.add_argument(
        "--skip_baselines",
        action="store_true",
        help="Skip baseline runs (only run lstm/gru)",
    )
    p.add_argument(
        "--skip_lstm_gru",
        action="store_true",
        help="Skip lstm/gru runs (only run baselines)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them",
    )
    return p.parse_args()


def _timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_with_streaming(cmd: str, logfile: str, tag: str) -> tuple[int, float]:
    """
    Run a command, streaming its output line-by-line to both:
      - the per-run log file (logfile)
      - stdout with a [tag] prefix (so it appears in launch_multiseed.log)

    Returns (returncode, elapsed_seconds).
    """
    t0 = time.time()
    print(f"  [{_timestamp()}] STARTED", flush=True)

    with open(logfile, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
        )
        for line in proc.stdout:
            line = line.rstrip("\n")
            logf.write(line + "\n")
            logf.flush()
            # Forward selected lines to main log (epoch progress, model starts, errors)
            line_lower = line.lower()
            if any(kw in line_lower for kw in [
                "epoch", "model", "training", "diagnost", "saving",
                "error", "exception", "traceback", "warning",
                "h_n", "envelope", "tau", "alpha",
                "starting", "finished", "completed",
            ]):
                print(f"  [{tag}] {line}", flush=True)
        proc.wait()

    elapsed = time.time() - t0
    print(f"  [{_timestamp()}] FINISHED (returncode={proc.returncode}, {elapsed:.1f}s)", flush=True)
    return proc.returncode, elapsed


def build_command(
    python_exe: str,
    script: str,
    outdir: str,
    seed: int,
    w_seed: int,
    common_args: str,
    extra_args: str,
    data_seed: int = None,
    alpha_method: str = None,
    min_samples_alpha: int = None,
) -> str:
    """Build the full command string."""
    parts = [
        python_exe,
        script,
        f"--outdir {outdir}",
        f"--seed {seed}",
        f"--w_seed {w_seed}",
    ]
    if data_seed is not None:
        parts.append(f"--data_seed {data_seed}")
    if alpha_method is not None:
        parts.append(f"--alpha_method {alpha_method}")
    if min_samples_alpha is not None:
        parts.append(f"--min_samples_alpha {min_samples_alpha}")
    if common_args.strip():
        parts.append(common_args.strip())
    if extra_args.strip():
        parts.append(extra_args.strip())
    return " ".join(parts)


def main():
    args = parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        print("[error] No seeds provided.")
        sys.exit(1)

    python_exe = sys.executable
    os.makedirs(args.logdir, exist_ok=True)

    print(f"[info] Seeds: {seeds}")
    print(f"[info] w_seed: {args.w_seed}")
    if args.data_seed is not None:
        print(f"[info] data_seed: {args.data_seed} (fixed dataset across runs)")
    else:
        print(f"[info] data_seed: not set (each --seed controls both data and model)")
    if args.alpha_method is not None:
        print(f"[info] alpha_method: {args.alpha_method}")
    if args.min_samples_alpha is not None:
        print(f"[info] min_samples_alpha: {args.min_samples_alpha}")
    print(f"[info] Baseline script: {args.baseline_script}")
    print(f"[info] LSTM/GRU script: {args.lstm_gru_script}")
    print(f"[info] Baselines outdir: {args.outdir_baselines}")
    print(f"[info] LSTM/GRU outdir: {args.outdir_lstm_gru}")
    print()

    total_runs = 0
    failed_runs = []

    for seed in seeds:
        print(f"{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

        # ── Baselines ──
        if not args.skip_baselines:
            bl_outdir = os.path.join(args.outdir_baselines, f"seed_{seed}")
            os.makedirs(bl_outdir, exist_ok=True)
            bl_log = os.path.join(args.logdir, f"baselines_seed_{seed}.log")

            cmd = build_command(
                python_exe,
                args.baseline_script,
                bl_outdir,
                seed,
                args.w_seed,
                args.common_args,
                args.baseline_extra,
                data_seed=args.data_seed,
                alpha_method=args.alpha_method,
                min_samples_alpha=args.min_samples_alpha,
            )

            print(f"\n[run] Baselines seed={seed}")
            print(f"  cmd: {cmd}")
            print(f"  log: {bl_log}")

            if args.dry_run:
                print("  [dry-run] skipped")
            else:
                tag = f"seed={seed}/baselines"
                rc, elapsed = run_with_streaming(cmd, bl_log, tag)
                total_runs += 1
                if rc != 0:
                    print(f"  [FAIL] returncode={rc} ({elapsed:.1f}s)")
                    failed_runs.append(("baselines", seed, rc))
                else:
                    print(f"  [OK] completed in {elapsed:.1f}s")

        # ── LSTM / GRU ──
        if not args.skip_lstm_gru:
            lg_outdir = os.path.join(args.outdir_lstm_gru, f"seed_{seed}")
            os.makedirs(lg_outdir, exist_ok=True)
            lg_log = os.path.join(args.logdir, f"lstm_gru_seed_{seed}.log")

            cmd = build_command(
                python_exe,
                args.lstm_gru_script,
                lg_outdir,
                seed,
                args.w_seed,
                args.common_args,
                args.lstm_gru_extra,
                data_seed=args.data_seed,
                alpha_method=args.alpha_method,
                min_samples_alpha=args.min_samples_alpha,
            )

            print(f"\n[run] LSTM/GRU seed={seed}")
            print(f"  cmd: {cmd}")
            print(f"  log: {lg_log}")

            if args.dry_run:
                print("  [dry-run] skipped")
            else:
                tag = f"seed={seed}/lstm_gru"
                rc, elapsed = run_with_streaming(cmd, lg_log, tag)
                total_runs += 1
                if rc != 0:
                    print(f"  [FAIL] returncode={rc} ({elapsed:.1f}s)")
                    failed_runs.append(("lstm_gru", seed, rc))
                else:
                    print(f"  [OK] completed in {elapsed:.1f}s")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"Total runs: {total_runs}")
    if failed_runs:
        print(f"Failed: {len(failed_runs)}")
        for kind, s, rc in failed_runs:
            print(f"  - {kind} seed={s} returncode={rc}")
    else:
        print("All runs succeeded.")

    # Clean up __pycache__
    import shutil
    pycache = os.path.join(os.path.dirname(os.path.abspath(__file__)), '__pycache__')
    if os.path.isdir(pycache):
        shutil.rmtree(pycache)

    if failed_runs:
        sys.exit(1)


if __name__ == "__main__":
    main()
