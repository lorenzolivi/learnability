#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_all_multiseed.py — Run all multi-seed plotting scripts in sequence.

Usage:
  python plot_all_multiseed.py \
    --inputdirs results/T1024/baselines/adamw_lagmax256 results/T1024/lstm_gru/adamw_lagmax256 \
    --outdir results/T1024/together/adamw_lagmax256

  # Skip specific plots:
  python plot_all_multiseed.py \
    --inputdirs ... --outdir ... \
    --skip envelope,tau

  # Only run specific plots:
  python plot_all_multiseed.py \
    --inputdirs ... --outdir ... \
    --only hn,envelope
"""

import os
import sys
import argparse
import subprocess
import time
import datetime


PLOT_STEPS = [
    {
        "name": "hn",
        "label": "H_N curves (mean±std)",
        "script": "plot_empirical_learnability_win.py",
        "extra_args": [],
    },
    {
        "name": "hn_percentile",
        "label": "H_N curves (percentile)",
        "script": "plot_empirical_learnability_win.py",
        "extra_args": ["--hn_view", "percentile"],
    },
    {
        "name": "hn_boxplot",
        "label": "H_N curves (boxplot)",
        "script": "plot_empirical_learnability_win.py",
        "extra_args": ["--hn_view", "boxplot"],
    },
    {
        "name": "envelope",
        "label": "Envelope scaling",
        "script": "plot_envelope.py",
        "extra_args": ["--show_fits", "1", "--save_fits", "1", "--fit_tempered", "1"],
    },
    {
        "name": "tau",
        "label": "Tau distributions (CCDF)",
        "script": "plot_tau.py",
        "extra_args": ["--ccdf", "--logx", "--logy"],
        "outdir_suffix": "/figures",
    },
    {
        "name": "N_vs_envelope",
        "label": "Sample complexity (N vs envelope)",
        "script": "plot_N_vs_envelope.py",
        "extra_args": [],
    },
    {
        "name": "alpha",
        "label": "Alpha estimation distributions",
        "script": "plot_alpha_estimation.py",
        "extra_args": [],
    },
    {
        "name": "noise_floor",
        "label": "Noise floor / detectability",
        "script": "plot_noise_floor.py",
        "extra_args": ["--N_budgets", "500,8000"],
        "outdir_suffix": "/figures",
    },
    {
        "name": "learning_curves",
        "label": "Learning curves",
        "script": "plot_learnability_learning_curves.py",
        "extra_args": [],
        "outdir_suffix": "/plots_learning_curves",
    },
    {
        "name": "master_fit",
        "label": "Master proportionality law",
        "script": "fit_master_proportionality.py",
        "extra_args": [],
    },
]


def _timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run all multi-seed plotting scripts in sequence."
    )
    p.add_argument(
        "--inputdirs",
        type=str,
        nargs="+",
        required=True,
        help="Input directories (e.g., baselines/adamw lstm_gru/adamw)",
    )
    p.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Base output directory for all plots",
    )
    p.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma-separated list of plot names to skip (e.g., 'tau,noise_floor')",
    )
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated list of plot names to run (skip all others)",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available plot names and exit",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # List mode
    if args.list:
        print("Available plot steps:")
        for step in PLOT_STEPS:
            print(f"  {step['name']:20s}  {step['label']}")
        sys.exit(0)

    # Determine which steps to run
    skip_set = set(s.strip() for s in args.skip.split(",") if s.strip())
    only_set = set(s.strip() for s in args.only.split(",") if s.strip())

    steps_to_run = []
    for step in PLOT_STEPS:
        if only_set and step["name"] not in only_set:
            continue
        if step["name"] in skip_set:
            continue
        steps_to_run.append(step)

    if not steps_to_run:
        print("[error] No plot steps selected. Use --list to see available names.")
        sys.exit(1)

    # Resolve script directory (same folder as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = sys.executable
    inputdirs_str = " ".join(args.inputdirs)

    print(f"[{_timestamp()}] plot_all_multiseed.py")
    print(f"[info] inputdirs: {args.inputdirs}")
    print(f"[info] outdir: {args.outdir}")
    print(f"[info] steps: {[s['name'] for s in steps_to_run]}")
    print()

    os.makedirs(args.outdir, exist_ok=True)

    n_ok = 0
    n_fail = 0
    failed = []

    for i, step in enumerate(steps_to_run, 1):
        name = step["name"]
        label = step["label"]
        script = os.path.join(script_dir, step["script"])
        outdir = args.outdir + step.get("outdir_suffix", "")

        os.makedirs(outdir, exist_ok=True)

        cmd_parts = [
            python_exe, script,
            "--inputdirs", *args.inputdirs,
            "--outdir", outdir,
            *step["extra_args"],
        ]
        cmd_str = " ".join(cmd_parts)

        print(f"{'='*60}")
        print(f"  [{i}/{len(steps_to_run)}] {label}  ({name})")
        print(f"{'='*60}")
        print(f"  cmd: {cmd_str}")

        if args.dry_run:
            print(f"  [dry-run] skipped")
            print()
            continue

        t0 = time.time()
        print(f"  [{_timestamp()}] started...")

        proc = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
        )

        elapsed = time.time() - t0

        if proc.returncode != 0:
            n_fail += 1
            failed.append(name)
            print(f"  [{_timestamp()}] FAILED (returncode={proc.returncode}, {elapsed:.1f}s)")
            # Print stderr/stdout for debugging
            if proc.stderr:
                for line in proc.stderr.strip().split("\n")[-10:]:
                    print(f"    stderr: {line}")
            if proc.stdout:
                for line in proc.stdout.strip().split("\n")[-5:]:
                    print(f"    stdout: {line}")
        else:
            n_ok += 1
            print(f"  [{_timestamp()}] OK ({elapsed:.1f}s)")
            # Print last few lines of stdout for confirmation
            if proc.stdout:
                for line in proc.stdout.strip().split("\n")[-3:]:
                    print(f"    {line}")

        print()

    # Summary
    print(f"{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    total = n_ok + n_fail
    print(f"  Total: {total}   OK: {n_ok}   Failed: {n_fail}")
    if failed:
        print(f"  Failed steps: {', '.join(failed)}")
    else:
        print(f"  All plots generated successfully.")
    print(f"  Output: {os.path.abspath(args.outdir)}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
