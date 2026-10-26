"""Run the paired C1-C5 QH-CTDE ablation matrix.

The condition is passed explicitly to train_ablation.py.  This is important:
without --condition, train_ablation.py runs all five conditions, which would
turn a 25-run matrix into 125 condition-runs.

Default design: 5 conditions x 5 identical seeds x selected topologies.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from multiprocessing import Pool


CONDITIONS = ["C1", "C2", "C3", "C4", "C5"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]


def run_single_job(job):
    condition, seed, config, steps, output_dir = job
    cmd = [
        sys.executable,
        "train_ablation.py",
        "--config", config,
        "--condition", condition,
        "--seed", str(seed),
        "--steps", str(steps),
        "--output_dir", output_dir,
    ]
    print(
        f"[START] condition={condition} seed={seed} config={config}",
        flush=True,
    )
    subprocess.run(cmd, check=True)
    print(
        f"[COMPLETE] condition={condition} seed={seed}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/qh_ctde.yaml",
        help="One common base config for all C1-C5 conditions.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS,
        default=CONDITIONS,
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Training-step budget passed to train_ablation.py.",
    )
    parser.add_argument(
        "--output-dir", default="./results_csv",
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Number of concurrent Python training processes.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    jobs = [
        (condition, seed, args.config, args.steps, args.output_dir)
        for condition in args.conditions
        for seed in args.seeds
    ]

    print(
        f"Total workload: {len(jobs)} runs "
        f"({len(args.conditions)} conditions x {len(args.seeds)} seeds).",
        flush=True,
    )
    print(
        "Each run receives exactly one --condition; "
        "therefore no accidental 5x duplication occurs.",
        flush=True,
    )

    # The jobs are independent, but the same seed is intentionally reused
    # across conditions so the later analysis can perform paired comparisons.
    with Pool(processes=min(args.workers, len(jobs))) as pool:
        pool.map(run_single_job, jobs)

    print("All requested ablation runs completed successfully.", flush=True)


if __name__ == "__main__":
    main()
