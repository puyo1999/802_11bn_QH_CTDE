"""
experiments/ablation.py
Ablation study: n_heads, hidden_dim, history_len 영향 분석

Usage:
    python experiments/ablation.py --config configs/drlca_attn.yaml
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import itertools
import numpy as np
import yaml

from src.envs.topology import topo_cw1, topo_cw3
from experiments.train import load_config, train_on_topology

ABLATION_GRID = {
    "n_heads":      [1, 2, 4],
    "hidden_dim":   [16, 32, 64],
    "history_len":  [5, 10, 20],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/drlca_attn.yaml")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    mode = cfg["env"].get("mode", "cw")
    topo = topo_cw1()   # 대표 토폴로지 Topo1

    results = []
    keys   = list(ABLATION_GRID.keys())
    values = list(ABLATION_GRID.values())

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        # cfg 패치
        for k, v in params.items():
            cfg["agent"][k] = v

        summary = train_on_topology(topo, cfg, mode, run_id=0)
        row = {**params,
               "d95_ms": summary["d95_mean"] * 1000,
               "thp":    summary["thp_mean"],
               "per":    summary["per_mean"]}
        results.append(row)
        print(f"  {params}  →  d95={row['d95_ms']:.1f}ms")

    # 결과 저장
    os.makedirs("results", exist_ok=True)
    import csv
    with open("results/ablation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\nAblation 결과 저장: results/ablation.csv")


if __name__ == "__main__":
    main()
