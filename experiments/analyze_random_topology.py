"""
experiments/analyze_random_topology.py
Summary statistics of the random-topology experiment.

Stage 1: coordination headroom H = 1 - Co-TDMA / oracle and the all-transmit SINR reference over
200 topologies, with an OLS fit of H on the standardized design variables.
Stage 2: gap decomposition of the learners over 20 topologies with paired t-statistics.

    python experiments/analyze_random_topology.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESIGN = yaml.safe_load(open(os.path.join(ROOT, "configs/topologies_random.yaml")))["topologies"]
VARS = ["d", "r_ratio", "n_aps", "overlap", "stas_per_ap"]


def load_rows(stage: str) -> list[dict]:
    rows = []
    for path in glob.glob(os.path.join(ROOT, f"experiments/oracle_gap_results_random_{stage}/jobs/*.json")):
        rows += json.load(open(path))["rows"]
    return rows


def per_topology(rows: list[dict]) -> dict:
    """{(load, topo): {policy: mean throughput / oracle over seeds}}."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["load"], r["topology"], r["seed"])][r["policy"]] = r["throughput"]
    out = defaultdict(lambda: defaultdict(list))
    for (load, topo, _), pol in by.items():
        ref = pol["exact_oracle"]
        for name, thp in pol.items():
            out[(load, topo)][name].append(thp / ref)
    return {k: {p: float(np.mean(v)) for p, v in d.items()} for k, d in out.items()}


def tstat(x: np.ndarray) -> float:
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def stage1() -> None:
    ratios = per_topology(load_rows("stage1"))
    loads = sorted({k[0] for k in ratios})
    print("== Stage 1: coordination headroom over 200 random topologies ==")
    for load in loads:
        topos = sorted(t for (l, t) in ratios if l == load)
        H = np.array([1 - ratios[(load, t)]["co_tdma"] for t in topos])
        sinr = np.array([ratios[(load, t)]["oracle"] for t in topos])
        print(f"λ={load}: n={len(H)} H median {100*np.median(H):.1f}% IQR [{100*np.percentile(H,25):.1f}, "
              f"{100*np.percentile(H,75):.1f}] max {100*H.max():.1f}% | share H<1%: {100*np.mean(H<0.01):.0f}% "
              f"H>10%: {100*np.mean(H>0.10):.0f}% | all-transmit min {100*sinr.min():.0f}% median {100*np.median(sinr):.0f}%")
        X = np.array([[DESIGN[t]["design"][v] for v in VARS] for t in topos], dtype=float)
        Z = (X - X.mean(0)) / X.std(0)
        A = np.column_stack([np.ones(len(Z)), Z])
        coef, *_ = np.linalg.lstsq(A, H, rcond=None)
        resid = H - A @ coef
        r2 = 1 - resid.var() / H.var()
        se = np.sqrt(np.diag(resid.var(ddof=A.shape[1]) * np.linalg.inv(A.T @ A)))
        print("   OLS H ~ z(vars): " + ", ".join(f"{v} {100*c:+.1f}pt (t {c/s:+.1f})"
                                               for v, c, s in zip(VARS, coef[1:], se[1:])) + f", R2 {r2:.2f}")
        for n in sorted({DESIGN[t]["design"]["n_aps"] for t in topos}):
            sel = [h for h, t in zip(H, topos) if DESIGN[t]["design"]["n_aps"] == n]
            print(f"   N={n}: H median {100*np.median(sel):.1f}%")


def stage2() -> None:
    rows = load_rows("stage2")
    if not rows:
        print("stage 2: no results yet")
        return
    ratios = per_topology(rows)
    print("\n== Stage 2: gap decomposition over random topologies (sampled actions) ==")
    for load in sorted({k[0] for k in ratios}):
        topos = sorted(t for (l, t) in ratios if l == load and "central+genie/sampled" in ratios[(l, t)])
        if not topos:
            continue
        get = lambda p: np.array([ratios[(load, t)][p] for t in topos])
        cot = get("co_tdma")
        print(f"λ={load}: n={len(topos)} topologies, H mean {100*(1-cot).mean():.1f}%")
        for key in ("mappo", "central"):
            for mcs in ("learned", "genie"):
                x = get(f"{key}+{mcs}/sampled")
                print(f"   {key}+{mcs}: {100*x.mean():.1f}% ± {100*x.std(ddof=1):.1f}")
        for key in ("mappo", "central"):
            d = get(f"{key}+genie/sampled") - get(f"{key}+learned/sampled")
            print(f"   ΔLA {key}: {100*d.mean():+.1f}pt (t {tstat(d):+.1f})")
        for mcs in ("learned", "genie"):
            d = get(f"central+{mcs}/sampled") - get(f"mappo+{mcs}/sampled")
            print(f"   ΔI {mcs}: {100*d.mean():+.1f}pt (t {tstat(d):+.1f})")
        opt = 1 - get("central+genie/sampled")
        print(f"   Δopt: {100*opt.mean():.1f}%")
        d = get("central+genie/sampled") - cot
        print(f"   central+genie - Co-TDMA: {100*d.mean():+.1f}pt (t {tstat(d):+.1f}), above in "
              f"{int((d > 0).sum())}/{len(d)} topologies")


if __name__ == "__main__":
    stage1()
    stage2()
