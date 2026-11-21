"""
experiments/la1_stats.py
Statistics of LA1 over drops (experiments/cellular_results/la1_D20_S1.csv):

  1. mean and 95% confidence interval (t over drops) of the geometric-mean rate relative to the oracle with genie
     CQI, per target selection, condition and CQI variant;
  2. paired differences over drops of ET selection against tuned OLLA, with 95% CI and t-statistic;
  3. OLLA with a target error rate tuned out of sample: two-fold cross-fitting over drops (target chosen on one
     half, evaluated on the other), against the in-sample choice of la1 that favors OLLA;
  4. bootstrap 95% CI over drops of the recovered share of the link-adaptation loss.

    python experiments/la1_stats.py
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cellular_results")
SRC = os.path.join(OUT, "la1_D20_S1.csv")
TARGETS = ["olla0.1", "olla0.2", "olla0.3", "olla0.5"]
ET = ["et_estimated", "et_multilag", "et_known", "et_signal", "et_calibrated"]
DELAYS, SPEEDS = [0, 4, 8], [3.0, 30.0, 60.0]


def load() -> dict:
    """{(selector, delay, speed, variant): {drop: relative geometric-mean rate in %}}."""
    with open(SRC, newline="") as handle:
        rows = list(csv.DictReader(handle))
    ref = {(r["drop"], r["delay"], r["speed"]): float(r["log_utility"]) for r in rows
           if r["selector"] == "oracle" and r["variant"] == "genie"}
    rel = defaultdict(dict)
    for r in rows:
        base = ref[(r["drop"], r["delay"], r["speed"])]
        rel[(r["selector"], int(r["delay"]), float(r["speed"]), r["variant"])][int(r["drop"])] = \
            100 * np.exp((float(r["log_utility"]) - base) / int(r["n_ues"]))
    return rel


def ci(x: np.ndarray) -> tuple[float, float]:
    """Mean and half-width of the 95% t confidence interval."""
    return float(np.mean(x)), float(stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x)))


def main() -> None:
    rel = load()
    drops = sorted(next(iter(rel.values())))
    arr = {k: np.array([v[d] for d in drops]) for k, v in rel.items()}
    rng = np.random.default_rng(0)
    halves = [drops[: len(drops) // 2], drops[len(drops) // 2:]]
    out = {}
    for selector in ("causal_greedy", "oracle"):
        print(f"\n== {selector}: mean +- 95% CI over {len(drops)} drops (geometric-mean rate / oracle, %)")
        for delay in DELAYS:
            for speed in SPEEDS:
                g = {v: arr[(selector, delay, speed, v)] for v in ["genie", "fresh"] + TARGETS + ET}
                # in-sample tuned target (as in la1) and two-fold cross-fitted target
                best_in = max(TARGETS, key=lambda v: g[v].mean())
                cross = np.empty(len(drops))
                chosen = []
                for fit, ev in ((halves[0], halves[1]), (halves[1], halves[0])):
                    t = max(TARGETS, key=lambda v: g[v][fit].mean())
                    chosen.append(t)
                    cross[ev] = g[t][ev]
                g["olla_tuned"], g["olla_cross"] = g[best_in], cross
                entry = {"tuned_in_sample": best_in, "tuned_cross": chosen,
                         "mean_ci": {v: ci(x) for v, x in g.items()}}
                # paired differences against tuned OLLA (in sample and cross-fitted)
                entry["paired_vs_tuned"] = {}
                for v in ET:
                    for base in ("olla_tuned", "olla_cross"):
                        diff = g[v] - g[base]
                        t_stat = float(stats.ttest_1samp(diff, 0.0).statistic)
                        entry["paired_vs_tuned"][f"{v}-{base}"] = {"mean_ci": ci(diff), "t": t_stat}
                # bootstrap CI of the recovered share of the LA loss (ratio of means over resampled drops)
                idx = rng.integers(0, len(drops), (2000, len(drops)))
                entry["recovered_share_ci"] = {}
                for v in ["olla_tuned", "olla_cross"] + ET:
                    num = (g[v][idx] - g["olla0.1"][idx]).mean(axis=1)
                    den = (g["genie"][idx] - g["olla0.1"][idx]).mean(axis=1)
                    share = num / den
                    point = (g[v].mean() - g["olla0.1"].mean()) / (g["genie"].mean() - g["olla0.1"].mean())
                    entry["recovered_share_ci"][v] = [float(point), float(np.percentile(share, 2.5)),
                                                      float(np.percentile(share, 97.5))]
                out[f"{selector}|d={delay}|v={speed:g}"] = entry
                m = entry["mean_ci"]
                pv = entry["paired_vs_tuned"]
                print(f"d={delay} v={speed:>2g}  OLLA {m['olla0.1'][0]:5.1f}+-{m['olla0.1'][1]:.1f}  "
                      f"tuned {m['olla_tuned'][0]:5.1f}+-{m['olla_tuned'][1]:.1f} ({best_in})  "
                      f"cross {m['olla_cross'][0]:5.1f}+-{m['olla_cross'][1]:.1f} ({'/'.join(chosen)})  "
                      f"ET {m['et_estimated'][0]:5.1f}+-{m['et_estimated'][1]:.1f}  "
                      f"ET-tuned {pv['et_estimated-olla_tuned']['mean_ci'][0]:+5.1f}"
                      f"+-{pv['et_estimated-olla_tuned']['mean_ci'][1]:.1f} (t={pv['et_estimated-olla_tuned']['t']:.1f})  "
                      f"share ET {100*entry['recovered_share_ci']['et_estimated'][0]:.0f}% "
                      f"[{100*entry['recovered_share_ci']['et_estimated'][1]:.0f}, "
                      f"{100*entry['recovered_share_ci']['et_estimated'][2]:.0f}]")
    with open(os.path.join(OUT, "la1_D20_S1_stats.json"), "w") as handle:
        json.dump(out, handle, indent=2)


if __name__ == "__main__":
    main()
