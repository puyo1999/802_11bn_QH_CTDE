"""Summary of E3 (MAPPO, 3 drops per condition): mean over drops and paired differences with 95% CI."""
import json, os
import numpy as np
from scipy import stats

R = os.path.dirname(os.path.abspath(__file__))
KEYS = ["full_reuse|ideal", "greedy_muting|ideal", "causal_full_reuse|ideal", "causal_greedy|ideal",
        "mappo|ideal|argmax", "oracle|olla", "causal_full_reuse|olla", "causal_greedy|olla", "mappo|olla|argmax",
        "mappo|ideal|sampled"]
out = {}


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return float(x.mean()), float(h), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if x.std(ddof=1) > 0 else np.nan


print(f"{'d':>2} {'v':>3} | " + " ".join(f"{k.replace('|ideal','').replace('|argmax','').replace('causal_','c_')[:13]:>13}" for k in KEYS))
for d in (0, 4, 8):
    for v in (3, 30, 60):
        t = json.load(open(os.path.join(R, f"e2_D3_delay{d}_v{v}_E300_g0.json")))["table"]
        drops = sorted(t["oracle|ideal"])
        val = {k: np.array([t[k][x] for x in drops]) for k in KEYS}
        best_causal = np.maximum(val["causal_full_reuse|ideal"], val["causal_greedy|ideal"])
        row = {k: ci(val[k]) for k in KEYS}
        row["mappo_minus_cfr"] = ci(val["mappo|ideal|argmax"] - val["causal_full_reuse|ideal"])
        row["mappo_minus_cg"] = ci(val["mappo|ideal|argmax"] - val["causal_greedy|ideal"])
        row["mappo_minus_best_causal"] = ci(val["mappo|ideal|argmax"] - best_causal)
        best_causal_olla = np.maximum(val["causal_full_reuse|olla"], val["causal_greedy|olla"])
        row["mappo_olla_minus_best_causal_olla"] = ci(val["mappo|olla|argmax"] - best_causal_olla)
        row["sampled_minus_argmax"] = ci(val["mappo|ideal|sampled"] - val["mappo|ideal|argmax"])
        row["sampled_minus_best_causal"] = ci(val["mappo|ideal|sampled"] - best_causal)
        row["olla_gap_mappo"] = ci(val["mappo|ideal|argmax"] - val["mappo|olla|argmax"])
        row["olla_gap_oracle"] = ci(100 - val["oracle|olla"])
        row["fr_minus_cfr"] = ci(val["full_reuse|ideal"] - val["causal_full_reuse|ideal"])
        out[f"d{d}_v{v}"] = row
        print(f"{d:>2} {v:>3} | " + " ".join(f"{row[k][0]:>13.1f}" for k in KEYS))
print()
print(f"{'d':>2} {'v':>3} | {'MAPPO-cFR':>16} {'MAPPO-cG':>16} {'MAPPO-bestc':>16} {'OLLA gap MAPPO':>16} {'OLLA gap orc':>14} {'FR-cFR':>14}")
for key, row in out.items():
    d, v = key[1:].split("_v")
    f = lambda m: f"{m[0]:+6.1f}+-{m[1]:4.1f}(t{m[2]:+5.1f})"
    print(f"{d:>2} {v:>3} | {f(row['mappo_minus_cfr']):>16} {f(row['mappo_minus_cg']):>16} {f(row['mappo_minus_best_causal']):>16} "
          f"{row['olla_gap_mappo'][0]:>8.1f}+-{row['olla_gap_mappo'][1]:4.1f} {row['olla_gap_oracle'][0]:>8.1f}+-{row['olla_gap_oracle'][1]:4.1f} "
          f"{row['fr_minus_cfr'][0]:>8.1f}+-{row['fr_minus_cfr'][1]:4.1f}")
json.dump(out, open(os.path.join(R, "e3_summary.json"), "w"), indent=1)
print()
print(f"{'d':>2} {'v':>3} | {'MAPPOolla-bestc_olla':>22} {'sampled-argmax':>22} {'sampled-bestc':>22}")
for key, row in out.items():
    d, v = key[1:].split("_v")
    f = lambda m: f"{m[0]:+6.1f}+-{m[1]:4.1f}(t{m[2]:+5.1f})"
    print(f"{d:>2} {v:>3} | {f(row['mappo_olla_minus_best_causal_olla']):>22} {f(row['sampled_minus_argmax']):>22} {f(row['sampled_minus_best_causal']):>22}")
json.dump(out, open(os.path.join(R, "e3_summary.json"), "w"), indent=1)
