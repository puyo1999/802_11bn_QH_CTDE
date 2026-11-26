"""Statistics of the full E1 runs for the paper: bootstrap 95% CIs over drops (all seeds and loads of a drop are
resampled together), for the throughput objective (e1_U4_D200_S3.csv) and the max-weight objective
(e1_maxweight_U4_D200_S1.csv, packet delays)."""
import csv, json, os
import numpy as np

R = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
B, STABLE = 2000, 0.01
POL = ["oracle", "full_reuse", "cluster_tdma", "greedy_muting"]


def load(name):
    rows = list(csv.DictReader(open(os.path.join(R, name))))
    by = {}
    for r in rows:
        by.setdefault((int(r["drop"]), int(r["seed"]), float(r["eta"])), {})[r["policy"]] = {k: float(v) for k, v in r.items() if k not in ("policy",)}
    etas = sorted({k[2] for k in by})
    cap = {(d, s): v["cluster_tdma"]["throughput"] for (d, s, e), v in by.items() if e == etas[-1]}
    recs = []
    for (d, s, e), v in by.items():
        o = v["oracle"]
        rec = {"drop": d, "eta": e, "H": 1 - v["cluster_tdma"]["throughput"] / o["throughput"],
               "rho": o["throughput"] / o["delivery"] / cap[(d, s)]}
        for p in POL:
            for k in ("throughput", "delivery", "delay_median", "delay_p95", "delay_censored"):
                rec[f"{k}_{p}"] = v[p][k]
        recs.append(rec)
    return recs, etas


def arr(recs, key):
    return np.array([r[key] for r in recs])


def boot(recs, fn):
    if not recs:
        return (np.nan, np.nan, np.nan)
    groups = {}
    for r in recs:
        groups.setdefault(r["drop"], []).append(r)
    keys = list(groups)
    reps = []
    for _ in range(B):
        sample = [r for k in rng.choice(keys, len(keys)) for r in groups[k]]
        reps.append(fn(sample))
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return (float(fn(recs)), float(lo), float(hi))


pct = lambda t: f"{100*t[0]:5.1f} [{100*t[1]:5.1f},{100*t[2]:5.1f}]"
out = {}
w, etas = load("e1_U4_D200_S3.csv")
sat = [r for r in w if r["eta"] == etas[-1]]
ratio = lambda rs: float(np.median(arr(rs, "throughput_oracle") / arr(rs, "throughput_cluster_tdma")))
rr = arr(sat, "throughput_oracle") / arr(sat, "throughput_cluster_tdma")
out["sat_ratio"] = {"median": boot(sat, ratio), "min": float(rr.min()), "max": float(rr.max())}
print("throughput objective, 200 drops x 3 seeds")
print(f"saturated oracle/TDMA median {out['sat_ratio']['median']} range [{rr.min():.2f},{rr.max():.2f}]")
out["by_eta"] = {}
for eta in etas:
    g = [r for r in w if r["eta"] == eta]
    e = {"H_median": boot(g, lambda rs: float(np.median(arr(rs, "H")))),
         "H_mean": boot(g, lambda rs: float(arr(rs, "H").mean())),
         "H_pos_share": float((arr(g, "H") > 0).mean()),
         "deliv_oracle": boot(g, lambda rs: float(arr(rs, "delivery_oracle").mean())),
         "deliv_tdma": boot(g, lambda rs: float(arr(rs, "delivery_cluster_tdma").mean())),
         "rel_full_reuse": float((arr(g, "throughput_full_reuse") / arr(g, "throughput_oracle")).mean()),
         "rel_greedy": float((arr(g, "throughput_greedy_muting") / arr(g, "throughput_oracle")).mean())}
    out["by_eta"][str(eta)] = e
    print(f"eta {eta:<6g} H med {pct(e['H_median'])} mean {pct(e['H_mean'])} H>0 {100*e['H_pos_share']:5.1f}% "
          f"deliv orc {pct(e['deliv_oracle'])} TDMA {pct(e['deliv_tdma'])} FR {100*e['rel_full_reuse']:5.1f} GM {100*e['rel_greedy']:5.1f}")
out["by_rho"] = {}
for lo, hi in ((0, 0.5), (0.5, 0.9), (0.9, 1.1), (1.1, 2), (2, 5), (5, 1e9)):
    g = [r for r in w if lo <= r["rho"] < hi]
    e = {"n": len(g), "H_median": boot(g, lambda rs: float(np.median(arr(rs, "H"))))}
    out["by_rho"][f"{lo}-{hi:g}"] = e
    print(f"rho {lo}-{hi:g}: n {len(g)} H med {pct(e['H_median'])}")

if os.path.exists(os.path.join(R, "e1_maxweight_U4_D200_S1.csv")):
    m, metas = load("e1_maxweight_U4_D200_S1.csv")
    print("\nmax-weight objective, 200 drops x 1 seed")
    out["maxweight"] = {"by_eta": {}}
    st = lambda p: (lambda rs: float((arr(rs, f"delay_censored_{p}") < STABLE).mean()))
    for eta in metas:
        g = [r for r in m if r["eta"] == eta]
        both = [r for r in g if r["delay_censored_oracle"] < STABLE and r["delay_censored_cluster_tdma"] < STABLE]
        e = {f"stable_{p}": boot(g, st(p)) for p in POL}
        e.update({"H_median": boot(g, lambda rs: float(np.median(arr(rs, "H")))), "n_both": len(both)})
        if len(both) >= 10:
            for k in ("delay_median", "delay_p95"):
                for p in ("oracle", "cluster_tdma", "greedy_muting"):
                    e[f"{k}_{p}"] = boot(both, lambda rs, k=k, p=p: float(np.median(arr(rs, f"{k}_{p}"))))
            e["p95_ratio"] = boot(both, lambda rs: float(np.median(arr(rs, "delay_p95_cluster_tdma") / arr(rs, "delay_p95_oracle"))))
            e["med_ratio"] = boot(both, lambda rs: float(np.median(arr(rs, "delay_median_cluster_tdma") / arr(rs, "delay_median_oracle"))))
        out["maxweight"]["by_eta"][str(eta)] = e
        line = (f"eta {eta:<6g} stable orc {pct(e['stable_oracle'])} TDMA {pct(e['stable_cluster_tdma'])} "
                f"FR {100*e['stable_full_reuse'][0]:5.1f} GM {100*e['stable_greedy_muting'][0]:5.1f} | H med {pct(e['H_median'])} | both {len(both)}")
        if "p95_ratio" in e:
            f = lambda t: f"{t[0]:.0f} [{t[1]:.0f},{t[2]:.0f}]"
            line += (f"\n      median delay orc {f(e['delay_median_oracle'])} TDMA {f(e['delay_median_cluster_tdma'])} GM {f(e['delay_median_greedy_muting'])}"
                     f" | p95 orc {f(e['delay_p95_oracle'])} TDMA {f(e['delay_p95_cluster_tdma'])} GM {f(e['delay_p95_greedy_muting'])}"
                     f" | ratio med {e['med_ratio'][0]:.1f} p95 {e['p95_ratio'][0]:.1f} [{e['p95_ratio'][1]:.1f},{e['p95_ratio'][2]:.1f}]")
        print(line)
    grid = metas[:-1]
    def stable_load(p):
        res = []
        for d in sorted({r["drop"] for r in m}):
            ok = {r["eta"]: r[f"delay_censored_{p}"] < STABLE for r in m if r["drop"] == d}
            k = 0.0
            for eta in grid:
                if ok.get(eta): k = eta
                else: break
            res.append(k)
        return np.array(res)
    so, stt = stable_load("oracle"), stable_load("cluster_tdma")
    sl = {"oracle_median": float(np.median(so)), "tdma_median": float(np.median(stt)),
          "oracle_hist": {str(e): int((so == e).sum()) for e in [0.0] + grid},
          "tdma_hist": {str(e): int((stt == e).sum()) for e in [0.0] + grid},
          "oracle_gt_tdma_share": float((so > stt).mean()), "equal_share": float((so == stt).mean()),
          "ratio_median": float(np.median(so[stt > 0] / stt[stt > 0]))}
    out["maxweight"]["stable_load"] = sl
    print("largest stable load per drop:", json.dumps(sl))
json.dump(out, open(os.path.join(R, "e1_stats.json"), "w"), indent=1)
