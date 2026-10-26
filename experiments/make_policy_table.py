"""Print the rows of tab:policy_comparison (LaTeX) and the derived numbers quoted in Section V-C.

Inputs
  results_csv/policy_comparison/policy_comparison_summary.csv   baselines and learned policies
  results_csv/harsh_tuning/h5_all/h5_summary.csv                 learned policies with ARF MCS
Every number is a mean over the evaluation seeds; nothing is rescaled.
"""
import csv
import sys
from collections import defaultdict

PC = "results_csv/policy_comparison/policy_comparison_summary.csv"
H5 = "results_csv/harsh_tuning/h5_all/h5_summary.csv"
TOPOS = ["T1", "T2", "T3", "T4", "T5", "T6"]


def load():
    val = defaultdict(list)    # (policy, topo) -> [(total, harsh, blocked)]
    for r in csv.DictReader(open(PC, encoding="utf-8")):
        val[(r["policy"], r["topology"])].append(
            (float(r["throughput_packets"]), float(r["throughput_harsh"]), float(r["blocked_per_step"])))
    for r in csv.DictReader(open(H5, encoding="utf-8")):
        if r["mode"] == "arf":
            val[(r["condition"] + "+ARF", r["topology"])].append(
                (float(r["throughput_packets"]), float(r["throughput_harsh"]), float("nan")))
    return val


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    val = load()
    rows = [("random", "Random"), ("greedy", "Greedy (ARF)"), ("round_robin", "Round-robin (ARF)"),
            ("C2", "C2 (CTDE)"), ("C4", "\\textbf{C4 (QH-CTDE)}"),
            ("C2+ARF", "C2 + ARF"), ("C4+ARF", "\\textbf{C4 + ARF}"), ("co_tdma", "Co-TDMA"),
            ("oracle", "All-transmit SINR heur."), ("exact_oracle", "Exact oracle (bound)")]
    for k, title in ((0, "Total"), (1, "Harsh regime")):
        print(f"\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for p, label in rows:
            cells = []
            for t in TOPOS:
                v = val.get((p, t))
                cells.append("--" if not v else f"{mean([x[k] for x in v]):.0f}")
            print(f"{label:24s} & " + " & ".join(cells) + " \\\\")
        print("\\midrule")
    print("\n% n seeds:", {p: [len(val.get((p, t), [])) for t in TOPOS] for p, _ in rows})
    # Normalise by the exact per-slot oracle (an upper bound); the all-transmit "oracle" is not one.
    ref = "exact_oracle" if any(("exact_oracle", t) in val for t in TOPOS) else "oracle"
    print(f"\n% percent of {ref} (total) and blocked/step")
    for t in TOPOS:
        o = mean([x[0] for x in val[(ref, t)]])
        parts = []
        for p in ("round_robin", "greedy", "co_tdma", "oracle", "C2", "C4", "C2+ARF", "C4+ARF"):
            v = val.get((p, t))
            if v:
                parts.append(f"{p}={100 * mean([x[0] for x in v]) / o:.1f}%")
        bl = {p: round(mean([x[2] for x in val[(p, t)]]), 3) for p in ("round_robin", "C2", "C4")
              if val.get((p, t))}
        print(f"% {t}: oracle={o:.0f} " + " ".join(parts) + f" blocked={bl}")
    print("\n% per-seed harsh throughput of C2/C4 and +ARF")
    for t in TOPOS:
        print(f"% {t}: " + " | ".join(f"{p}: " + ",".join(f"{x[1]:.0f}" for x in val.get((p, t), []))
                                      for p in ("C2", "C4", "C2+ARF", "C4+ARF")))


if __name__ == "__main__":
    sys.exit(main())
