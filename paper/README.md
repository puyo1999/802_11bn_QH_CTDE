# Paper draft — Exact-oracle gap analysis (IEEE Access)

`main.tex` is a draft for IEEE Access. Every number in it comes from the committed result files below;
items still open are marked in red with `\todo{...}`.

## Build

1. Download the IEEE Access LaTeX template from the IEEE Author Center and copy `ieeeaccess.cls`,
   `spotcolor.sty`, `IEEEtran.bst` and the logo files (`logo.png`, `notaglinelogo.png`, `bullet.png`) into `paper/`.
2. Regenerate the figure and table rows from the result CSVs:
   ```
   python paper/figures/make_figures.py
   ```
3. Compile: `pdflatex main && bibtex main && pdflatex main && pdflatex main` (run inside `paper/`).

## Data sources

| Paper item | Source | Produced by |
|---|---|---|
| Fig. 1 (headroom vs load) | `experiments/oracle_gap_results_envv2_headroom/oracle_gap.csv` (7 loads × 6 topologies × 3 seeds) | `experiments/oracle_gap.py --learners --loads 0.25 0.5 1 2 3 5 8 --seeds 3` |
| Tables I–II (gap decomposition, λ = 5 and 8) | `experiments/oracle_gap_results_envv2_full/jobs/*.json` (5 seeds, 1000 episodes) | `experiments/oracle_gap.py --loads 5 8 --topos T1 T3 T4 T5 T6 --seeds 5 --learners mappo central --mcs learned arf genie --episodes 1000 --out experiments/oracle_gap_results_envv2_full` |

## Open items before submission

- λ = 3 learner runs were dropped as low-impact. The single-seed pilot (`experiments/oracle_gap_results_envv2_mcs_pilot/`) is superseded by the 5-seed results.
- Verify every entry of `references.bib` against publisher records; complete `yan2025marl`.
- Related work on Co-SR / Co-TDMA analysis and Wi-Fi MARL for spatial reuse.
- System-model figure (topologies T1–T6).
- Fading channel model or a justification of the deterministic path-loss model.
- Author, affiliation, funding, biography.
