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

## Status

- `references.bib`: all 20 entries verified on 2026-10-26 (source noted per entry). Pages for `vandenoord2017vqvae`
  are omitted because only a secondary record lists them. Three related-work entries are arXiv preprints with no
  journal or conference version found (`wilhelmi2023csr`, `nunez2024csr`, `yu2025hmarl`).
- Related work covers Co-SR analysis, ML for Wi-Fi, learned MAPC scheduling, and the MILP upper bound of Wojnar et al.
- Fig. `fig_topologies` (T1–T6) is generated from `configs/topologies.yaml`.
- λ = 3 learner runs were dropped as low-impact. The single-seed pilot (`experiments/oracle_gap_results_envv2_mcs_pilot/`) is superseded by the 5-seed results.

## Open items before submission

- Rayleigh fading robustness results (sweep in `experiments/oracle_gap_results_envv2_fading/`) and the corresponding paper section.
- Author, affiliation, funding, biography.
