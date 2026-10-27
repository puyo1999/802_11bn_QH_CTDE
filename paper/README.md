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
| Fig. `fig_learning_curves` (training convergence, T3/T6) | training curves in `experiments/oracle_gap_results_envv2_full/jobs/*.json` | `paper/figures/make_figures.py` |
| Tables I–II (gap decomposition, λ = 5 and 8) | `experiments/oracle_gap_results_envv2_full/jobs/*.json` (5 seeds, 1000 episodes) | `experiments/oracle_gap.py --loads 5 8 --topos T1 T3 T4 T5 T6 --seeds 5 --learners mappo central --mcs learned arf genie --episodes 1000 --out experiments/oracle_gap_results_envv2_full` |

## Status

- `references.bib`: all 20 entries verified on 2026-10-26 (source noted per entry). Pages for `vandenoord2017vqvae`
  are omitted because only a secondary record lists them. Three related-work entries are arXiv preprints with no
  journal or conference version found (`wilhelmi2023csr`, `nunez2024csr`, `yu2025hmarl`).
- Related work covers Co-SR analysis, ML for Wi-Fi, learned MAPC scheduling, and the MILP upper bound of Wojnar et al.
- Fig. `fig_topologies` (T1–T6) is generated from `configs/topologies.yaml`.
- λ = 3 learner runs were dropped as low-impact. The single-seed pilot (`experiments/oracle_gap_results_envv2_mcs_pilot/`) is superseded by the 5-seed results.

- Rayleigh fading robustness (Table III): `experiments/oracle_gap_results_envv2_fading/` (T3, T6 × λ ∈ {5, 8} × 3 seeds)
  compared with the deterministic channel on the same seeds; baseline headroom sweep under fading in
  `experiments/oracle_gap_results_envv2_fading_headroom/`.

- Significance: differences from Co-TDMA are reported with t-statistics over the five per-seed differences
  (T3 at λ = 5 is not significant, t = −0.93). Learned-MCS variants are still improving at 1000 episodes
  (episodes 900–999 vs 500–599: +9.7% MAPPO, +9.6% Central PPO), so Δ_LA is reported at a fixed training budget.
- Oracle runtime (Python, one core, 2.8 GHz Xeon, λ = 5): 1.3 / 3.0 / 5.8 ms per slot in T5 / T3 / T6.

## Open items before submission

- Random-topology generalization (plan: `docs/random_topology_experiment_plan.md`).
- Funding statement and acknowledgment (none yet; placeholders are comments in `main.tex`).
- Repository URL in the Code Availability section once the repository is made public.
