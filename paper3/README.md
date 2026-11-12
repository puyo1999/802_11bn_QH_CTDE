# Paper 3 draft — Exact-oracle gap analysis of learned multi-cell schedulers in 5G NR (IEEE Access)

Title: *When Does Multi-Cell Coordination Pay Off? An Exact-Oracle Gap Analysis of Learned Schedulers in 5G NR*

Overleaf project name: `MultiCell_OracleGap_5GNR`

Plan: `docs/paper3_5g_multicell_plan.md`. Companion papers: `paper/` (IEEE 802.11bn exact-oracle gap analysis),
`paper2/` (MAPC-assisted roaming).

## Status

| Section | State |
|---|---|
| Abstract | Written; quantitative results marked `\todo` |
| I. Introduction | Written; companion-paper citation marked `\todo` |
| II. Related Work | Written; 13 verified references |
| III. System Model | Written: layout and cluster, UMi channel (TR 38.901), subbands, CQI table (TS 38.214), attenuated Shannon thresholds (TR 36.942), traffic, PF objective, observations; Fig. `fig:topology` |
| IV. Problem Formulation | Written: objective, Dec-POMDP, offered load η, headroom H |
| V. Method | Written: per-subband exact oracle with Algorithm 1, references, learners, OLLA, gap decomposition |
| VI. Experimental Setup | Written: experiments E0–E5, Table I; PPO hyperparameters to be copied once the companion paper is citable |
| VII–IX. Results, Discussion, Conclusion | Outlined (`\todo`), waiting for the experiments |

## Files

| Path | Source |
|---|---|
| `figures/paper3_topology.{pdf,png}` | copy of `docs/figures/paper3_topology.*`, made by `docs/figures/make_paper3_topology.py` |
| `figures/fig_sinr_cdf.{pdf,png}` | E0, recomputed from `src/cellular` (50 drops) by `figures/make_figures.py` |
| `figures/fig_headroom.{pdf,png}` | E1 pilot, `experiments/cellular_results/e1_U4_D20_S1.csv` (20 drops, 1 seed), by `figures/make_figures.py` |
| `figures/fig_la0.{pdf,png}` | LA0 pre-validation, `experiments/cellular_results/la0_D20_S1.csv` (20 drops, 1 seed), by `figures/make_figures.py` |

The result figures come from pilot runs and are not yet placed in `main.tex`; they are regenerated with
`python paper3/figures/make_figures.py` after the full runs.
| `photo/kpy.eps`, `photo/kty.eps` | copy of `paper/photo/` |
| `references.bib` | verified subset of `paper/references.bib` |

The folder is self-contained so that it can be uploaded to Overleaf as is.

## Rules

- References: only entries verified against a primary record enter `references.bib` (same practice as `paper/`).
- Channel and CQI parameters are verified against the specifications before they enter the text. Verified so far:
  ETSI TR 138 901 V17.0.0 (Tables 7.4.1-1, 7.4.2-1, 7.8-1), ETSI TS 138 214 V16.2.0 (Table 5.2.2.1-2),
  ETSI TR 136 942 V15.0.0 (Annex A, Table A.1).

## Build

Copy the IEEE Access template files (`ieeeaccess.cls`, `spotcolor.sty`, `IEEEtran.bst`, logo files) into `paper3/`,
then run `pdflatex main && bibtex main && pdflatex main && pdflatex main` inside `paper3/`.
