# Paper 2 draft — MAPC-assisted roaming (IEEE Access)

Follow-up to `paper/` (exact-oracle gap analysis). Model and placement plan: `docs/mobility_association_model.md`.

## Status

| Section | State |
|---|---|
| Abstract, I. Introduction | Written; quantitative results and roaming citations marked `\todo` |
| II. Related Work | MAPC part written; roaming and handover parts outlined (citations to verify) |
| III. System Model | Written: Gauss–Markov mobility, filtered RSSI, hysteresis/TTT/threshold roaming, handover interruption, queue transfer |
| IV. Analysis | Written: closed-form mis-association distance, filtered-noise variance, flapping probability, roaming loss |
| V. Problem Formulation | Written: fixed-association objective, serving-AP switching with switching cost |
| VI. Method | Outlined (`\todo`) |
| VII–X. Setup, Results, Discussion, Conclusion | Placeholders |

## Key closed forms (Section IV)

- Mis-association distance beyond the equal-RSSI boundary (two APs at distance L, path-loss exponent 3):
  `(L/2) tanh(Δ_hys ln10 / 60)` — 1.15 m for L = 20 m, Δ_hys = 3 dB.
- Roaming-threshold position: `10^((P_tx − 46.67 − R_roam)/30)` — about 19 m for R_roam = −70 dBm, so the
  threshold, not the hysteresis, sets the window in a 20 m deployment.
- Filtered RSSI-difference variance: `2α/(2−α) σ_X²`, lag-one correlation `1 − α`.

## Rules

- References: only entries verified against a primary record enter `references.bib` (same practice as `paper/`).
- Figures go to `paper2/figures/`; author photos are taken from `paper/photo/` via `\graphicspath`.

## Build

Copy the IEEE Access template files (`ieeeaccess.cls`, `spotcolor.sty`, `IEEEtran.bst`, logo files) into `paper2/`,
then run `pdflatex main && bibtex main && pdflatex main && pdflatex main` inside `paper2/`.
