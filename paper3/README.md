# Paper 3 draft — Exact-oracle gap analysis of learned multi-cell schedulers in 5G NR (IEEE Access)

Title: *When Does Multi-Cell Coordination Pay Off? An Exact-Oracle Gap Analysis of Learned Schedulers in 5G NR*

Overleaf project name: `MultiCell_OracleGap_5GNR`

Plan: `docs/paper3_5g_multicell_plan.md`. Companion papers: `paper/` (IEEE 802.11bn exact-oracle gap analysis),
`paper2/` (MAPC-assisted roaming).

## Status

| Section | State |
|---|---|
| Abstract, I. Introduction | Framing written; results, contributions and organization marked `\todo` |
| II. Related Work | Outlined (citations to verify) |
| III. System Model | Outlined; Fig. `fig:topology` placed |
| IV–X | Outlined (`\todo`) |

## Files

| Path | Source |
|---|---|
| `figures/paper3_topology.{pdf,png}` | copy of `docs/figures/paper3_topology.*`, made by `docs/figures/make_paper3_topology.py` |
| `photo/kpy.eps`, `photo/kty.eps` | copy of `paper/photo/` |
| `references.bib` | verified subset of `paper/references.bib` |

The folder is self-contained so that it can be uploaded to Overleaf as is.

## Rules

- References: only entries verified against a primary record enter `references.bib` (same practice as `paper/`).
- Channel and CQI parameters (TR 38.901, TS 38.214) are verified against the specifications before they enter the text.

## Build

Copy the IEEE Access template files (`ieeeaccess.cls`, `spotcolor.sty`, `IEEEtran.bst`, logo files) into `paper3/`,
then run `pdflatex main && bibtex main && pdflatex main && pdflatex main` inside `paper3/`.
