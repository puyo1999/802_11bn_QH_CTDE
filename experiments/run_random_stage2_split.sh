#!/usr/bin/env bash
# Stage 2 split across two machines: pass the topology IDs to run on this machine.
#   server: experiments/run_random_stage2_split.sh L00 L01 L02 L03 L04 L05 L06 L07 L08 L09
#   PC:     experiments/run_random_stage2_split.sh L10 L11 L12 L13 L14 L15 L16 L17 L18 L19
# Results go to the shared jobs/ directory; finished jobs are skipped on a re-run.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
$PY experiments/oracle_gap.py --topology-file configs/topologies_random.yaml --topos "$@" \
    --loads 5 8 --seeds 1 --learners mappo central --mcs learned genie --episodes 1000 \
    --out experiments/oracle_gap_results_random_stage2
