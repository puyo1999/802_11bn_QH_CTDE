#!/usr/bin/env bash
# Random-topology experiment (docs/random_topology_experiment_plan.md).
# Stage 1: baselines on 200 topologies; stage 2: learners on 20 topologies. Both resume from jobs/.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
$PY experiments/oracle_gap.py --topology-file configs/topologies_random.yaml --topos group:stage1 \
    --loads 1 5 8 --seeds 3 --learners --out experiments/oracle_gap_results_random_stage1
$PY experiments/oracle_gap.py --topology-file configs/topologies_random.yaml --topos group:stage2 \
    --loads 5 8 --seeds 1 --learners mappo central --mcs learned genie --episodes 1000 \
    --out experiments/oracle_gap_results_random_stage2
