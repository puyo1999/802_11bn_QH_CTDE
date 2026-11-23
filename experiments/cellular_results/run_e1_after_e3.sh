#!/usr/bin/env bash
# waits for the last E3 condition, then runs E1 with the throughput objective (200 drops, 3 seeds)
# and with the max-weight objective (200 drops, 1 seed)
cd "$(dirname "$0")/../.."
while [ ! -f experiments/cellular_results/e2_D3_delay8_v60_E300_g0.json ]; do sleep 120; done
sleep 60
.venv/Scripts/python.exe -u experiments/cellular_gap.py e1 --drops 200 --seeds 3 > experiments/cellular_results/e1_full.log 2>&1
.venv/Scripts/python.exe -u experiments/cellular_gap.py e1 --drops 200 --seeds 1 --objective maxweight > experiments/cellular_results/e1_maxweight.log 2>&1
