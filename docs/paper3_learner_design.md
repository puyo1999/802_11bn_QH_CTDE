# paper3 learner experiments (E2–E5): design and compute budget

Status: 2026-11-21. Environment, learners and pilot harness are implemented; the scale of E2–E5 is a decision for
the authors (Section 5).

## 1. What is implemented

| Piece | File | Check |
|---|---|---|
| Multi-agent environment: agent = (cell, subband), action = (UE slot or silence, CQI), observation of paper3 eq. (obs), reward = slot PF utility / number of agents | `src/cellular/marl_env.py` | `test_marl_env_actions_map_to_own_ues` (actions map to the cell's own UEs, slot U = silence, ideal CQI never fails when it transmits) |
| CQI control: learned (actor picks CQI 1..15), OLLA on the delayed report, ideal (genie CQI for the chosen targets) | same | same |
| 0-bit learner with a shared actor (paper text: agents share parameters, subband one-hot tells them apart) | `SharedMAPPO` in `src/agents/mappo.py` | same distributions and update as companion MAPPO; only the actor is shared |
| Full-information learner | `CentralPPO` (companion, unchanged) | — |
| Vectorized SINR of the simulator (speed) | `CellularSim.sinr` | `test_vectorized_sinr_equals_loop_bitwise` (C = 3 and 7) |
| Training/evaluation harness with per-job files | `experiments/cellular_learn.py` | smoke run |

Evaluation uses the full episode of LA0/LA1 (300 warm-up + 1000 measured slots, evaluation seed 0), so learners
and references are compared on the same fading sequence. Training uses shorter episodes (default 400 slots) with
other seeds.

## 2. Finding that shapes E2: the coordination headroom under PF with four subbands is about 1–2%

From LA0 (20 drops, d = 4, genie CQI), the geometric-mean rate relative to the oracle is 97.9% for full reuse and
99.0–99.2% for greedy muting. The pilot drop gives 99.4% and 99.9%. With OLLA on delayed reports at 3 km/h, full
reuse (86.0%) is above the targets of the oracle (82.7%), because silencing makes the all-transmit report
pessimistic.

Consequences:
- In the E2 setting, the optimization gap Δ_opt and the information gap Δ_I of a learner are bounded by a few
  points: a learner can gain at most ~2% over full reuse. Any larger loss is learning failure or link adaptation.
- The paper's question "when does coordination pay off" has a clear partial answer already: under PF with
  frequency-selective subbands and equal power, full reuse is near-optimal; the gain of coordination is over
  cluster TDMA (E1), not over full reuse.
- E2 should therefore report the learners' loss against full reuse and greedy muting as well as against the
  oracle, and E5 (C = 7) or B = 1 settings are where any headroom would show.

## 3. Compute measurements (Intel i3-4030U, 1 thread per process, while 4 other simulations ran)

| Learner | ms per slot (act + env) | env alone |
|---|---|---|
| MAPPO, separate actors | 21–29 | 3–4 |
| SharedMAPPO | 16 | 3 |
| CentralPPO (12-step autoregressive) | 57–110 | 3–4 |

The PyTorch overhead per call dominates; the environment itself is ~3 ms per slot.

## 4. Budget of the plan in the paper

Plan text: 1000 training episodes per drop; E2 20 drops × 2 learners × 3 CQI controls; E3 10 drops × 9
conditions; E5 C = 7.

- One training of 1000 × 1300 slots: ~6 h (SharedMAPPO) to ~25 h (CentralPPO).
- E2 alone: 120 trainings ≈ 1500 CPU-hours ≈ 2 weeks on 4 processes. E3 adds ×4.5. Not feasible on this laptop.

## 5. Options (decision for the authors)

A. **Reduced scale on the laptop** (recommended to start): training episodes of 400 slots, 300–500 episodes
   (set from the pilot learning curve), E2 on 5 drops, E3 on the 4 corner conditions (d ∈ {0, 8} ×
   v ∈ {3, 60}) on 3 drops, E5 on 2 drops with ideal CQI only. Roughly 60–100 CPU-hours (≈ 1–1.5 days on 4
   processes). The paper states the reduced scale.
B. **Batch the environments**: run 16 environments per actor call to amortize the PyTorch overhead (expected
   3–5× faster). About half a day of implementation and tests; makes option A ≈ 2–3× cheaper.
C. **GPU / cloud machine**: the plan as written; needs a machine with many cores (the environment is CPU bound).

Recommended: A with the pilot deciding the episode budget, then B if the pilot shows learning needs more episodes.

## 6. Pilot (running)

Drop 0, d = 4, v = 3 km/h, ideal CQI, 300 episodes × 400 slots, SharedMAPPO and CentralPPO; result in
`experiments/cellular_results/learn_pilot_d0.json`. Pass criterion: the learners reach at least full reuse
(99.4% of the oracle on this drop) with argmax actions; if not, the learning setup (reward scale, entropy,
episode budget) needs work before any scaled run.
