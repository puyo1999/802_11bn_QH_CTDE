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

## 6. Pilot (drop 0, d = 4, v = 3 km/h, ideal CQI, training episodes of 400 slots)

Geometric-mean rate relative to the oracle with genie CQI on the evaluation episode (seed 0):

| Policy | Sampled | Argmax | Training |
|---|---|---|---|
| Full reuse / greedy muting (current channel) | 99.4 / 99.9 | | |
| Causal full reuse / causal greedy (delayed reports, as LA0C) | 97.6 / 97.9 | | |
| Uniform random UE, no silence / with silence | 67.8 / 56.3 | | |
| SharedMAPPO, γ = 0.99, 300 episodes | 59.2 | 5.0 | 27 min |
| CentralPPO, γ = 0.99, 300 episodes | 56.9 | 4.7 | 112 min |
| SharedMAPPO, γ = 0, 300 episodes | 94.8 | 97.0 | 12 min |
| SharedMAPPO, γ = 0, 1000 episodes | 96.3 | 96.9 | 36 min |
| CentralPPO, γ = 0, 300 episodes | 69.9 | 69.5 | 55 min |
| CentralPPO, γ = 0, 1000 episodes | 79.1 | 79.0 | 196 min |
| SharedCentralPPO, γ = 0, 300 episodes | 93.0 | 96.6 | 65 min |

Findings:
- With the discount 0.99 of the companion study, neither learner learns: both stay at the level of a random
  policy, and argmax actions starve UEs. The reward of a slot (Σ w_u r_u with PF weights) is already the
  per-slot objective of the oracle; a long discount adds the variance of ~100 future slots to the advantage of
  12 agents and buries the signal.
- With γ = 0 the learner maximizes the expected slot utility, which is the objective of the per-slot oracle,
  and SharedMAPPO learns within ~100 episodes; the curve is flat from ~200 episodes on (slot throughput 7.78 →
  7.93 between the 2nd and 10th tenth of 1000 episodes).
- The fair reference for a learner that observes delayed reports is the causal reference, not the reference
  that selects targets from the current channel: SharedMAPPO is 0.7–1.6 points below causal full reuse and
  causal greedy, against 2.5–3.6 points below their current-channel versions. The pass criterion is revised
  accordingly: a learner passes if it reaches the causal references.

- CentralPPO learns only slightly with γ = 0 (slot throughput 5.49 → 5.77 over 300 episodes; 69.9%, barely
  above a random choice of UEs). Its autoregressive actor decides 12 agents in sequence from the joint
  observation of 156 inputs, which learns far more slowly than the shared actor that sees 13 inputs. Before E2
  it needs work: more episodes, or a shared per-agent encoder in place of the flat joint observation. This is
  the next step on the learner side.
- SharedCentralPPO (`src/agents/mappo.py`) keeps the autoregressive joint policy and the joint embedding of
  CentralPPO but uses one head for all agents that also receives the agent's own observation. With γ = 0 it
  reaches 93.0% (argmax 96.6%) in 300 episodes, against 69.9% for CentralPPO after 300 and 79.1% after 1000
  episodes; its curve still rises at 300 episodes (slot throughput 7.38 → 7.55 between the 6th and 10th tenth),
  so it needs more episodes than SharedMAPPO. SharedCentralPPO replaces CentralPPO as the full-information
  learner of paper3.

Decision proposed for the paper: γ = 0 for all learners (stated and justified by the per-slot oracle), 300–500
training episodes of 400 slots, and the causal references in the E2 table next to the current-channel ones.

## 7. Reduced E2 with SharedMAPPO (5 drops, d = 4, v = 3 km/h, γ = 0, 300 episodes of 400 slots)

Mean over drops of the geometric-mean rate relative to the oracle with genie CQI (%), from
`experiments/cellular_results/e2_D5_delay4_v3_E300_g0.json`:

| CQI | Oracle | Full reuse | Greedy | Causal full reuse | Causal greedy | MAPPO sampled | MAPPO argmax |
|---|---|---|---|---|---|---|---|
| Genie / ideal | 100.0 | 98.6 | 99.3 | 97.1 | 96.9 | 93.9 | 96.4 |
| OLLA | 83.0 | 86.6 | 84.9 | 84.9 | 83.2 | 81.9 | 84.4 |
| Learned | — | — | — | — | — | 39.4 | 42.4 |

- Ideal CQI: MAPPO (argmax) is 0.5–0.7 points below the causal references, which see the same delayed
  reports, and 2.2–2.9 points below the references on the current channel. Of its 3.6-point loss to the oracle,
  about 3 points are the information of the delayed observation (the loss of the causal references) and about
  0.6 points are the learner. Sampling the actions costs a further 2.5 points.
- OLLA: MAPPO (84.4) lies between causal full reuse (84.9) and causal greedy (83.2); the 12-point drop from
  ideal CQI equals that of the references (causal full reuse 97.1 → 84.9), i.e., the operating cost of OLLA.
- Learned CQI: 42.4%, a link-adaptation gap of 54 points against ideal CQI. The curves flatten (slot
  throughput of the last two tenths within 0.1–0.2), so more episodes would not close it; this repeats the
  companion finding that rate selection dominates the loss of learners.
- Evaluation with argmax is 2.5 points above sampled actions for ideal and OLLA CQI; the paper text says the
  learners are evaluated with sampled actions, which should be revised to deterministic execution.
