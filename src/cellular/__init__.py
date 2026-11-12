"""
src/cellular
Multi-cell coordinated scheduling in 5G NR for paper3 (docs/paper3_5g_multicell_plan.md).

    layout     hexagonal 7-cell layout with geographical wraparound and UE drops
    channel    3GPP TR 38.901 UMi-Street Canyon path loss, LOS probability, shadowing, Gauss-Markov fading
    cqi        TS 38.214 four-bit CQI table, attenuated-Shannon SINR thresholds (TR 36.942), OLLA
    sim        slot simulator: decisions -> SINR -> deliveries, backlog, PF averages
    oracle     exact per-subband oracle (enumeration of target UE or silence per cluster cell)
    baselines  full reuse, cluster TDMA, greedy muting
"""
