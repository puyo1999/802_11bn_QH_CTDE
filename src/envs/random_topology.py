"""
src/envs/random_topology.py
Random topologies sampled over geometric design variables (docs/random_topology_experiment_plan.md).

SINR depends on distance ratios when interference dominates and on absolute distance only near the
noise floor, so the generator samples the variables that drive the result instead of scattering APs
over a fixed area:

  d            minimum AP-AP distance (m)           density / interference strength
  r_ratio      STA serving distance / d             cell-edge position (SIR)
  n_aps        number of APs                         coordination scale
  overlap      share of STAs in an overlap region    hard-overlap constraint
  stas_per_ap  STAs per AP                           scheduling choices

The design points come from a Latin hypercube, so a small sample still covers each variable's range.
"""
from __future__ import annotations

import numpy as np

RANGES = {
    "d": (10.0, 40.0),
    "r_ratio": (0.1, 0.5),
    "n_aps": (3, 6),
    "overlap": (0.0, 0.3),
    "stas_per_ap": (1, 3),
}


def latin_hypercube(n: int, dims: int, rng: np.random.Generator) -> np.ndarray:
    """n points in [0, 1)^dims, one point per 1/n stratum along every dimension."""
    strata = np.stack([rng.permutation(n) for _ in range(dims)], axis=1)
    return (strata + rng.random((n, dims))) / n


def design_points(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    unit = latin_hypercube(n, len(RANGES), rng)
    points = []
    for u in unit:
        point = {}
        for (name, (lo, hi)), x in zip(RANGES.items(), u):
            if isinstance(lo, int):
                point[name] = int(lo + np.floor(x * (hi - lo + 1)))
            else:
                point[name] = round(float(lo + x * (hi - lo)), 3)
        points.append(point)
    return points


def place_aps(n_aps: int, d: float, rng: np.random.Generator) -> np.ndarray:
    """Random sequential placement, rescaled so that the minimum AP-AP distance is exactly d."""
    side = 1.6 * d * np.ceil(np.sqrt(n_aps))
    while True:
        aps = [rng.uniform(0, side, 2)]
        for _ in range(10_000):
            if len(aps) == n_aps:
                break
            cand = rng.uniform(0, side, 2)
            if min(np.linalg.norm(cand - a) for a in aps) >= d:
                aps.append(cand)
        if len(aps) == n_aps:
            aps = np.array(aps)
            dist = min(np.linalg.norm(aps[i] - aps[j]) for i in range(n_aps) for j in range(i))
            return (aps - aps.mean(axis=0)) * (d / dist)


def build_spec(point: dict, seed: int) -> dict:
    """YAML spec (configs/topologies.yaml schema) for one design point."""
    rng = np.random.default_rng(seed)
    n, d, m = point["n_aps"], point["d"], point["stas_per_ap"]
    aps = place_aps(n, d, rng)
    n_overlap = int(round(point["overlap"] * n * m))
    overlap_slots = set(rng.choice(n * m, size=n_overlap, replace=False).tolist())
    stas = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for k in range(m):
            if i * m + k in overlap_slots:
                # Overlap STA: near the boundary with the nearest other AP, overlapping that AP.
                j = min(others, key=lambda o: np.linalg.norm(aps[o] - aps[i]))
                along = aps[i] + rng.uniform(0.4, 0.5) * (aps[j] - aps[i])
                normal = np.array([-(aps[j] - aps[i])[1], (aps[j] - aps[i])[0]])
                pos = along + rng.uniform(-0.1, 0.1) * normal
                stas.append({"ap": i, "position": _round(pos), "overlap_with": [j],
                             "label": f"S{i}_{k}_ov{j}"})
            else:
                theta = rng.uniform(0, 2 * np.pi)
                pos = aps[i] + point["r_ratio"] * d * np.array([np.cos(theta), np.sin(theta)])
                stas.append({"ap": i, "position": _round(pos), "label": f"S{i}_{k}"})
    return {"name": "random", "description": "random design point", "design": dict(point),
            "ap_positions": [_round(a) for a in aps], "stas": stas}


def _round(xy) -> list[float]:
    return [round(float(v), 2) for v in xy]
