"""
src/envs/topology.py
논문 Fig.7 / Fig.11 고정 토폴로지 6종 + 랜덤 토폴로지 생성기
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import yaml


@dataclass
class Topology:
    """
    n_aps:          AP 수
    ap_positions:   shape (n_aps, 2)  — 미터 단위
    sta_positions:  list[list[np.ndarray]]  — sta_positions[i] = i번 AP의 STA 좌표 목록
    neighbors:      dict[int, list[int]]    — 직접 감지 가능한 이웃 AP 집합
    name:           토폴로지 이름
    """
    n_aps:         int
    ap_positions:  np.ndarray
    sta_positions: list
    neighbors:     dict
    name:          str = "unnamed"
    shared_sta_groups: list[list[tuple[int, int]]] = field(default_factory=list)


def _neighbors_from_positions(
    ap_positions: np.ndarray,
    cca_thresh_dbm: float = -82.0,
    tx_power_dbm: float   = 10.0,
    noise_dbm: float      = -95.0,
) -> dict[int, list[int]]:
    """
    거리 기반으로 이웃 AP 집합 계산.
    수신 전력이 CCA threshold 이상이면 이웃으로 간주.
    """
    import math
    n = len(ap_positions)
    nbrs: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = float(np.linalg.norm(ap_positions[i] - ap_positions[j]))
            dist = max(dist, 0.5)
            pl_db  = -46.67 - 30.0 * math.log10(dist)
            rx_dbm = tx_power_dbm + pl_db
            if rx_dbm >= cca_thresh_dbm:
                nbrs[i].append(j)
    return nbrs


# ── YAML-backed fixed topologies (legacy function names remain public) ────

_TOPOLOGY_FILE = Path(__file__).resolve().parents[2] / "configs" / "topologies.yaml"


def load_fixed_topologies(path: str | Path = _TOPOLOGY_FILE) -> dict[str, Topology]:
    """Build fixed topologies from YAML, including intentional shared STAs."""
    with Path(path).open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    result: dict[str, Topology] = {}
    for key, spec in raw.get("topologies", {}).items():
        ap = np.asarray(spec["ap_positions"], dtype=np.float32)
        sta = [[np.asarray(point, dtype=np.float32) for point in stations]
               for stations in spec["sta_positions"]]
        if len(ap) != len(sta):
            raise ValueError(f"{key}: AP and STA list lengths differ")
        groups = [[tuple(member) for member in group]
                  for group in spec.get("shared_sta_groups", [])]
        for group in groups:
            if len(group) < 2:
                raise ValueError(f"{key}: shared STA group must contain two or more entries")
            positions = [sta[ap_idx][sta_idx] for ap_idx, sta_idx in group]
            if not all(np.allclose(positions[0], point) for point in positions[1:]):
                raise ValueError(f"{key}: shared STA coordinates must be identical")
        result[key] = Topology(len(ap), ap, sta, _neighbors_from_positions(ap),
                               spec.get("name", key), groups)
    return result


def _fixed(name: str) -> Topology:
    return load_fixed_topologies()[name]

def topo_cw1() -> Topology:
    return _fixed("T1")


def topo_cw2() -> Topology:
    return _fixed("T2")


def topo_cw3() -> Topology:
    return _fixed("T3")


def topo_cw4() -> Topology:
    return _fixed("T4")


def topo_cw5() -> Topology:
    return _fixed("T5")


def topo_cw6() -> Topology:
    return _fixed("T6")


FIXED_TOPOS_CW = [
    topo_cw1, topo_cw2, topo_cw3,
    topo_cw4, topo_cw5, topo_cw6,
]


def random_topology(
    n_aps: int = 4,
    area: tuple[float, float] = (60.0, 40.0),
    stas_per_ap: tuple[int, int] = (1, 3),
    sta_radius: tuple[float, float] = (2.0, 5.0),
    rng: np.random.Generator | None = None,
) -> Topology:
    """
    논문 Section V-F: 60m×40m 공간에 4 AP, 각 1~3 STA (2~5m 반경)
    """
    if rng is None:
        rng = np.random.default_rng()

    ap_pos = rng.uniform(
        low=[0., 0.], high=list(area),
        size=(n_aps, 2)
    ).astype(np.float32)

    sta_pos: list[list[np.ndarray]] = []
    for i in range(n_aps):
        n_sta = int(rng.integers(stas_per_ap[0], stas_per_ap[1] + 1))
        stas = []
        for _ in range(n_sta):
            r     = rng.uniform(*sta_radius)
            theta = rng.uniform(0, 2 * np.pi)
            s = ap_pos[i] + np.array([r * np.cos(theta),
                                       r * np.sin(theta)], dtype=np.float32)
            stas.append(s)
        sta_pos.append(stas)

    nbrs = _neighbors_from_positions(ap_pos)
    return Topology(n_aps, ap_pos, sta_pos, nbrs, name="random")


def get_all_fixed_topos() -> list[Topology]:
    return [fn() for fn in FIXED_TOPOS_CW]
