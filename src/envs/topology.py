"""
src/envs/topology.py
논문 Fig.7 / Fig.11 고정 토폴로지 6종 + 랜덤 토폴로지 생성기
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


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


# ── 논문 고정 토폴로지 (Fig.7 CW 최적화 / Fig.11 CCA 최적화) ─

def topo_cw1() -> Topology:
    """Topo1: 3-AP, 20m 간격, AP1-AP2 직접, AP1-AP3 hidden"""
    ap = np.array([[0., 0.], [20., 0.], [40., 0.]], dtype=np.float32)
    sta = [
        [np.array([0., -10.], dtype=np.float32)],
        [np.array([20., -10.], dtype=np.float32)],
        [np.array([40., -10.], dtype=np.float32)],
    ]
    nbrs = _neighbors_from_positions(ap)
    return Topology(3, ap, sta, nbrs, name="CW_Topo1")


def topo_cw2() -> Topology:
    """Topo2: 3-AP, 20m 간격, 반대 방향"""
    ap = np.array([[40., 0.], [20., 0.], [0., 0.]], dtype=np.float32)
    sta = [
        [np.array([40., -10.], dtype=np.float32)],
        [np.array([20., -10.], dtype=np.float32)],
        [np.array([0., -10.], dtype=np.float32)],
    ]
    nbrs = _neighbors_from_positions(ap)
    return Topology(3, ap, sta, nbrs, name="CW_Topo2")


def topo_cw3() -> Topology:
    """Topo3: 4-AP 정사각형, 30m"""
    ap = np.array([
        [0., 0.], [30., 0.], [0., -30.], [30., -30.]
    ], dtype=np.float32)
    sta = [[np.array([x + 5., y - 5.], dtype=np.float32)]
           for x, y in ap]
    nbrs = _neighbors_from_positions(ap)
    return Topology(4, ap, sta, nbrs, name="CW_Topo3")


def topo_cw4() -> Topology:
    """Topo4: 4-AP, 비대칭 배치"""
    ap = np.array([
        [0., 0.], [30., 0.], [30., -30.], [0., -30.]
    ], dtype=np.float32)
    sta = [[np.array([x + 5., y - 5.], dtype=np.float32)]
           for x, y in ap]
    nbrs = _neighbors_from_positions(ap)
    return Topology(4, ap, sta, nbrs, name="CW_Topo4")


def topo_cw5() -> Topology:
    """Topo5: 4-AP + 2 extra STA, 밀집"""
    ap = np.array([
        [0., 0.], [10., 0.], [20., 0.], [20., -20.]
    ], dtype=np.float32)
    sta = [
        [np.array([0., -8.], dtype=np.float32),
         np.array([0., -16.], dtype=np.float32)],
        [np.array([10., -8.], dtype=np.float32)],
        [np.array([20., -8.], dtype=np.float32)],
        [np.array([20., -28.], dtype=np.float32)],
    ]
    nbrs = _neighbors_from_positions(ap)
    return Topology(4, ap, sta, nbrs, name="CW_Topo5")


def topo_cw6() -> Topology:
    """Topo6: 4-AP, 분산 배치"""
    ap = np.array([
        [0., 0.], [20., 0.], [40., 0.], [20., -20.]
    ], dtype=np.float32)
    sta = [[np.array([x + 3., y - 8.], dtype=np.float32)]
           for x, y in ap]
    nbrs = _neighbors_from_positions(ap)
    return Topology(4, ap, sta, nbrs, name="CW_Topo6")


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
