"""
src/envs/topology_builder.py
YAML 토폴로지 사양(dict)을 Topology 객체로 변환한다.

책임 범위는 '사양 → 객체' 변환 단 하나이다.
파일 입출력과 ID 조회는 src/envs/topology_registry.py가 담당한다.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.envs.topology import Topology


def _validate_spec(topo_id: str, spec: dict[str, Any]) -> None:
    if "ap_positions" not in spec:
        raise KeyError(f"[{topo_id}] ap_positions 항목이 없습니다")
    if not spec.get("stas"):
        raise KeyError(f"[{topo_id}] stas 항목이 비어 있습니다")

    n_aps = len(spec["ap_positions"])
    for idx, sta in enumerate(spec["stas"]):
        ap_idx = sta.get("ap")
        if ap_idx is None:
            raise KeyError(f"[{topo_id}] stas[{idx}]에 ap 필드가 없습니다")
        if not 0 <= ap_idx < n_aps:
            raise ValueError(f"[{topo_id}] stas[{idx}].ap={ap_idx} 가 AP 범위를 벗어납니다")
        if "position" not in sta:
            raise KeyError(f"[{topo_id}] stas[{idx}]에 position 필드가 없습니다")
        for peer in sta.get("overlap_with", []) or []:
            if not 0 <= peer < n_aps:
                raise ValueError(
                    f"[{topo_id}] stas[{idx}].overlap_with={peer} 가 AP 범위를 벗어납니다"
                )
            if peer == ap_idx:
                raise ValueError(
                    f"[{topo_id}] stas[{idx}].overlap_with 에 서빙 AP 자신({ap_idx})이 포함되었습니다"
                )


def neighbors_by_rssi(
    ap_positions: np.ndarray,
    cca_thresh_dbm: float = -82.0,
    tx_power_dbm: float = 10.0,
    path_loss_intercept: float = -46.67,
    path_loss_exponent: float = 30.0,
    min_distance_m: float = 0.5,
) -> dict[int, list[int]]:
    """수신 전력이 CCA 임계치 이상인 AP를 이웃으로 판정한다."""
    n = len(ap_positions)
    nbrs: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = max(float(np.linalg.norm(ap_positions[i] - ap_positions[j])), min_distance_m)
            rx_dbm = tx_power_dbm + path_loss_intercept - path_loss_exponent * math.log10(dist)
            if rx_dbm >= cca_thresh_dbm:
                nbrs[i].append(j)
    return nbrs


def neighbors_by_radius(ap_positions: np.ndarray, radius_m: float = 45.0) -> dict[int, list[int]]:
    """단순 거리 임계 기반 이웃 판정."""
    n = len(ap_positions)
    return {
        i: [
            j for j in range(n)
            if j != i and float(np.linalg.norm(ap_positions[i] - ap_positions[j])) <= radius_m
        ]
        for i in range(n)
    }


def resolve_neighbors(ap_positions: np.ndarray, rule: dict[str, Any]) -> dict[int, list[int]]:
    """neighbor_rule 사양에 따라 이웃 집합을 산출한다."""
    method = (rule or {}).get("method", "rssi")
    params = {k: v for k, v in (rule or {}).items() if k != "method"}

    if method == "rssi":
        return neighbors_by_rssi(ap_positions, **{
            k: v for k, v in params.items()
            if k in {"cca_thresh_dbm", "tx_power_dbm", "path_loss_intercept",
                     "path_loss_exponent", "min_distance_m"}
        })
    if method == "radius":
        return neighbors_by_radius(ap_positions, radius_m=params.get("radius_m", 45.0))
    if method == "explicit":
        explicit = params.get("neighbors")
        if explicit is None:
            raise KeyError("neighbor_rule.method='explicit' 에는 neighbors 사양이 필요합니다")
        return {int(k): list(v) for k, v in explicit.items()}
    raise ValueError(f"지원하지 않는 neighbor_rule.method: {method}")


def build_topology(
    topo_id: str,
    spec: dict[str, Any],
    defaults: dict[str, Any] | None = None,
) -> Topology:
    """
    단일 토폴로지 사양을 Topology 객체로 변환한다.

    STA는 서빙 AP별로 그룹화되며, 각 AP 내부 순서는 YAML 기재 순서를 따른다.
    이 순서가 곧 에이전트 액션의 STA 인덱스가 되므로 결정적으로 유지된다.
    """
    defaults = defaults or {}
    _validate_spec(topo_id, spec)

    ap_positions = np.asarray(spec["ap_positions"], dtype=np.float32)
    n_aps = len(ap_positions)

    sta_positions: list[list[np.ndarray]] = [[] for _ in range(n_aps)]
    sta_overlap: list[list[list[int]]] = [[] for _ in range(n_aps)]
    sta_labels: list[list[str]] = [[] for _ in range(n_aps)]

    for sta in spec["stas"]:
        ap_idx = int(sta["ap"])
        sta_positions[ap_idx].append(np.asarray(sta["position"], dtype=np.float32))
        sta_overlap[ap_idx].append([int(p) for p in (sta.get("overlap_with") or [])])
        sta_labels[ap_idx].append(str(sta.get("label", f"AP{ap_idx}_S{len(sta_labels[ap_idx])}")))

    empty_aps = [i for i, stas in enumerate(sta_positions) if not stas]
    if empty_aps:
        raise ValueError(f"[{topo_id}] STA가 배정되지 않은 AP가 있습니다: {empty_aps}")

    rule = {**(defaults.get("neighbor_rule") or {}), **(spec.get("neighbor_rule") or {})}
    neighbors = resolve_neighbors(ap_positions, rule)

    metadata = {
        "id": topo_id,
        "description": spec.get("description", ""),
        "sta_labels": sta_labels,
        "neighbor_rule": rule,
        "stas_per_ap": [len(s) for s in sta_positions],
    }

    return Topology(
        n_aps=n_aps,
        ap_positions=ap_positions,
        sta_positions=sta_positions,
        neighbors=neighbors,
        name=spec.get("name", topo_id),
        sta_overlap=sta_overlap,
        metadata=metadata,
    )
