"""
src/envs/topology_registry.py
topologies.yaml을 읽어 토폴로지 ID/그룹 단위 조회를 제공한다.

기존 FIXED_TOPOS_CW 리스트를 대체하되, 하위 호환 어댑터를 함께 제공한다.
"""

from __future__ import annotations

import functools
from typing import Any

from src.config.loader import read_yaml
from src.envs.topology import Topology
from src.envs.topology_builder import build_topology

DEFAULT_TOPOLOGY_FILE = "configs/topologies.yaml"


class TopologyRegistry:
    """토폴로지 사양 파일 하나에 대응하는 조회 객체."""

    def __init__(self, spec_file: str = DEFAULT_TOPOLOGY_FILE):
        self.spec_file = spec_file
        raw = read_yaml(spec_file)
        self._defaults: dict[str, Any] = raw.get("defaults", {}) or {}
        self._specs: dict[str, Any] = raw.get("topologies", {}) or {}
        self._groups: dict[str, list[str]] = raw.get("groups", {}) or {}
        if not self._specs:
            raise ValueError(f"토폴로지 정의가 비어 있습니다: {spec_file}")

    # ── 조회 ────────────────────────────────────────────────────
    def ids(self) -> list[str]:
        return list(self._specs.keys())

    def group_ids(self, group: str) -> list[str]:
        if group not in self._groups:
            raise KeyError(f"정의되지 않은 토폴로지 그룹: {group} (가능: {list(self._groups)})")
        return list(self._groups[group])

    def groups_of(self, topo_id: str) -> list[str]:
        """해당 토폴로지가 속한 그룹 목록 ('all' 제외)."""
        return [g for g, members in self._groups.items() if g != "all" and topo_id in members]

    def get(self, topo_id: str) -> Topology:
        if topo_id not in self._specs:
            raise KeyError(f"정의되지 않은 토폴로지 ID: {topo_id} (가능: {self.ids()})")
        return build_topology(topo_id, self._specs[topo_id], self._defaults)

    def get_many(self, topo_ids: list[str]) -> list[Topology]:
        return [self.get(t) for t in topo_ids]

    def all(self) -> list[Topology]:
        return self.get_many(self.ids())

    def resolve(self, selector: str | list[str]) -> list[Topology]:
        """
        'T3', ['T1','T3'], 'group:with_overlap', 'all' 형태의 선택자를 해석한다.
        """
        if isinstance(selector, list):
            return self.get_many(selector)
        if selector.startswith("group:"):
            return self.get_many(self.group_ids(selector.split(":", 1)[1]))
        if selector == "all":
            return self.all()
        return [self.get(selector)]

    # ── 요약 ────────────────────────────────────────────────────
    def summary(self) -> list[dict[str, Any]]:
        """토폴로지별 규모 요약 — 논문 Table용 집계에 사용한다."""
        rows = []
        for topo_id in self.ids():
            topo = self.get(topo_id)
            rows.append({
                "id": topo_id,
                "name": topo.name,
                "n_aps": topo.n_aps,
                "n_stas": sum(len(s) for s in topo.sta_positions),
                "max_stas_per_ap": topo.max_stas_per_ap(),
                "n_overlap_stas": topo.n_overlap_stas(),
                "mean_degree": sum(len(v) for v in topo.neighbors.values()) / topo.n_aps,
            })
        return rows


@functools.lru_cache(maxsize=4)
def get_registry(spec_file: str = DEFAULT_TOPOLOGY_FILE) -> TopologyRegistry:
    """동일 사양 파일에 대한 레지스트리를 캐시하여 재사용한다."""
    return TopologyRegistry(spec_file)


def load_topology(topo_id: str, spec_file: str = DEFAULT_TOPOLOGY_FILE) -> Topology:
    """단일 토폴로지 조회 단축 함수."""
    return get_registry(spec_file).get(topo_id)


def load_topologies(
    selector: str | list[str] = "all",
    spec_file: str = DEFAULT_TOPOLOGY_FILE,
) -> list[Topology]:
    """선택자 기반 다중 조회 단축 함수."""
    return get_registry(spec_file).resolve(selector)
