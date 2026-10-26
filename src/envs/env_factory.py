"""
src/envs/env_factory.py
EnvConfig + Topology → 환경 인스턴스 생성.

기존 학습 스크립트는 OBSSEnv(topology=topo, mode="qh", interact_slots=55556, ...)
형태로 상수를 코드에 직접 기재했다. 본 모듈은 모든 인자를 설정 객체에서 취해
동일한 실험을 YAML만으로 재현 가능하게 만든다.
"""

from __future__ import annotations

import importlib
from typing import Any

from src.config.schema import EnvConfig
from src.envs.topology import Topology
from src.envs.topology_registry import load_topology

# 환경 구현체 레지스트리: impl 키 → (모듈 경로, 클래스명)
ENV_IMPLEMENTATIONS: dict[str, tuple[str, str]] = {
    "qh_v2":  ("src.envs.qh_obss_env_v2", "QHOBSSEnv"),
    "qh":     ("src.envs.qh_obss_env_mod1", "OBSSEnv"),
    "ctde":   ("src.envs.ctde_obss_env_mod1", "OBSSEnv"),
    "legacy": ("src.envs.obss_env", "OBSSEnv"),
}

# 구현체별로 생성자가 실제로 받는 인자 이름 집합.
# 설정에는 공통 파라미터를 모두 두되, 전달 시 구현체가 모르는 인자는 걸러낸다.
_COMMON_KWARGS = {
    "mode", "interact_slots", "max_episodes", "seed", "queue_max_size",
}
_IMPL_EXTRA_KWARGS: dict[str, set[str]] = {
    "qh_v2":  {"k_stas", "n_mcs", "tx_power_dbm", "noise_dbm",
               "harsh_mode_enabled", "harsh_start_ratio", "harsh_interference_gain",
               "overlap", "reward", "traffic", "allow_mute"},
    "qh":     {"k_stas", "k_stas_per_ap", "poisson_lambda", "poisson_arrival_rate", "traffic_type",
               "allow_mute"},
    "ctde":   {"poisson_arrival_rate", "traffic_type"},
    "legacy": {"poisson_arrival_rate", "traffic_type"},
}


def resolve_env_class(impl: str):
    """impl 키에 해당하는 환경 클래스를 동적으로 임포트한다."""
    if impl not in ENV_IMPLEMENTATIONS:
        raise KeyError(f"지원하지 않는 env.impl: {impl} (가능: {list(ENV_IMPLEMENTATIONS)})")
    module_path, class_name = ENV_IMPLEMENTATIONS[impl]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_env_kwargs(cfg: EnvConfig, seed_offset: int = 0) -> dict[str, Any]:
    """EnvConfig에서 해당 구현체가 수용하는 생성자 인자만 추출한다."""
    allowed = _COMMON_KWARGS | _IMPL_EXTRA_KWARGS.get(cfg.impl, set())

    candidates: dict[str, Any] = {
        "mode": cfg.mode,
        "interact_slots": cfg.interact_slots,
        "max_episodes": cfg.max_episodes,
        "seed": cfg.seed + seed_offset,
        "queue_max_size": cfg.queue_max_size,
        "k_stas": cfg.k_stas,
        # 구형 구현체(qh/ctde/legacy)는 traffic_load_ratio 개념이 없으므로
        # ratio>=0.5 를 saturated, 그 외를 poisson으로 근사 변환한다.
        "traffic_type": "saturated" if cfg.traffic.get("traffic_load_ratio", 0.7) >= 0.5 else "poisson",
        "poisson_lambda": cfg.traffic.get("poisson_lambda_min", 0.1),
        "poisson_arrival_rate": cfg.traffic.get("poisson_lambda_min", 0.1),
        "traffic": cfg.traffic,
        "n_mcs": cfg.n_mcs,
        "tx_power_dbm": cfg.tx_power_dbm,
        "noise_dbm": cfg.noise_dbm,
        "harsh_mode_enabled": cfg.harsh_mode_enabled,
        "harsh_start_ratio": cfg.harsh_start_ratio,
        "harsh_interference_gain": cfg.harsh_interference_gain,
        "overlap": cfg.overlap,
        "reward": cfg.reward,
        "allow_mute": cfg.allow_mute,
    }
    return {k: v for k, v in candidates.items() if k in allowed}


def apply_harsh_mode_settings(env, cfg: EnvConfig) -> None:
    """
    가혹 간섭 조건 파라미터를 환경 인스턴스에 주입한다.
    환경 코드가 상수로 들고 있던 배율(5.0 / 10.0)과 전환 시점(max_episodes // 2)을
    설정값으로 노출하여, 시나리오별 스윕이 가능하도록 한다.
    """
    env.harsh_mode_enabled = cfg.harsh_mode_enabled
    env.harsh_start_ratio = cfg.harsh_start_ratio
    env.harsh_interference_gain = cfg.harsh_interference_gain
    env.harsh_start_step = int(cfg.max_episodes * cfg.harsh_start_ratio)
    env.is_harsh_interference = False


def make_env(
    cfg: EnvConfig,
    topology: Topology | None = None,
    seed_offset: int = 0,
    topology_file: str = "configs/topologies.yaml",
):
    """
    설정 객체로부터 환경 인스턴스를 생성한다.

    Args:
        cfg:           환경 설정
        topology:      직접 지정할 토폴로지. None이면 cfg.topology ID로 조회한다.
        seed_offset:   다중 시드 반복 실험용 오프셋
        topology_file: 토폴로지 사양 파일 경로
    """
    topo = topology if topology is not None else load_topology(cfg.topology, topology_file)
    env_cls = resolve_env_class(cfg.impl)
    env = env_cls(topology=topo, **build_env_kwargs(cfg, seed_offset))
    apply_harsh_mode_settings(env, cfg)
    env.topology_id = topo.metadata.get("id", topo.name)
    return env
