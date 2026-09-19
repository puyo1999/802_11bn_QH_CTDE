"""
src/envs/space_validator.py
환경의 관측/액션 공간과 설정·토폴로지 간 정합성 검증.

기존 코드에서는 agent.state_dim을 YAML에 9 또는 10으로 고정 기재한 뒤
환경이 실제로는 다른 차원을 반환하여 런타임에 형상 불일치가 발생했고,
train_ctde.py는 이를 init_obs.shape[-1]로 사후 보정하고 있었다.
본 모듈은 학습 시작 전에 차원을 산출·검증하여 조용한 불일치를 차단한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.envs.topology import Topology


@dataclass
class SpaceSpec:
    """검증을 통과한 공간 사양. 에이전트 생성의 유일한 차원 출처가 된다."""
    n_aps: int
    local_obs_dim: int          # 개별 AP가 실행 시점에 관측하는 차원 (Local Actor 입력)
    global_state_dim: int       # 중앙 critic이 학습 시점에 관측하는 차원 (Global Critic 입력)
    max_stas_per_ap: int
    n_stas_per_ap: list[int]
    action_dims_per_ap: list[int]
    n_overlap_stas: int

    def describe(self) -> str:
        return (
            f"n_aps={self.n_aps}  local_obs_dim={self.local_obs_dim}  "
            f"global_state_dim={self.global_state_dim}  "
            f"max_stas_per_ap={self.max_stas_per_ap}  "
            f"action_dims_per_ap={self.action_dims_per_ap}  "
            f"overlap_stas={self.n_overlap_stas}"
        )


def infer_local_obs_dim(env) -> int:
    """환경이 실제로 반환하는 관측에서 AP당 차원을 추론한다."""
    obs, _ = env.reset()
    obs = np.asarray(obs)
    if obs.ndim != 2:
        raise ValueError(f"관측 배열은 (n_aps, obs_dim) 형상이어야 합니다 (현재: {obs.shape})")
    if obs.shape[0] != env.n_aps:
        raise ValueError(
            f"관측 첫 축이 AP 수와 불일치합니다: obs.shape[0]={obs.shape[0]}, n_aps={env.n_aps}"
        )
    return int(obs.shape[-1])


def infer_action_dims(env) -> list[int]:
    """MultiDiscrete 액션 공간의 AP당 분기 크기를 추출한다."""
    space = env.action_space
    nvec = getattr(space, "nvec", None)
    if nvec is None:
        raise TypeError(f"MultiDiscrete 액션 공간이 필요합니다 (현재: {type(space).__name__})")
    nvec = np.asarray(nvec).tolist()
    if len(nvec) % env.n_aps != 0:
        raise ValueError(f"액션 차원 {len(nvec)} 이 AP 수 {env.n_aps} 로 나누어떨어지지 않습니다")
    per_ap = len(nvec) // env.n_aps
    return [int(v) for v in nvec[:per_ap]]


def validate_spaces(
    env,
    topology: Topology,
    declared_state_dim: int | None = None,
    strict: bool = True,
) -> SpaceSpec:
    """
    환경 공간을 검사하고 SpaceSpec을 반환한다.

    Args:
        declared_state_dim: YAML에 명시된 agent.state_dim. None이면 검증을 생략하고
                            환경 실측값을 사용한다.
        strict:             True이면 선언값 불일치 시 예외, False이면 경고 후 실측값 채택.
    """
    local_obs_dim = infer_local_obs_dim(env)
    action_dims = infer_action_dims(env)
    n_stas = [len(s) for s in topology.sta_positions]

    if declared_state_dim is not None and declared_state_dim != local_obs_dim:
        message = (
            f"[{topology.name}] agent.state_dim={declared_state_dim} 이 "
            f"환경 실측 관측 차원 {local_obs_dim} 과 일치하지 않습니다. "
            f"YAML의 agent.state_dim을 제거하여 자동 추론에 위임하거나 "
            f"{local_obs_dim} 으로 수정하십시오."
        )
        if strict:
            raise ValueError(message)
        print(f"[경고] {message}")

    max_stas = topology.max_stas_per_ap()
    sta_branch = action_dims[0] if action_dims else 0
    if sta_branch and sta_branch < max_stas:
        raise ValueError(
            f"[{topology.name}] 액션 공간의 STA 분기 크기 {sta_branch} 가 "
            f"최대 STA 수 {max_stas} 보다 작아 일부 STA를 선택할 수 없습니다."
        )

    return SpaceSpec(
        n_aps=env.n_aps,
        local_obs_dim=local_obs_dim,
        global_state_dim=local_obs_dim * env.n_aps,
        max_stas_per_ap=max_stas,
        n_stas_per_ap=n_stas,
        action_dims_per_ap=action_dims,
        n_overlap_stas=topology.n_overlap_stas(),
    )


def build_adjacency(topology: Topology, include_self: bool = True) -> np.ndarray:
    """어텐션 마스크용 인접 행렬. 기존 학습 스크립트의 중복 구현을 대체한다."""
    n = topology.n_aps
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if include_self:
            adj[i, i] = 1.0
        for j in topology.neighbors[i]:
            adj[i, j] = 1.0
    return adj
