"""
src/config/schema.py
설정 딕셔너리를 타입 안정적인 dataclass로 변환한다.

딕셔너리 직접 접근(cfg["train"].get("hidden_dim", 128))이 코드 전반에 흩어지면
기본값이 파일마다 달라지는 문제가 발생한다. 본 모듈이 기본값의 단일 출처가 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


def _section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key) or {}
    if not isinstance(value, dict):
        raise TypeError(f"설정 섹션 '{key}'는 매핑이어야 합니다 (현재: {type(value).__name__})")
    return value


@dataclass
class EnvConfig:
    """환경 생성 파라미터. OBSSEnv 계열 생성자 인자와 1:1 대응한다."""
    impl: str = "qh"                    # qh | ctde | legacy
    mode: str = "qh"                    # qh | cw | cca | joint
    topology: str = "T1"                # topologies.yaml의 토폴로지 ID
    interact_slots: int = 55_556
    max_episodes: int = 600
    seed: int = 42
    queue_max_size: int = 100
    k_stas: int = 4
    harsh_mode_enabled: bool = True
    harsh_start_ratio: float = 0.5      # 에피소드의 몇 지점부터 가혹 조건을 적용할지
    harsh_interference_gain: float = 5.0
    n_mcs: int = 16
    tx_power_dbm: float = 15.0
    noise_dbm: float = -95.0
    overlap: dict = field(default_factory=dict)   # src/envs/overlap.py OverlapConfig
    reward: dict = field(default_factory=dict)    # src/envs/reward.py RewardConfig
    traffic: dict = field(default_factory=dict)   # src/envs/traffic.py TrafficConfig (traffic_load_ratio 등)
    allow_mute: bool = False            # STA 분기에 명시적 mute 행동(인덱스 K) 추가
    fading: dict = field(default_factory=dict)    # 소규모 페이딩 {"model": "none" | "rayleigh"}

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "EnvConfig":
        raw = _section(cfg, "env")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class AgentConfig:
    """에이전트 신경망 및 학습 하이퍼파라미터."""
    state_dim: int | None = None        # None이면 환경 관측 차원에서 자동 추론
    hidden_dim: int = 128
    n_mcs: int = 16
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 32
    mem_size: int = 5000
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.995
    use_quantization: bool = True
    vq_bits: int = 4
    use_dsm: bool = True
    gamma_margin: float = 0.8
    device: str = "cpu"

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "AgentConfig":
        raw = _section(cfg, "agent")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class TrainConfig:
    """학습 루프 제어 파라미터."""
    max_episodes: int = 300
    n_runs: int = 5
    eval_steps: int = 200
    rollout_length: int = 128
    ppo: dict[str, Any] = field(default_factory=dict)
    warmup_steps: int = 500
    update_every: int = 1
    target_update_every: int = 5
    grad_clip: float = 1.0
    log_interval: int = 10
    save_interval: int = 100
    device: str = "cpu"
    result_dir: str = "results"

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "TrainConfig":
        raw = _section(cfg, "train")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class CommunicationDelayConfig:
    """Simulation-only AP-to-server observation delay in decision cycles."""
    enabled: bool = False
    normal_delay_steps: int = 0
    congested_delay_steps: int = 2
    jitter_max_steps: int = 1
    max_delay_steps: int = 3
    normal_to_congested_prob: float = 0.05
    congested_to_normal_prob: float = 0.25
    initial_congested_prob: float = 0.1

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CommunicationDelayConfig":
        raw = cfg.get("communication_delay") or {}
        if not isinstance(raw, dict):
            raise TypeError("communication_delay must be a mapping")
        known = cls.__dataclass_fields__
        result = cls(**{key: value for key, value in raw.items() if key in known})
        for key in ("normal_delay_steps", "congested_delay_steps", "jitter_max_steps", "max_delay_steps"):
            if getattr(result, key) < 0:
                raise ValueError(f"communication_delay.{key} must be non-negative")
        if result.normal_delay_steps > result.max_delay_steps or result.congested_delay_steps > result.max_delay_steps:
            raise ValueError("base delay values cannot exceed max_delay_steps")
        for key in ("normal_to_congested_prob", "congested_to_normal_prob", "initial_congested_prob"):
            if not 0.0 <= getattr(result, key) <= 1.0:
                raise ValueError(f"communication_delay.{key} must be in [0, 1]")
        return result


@dataclass
class LoggingConfig:
    run_name: str = "run"
    use_wandb: bool = False
    use_tb: bool = True
    log_dir: str = "runs"

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "LoggingConfig":
        raw = _section(cfg, "logging")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class ExperimentConfig:
    """실험 하나를 완전히 기술하는 최상위 설정 객체."""
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    communication_delay: CommunicationDelayConfig = field(default_factory=CommunicationDelayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    topology_file: str = "configs/topologies.yaml"
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            env=EnvConfig.from_dict(cfg),
            agent=AgentConfig.from_dict(cfg),
            train=TrainConfig.from_dict(cfg),
            communication_delay=CommunicationDelayConfig.from_dict(cfg),
            logging=LoggingConfig.from_dict(cfg),
            topology_file=cfg.get("topology_file", "configs/topologies.yaml"),
            raw=cfg,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("raw", None)
        return data
