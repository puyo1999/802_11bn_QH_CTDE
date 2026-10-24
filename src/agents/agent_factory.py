"""
src/agents/agent_factory.py
AgentConfig + SpaceSpec -> AP별 QHCTDEAgent 리스트.

state_dim/n_stas는 여기서만 SpaceSpec으로부터 결정된다. YAML에 하드코딩하지 않는다.
"""

from __future__ import annotations

from src.agents.qh_ctde_agent import QHCTDEAgent
from src.config.schema import AgentConfig
from src.envs.space_validator import SpaceSpec


def build_agents(space: SpaceSpec, cfg: AgentConfig) -> list[QHCTDEAgent]:
    return [
        QHCTDEAgent(
            ap_id=i,
            state_dim=space.local_obs_dim,
            n_stas=space.max_stas_per_ap,
            n_mcs=cfg.n_mcs,
            hidden_dim=cfg.hidden_dim,
            use_quantization=cfg.use_quantization,
            vq_bits=cfg.vq_bits,
            use_dsm=cfg.use_dsm,
            gamma_margin=cfg.gamma_margin,
            lr=cfg.lr,
            gamma=cfg.gamma,
            tau=cfg.tau,
            eps_start=cfg.eps_start,
            eps_end=cfg.eps_end,
            eps_decay=cfg.eps_decay,
            batch_size=cfg.batch_size,
            mem_size=cfg.mem_size,
            device=cfg.device,
        )
        for i in range(space.n_aps)
    ]
