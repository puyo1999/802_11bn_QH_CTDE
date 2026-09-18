"""Parameter-driven environment construction and state contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.envs.qh_obss_env_mod1 import OBSSEnv as SchedulingEnv
from src.envs.topology import Topology


@dataclass(frozen=True)
class StateContract:
    local_dim: int
    global_dim: int
    n_aps: int


def make_scheduling_env(topology: Topology, config: dict[str, Any], *, episode_horizon: int | None = None) -> SchedulingEnv:
    """Create the scheduling environment using only ``env`` YAML parameters."""
    env = config["env"]
    traffic = env.get("traffic", {})
    return SchedulingEnv(
        topology=topology,
        mode=env.get("mode", "qh"),
        interact_slots=env.get("interact_slots", 55_556),
        max_episodes=episode_horizon or env.get("max_episodes", 1),
        seed=env.get("seed"),
        traffic_type=traffic.get("type", env.get("traffic_type", "saturated")),
        poisson_arrival_rate=traffic.get("poisson_arrival_rate", env.get("poisson_arrival_rate", 0.1)),
        queue_max_size=traffic.get("queue_max_size", env.get("queue_max_size", 100)),
        k_stas=env.get("k_stas", env.get("k_stas_per_ap", 4)),
    )


def state_contract(env: SchedulingEnv) -> StateContract:
    """Derive, rather than duplicate, local/global state dimensions."""
    local_dim = int(env.observation_space.shape[-1])
    return StateContract(local_dim=local_dim, global_dim=env.n_aps * local_dim, n_aps=env.n_aps)


def validate_state_contract(env: SchedulingEnv, config: dict[str, Any]) -> StateContract:
    contract = state_contract(env)
    declared = config.get("agent", {}).get("state_dim")
    if declared is not None and int(declared) != contract.local_dim:
        raise ValueError(
            f"agent.state_dim={declared}, but environment emits {contract.local_dim}; "
            "remove the override or set it to 2 * env.k_stas + 2"
        )
    return contract
