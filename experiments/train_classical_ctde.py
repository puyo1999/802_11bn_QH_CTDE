"""Configuration-driven classical CTDE training entry point."""
from __future__ import annotations
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.agents.classical_ctde import ClassicalCTDE
from src.config import load_config
from src.envs.factory import make_scheduling_env, validate_state_contract
from src.envs.topology import load_fixed_topologies


def train_topology(topo, cfg: dict) -> list[float]:
    horizon = cfg["train"].get("horizon", 32)
    env = make_scheduling_env(topo, cfg, episode_horizon=horizon)
    contract = validate_state_contract(env, cfg)
    acfg = cfg["agent"]
    learner = ClassicalCTDE(contract.local_dim, contract.n_aps, cfg["env"]["k_stas"],
                            hidden_dim=acfg.get("hidden_dim", 128), gamma=acfg.get("gamma", .99),
                            lr=acfg.get("lr", 3e-4), device=cfg["train"].get("device", "cpu"))
    rewards = []
    valid_counts = [len(stas) for stas in topo.sta_positions]
    for _ in range(cfg["train"]["max_episodes"]):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            action, log_prob = learner.select_actions(obs, valid_counts)
            next_obs, reward, done, _, _ = env.step(action)
            learner.update(obs, log_prob, reward, next_obs, done)
            obs, total = next_obs, total + reward
        rewards.append(total)
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/classical_ctde.yaml")
    parser.add_argument("--topology", default="T1", choices=[f"T{i}" for i in range(1, 7)])
    args = parser.parse_args()
    cfg = load_config(args.config)
    topo = load_fixed_topologies()[args.topology]
    rewards = train_topology(topo, cfg)
    print(f"{topo.name}: {len(rewards)} episodes, final reward={rewards[-1]:.4f}")


if __name__ == "__main__":
    main()
