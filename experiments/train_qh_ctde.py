"""Train and evaluate the QH-CTDE centralized PPO implementation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.ctde_ppo import CTDEActorCritic
from src.config import load_config
from src.config.schema import ExperimentConfig
from src.envs.env_factory import make_env
from src.envs.topology_registry import load_topologies
from src.envs.space_validator import build_adjacency, validate_spaces
from src.training.centralized_execution import CentralizedExecutor
from src.training.communication_delay import ObservationDelayChannel
from src.training.ppo import CTDEPPOTrainer, PPOConfig, PPORollout


def _new_env(topo, cfg: dict, horizon: int, seed_offset: int = 0):
    experiment = ExperimentConfig.from_dict(cfg)
    env_config = replace(experiment.env, max_episodes=horizon,
                         seed=experiment.env.seed + seed_offset)
    if env_config.impl != "qh_v2":
        raise ValueError("CTDE PPO entrypoint requires env.impl=qh_v2")
    env = make_env(env_config, topology=topo, seed_offset=0,
                   topology_file=experiment.topology_file)
    declared_dim = experiment.agent.state_dim
    space = validate_spaces(env, topo, declared_state_dim=declared_dim, strict=True)
    expected_overlap = env_config.overlap
    overlap_cfg = env.overlap_policy.cfg
    if (overlap_cfg.enabled != bool(expected_overlap.get("enabled", True))
            or overlap_cfg.mode != expected_overlap.get("mode", "hard")
            or overlap_cfg.expose_in_obs != bool(expected_overlap.get("expose_in_obs", True))):
        raise RuntimeError("env_factory did not propagate the configured overlap settings")
    expected_load = float(env_config.traffic.get("traffic_load_ratio", 0.7))
    if abs(env.traffic.cfg.traffic_load_ratio - expected_load) > 1e-12:
        raise RuntimeError("env_factory did not propagate traffic_load_ratio")
    env.space_spec = space
    return env


def build_ctde_trainer(env, cfg: dict) -> CTDEPPOTrainer:
    """Build the shared local policy, joint-state critic, and PPO optimizer."""
    experiment = ExperimentConfig.from_dict(cfg)
    agent_cfg = experiment.agent
    train_cfg = experiment.train
    delay_enabled = experiment.communication_delay.enabled
    model = CTDEActorCritic(
        observation_dim=env.space_spec.local_obs_dim + int(delay_enabled),
        n_agents=env.space_spec.n_aps,
        n_stas=env.space_spec.action_dims_per_ap[0],
        n_mcs=env.space_spec.action_dims_per_ap[1],
        hidden_dim=agent_cfg.hidden_dim,
        vq_bits=agent_cfg.vq_bits,
        adjacency=torch.as_tensor(build_adjacency(env.topo), dtype=torch.bool),
        mute_index=env.mute_index,
    )
    return CTDEPPOTrainer(
        model,
        lr=agent_cfg.lr,
        ppo_config=PPOConfig.from_dict(train_cfg.ppo),
        device=train_cfg.device,
    )


def train_qh_ctde(topo, cfg: dict, seed_offset: int = 0,
                  execution_mode: str = "auto",
                  episode_metrics: list[dict] | None = None):
    """Collect full rollouts, estimate GAE centrally, then perform PPO updates."""
    experiment = ExperimentConfig.from_dict(cfg)
    train_cfg = experiment.train
    agent_cfg = experiment.agent
    horizon = train_cfg.rollout_length
    n_episodes = experiment.train.max_episodes
    env = _new_env(topo, cfg, horizon, seed_offset)
    trainer = build_ctde_trainer(env, cfg)
    valid_counts = [len(stas) for stas in topo.sta_positions]
    executor = CentralizedExecutor(
        trainer.model, valid_counts,
        threshold=agent_cfg.gamma_margin,
        device=train_cfg.device,
        execution_mode=execution_mode,
        use_age_feature=experiment.communication_delay.enabled,
    )
    reward_history: list[float] = []
    seed_base = int(cfg.get("env", {}).get("seed", 42)) + seed_offset
    gamma = agent_cfg.gamma
    ppo_config = PPOConfig.from_dict(train_cfg.ppo)
    gae_lambda = ppo_config.gae_lambda

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + episode)
        delay_channel = None
        if experiment.communication_delay.enabled:
            delay_channel = ObservationDelayChannel(experiment.communication_delay, len(valid_counts))
            delay_channel.reset(obs, seed=seed_base + episode + 50_000)
        decision_step = 0
        rollout = PPORollout.empty()
        episode_reward = 0.0
        overlap_stats = {
            "centralized": {"steps": 0, "blocked": 0},
            "decentralized": {"steps": 0, "blocked": 0},
        }
        done = False
        while not done:
            delayed_obs = None
            age_steps = None
            if delay_channel is not None:
                request_central = (
                    execution_mode in {"joint_raw", "joint_vq"}
                    or (execution_mode == "auto"
                        and float(obs[0, -1]) <= agent_cfg.gamma_margin)
                )
                if request_central:
                    delay_channel.transmit(obs, decision_step)
                delayed_obs, age_steps = delay_channel.receive(decision_step)
            execution = executor.act(
                obs, deterministic=False,
                delayed_observations=delayed_obs,
                observation_age_steps=age_steps,
            )
            action_tensor = torch.as_tensor(execution.actions, dtype=torch.int32).reshape(-1).cpu().numpy()
            next_obs, reward, done, _, info = env.step(action_tensor)
            mode_name = "centralized" if execution.centralized_policy else "decentralized"
            overlap_stats[mode_name]["steps"] += 1
            overlap_stats[mode_name]["blocked"] += int(info.get("overlap_blocked_step", 0))
            critic_obs = torch.as_tensor(execution.critic_observation, dtype=torch.float32, device=trainer.device).unsqueeze(0)
            with torch.no_grad():
                value = float(trainer.model.value(critic_obs).item())
            rollout.add(
                policy_obs=execution.policy_observations,
                critic_obs=execution.critic_observation,
                actions=execution.actions,
                log_probs=execution.old_log_probs,
                reward=reward,
                value=value,
                done=done,
                quantized_policy=execution.quantized_policy,
                centralized_policy=execution.centralized_policy,
            )
            episode_reward += float(reward)
            obs = next_obs
            decision_step += 1
        losses = trainer.update(rollout, np.asarray(valid_counts), gamma, gae_lambda)
        reward_history.append(episode_reward)
        if episode_metrics is not None:
            central = overlap_stats["centralized"]
            local = overlap_stats["decentralized"]
            episode_metrics.append({
                "episode": episode + 1,
                "reward": episode_reward,
                "centralized_steps": central["steps"],
                "centralized_blocked_events": central["blocked"],
                "centralized_blocked_events_per_step": (
                    central["blocked"] / central["steps"] if central["steps"] else None),
                "decentralized_steps": local["steps"],
                "decentralized_blocked_events": local["blocked"],
                "decentralized_blocked_events_per_step": (
                    local["blocked"] / local["steps"] if local["steps"] else None),
            })
        if train_cfg.log_interval and (episode + 1) % train_cfg.log_interval == 0:
            overlap_text = " ".join(
                f"{mode}={stats['blocked']} events/{stats['steps']} steps"
                for mode, stats in overlap_stats.items()
            )
            print(f"  [{topo.name}] episode {episode + 1}/{n_episodes}: "
                  f"reward={episode_reward:.3f}; central={executor.central_steps}; "
                  f"fallback={executor.local_steps}; backhaul={executor.total_transmitted_bits} bits; "
                  f"overlap_blocked central/decentral=({overlap_text}); "
                  f"actor_loss={losses['actor_loss']:.4f}; critic_loss={losses['critic_loss']:.4f}")
    env.close()
    return trainer.model, reward_history


def eval_qh_ctde(env, model: CTDEActorCritic, topo, test_steps: int = 200,
                 cfg: dict | None = None) -> tuple[float, int]:
    """Run deterministic joint actions; centralized execution transmits VQ symbols."""
    del topo
    cfg = cfg or {}
    experiment = ExperimentConfig.from_dict(cfg)
    device = experiment.train.device
    agent_cfg = experiment.agent
    delay_config = experiment.communication_delay
    valid_counts = [len(stas) for stas in env.topo.sta_positions]
    executor = CentralizedExecutor(model, valid_counts,
                                   agent_cfg.gamma_margin, device,
                                   use_age_feature=delay_config.enabled)
    obs, _ = env.reset()
    delay_channel = None
    if delay_config.enabled:
        delay_channel = ObservationDelayChannel(delay_config, len(valid_counts))
        delay_channel.reset(obs, seed=experiment.env.seed + 50_000)
    total_packets = 0.0
    for step in range(test_steps):
        delayed_obs = age_steps = None
        if delay_channel is not None:
            request_central = float(obs[0, -1]) <= agent_cfg.gamma_margin
            if request_central:
                delay_channel.transmit(obs, step)
            delayed_obs, age_steps = delay_channel.receive(step)
        execution = executor.act(
            obs, deterministic=True,
            delayed_observations=delayed_obs,
            observation_age_steps=age_steps,
        )
        action = np.asarray(execution.actions, dtype=np.int32).reshape(-1)
        obs, _, done, _, info = env.step(action)
        total_packets += float(info.get("system_throughput", 0.0))
        if done:
            break
    return total_packets, executor.total_transmitted_bits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qh_ctde.yaml")
    parser.add_argument("--topology", default=None, help="Override env.topology selector")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.topology:
        cfg["env"]["topology"] = args.topology

    seed = int(cfg.get("env", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    selector = cfg.get("env", {}).get("topology", "all")
    topology_file = cfg.get("topology_file", "configs/topologies.yaml")
    topologies = load_topologies(selector, topology_file)
    experiment = ExperimentConfig.from_dict(cfg)
    train_cfg = experiment.train
    eval_steps = train_cfg.eval_steps
    all_rewards: dict[str, list[float]] = {}
    result_dir = train_cfg.result_dir
    os.makedirs(result_dir, exist_ok=True)
    print("CTDE PPO: centralized critic + GAE; joint server action uses VQ symbol transport")

    for topo in topologies:
        print(f"\n[{topo.name}] training centralized PPO")
        model, history = train_qh_ctde(topo, cfg)
        all_rewards[topo.name] = history
        eval_env = _new_env(topo, cfg, eval_steps)
        packets, bits = eval_qh_ctde(eval_env, model, topo, eval_steps, cfg)
        eval_env.close()
        print(f"  deterministic evaluation: {packets:.0f} packets; "
              f"compressed backhaul payload: {bits} bits")
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in topo.name)
        torch.save(model.state_dict(), os.path.join(result_dir, f"qh_ctde_ppo_{safe_name}.pt"))

    np.savez_compressed(os.path.join(result_dir, "qh_ctde_ppo_rewards.npz"),
                        **{name: np.asarray(values) for name, values in all_rewards.items()})


if __name__ == "__main__":
    main()
