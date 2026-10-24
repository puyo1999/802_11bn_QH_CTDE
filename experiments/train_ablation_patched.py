"""Run the five-condition QH-CTDE ablation with paired seeds.

C1: Local actor + local-observation critic
C2: Local actor + centralized critic
C3: Joint actor + topology-masked attention + centralized critic
C4: Joint actor + VQ observation transport + centralized critic
C5: Dummy joint actor + centralized critic

The existing QH-CTDE environment and PPO trainer are reused.  This driver only
adds the condition-specific actor/critic hooks and experiment logging.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.agents.ctde_ppo import CTDEActorCritic
from src.config import dump_config, load_config
from src.config.schema import ExperimentConfig
from src.envs.env_factory import make_env
from src.envs.overlap_exposure import OverlapExposureTracker
from src.envs.space_validator import build_adjacency, validate_spaces
from src.envs.topology_registry import load_topologies
from src.training.centralized_execution import CentralizedExecutor
from src.training.communication_delay import ObservationDelayChannel
from src.training.ppo import CTDEPPOTrainer, PPOConfig, PPORollout


CONDITIONS = {
    "C1": dict(label="Local actor + Local critic", execution_mode="local",
               critic="local", actor="local"),
    "C2": dict(label="Local actor + Central critic", execution_mode="local",
               critic="central", actor="local"),
    "C3": dict(label="Joint actor + Attention", execution_mode="joint_raw",
               critic="central", actor="joint_attention"),
    "C4": dict(label="Joint actor + VQ", execution_mode="joint_vq",
               critic="central", actor="joint_attention"),
    "C5": dict(label="Dummy joint capacity control", execution_mode="joint_raw",
               critic="central", actor="dummy_joint"),
}


class AblationActorCritic(CTDEActorCritic):
    """Existing CTDE model with explicit C1/C5 ablation hooks."""

    def __init__(self, *args, critic_type="central",
                 actor_type="joint_attention", **kwargs):
        super().__init__(*args, **kwargs)
        if critic_type not in {"central", "local"}:
            raise ValueError("critic_type must be 'central' or 'local'")
        if actor_type not in {"local", "joint_attention", "dummy_joint"}:
            raise ValueError("unsupported actor_type")

        self.critic_type = critic_type
        self.actor_type = actor_type

        if critic_type == "local":
            hidden_dim = self.critic[0].out_features
            self.local_value = nn.Sequential(
                nn.Linear(self.observation_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

    def actor_logits(self, local_observations, valid_sta_counts=None,
                     centralized=False):
        features = self.local_encoder(local_observations)

        if self.actor_type == "dummy_joint":
            # C5 keeps the joint-attention capacity but blocks cross-AP
            # information. The attention mask permits only self-attention.
            dummy_mask = torch.ones_like(self.attention_mask)
            dummy_mask.fill_diagonal_(False)
            context, _ = self.joint_attention(
                features, features, features,
                attn_mask=dummy_mask, need_weights=False,
            )
            joint_features = self.joint_actor(
                torch.cat((features, context), dim=-1)
            )
            sta_logits = self.joint_sta_head(joint_features)
            mcs_logits = self.joint_mcs_head(joint_features)
        else:
            context, _ = self.joint_attention(
                features, features, features,
                attn_mask=self.attention_mask, need_weights=False,
            )
            joint_features = self.joint_actor(
                torch.cat((features, context), dim=-1)
            )
            local_sta = self.local_sta_head(features)
            local_mcs = self.local_mcs_head(features)
            joint_sta = self.joint_sta_head(joint_features)
            joint_mcs = self.joint_mcs_head(joint_features)

            if isinstance(centralized, torch.Tensor):
                mode_mask = centralized.to(
                    device=features.device, dtype=torch.bool
                ).view(-1, 1, 1)
                sta_logits = torch.where(mode_mask, joint_sta, local_sta)
                mcs_logits = torch.where(mode_mask, joint_mcs, local_mcs)
            elif centralized:
                sta_logits, mcs_logits = joint_sta, joint_mcs
            else:
                sta_logits, mcs_logits = local_sta, local_mcs

        if valid_sta_counts is not None:
            sta_ids = torch.arange(self.n_stas, device=sta_logits.device)
            mask = sta_ids.view(1, 1, -1) >= valid_sta_counts.view(
                -1, self.n_agents, 1
            )
            sta_logits = sta_logits.masked_fill(mask, -1.0e9)
        return sta_logits, mcs_logits

    def value(self, observations):
        if self.critic_type == "central":
            return super().value(observations)

        # C1 uses only o_i. The PPO implementation expects one scalar value
        # per rollout sample, so the per-AP local estimates are averaged.
        if observations.ndim == 2:
            observations = observations.unsqueeze(0)
        if observations.ndim != 3:
            raise ValueError("local critic expects [B,N,D] observations")
        values = self.local_value(observations).squeeze(-1)
        return values.mean(dim=-1)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def new_env(topo, cfg, horizon, seed_offset=0):
    experiment = ExperimentConfig.from_dict(cfg)
    env_config = replace(
        experiment.env,
        max_episodes=horizon,
        seed=experiment.env.seed + seed_offset,
    )
    if env_config.impl != "qh_v2":
        raise ValueError("QH-CTDE ablation requires env.impl=qh_v2")
    env = make_env(
        env_config, topology=topo, seed_offset=0,
        topology_file=experiment.topology_file,
    )
    space = validate_spaces(
        env, topo, declared_state_dim=experiment.agent.state_dim, strict=True
    )
    env.space_spec = space
    return env


def build_trainer(env, cfg, condition):
    experiment = ExperimentConfig.from_dict(cfg)
    agent_cfg = experiment.agent
    train_cfg = experiment.train
    delay_enabled = experiment.communication_delay.enabled
    spec = CONDITIONS[condition]

    model = AblationActorCritic(
        observation_dim=env.space_spec.local_obs_dim + int(delay_enabled),
        n_agents=env.space_spec.n_aps,
        n_stas=env.space_spec.action_dims_per_ap[0],
        n_mcs=env.space_spec.action_dims_per_ap[1],
        hidden_dim=agent_cfg.hidden_dim,
        vq_bits=agent_cfg.vq_bits,
        adjacency=torch.as_tensor(
            build_adjacency(env.topo), dtype=torch.bool
        ),
        critic_type=spec["critic"],
        actor_type=spec["actor"],
    )
    return CTDEPPOTrainer(
        model, lr=agent_cfg.lr,
        ppo_config=PPOConfig.from_dict(train_cfg.ppo),
        device=train_cfg.device,
    )


def train_condition(topo, cfg, condition, seed):
    experiment = ExperimentConfig.from_dict(cfg)
    train_cfg = experiment.train
    agent_cfg = experiment.agent
    delay_cfg = experiment.communication_delay

    env = new_env(topo, cfg, train_cfg.rollout_length, seed_offset=seed)
    trainer = build_trainer(env, cfg, condition)
    valid_counts = [len(stas) for stas in topo.sta_positions]
    spec = CONDITIONS[condition]

    executor = CentralizedExecutor(
        trainer.model, valid_counts, threshold=agent_cfg.gamma_margin,
        device=train_cfg.device, execution_mode=spec["execution_mode"],
        use_age_feature=delay_cfg.enabled,
    )

    reward_history = []
    overlap_rows = []
    episode_diag = []
    seed_base = int(cfg.get("env", {}).get("seed", 42)) + seed
    gamma = agent_cfg.gamma
    gae_lambda = PPOConfig.from_dict(train_cfg.ppo).gae_lambda


    # C4 only: initialize the VQ codebook from multiple real reset observations.
    # Keep the existing checkpoint-protection logic in ObservationCodec.
    if condition == "C4" and not trainer.model.codec._initialized:
        init_observations = []

        # 8 reset samples provide 8 * n_agents observation vectors.
        for init_idx in range(8):
            init_obs, _ = env.reset(seed=seed_base + init_idx)

            # Match the model's observation dimension when age features are enabled.
            if delay_cfg.enabled:
                zero_age = np.zeros(
                    (init_obs.shape[0], 1), dtype=np.float32
                )
                init_obs = np.concatenate(
                    [init_obs.astype(np.float32), zero_age], axis=-1
                )

            init_observations.append(init_obs)

        init_array = np.concatenate(init_observations, axis=0)
        init_tensor = torch.as_tensor(
            init_array, dtype=torch.float32, device=trainer.device
        ).unsqueeze(0)

        with torch.no_grad():
            trainer.model.encode_observations(init_tensor)

        print(
            f"[C4 VQ init] initialized={trainer.model.codec._initialized}, "
            f"samples={init_array.shape[0]}, "
            f"codebook_size={trainer.model.codec.codebook_size}"
        )

    # Restore the initial training state before the actual rollout loop.
    env.reset(seed=seed_base)

    try:
        for episode in range(train_cfg.max_episodes):
            obs, _ = env.reset(seed=seed_base + episode)
            delay_channel = None
            if delay_cfg.enabled:
                delay_channel = ObservationDelayChannel(
                    delay_cfg, len(valid_counts)
                )
                delay_channel.reset(
                    obs, seed=seed_base + episode + 50_000
                )

            decision_step = 0
            rollout = PPORollout.empty()
            episode_reward = 0.0
            mcs_sum = 0.0
            thp_positive_steps = 0
            episode_throughput = 0.0
            overlap_stats = {
                "centralized": {"steps": 0, "blocked": 0},
                "decentralized": {"steps": 0, "blocked": 0},
            }

            done = False
            while not done:
                delayed_obs = age_steps = None
                if delay_channel is not None:
                    request_central = (
                        spec["execution_mode"] in {"joint_raw", "joint_vq"}
                        or (
                            spec["execution_mode"] == "auto"
                            and float(obs[0, -1]) <= agent_cfg.gamma_margin
                        )
                    )
                    if request_central:
                        delay_channel.transmit(obs, decision_step)
                    delayed_obs, age_steps = delay_channel.receive(
                        decision_step
                    )

                execution = executor.act(
                    obs, deterministic=False,
                    delayed_observations=delayed_obs,
                    observation_age_steps=age_steps,
                )
                action = np.asarray(execution.actions, dtype=np.int32).reshape(-1)
                next_obs, reward, done, _, info = env.step(action)

                mcs_sum += float(np.mean(action[1::2]))
                step_thp = float(info.get("system_throughput", 0.0))
                episode_throughput += step_thp
                thp_positive_steps += int(step_thp > 0)

                mode_name = (
                    "centralized" if execution.centralized_policy
                    else "decentralized"
                )
                overlap_stats[mode_name]["steps"] += 1
                overlap_stats[mode_name]["blocked"] += int(
                    info.get("overlap_blocked_step", 0)
                )

                critic_obs = torch.as_tensor(
                    execution.critic_observation, dtype=torch.float32,
                    device=trainer.device,
                ).unsqueeze(0)
                with torch.no_grad():
                    value = float(trainer.model.value(critic_obs).item())

                rollout.add(
                    policy_obs=execution.policy_observations,
                    critic_obs=execution.critic_observation,
                    actions=execution.actions,
                    log_probs=execution.old_log_probs,
                    reward=reward, value=value, done=done,
                    quantized_policy=execution.quantized_policy,
                    centralized_policy=execution.centralized_policy,
                )
                episode_reward += float(reward)
                obs = next_obs
                decision_step += 1

            losses = trainer.update(
                rollout, np.asarray(valid_counts), gamma, gae_lambda
            )
            reward_history.append(episode_reward)
            episode_diag.append({
                "topology": topo.metadata.get("id", topo.name),
                "condition_id": condition, "seed": seed, "episode": episode + 1,
                "reward": float(episode_reward),
                "mean_selected_mcs": mcs_sum / max(decision_step, 1),
                "throughput_packets": episode_throughput,
                "thp_positive_step_fraction": thp_positive_steps / max(decision_step, 1),
            })

            for mode, stats in overlap_stats.items():
                overlap_rows.append({
                    "topology": topo.metadata.get("id", topo.name),
                    "condition_id": condition, "seed": seed,
                    "episode": episode + 1, "mode": mode,
                    "steps": stats["steps"],
                    "blocked_events": stats["blocked"],
                    "blocked_events_per_step": (
                        stats["blocked"] / stats["steps"]
                        if stats["steps"] else ""
                    ),
                })

            if train_cfg.log_interval and (episode + 1) % train_cfg.log_interval == 0:
                print(
                    f"[{topo.name}][{condition}][seed={seed}] "
                    f"episode {episode + 1}/{train_cfg.max_episodes}: "
                    f"reward={episode_reward:.3f}; "
                    f"mean_mcs={episode_diag[-1]['mean_selected_mcs']:.2f}; "
                    f"thp_steps={episode_diag[-1]['thp_positive_step_fraction']:.2f}; "
                    f"actor_loss={losses['actor_loss']:.4f}; "
                    f"critic_loss={losses['critic_loss']:.4f}",
                    flush=True,
                )
    finally:
        env.close()

    return trainer.model, reward_history, overlap_rows, episode_diag


def first_present(info: dict[str, Any], names):
    for name in names:
        if name in info:
            return info[name]
    return None


def evaluate(model, topo, cfg, condition, seed):
    experiment = ExperimentConfig.from_dict(cfg)
    train_cfg = experiment.train
    agent_cfg = experiment.agent
    delay_cfg = experiment.communication_delay
    eval_seed = seed + 10_000

    env = new_env(topo, cfg, train_cfg.eval_steps, seed_offset=eval_seed)
    executor = CentralizedExecutor(
        model, [len(stas) for stas in topo.sta_positions],
        threshold=agent_cfg.gamma_margin, device=train_cfg.device,
        execution_mode=CONDITIONS[condition]["execution_mode"],
        use_age_feature=delay_cfg.enabled,
    )

    obs, _ = env.reset(seed=eval_seed)
    delay_channel = None
    if delay_cfg.enabled:
        delay_channel = ObservationDelayChannel(
            delay_cfg, len(topo.sta_positions)
        )
        delay_channel.reset(obs, seed=eval_seed + 50_000)

    throughput = blocked = steps = 0
    tracker = OverlapExposureTracker.for_topology(topo)
    per_ap_throughput = np.zeros(len(topo.sta_positions), dtype=np.float64)
    exposure = {}

    try:
        for step in range(train_cfg.eval_steps):
            delayed_obs = age_steps = None
            if delay_channel is not None:
                request_central = CONDITIONS[condition]["execution_mode"] in {
                    "joint_raw", "joint_vq"
                }
                if request_central:
                    delay_channel.transmit(obs, step)
                delayed_obs, age_steps = delay_channel.receive(step)

            decision = executor.act(
                obs, deterministic=True,
                delayed_observations=delayed_obs,
                observation_age_steps=age_steps,
            )
            action = np.asarray(decision.actions, dtype=np.int32).reshape(-1)
            tracker.record_step(action)  # env.step()에 넘기는 것과 같은 액션
            obs, _, done, _, info = env.step(action)

            throughput += float(info.get("system_throughput", 0.0))
            blocked += int(info.get("overlap_blocked_step", 0))
            per_ap_throughput += np.asarray(info.get("thp", []), dtype=np.float64)
            steps += 1

            if done:
                break

        # env 상태가 살아 있는 지금 overlap STA의 잔여 큐를 읽는다.
        if tracker.is_active():
            exposure = tracker.summary(env)
    finally:
        env.close()

    # per-AP 누적 throughput 기준 Jain 공정성. throughput 합이 0이면 정의되지 않으므로
    # 1.0으로 두지 않고 빈 값으로 남긴다.
    sq_sum = float(np.sum(per_ap_throughput ** 2))
    fairness = (
        float(per_ap_throughput.sum() ** 2 / (len(per_ap_throughput) * sq_sum))
        if sq_sum > 0 else ""
    )

    return {
        "topology": topo.metadata.get("id", topo.name),
        "condition_id": condition, "seed": seed,
        "evaluation_seed": eval_seed,
        "throughput_packets": throughput,
        "blocked_events": blocked,
        "evaluation_steps": steps,
        "blocked_events_per_step": blocked / max(steps, 1),
        "backhaul_bits": executor.total_transmitted_bits,
        "backhaul_bits_per_step": executor.total_transmitted_bits / max(steps, 1),
        # overlap STA가 없는 토폴로지(T1/T2/T4/T5)에서는 빈 값 (0과 구분).
        "overlap_sta_selections_eval": exposure.get("overlap_sta_selected_events", ""),
        "overlap_sta_final_queue_sum": exposure.get("overlap_sta_final_queue_sum", ""),
        "fairness_mean": fairness,
    }


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qh_ctde.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", choices=list(CONDITIONS), default=None)
    parser.add_argument("--topologies", nargs="+", default=["T3", "T6"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output_dir", default="./results_csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        # Pipeline compatibility: --steps is a total training-step budget.
        # The existing implementation updates PPO once per episode, so convert
        # the requested budget to an episode count conservatively.
        rollout_length = max(int(cfg["train"]["rollout_length"]), 1)
        cfg["train"]["max_episodes"] = max(
            1, int(np.ceil(args.steps / rollout_length))
        )

    topologies = load_topologies(
        args.topologies, cfg.get("topology_file", "configs/topologies.yaml")
    )
    conditions = [args.condition] if args.condition else list(CONDITIONS)

    os.makedirs(args.output_dir, exist_ok=True)
    dump_config(cfg, os.path.join(args.output_dir, "resolved_config.yaml"))

    manifest = {
        "seed": args.seed, "conditions": conditions,
        "topologies": args.topologies,
        "max_episodes": cfg["train"]["max_episodes"],
        "rollout_length": cfg["train"]["rollout_length"],
        "eval_steps": cfg["train"]["eval_steps"],
    }
    with open(
        os.path.join(args.output_dir, f"manifest_seed_{args.seed}.json"),
        "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)

    for topo in topologies:
        for condition in conditions:
            topology_id = topo.metadata.get("id", topo.name)
            print(
                f"\n=== {condition}: {CONDITIONS[condition]['label']} | "
                f"topology={topology_id} | seed={args.seed} ===",
                flush=True,
            )
            run_cfg = {
                **cfg, "env": {**cfg.get("env", {}), "seed": args.seed}
            }
            seed_everything(args.seed)

            model, rewards, overlap_rows, episode_diag = train_condition(
                topo, run_cfg, condition, args.seed
            )
            evaluation = evaluate(model, topo, run_cfg, condition, args.seed)

            prefix = f"{topology_id}_{condition}_seed{args.seed}"
            write_csv(
                os.path.join(args.output_dir, f"{prefix}_training_rewards.csv"),
                [{
                    "topology": topology_id, "condition_id": condition,
                    "seed": args.seed, "episode": i + 1,
                    "reward": float(r),
                } for i, r in enumerate(rewards)]
            )
            write_csv(
                os.path.join(args.output_dir, f"{prefix}_training_overlap.csv"),
                overlap_rows,
            )
            write_csv(
                os.path.join(args.output_dir, f"{prefix}_training_diagnostics.csv"),
                episode_diag,
            )
            write_csv(
                os.path.join(args.output_dir, f"{prefix}_evaluation.csv"),
                [evaluation],
            )
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, f"{prefix}_model.pt"),
            )

            print(
                f"[DONE] throughput={evaluation['throughput_packets']:.1f}; "
                f"blocked/step={evaluation['blocked_events_per_step']:.6f}; "
                f"backhaul={evaluation['backhaul_bits_per_step']:.1f} bits/step",
                flush=True,
            )


if __name__ == "__main__":
    main()
