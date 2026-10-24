"""Rollout storage, GAE and clipped PPO updates for CTDE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.ctde_ppo import CTDEActorCritic
from src.debug_c4 import log


@dataclass(frozen=True)
class PPOConfig:
    """PPO-specific hyperparameters parsed from ``train.ppo``."""
    clip_ratio: float = 0.2
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4
    minibatch_size: int = 64
    grad_clip: float = 0.5

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "PPOConfig":
        values = values or {}
        if not isinstance(values, dict):
            raise TypeError("train.ppo must be a mapping")
        config = cls(**{key: value for key, value in values.items()
                        if key in cls.__dataclass_fields__})
        if not 0.0 < config.clip_ratio < 1.0:
            raise ValueError("PPO clip_ratio must be between 0 and 1")
        if not 0.0 <= config.gae_lambda <= 1.0:
            raise ValueError("PPO gae_lambda must be between 0 and 1")
        if config.update_epochs < 1 or config.minibatch_size < 1:
            raise ValueError("PPO update_epochs and minibatch_size must be positive")
        if config.grad_clip <= 0.0:
            raise ValueError("PPO grad_clip must be positive")
        return config


@dataclass
class PPORollout:
    policy_obs: list[np.ndarray]
    critic_obs: list[np.ndarray]
    actions: list[np.ndarray]
    log_probs: list[np.ndarray]
    rewards: list[float]
    values: list[float]
    dones: list[bool]
    quantized_policy: list[bool]
    centralized_policy: list[bool]

    @classmethod
    def empty(cls):
        return cls([], [], [], [], [], [], [], [], [])

    def add(self, policy_obs, critic_obs, actions, log_probs, reward, value, done, quantized_policy, centralized_policy):
        self.policy_obs.append(np.asarray(policy_obs, dtype=np.float32))
        self.critic_obs.append(np.asarray(critic_obs, dtype=np.float32))
        self.actions.append(np.asarray(actions, dtype=np.int64))
        self.log_probs.append(np.asarray(log_probs, dtype=np.float32))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))
        self.quantized_policy.append(bool(quantized_policy))
        self.centralized_policy.append(bool(centralized_policy))

    def gae(self, last_value: float, gamma: float, gae_lambda: float):
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values + [last_value], dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[t] = gae
        return advantages, advantages + values[:-1]


class CTDEPPOTrainer:
    """Optimizes decentralized actor and centralized value critic with PPO/GAE."""

    def __init__(self, model: CTDEActorCritic, *, lr: float,
                 ppo_config: PPOConfig, device: str):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.config = ppo_config
        self.actor_optimizer = optim.Adam(
            list(model.local_encoder.parameters()) + list(model.joint_attention.parameters())
            + list(model.local_sta_head.parameters()) + list(model.local_mcs_head.parameters())
            + list(model.joint_actor.parameters()) + list(model.joint_sta_head.parameters())
            + list(model.joint_mcs_head.parameters()) + list(model.codec.parameters()), lr=lr)
        self.critic_optimizer = optim.Adam(model.critic.parameters(), lr=lr)

    def update(self, rollout: PPORollout, valid_sta_counts: np.ndarray,
               gamma: float, gae_lambda: float,
               last_value: float | None = None) -> dict[str, float]:
        t_count = len(rollout.rewards)
        if t_count == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0}
        if last_value is None:
            if rollout.dones[-1]:
                last_value = 0.0
            else:
                raise ValueError("A non-terminal rollout must provide V(next_joint_observation) as last_value")
        advantages, returns = rollout.gae(last_value, gamma, gae_lambda)
        obs = torch.as_tensor(np.asarray(rollout.policy_obs), dtype=torch.float32, device=self.device)
        joint_obs = torch.as_tensor(np.asarray(rollout.critic_obs), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(rollout.actions), dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(rollout.log_probs), dtype=torch.float32, device=self.device)
        quantized_mask = torch.as_tensor(np.asarray(rollout.quantized_policy), dtype=torch.bool, device=self.device)
        centralized_mask = torch.as_tensor(np.asarray(rollout.centralized_policy), dtype=torch.bool, device=self.device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        log("PPO update start steps=%d quantized_steps=%d centralized_steps=%d obs_shape=%s action_shape=%s old_logp_mean=%.8g old_logp_std=%.8g",
            t_count, int(quantized_mask.sum().item()), int(centralized_mask.sum().item()),
            tuple(obs.shape), tuple(actions.shape), float(old_log_probs.mean().item()),
            float(old_log_probs.std(unbiased=False).item()))
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        counts = torch.as_tensor(valid_sta_counts, dtype=torch.long, device=self.device)
        totals = {"actor_loss": 0.0, "critic_loss": 0.0, "vq_loss": 0.0, "updates": 0}
        for _ in range(self.config.update_epochs):
            order = torch.randperm(t_count, device=self.device)
            for ids in order.split(self.config.minibatch_size):
                raw_obs = obs[ids]
                quantized, _, vq_loss = self.model.encode_observations(raw_obs)
                actor_obs = torch.where(quantized_mask[ids, None, None], quantized, raw_obs)
                log_probs, entropy = self.model.evaluate_actions(
                    actor_obs, actions[ids], counts.unsqueeze(0).expand(len(ids), -1),
                    centralized=centralized_mask[ids],
                )
                logp_delta = log_probs - old_log_probs[ids]
                ratio = torch.exp(logp_delta)
                log("PPO minibatch ids=%s vq_loss=%.8g logp_delta_mean=%.8g logp_delta_absmax=%.8g ratio_mean=%.8g ratio_min=%.8g ratio_max=%.8g actor_obs_mse_vs_raw=%.8g",
                    ids.detach().cpu().tolist(), float(vq_loss.detach().item()),
                    float(logp_delta.detach().mean().item()),
                    float(logp_delta.detach().abs().max().item()),
                    float(ratio.detach().mean().item()), float(ratio.detach().min().item()),
                    float(ratio.detach().max().item()),
                    float((actor_obs - raw_obs).detach().square().mean().item()))
                adv_batch = adv[ids].unsqueeze(-1).expand_as(ratio)
                unclipped = ratio * adv_batch
                clipped = torch.clamp(ratio, 1.0-self.config.clip_ratio,
                                      1.0+self.config.clip_ratio) * adv_batch
                actor_loss = (-torch.minimum(unclipped, clipped).mean()
                              - self.config.entropy_coef * entropy.mean() + 0.1 * vq_loss)
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_parameters = (
                    list(self.model.local_encoder.parameters())
                    + list(self.model.joint_attention.parameters())
                    + list(self.model.local_sta_head.parameters())
                    + list(self.model.local_mcs_head.parameters())
                    + list(self.model.joint_actor.parameters())
                    + list(self.model.joint_sta_head.parameters())
                    + list(self.model.joint_mcs_head.parameters())
                    + list(self.model.codec.parameters())
                )
                grad_norm = nn.utils.clip_grad_norm_(actor_parameters, self.config.grad_clip)
                codebook_grad_norm = (
                    float(self.model.codec.codebook.grad.detach().norm().item())
                    if self.model.codec.codebook.grad is not None else None
                )
                self.actor_optimizer.step()
                log("PPO actor_step actor_loss=%.8g grad_norm_before_clip=%.8g codec_codebook_grad_norm=%s",
                    float(actor_loss.detach().item()), float(grad_norm.detach().item()),
                    codebook_grad_norm)

                values = self.model.value(joint_obs[ids])
                critic_loss = 0.5 * (values - returns_t[ids]).square().mean() * self.config.value_coef
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.config.grad_clip)
                self.critic_optimizer.step()
                totals["actor_loss"] += float(actor_loss.detach())
                totals["critic_loss"] += float(critic_loss.detach())
                totals["vq_loss"] += float(vq_loss.detach())
                totals["updates"] += 1
        n = max(int(totals["updates"]), 1)
        return {k: v/n for k,v in totals.items() if k != "updates"}
