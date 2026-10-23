"""0-bit MAPPO for the qh scheduling environment.

Each AP has its own actor that sees only its local observation (no inter-AP messages
at execution time — the "0-bit" end of the coordination-rate curve). A centralized
critic sees the concatenated observations during training only (CTDE).

The STA head has K + 1 outputs when the env allows muting; index K is "mute". The MCS
log-probability is dropped for muted slots because the MCS choice has no effect there.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_sta_actions: int, n_mcs: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.sta = nn.Linear(hidden_dim, n_sta_actions)
        self.mcs = nn.Linear(hidden_dim, n_mcs)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        return self.sta(h), self.mcs(h)


@dataclass
class Rollout:
    obs: list
    actions: list
    log_probs: list
    rewards: list
    values: list
    dones: list


class MAPPO:
    def __init__(self, obs_dim: int, n_aps: int, k_stas: int, valid_sta_counts: list[int], *,
                 allow_mute: bool = True, n_mcs: int = 16, hidden_dim: int = 64, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95, clip: float = 0.2, epochs: int = 4,
                 entropy_coef: float = 0.01, seed: int = 0, device: str = "cpu"):
        torch.manual_seed(seed)
        self.n_aps, self.k_stas, self.n_mcs = n_aps, k_stas, n_mcs
        self.allow_mute = allow_mute
        self.gamma, self.gae_lambda, self.clip, self.epochs, self.entropy_coef = gamma, gae_lambda, clip, epochs, entropy_coef
        self.device = torch.device(device)
        n_sta_actions = k_stas + 1 if allow_mute else k_stas

        # Invalid (padding) STAs are masked; mute stays available.
        mask = torch.zeros(n_aps, n_sta_actions, dtype=torch.bool)
        for i, n_valid in enumerate(valid_sta_counts):
            mask[i, min(n_valid, k_stas):k_stas] = True
        self.sta_mask = mask.to(self.device)

        self.actors = nn.ModuleList([Actor(obs_dim, hidden_dim, n_sta_actions, n_mcs) for _ in range(n_aps)]).to(self.device)
        self.critic = nn.Sequential(nn.Linear(obs_dim * n_aps, hidden_dim), nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.actors.parameters()) + list(self.critic.parameters()), lr=lr)

    def _dists(self, obs: torch.Tensor):
        """obs: (B, n_aps, obs_dim) -> per-AP categorical distributions, each batched over B."""
        sta_logits, mcs_logits = zip(*(actor(obs[:, i]) for i, actor in enumerate(self.actors)))
        sta_logits = torch.stack(sta_logits, dim=1).masked_fill(self.sta_mask, -1e9)
        mcs_logits = torch.stack(mcs_logits, dim=1)
        return torch.distributions.Categorical(logits=sta_logits), torch.distributions.Categorical(logits=mcs_logits)

    def _joint_log_prob_entropy(self, obs: torch.Tensor, actions: torch.Tensor):
        sta_dist, mcs_dist = self._dists(obs)
        sta, mcs = actions[..., 0], actions[..., 1]
        transmits = (sta != self.k_stas).float() if self.allow_mute else torch.ones_like(sta, dtype=torch.float32)
        log_prob = sta_dist.log_prob(sta) + transmits * mcs_dist.log_prob(mcs)
        entropy = sta_dist.entropy() + transmits * mcs_dist.entropy()
        return log_prob.sum(-1), entropy.sum(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        sta_dist, mcs_dist = self._dists(o)
        if deterministic:
            sta, mcs = sta_dist.probs.argmax(-1), mcs_dist.probs.argmax(-1)
        else:
            sta, mcs = sta_dist.sample(), mcs_dist.sample()
        actions = torch.stack([sta, mcs], dim=-1)  # (1, n_aps, 2)
        log_prob, _ = self._joint_log_prob_entropy(o, actions)
        value = self.critic(o.reshape(1, -1)).squeeze()
        return actions[0].reshape(-1).cpu().numpy().astype(np.int32), float(log_prob), float(value)

    def update(self, rollout: Rollout, last_value: float = 0.0) -> dict[str, float]:
        rewards = np.asarray(rollout.rewards, dtype=np.float32)
        values = np.asarray(rollout.values + [last_value], dtype=np.float32)
        dones = np.asarray(rollout.dones, dtype=np.float32)
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            adv[t] = gae
        returns = adv + values[:-1]

        obs = torch.as_tensor(np.asarray(rollout.obs), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(rollout.actions).reshape(len(rewards), self.n_aps, 2), device=self.device)
        old_log_probs = torch.as_tensor(rollout.log_probs, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        returns_t = torch.as_tensor(returns, device=self.device)

        for _ in range(self.epochs):
            log_probs, entropy = self._joint_log_prob_entropy(obs, actions)
            ratio = torch.exp(log_probs - old_log_probs)
            actor_loss = -torch.min(ratio * adv_t, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t).mean()
            critic_loss = (self.critic(obs.reshape(len(rewards), -1)).squeeze(-1) - returns_t).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.actors.parameters()) + list(self.critic.parameters()), 0.5)
            self.optimizer.step()
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(), "entropy": entropy.mean().item()}


def train_mappo(env, learner: MAPPO, episodes: int, reward_scale: float = 1.0, seed: int = 0,
                reward_fn: str = "throughput") -> list[float]:
    """One episode = one on-policy rollout of env.max_episodes steps. Returns per-episode throughput."""
    history = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        rollout = Rollout([], [], [], [], [], [])
        done, total = False, 0.0
        while not done:
            action, log_prob, value = learner.act(obs)
            next_obs, env_reward, done, _, info = env.step(action)
            reward = info["system_throughput"] if reward_fn == "throughput" else env_reward
            rollout.obs.append(obs); rollout.actions.append(action); rollout.log_probs.append(log_prob)
            rollout.rewards.append(reward * reward_scale); rollout.values.append(value); rollout.dones.append(float(done))
            total += info["system_throughput"]
            obs = next_obs
        learner.update(rollout)
        history.append(total)
    return history
