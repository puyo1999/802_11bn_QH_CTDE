"""PPO learners for the qh scheduling environment, at the two ends of the coordination-rate curve.

* MAPPO        — 0-bit: each AP has its own actor that sees only its local observation
                 (no inter-AP messages at execution time). A centralized critic sees the
                 concatenated observations during training only (CTDE).
* CentralPPO   — ∞-bit: one actor sees every AP's observation and picks the joint action
                 autoregressively (AP i conditions on the choices of APs < i), so it can
                 select a joint transmission set deterministically. This is the learned
                 counterpart of the genie oracle and separates "missing information" from
                 "failed learning" when 0-bit MAPPO falls short.

The STA head has K + 1 outputs when the env allows muting; index K is "mute". The MCS
log-probability is dropped for muted slots because the MCS choice has no effect there.

`ClockObs` appends a one-hot slot phase (step mod period) to every AP's observation. APs
share a TSF timer in Wi-Fi, so this is a 0-bit common time reference, not a message.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


def _mlp(in_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())


class Actor(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_sta_actions: int, n_mcs: int):
        super().__init__()
        self.body = _mlp(obs_dim, hidden_dim)
        self.sta = nn.Linear(hidden_dim, n_sta_actions)
        self.mcs = nn.Linear(hidden_dim, n_mcs)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        return self.sta(h), self.mcs(h)


@dataclass
class Rollout:
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)


class _PPOBase:
    """Shared PPO machinery. Subclasses provide the actor parameters, `_sample` and `_evaluate`."""

    def __init__(self, obs_dim: int, n_aps: int, k_stas: int, valid_sta_counts: list[int], *,
                 allow_mute: bool = True, n_mcs: int = 16, hidden_dim: int = 128, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95, clip: float = 0.2, epochs: int = 4,
                 minibatches: int = 4, entropy_coef: float = 0.01, seed: int = 0, device: str = "cpu"):
        torch.manual_seed(seed)
        self.n_aps, self.k_stas, self.n_mcs = n_aps, k_stas, n_mcs
        self.allow_mute = allow_mute
        self.gamma, self.gae_lambda, self.clip = gamma, gae_lambda, clip
        self.epochs, self.minibatches = epochs, minibatches
        self.entropy_coef = self.entropy_coef_start = entropy_coef
        self.lr_start = lr
        self.device = torch.device(device)
        self.n_sta_actions = k_stas + 1 if allow_mute else k_stas

        # Invalid (padding) STAs are masked; mute stays available.
        mask = torch.zeros(n_aps, self.n_sta_actions, dtype=torch.bool)
        for i, n_valid in enumerate(valid_sta_counts):
            mask[i, min(n_valid, k_stas):k_stas] = True
        self.sta_mask = mask.to(self.device)

        self._build_actor(obs_dim, hidden_dim)
        self.critic = nn.Sequential(_mlp(obs_dim * n_aps, hidden_dim), nn.Linear(hidden_dim, 1)).to(self.device)
        self.params = list(self._actor_parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=lr)

    def _transmits(self, sta: torch.Tensor) -> torch.Tensor:
        return (sta != self.k_stas).float() if self.allow_mute else torch.ones_like(sta, dtype=torch.float32)

    def set_progress(self, frac: float) -> None:
        """Linear annealing of the learning rate and entropy bonus over training (frac in [0, 1])."""
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr_start * max(1.0 - frac, 0.05)
        self.entropy_coef = self.entropy_coef_start * max(1.0 - frac, 0.0)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions, log_prob = self._sample(o, deterministic)  # (1, n_aps, 2), (1,)
        value = self.critic(o.reshape(1, -1)).squeeze()
        return actions[0].reshape(-1).cpu().numpy().astype(np.int32), float(log_prob), float(value)

    def update(self, rollouts: list[Rollout]) -> dict[str, float]:
        obs, actions, old_log_probs, advs, returns = [], [], [], [], []
        for rollout in rollouts:
            rewards = np.asarray(rollout.rewards, dtype=np.float32)
            values = np.asarray(rollout.values + [0.0], dtype=np.float32)
            dones = np.asarray(rollout.dones, dtype=np.float32)
            adv, gae = np.zeros_like(rewards), 0.0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                adv[t] = gae
            obs.extend(rollout.obs); actions.extend(rollout.actions); old_log_probs.extend(rollout.log_probs)
            advs.append(adv); returns.append(adv + values[:-1])

        n = len(obs)
        obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(actions).reshape(n, self.n_aps, 2), dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(np.concatenate(advs), device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        returns_t = torch.as_tensor(np.concatenate(returns), device=self.device)

        for _ in range(self.epochs):
            for idx in torch.randperm(n, device=self.device).chunk(self.minibatches):
                log_probs, entropy = self._evaluate(obs[idx], actions[idx])
                ratio = torch.exp(log_probs - old_log_probs[idx])
                clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
                actor_loss = -torch.min(ratio * adv_t[idx], clipped * adv_t[idx]).mean()
                critic_loss = (self.critic(obs[idx].reshape(len(idx), -1)).squeeze(-1) - returns_t[idx]).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, 0.5)
                self.optimizer.step()
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(), "entropy": entropy.mean().item()}


class MAPPO(_PPOBase):
    """0-bit: independent local actors, centralized critic."""

    def _build_actor(self, obs_dim: int, hidden_dim: int) -> None:
        self.actors = nn.ModuleList([Actor(obs_dim, hidden_dim, self.n_sta_actions, self.n_mcs)
                                     for _ in range(self.n_aps)]).to(self.device)

    def _actor_parameters(self):
        return self.actors.parameters()

    def _dists(self, obs: torch.Tensor):
        """obs: (B, n_aps, obs_dim) -> per-AP categorical distributions, each batched over B."""
        sta_logits, mcs_logits = zip(*(actor(obs[:, i]) for i, actor in enumerate(self.actors)))
        sta_logits = torch.stack(sta_logits, dim=1).masked_fill(self.sta_mask, -1e9)
        mcs_logits = torch.stack(mcs_logits, dim=1)
        return torch.distributions.Categorical(logits=sta_logits), torch.distributions.Categorical(logits=mcs_logits)

    def _evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        sta_dist, mcs_dist = self._dists(obs)
        sta, mcs = actions[..., 0], actions[..., 1]
        transmits = self._transmits(sta)
        log_prob = sta_dist.log_prob(sta) + transmits * mcs_dist.log_prob(mcs)
        entropy = sta_dist.entropy() + transmits * mcs_dist.entropy()
        return log_prob.sum(-1), entropy.sum(-1)

    def _sample(self, obs: torch.Tensor, deterministic: bool):
        sta_dist, mcs_dist = self._dists(obs)
        if deterministic:
            sta, mcs = sta_dist.probs.argmax(-1), mcs_dist.probs.argmax(-1)
        else:
            sta, mcs = sta_dist.sample(), mcs_dist.sample()
        actions = torch.stack([sta, mcs], dim=-1)
        return actions, self._evaluate(obs, actions)[0]


class CentralPPO(_PPOBase):
    """∞-bit: one autoregressive actor over the joint action, conditioned on all observations."""

    def _build_actor(self, obs_dim: int, hidden_dim: int) -> None:
        prev_dim = self.n_aps * (self.n_sta_actions + self.n_mcs)  # one-hot choices of earlier APs
        self.trunk = _mlp(obs_dim * self.n_aps, hidden_dim).to(self.device)
        self.heads = nn.ModuleList([Actor(hidden_dim + prev_dim + self.n_aps, hidden_dim, self.n_sta_actions, self.n_mcs)
                                    for _ in range(self.n_aps)]).to(self.device)

    def _actor_parameters(self):
        return list(self.trunk.parameters()) + list(self.heads.parameters())

    def _step_dists(self, h: torch.Tensor, prev: torch.Tensor, i: int):
        ap_id = torch.zeros(len(h), self.n_aps, device=self.device)
        ap_id[:, i] = 1.0
        sta_logits, mcs_logits = self.heads[i](torch.cat([h, prev, ap_id], dim=-1))
        sta_logits = sta_logits.masked_fill(self.sta_mask[i], -1e9)
        return torch.distributions.Categorical(logits=sta_logits), torch.distributions.Categorical(logits=mcs_logits)

    def _write_prev(self, prev: torch.Tensor, i: int, sta: torch.Tensor, mcs: torch.Tensor) -> torch.Tensor:
        width = self.n_sta_actions + self.n_mcs
        prev = prev.clone()
        prev[:, i * width:i * width + self.n_sta_actions] = nn.functional.one_hot(sta, self.n_sta_actions).float()
        prev[:, i * width + self.n_sta_actions:(i + 1) * width] = nn.functional.one_hot(mcs, self.n_mcs).float()
        return prev

    def _rollout(self, obs: torch.Tensor, actions: torch.Tensor | None, deterministic: bool = False):
        """Sample (actions=None) or teacher-force stored actions; returns actions, log-prob, entropy."""
        h = self.trunk(obs.reshape(len(obs), -1))
        prev = torch.zeros(len(obs), self.n_aps * (self.n_sta_actions + self.n_mcs), device=self.device)
        chosen, log_prob, entropy = [], 0.0, 0.0
        for i in range(self.n_aps):
            sta_dist, mcs_dist = self._step_dists(h, prev, i)
            if actions is not None:
                sta, mcs = actions[:, i, 0], actions[:, i, 1]
            elif deterministic:
                sta, mcs = sta_dist.probs.argmax(-1), mcs_dist.probs.argmax(-1)
            else:
                sta, mcs = sta_dist.sample(), mcs_dist.sample()
            transmits = self._transmits(sta)
            log_prob = log_prob + sta_dist.log_prob(sta) + transmits * mcs_dist.log_prob(mcs)
            entropy = entropy + sta_dist.entropy() + transmits * mcs_dist.entropy()
            prev = self._write_prev(prev, i, sta, mcs)
            chosen.append(torch.stack([sta, mcs], dim=-1))
        return torch.stack(chosen, dim=1), log_prob, entropy

    def _evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        _, log_prob, entropy = self._rollout(obs, actions)
        return log_prob, entropy

    def _sample(self, obs: torch.Tensor, deterministic: bool):
        actions, log_prob, _ = self._rollout(obs, None, deterministic)
        return actions, log_prob


class ClockObs:
    """Env wrapper that appends a one-hot slot phase (step mod period) to each AP's observation."""

    def __init__(self, env, period: int):
        self.env, self.period = env, period

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        phase = np.zeros((obs.shape[0], self.period), dtype=np.float32)
        phase[:, self.env._step % self.period] = 1.0
        return np.concatenate([obs, phase], axis=1)

    @property
    def obs_dim(self) -> int:
        return self.env.observation_space.shape[-1] + self.period

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._augment(obs), reward, done, truncated, info


def train_mappo(env, learner: _PPOBase, episodes: int, reward_scale: float = 1.0, seed: int = 0,
                reward_fn: str = "throughput", episodes_per_update: int = 1) -> list[float]:
    """Collects `episodes_per_update` on-policy episodes per PPO update. Returns per-episode throughput."""
    history, batch = [], []
    for ep in range(episodes):
        learner.set_progress(ep / max(episodes, 1))
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        rollout = Rollout()
        done, total = False, 0.0
        while not done:
            action, log_prob, value = learner.act(obs)
            next_obs, env_reward, done, _, info = env.step(action)
            reward = info["system_throughput"] if reward_fn == "throughput" else env_reward
            rollout.obs.append(obs); rollout.actions.append(action); rollout.log_probs.append(log_prob)
            rollout.rewards.append(reward * reward_scale); rollout.values.append(value); rollout.dones.append(float(done))
            total += info["system_throughput"]
            obs = next_obs
        batch.append(rollout)
        history.append(total)
        if len(batch) == episodes_per_update or ep == episodes - 1:
            learner.update(batch)
            batch = []
    return history
