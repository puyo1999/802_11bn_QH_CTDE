"""src/utils/replay_buffer.py"""

import random
from collections import deque
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, int(action), float(reward), next_obs, float(done)))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.array(obs),      dtype=torch.float32).to(device),
            torch.tensor(actions,            dtype=torch.long).to(device),
            torch.tensor(rewards,            dtype=torch.float32).to(device),
            torch.tensor(np.array(next_obs), dtype=torch.float32).to(device),
            torch.tensor(dones,              dtype=torch.float32).to(device),
        )

    def __len__(self):
        return len(self.buf)
