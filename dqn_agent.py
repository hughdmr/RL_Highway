import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_sizes: Tuple[int, int] = (256, 256)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.model = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 128,
        replay_size: int = 100_000,
        learning_starts: int = 5_000,
        train_freq: int = 1,
        target_update_interval: int = 1_000,
        grad_clip_norm: float = 10.0,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.grad_clip_norm = grad_clip_norm

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_size)

        self.action_dim = action_dim
        self.total_steps = 0

    @staticmethod
    def flatten_obs(obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        return arr.reshape(-1)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(
            Transition(
                state=state.astype(np.float32),
                action=int(action),
                reward=float(reward),
                next_state=next_state.astype(np.float32),
                done=float(done),
            )
        )

    def maybe_train_step(self) -> float:
        self.total_steps += 1

        if self.total_steps < self.learning_starts:
            return float("nan")
        if self.total_steps % self.train_freq != 0:
            return float("nan")
        if len(self.replay_buffer) < self.batch_size:
            return float("nan")

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(
            np.stack([t.next_state for t in batch]), dtype=torch.float32, device=self.device
        )
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(dim=1, keepdim=True)[0]
            targets = rewards + (1.0 - dones) * self.gamma * next_q_values

        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        if self.total_steps % self.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_q_net": self.target_q_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_q_net.load_state_dict(checkpoint["target_q_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = int(checkpoint.get("total_steps", 0))
