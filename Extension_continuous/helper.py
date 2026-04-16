import gymnasium as gym
import highway_env  # registers highway-v0, etc.
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from config import ENV_CONFIG


SEED = 42

def make_env(rank: int = 0, seed: int = SEED) -> gym.Env:
    """Factory for a single monitored highway env."""
    env = gym.make("highway-v0", config=ENV_CONFIG)
    env = Monitor(env)
    env.reset(seed=seed + rank)
    return env



class EpisodeMetricsCallback(BaseCallback):
    """Records per-episode reward & crash rate to a CSV file."""

    def __init__(self, log_path: str = "metrics.csv", verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.records: list[dict] = []
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int]  = []
        self._crashes: list[bool] = []

    def _on_step(self) -> bool:
        infos   = self.locals.get("infos", [])
        dones   = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        for i, (done, info, reward) in enumerate(zip(dones, infos, rewards)):
            self._episode_rewards.append(reward)
            self._episode_lengths.append(1)
            if done:
                crashed = info.get("crashed", False)
                ep_reward = sum(self._episode_rewards)
                ep_len    = len(self._episode_lengths)
                self.records.append({
                    "total_steps": self.num_timesteps,
                    "episode_reward": ep_reward,
                    "episode_length": ep_len,
                    "crashed": int(crashed),
                })
                self._episode_rewards.clear()
                self._episode_lengths.clear()
        return True

    def _on_training_end(self) -> None:
        pd.DataFrame(self.records).to_csv(self.log_path, index=False)
        print(f"Metrics saved → {self.log_path}")



def evaluate_model(model, env_config: dict, n_episodes: int = 50, seed: int = 100):
    """Run deterministic evaluation, return (rewards, crash_rate)."""
    env = gym.make("highway-v0", config=env_config)
    env.reset(seed=seed)
    rewards, crashes = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_r, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += r
        rewards.append(total_r)
        crashes.append(int(info.get("crashed", False)))
    env.close()
    return np.array(rewards), np.mean(crashes) * 100