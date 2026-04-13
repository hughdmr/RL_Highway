import argparse
import csv
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import highway_env
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID


class EpisodeMetricsCallback(BaseCallback):
    """Records per-episode reward and length during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._ep_reward = 0.0
        self._ep_length = 0
        self.episode_rewards: list = []
        self.episode_lengths: list = []
        self.episode_steps: list = []

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals["rewards"][0])
        self._ep_length += 1
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_length)
            self.episode_steps.append(self.num_timesteps)
            if self.verbose and len(self.episode_rewards) % 10 == 0:
                print(
                    f"episode={len(self.episode_rewards)} "
                    f"step={self.num_timesteps} "
                    f"reward={self._ep_reward:.3f}"
                )
            self._ep_reward = 0.0
            self._ep_length = 0
        return True


def make_env(seed: int):
    _ = highway_env.__name__
    env = gym.make(SHARED_CORE_ENV_ID)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env.reset(seed=seed)
    return env


def train(config: argparse.Namespace) -> dict:
    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(config.seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    algo = config.algo.upper()
    if algo == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.lr,
            batch_size=config.batch_size,
            gamma=config.gamma,
            n_steps=config.n_steps,
            n_epochs=config.n_epochs,
            policy_kwargs={"net_arch": [256, 256]},
            seed=config.seed,
            device=device,
            verbose=0,
        )
    else:
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=config.lr,
            batch_size=config.batch_size,
            buffer_size=config.replay_size,
            learning_starts=config.learning_starts,
            gamma=config.gamma,
            target_update_interval=config.target_update_interval,
            train_freq=config.train_freq,
            gradient_steps=1,
            policy_kwargs={"net_arch": [256, 256]},
            exploration_fraction=config.eps_decay_fraction,
            exploration_initial_eps=config.eps_start,
            exploration_final_eps=config.eps_end,
            seed=config.seed,
            device=device,
            verbose=0,
        )

    callback = EpisodeMetricsCallback(verbose=1)
    model.learn(total_timesteps=config.total_timesteps, callback=callback)

    model_path = str(run_dir / "model")
    model.save(model_path)

    metrics_csv = run_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode", "global_step", "episode_reward", "episode_length"]
        )
        writer.writeheader()
        for i, (r, l, s) in enumerate(
            zip(callback.episode_rewards, callback.episode_lengths, callback.episode_steps), start=1
        ):
            writer.writerow(
                {"episode": i, "global_step": s, "episode_reward": r, "episode_length": l}
            )

    best_reward = max(callback.episode_rewards) if callback.episode_rewards else float("nan")
    summary = {
        "run_name": config.run_name,
        "env_id": SHARED_CORE_ENV_ID,
        "total_timesteps": config.total_timesteps,
        "episodes": len(callback.episode_rewards),
        "best_episode_reward": best_reward,
        "model_path": model_path + ".zip",
        "metrics_csv": str(metrics_csv),
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training completed.")
    print(json.dumps(summary, indent=2))
    env.close()
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Stable-Baselines3 agent on highway-v0.")
    parser.add_argument("--algo", type=str, default="DQN", choices=["DQN", "PPO"], help="Algorithme SB3 (DQN ou PPO)")
    parser.add_argument("--run-name", type=str, default="", help="Nom du run (défaut: sb3_dqn ou sb3_ppo selon --algo)")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--total-timesteps", "--total_timesteps", type=int, default=200_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=128)

    # DQN-only
    parser.add_argument("--replay-size", "--replay_size", type=int, default=100_000)
    parser.add_argument("--learning-starts", "--learning_starts", type=int, default=5_000)
    parser.add_argument("--train-freq", "--train_freq", type=int, default=1)
    parser.add_argument("--target-update-interval", "--target_update_interval", type=int, default=1_000)
    parser.add_argument("--eps-start", "--eps_start", type=float, default=1.0)
    parser.add_argument("--eps-end", "--eps_end", type=float, default=0.05)
    parser.add_argument("--eps-decay-fraction", "--eps_decay_fraction", type=float, default=0.5)

    # PPO-only
    parser.add_argument("--n-steps", "--n_steps", type=int, default=512, help="Horizon de rollout PPO")
    parser.add_argument("--n-epochs", "--n_epochs", type=int, default=10, help="Epochs par update PPO")

    # Be robust to accidental whitespace-only tokens from shell line-continuation formatting.
    if argv is None:
        argv = sys.argv[1:]
    argv = [arg for arg in argv if arg.strip()]
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if not args.run_name:
        args.run_name = f"sb3_{args.algo.lower()}"
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
