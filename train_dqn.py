import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import highway_env
import numpy as np
import torch

from dqn_agent import DQNAgent
from safe_driving_wrapper import SafeDrivingRewardWrapper
from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_schedule(step: int, total_steps: int, eps_start: float, eps_end: float, decay_fraction: float) -> float:
    decay_steps = max(1, int(total_steps * decay_fraction))
    progress = min(1.0, step / decay_steps)
    return eps_start + progress * (eps_end - eps_start)


def make_env(seed: int):
    _ = highway_env.__name__
    env = gym.make(SHARED_CORE_ENV_ID)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env = SafeDrivingRewardWrapper(env)
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    return env, obs, info


def train(config: argparse.Namespace) -> Dict[str, float]:
    set_seed(config.seed)

    run_dir = Path(config.output_dir) / config.run_name
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    env, obs, _ = make_env(config.seed)

    state_dim = int(np.asarray(obs, dtype=np.float32).size)
    action_dim = int(env.action_space.n)

    if not config.cpu and torch.cuda.is_available():
        device = "cuda"
    elif not config.cpu and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        gamma=config.gamma,
        lr=config.lr,
        batch_size=config.batch_size,
        replay_size=config.replay_size,
        learning_starts=config.learning_starts,
        train_freq=config.train_freq,
        target_update_interval=config.target_update_interval,
    )

    metrics: List[Dict[str, float]] = []
    losses_window: List[float] = []
    best_reward = -float("inf")

    state = agent.flatten_obs(obs)
    episode_reward = 0.0
    episode_length = 0
    episode_idx = 0

    for step in range(1, config.total_timesteps + 1):
        epsilon = linear_schedule(
            step=step,
            total_steps=config.total_timesteps,
            eps_start=config.eps_start,
            eps_end=config.eps_end,
            decay_fraction=config.eps_decay_fraction,
        )
        action = agent.select_action(state, epsilon)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        next_state = agent.flatten_obs(next_obs)

        agent.store_transition(state, action, reward, next_state, done)
        loss = agent.maybe_train_step()
        if np.isfinite(loss):
            losses_window.append(loss)

        episode_reward += float(reward)
        episode_length += 1
        state = next_state

        if done:
            episode_idx += 1
            avg_loss = float(np.mean(losses_window)) if losses_window else float("nan")

            row = {
                "episode": episode_idx,
                "global_step": step,
                "episode_reward": episode_reward,
                "episode_length": episode_length,
                "epsilon": epsilon,
                "avg_loss": avg_loss,
            }
            metrics.append(row)

            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save(str(checkpoints_dir / "best_model.pt"))

            if episode_idx % config.log_every_episodes == 0:
                print(
                    f"episode={episode_idx} step={step} reward={episode_reward:.3f} "
                    f"len={episode_length} eps={epsilon:.3f} loss={avg_loss:.5f}"
                )

            obs, _ = env.reset()
            state = agent.flatten_obs(obs)
            episode_reward = 0.0
            episode_length = 0
            losses_window.clear()

        if step % config.checkpoint_every_steps == 0:
            agent.save(str(checkpoints_dir / f"model_step_{step}.pt"))

    final_ckpt = checkpoints_dir / "last_model.pt"
    agent.save(str(final_ckpt))
    env.close()

    metrics_csv = run_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode", "global_step", "episode_reward", "episode_length", "epsilon", "avg_loss"],
        )
        writer.writeheader()
        writer.writerows(metrics)

    summary = {
        "run_name": config.run_name,
        "env_id": SHARED_CORE_ENV_ID,
        "total_timesteps": config.total_timesteps,
        "episodes": len(metrics),
        "best_episode_reward": best_reward,
        "final_checkpoint": str(final_ckpt),
        "best_checkpoint": str(checkpoints_dir / "best_model.pt"),
        "metrics_csv": str(metrics_csv),
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training completed.")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a from-scratch DQN on highway-v0.")
    parser.add_argument("--run-name", type=str, default="dqn_scratch")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)

    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-fraction", type=float, default=0.5)

    parser.add_argument("--log-every-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-every-steps", type=int, default=50_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
