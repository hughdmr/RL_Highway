"""
Train a Stable-Baselines3 agent (DQN ou PPO) on highway-v0.

Usage:
    # SB3 DQN (défaut)
    python train_sb3.py --algo DQN

    # SB3 PPO
    python train_sb3.py --algo PPO

    # PPO avec hyperparams custom
    python train_sb3.py --algo PPO --total-timesteps 500000 --n-steps 1024 --lr 3e-4

Checkpoints sauvegardés dans results/<run_name>/checkpoints/ :
  - best_model.zip     : meilleur modèle selon l'évaluation périodique
  - model_step_N.zip   : checkpoint toutes les --checkpoint-every-steps steps
  - ../model.zip       : dernier modèle (fin du training)
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import highway_env
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
<<<<<<< HEAD
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
=======
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
>>>>>>> 11a6390 (training in hard conditions)

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID


class EpisodeMetricsCallback(BaseCallback):
    """Records per-episode reward, length, and env metrics during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_metric_sums: dict[str, float] = {}
        self._ep_metric_counts: dict[str, int] = {}
        self._ep_crashed = False
        self._ep_lane_changes = 0
        self._prev_lane_id: int | None = None
        self.episode_rows: list[dict[str, Any]] = []
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_steps: list[int] = []
        self.metric_columns: list[str] = []
        self._metric_column_set: set[str] = set()

    def _register_metric_column(self, column_name: str) -> None:
        if column_name not in self._metric_column_set:
            self._metric_column_set.add(column_name)
            self.metric_columns.append(column_name)

    def _accumulate_info_metrics(self, info: dict[str, Any]) -> None:
        for key, value in info.items():
            if isinstance(value, bool):
                numeric_value = float(value)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                numeric_value = float(value)
            else:
                continue

            if np.isnan(numeric_value):
                continue

            self._ep_metric_sums[key] = self._ep_metric_sums.get(key, 0.0) + numeric_value
            self._ep_metric_counts[key] = self._ep_metric_counts.get(key, 0) + 1

            if key == "crashed" and bool(value):
                self._ep_crashed = True

    def _step_info(self) -> dict[str, Any]:
        infos = self.locals.get("infos", [])
        if isinstance(infos, (list, tuple)) and infos:
            info = infos[0]
            return info if isinstance(info, dict) else {}
        if isinstance(infos, dict):
            return infos
        return {}

    def _extract_lane_id(self, info: dict[str, Any]) -> int | None:
        lane_index = info.get("lane_index")
        if isinstance(lane_index, (tuple, list)) and lane_index:
            candidate = lane_index[-1]
            if isinstance(candidate, (int, np.integer)):
                return int(candidate)
        if isinstance(lane_index, (int, np.integer)):
            return int(lane_index)
        return None

    def _update_lane_changes(self, info: dict[str, Any]) -> None:
        lane_id = self._extract_lane_id(info)
        if lane_id is None:
            return
        if self._prev_lane_id is not None and lane_id != self._prev_lane_id:
            self._ep_lane_changes += 1
        self._prev_lane_id = lane_id

    def _build_episode_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "episode": len(self.episode_rewards) + 1,
            "global_step": self.num_timesteps,
            "episode_reward": self._ep_reward,
            "episode_length": self._ep_length,
        }

        for key, total in self._ep_metric_sums.items():
            count = self._ep_metric_counts.get(key, 0)
            if count > 0:
                column_name = f"episode_{key}_mean"
                row[column_name] = total / count
                self._register_metric_column(column_name)

        if "crashed" in self._ep_metric_sums:
            row["episode_crashed"] = float(self._ep_crashed)
            self._register_metric_column("episode_crashed")

        return row

    def _on_step(self) -> bool:
        info = self._step_info()
        self._ep_reward += float(self.locals["rewards"][0])
        self._ep_length += 1
        self._accumulate_info_metrics(info)
        self._update_lane_changes(info)
        if self.locals["dones"][0]:
            row = self._build_episode_row()
            self.episode_rows.append(row)
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_length)
            self.episode_steps.append(self.num_timesteps)
            if self.verbose and len(self.episode_rewards) % 10 == 0:
                speed_mean = row.get("episode_speed_mean", None)
                speed_str = f"speed_mean={speed_mean:.3f}" if speed_mean is not None else "speed_mean=N/A"
                print(
                    f"episode={len(self.episode_rewards)} "
                    f"step={self.num_timesteps} "
                    f"reward={self._ep_reward:.3f} "
                    f"{speed_str} "
                )
            self._ep_reward = 0.0
            self._ep_length = 0
            self._ep_metric_sums.clear()
            self._ep_metric_counts.clear()
            self._ep_crashed = False
            self._ep_lane_changes = 0
            self._prev_lane_id = None
        return True


class BestModelCheckpointCallback(BaseCallback):
    """Saves the best model according to episode reward."""

    def __init__(self, checkpoints_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.checkpoints_dir = checkpoints_dir
        self.best_reward = -float("inf")
        self._ep_reward = 0.0

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            if self._ep_reward > self.best_reward:
                self.best_reward = self._ep_reward
                best_base = self.checkpoints_dir / "best_model"
                self.model.save(str(best_base))
                best_zip = best_base.with_suffix(".zip")
                best_pt = self.checkpoints_dir / "best_model.pt"
                if best_zip.exists():
                    shutil.copyfile(best_zip, best_pt)
            self._ep_reward = 0.0
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
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

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

<<<<<<< HEAD
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    eval_env = make_env(config.seed + 1)
    metrics_cb = EpisodeMetricsCallback(verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoints_dir),
        log_path=str(checkpoints_dir),
        eval_freq=max(config.checkpoint_every_steps // 4, 1000),
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=config.checkpoint_every_steps,
        save_path=str(checkpoints_dir),
        name_prefix="model_step",
        verbose=0,
    )
    callback = CallbackList([metrics_cb, eval_cb, checkpoint_cb])
=======
    metrics_callback = EpisodeMetricsCallback(verbose=1)
    best_model_callback = BestModelCheckpointCallback(checkpoints_dir=checkpoints_dir)
    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_every_steps,
        save_path=str(checkpoints_dir),
        name_prefix="model_step",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callback = CallbackList([metrics_callback, best_model_callback, checkpoint_callback])
>>>>>>> 11a6390 (training in hard conditions)
    model.learn(total_timesteps=config.total_timesteps, callback=callback)
    eval_env.close()

    model_path = str(run_dir / "model")
    model.save(model_path)  # last model

    metrics_csv = run_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["episode", "global_step", "episode_reward", "episode_length"] + metrics_callback.metric_columns
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
<<<<<<< HEAD
        for i, (r, l, s) in enumerate(
            zip(metrics_cb.episode_rewards, metrics_cb.episode_lengths, metrics_cb.episode_steps), start=1
        ):
            writer.writerow(
                {"episode": i, "global_step": s, "episode_reward": r, "episode_length": l}
            )

    best_reward = max(metrics_cb.episode_rewards) if metrics_cb.episode_rewards else float("nan")
=======
        for row in metrics_callback.episode_rows:
            writer.writerow(row)

    last_base = checkpoints_dir / "last_model"
    model.save(str(last_base))
    last_ckpt = last_base.with_suffix(".zip")
    last_ckpt_pt = checkpoints_dir / "last_model.pt"
    if last_ckpt.exists():
        shutil.copyfile(last_ckpt, last_ckpt_pt)

    best_reward = max(metrics_callback.episode_rewards) if metrics_callback.episode_rewards else float("nan")
    speed_means = [row["episode_speed_mean"] for row in metrics_callback.episode_rows if "episode_speed_mean" in row]
    crash_flags = [row["episode_crashed"] for row in metrics_callback.episode_rows if "episode_crashed" in row]
    lane_changes = [row["episode_lane_changes"] for row in metrics_callback.episode_rows if "episode_lane_changes" in row]
>>>>>>> 11a6390 (training in hard conditions)
    summary = {
        "run_name": config.run_name,
        "env_id": SHARED_CORE_ENV_ID,
        "total_timesteps": config.total_timesteps,
<<<<<<< HEAD
        "episodes": len(metrics_cb.episode_rewards),
=======
        "episodes": len(metrics_callback.episode_rewards),
>>>>>>> 11a6390 (training in hard conditions)
        "best_episode_reward": best_reward,
        "mean_episode_speed": float(np.mean(speed_means)) if speed_means else float("nan"),
        "crash_rate": float(np.mean(crash_flags)) if crash_flags else float("nan"),
        "mean_episode_lane_changes": float(np.mean(lane_changes)) if lane_changes else float("nan"),
        "tracked_metric_columns": metrics_callback.metric_columns,
        "model_path": model_path + ".zip",
<<<<<<< HEAD
        "best_checkpoint": str(checkpoints_dir / "best_model.zip"),
=======
        "checkpoints_dir": str(checkpoints_dir),
        "best_checkpoint": str(checkpoints_dir / "best_model.pt"),
        "last_checkpoint": str(last_ckpt),
        "last_checkpoint_pt": str(last_ckpt_pt),
        "checkpoint_every_steps": config.checkpoint_every_steps,
>>>>>>> 11a6390 (training in hard conditions)
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

    parser.add_argument("--n-steps", "--n_steps", type=int, default=512, help="Horizon de rollout PPO")
    parser.add_argument("--n-epochs", "--n_epochs", type=int, default=10, help="Epochs par update PPO")

<<<<<<< HEAD
    parser.add_argument("--checkpoint-every-steps", "--checkpoint_every_steps", type=int, default=50_000,
                        help="Sauvegarde un checkpoint toutes les N steps")

=======
    parser.add_argument("--checkpoint-every-steps", "--checkpoint_every_steps", type=int, default=50_000)

    # Be robust to accidental whitespace-only tokens from shell line-continuation formatting.
>>>>>>> 11a6390 (training in hard conditions)
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
