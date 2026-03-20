import argparse
import json
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import highway_env
import numpy as np
import torch

from dqn_agent import DQNAgent
from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID


def make_env(seed: int):
    _ = highway_env.__name__
    env = gym.make(SHARED_CORE_ENV_ID)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    obs, _ = env.reset(seed=seed)
    return env, obs


def evaluate(checkpoint_path: str, episodes: int, seed: int, cpu: bool) -> Dict[str, float]:
    device = "cuda" if torch.cuda.is_available() and not cpu else "cpu"

    env, obs = make_env(seed)
    state_dim = int(np.asarray(obs, dtype=np.float32).size)
    action_dim = int(env.action_space.n)

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    agent.load(checkpoint_path)

    rewards: List[float] = []
    lengths: List[int] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        state = agent.flatten_obs(obs)
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            action = agent.select_action(state, epsilon=0.0)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            state = agent.flatten_obs(next_obs)
            ep_reward += float(reward)
            ep_len += 1

        rewards.append(ep_reward)
        lengths.append(ep_len)

    env.close()

    result = {
        "checkpoint": checkpoint_path,
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN checkpoint on highway-v0.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate(args.checkpoint, args.episodes, args.seed, args.cpu)
    print(json.dumps(results, indent=2))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
