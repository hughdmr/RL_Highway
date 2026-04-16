"""
Evaluate DQN-scratch, SB3 DQN, and SB3 PPO across multiple seeds.

Outputs a JSON comparison table and prints a summary to stdout.
PPO is optional : omit --ppo-model pour comparer seulement DQN-scratch et SB3-DQN.

Usage:
    python evaluate_multiseed.py \
        --dqn-checkpoint results/dqn_scratch/checkpoints/best_model.pt \
        --sb3-model results/sb3_dqn/model.zip \
        --ppo-model results/sb3_ppo/model.zip \
        --seeds 100 200 300 \
        --episodes 50 \
        --output-json results/comparison_multiseed.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import highway_env
import numpy as np
import torch

from dqn_agent import DQNAgent
from eval_shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID


def make_env(seed: int):
    _ = highway_env.__name__
    env = gym.make(SHARED_CORE_ENV_ID)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    env.reset(seed=seed)
    return env


def run_episodes(env, step_fn, episodes: int, base_seed: int) -> Dict:
    rewards: List[float] = []
    lengths: List[int] = []
    crashed: List[bool] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        ep_reward = 0.0
        ep_len = 0
        ep_crashed = False

        while not done:
            action = step_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)
            ep_len += 1
            if info.get("crashed", False):
                ep_crashed = True

        rewards.append(ep_reward)
        lengths.append(ep_len)
        crashed.append(ep_crashed)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "crash_rate": float(np.mean(crashed)),
    }


def evaluate_dqn_scratch(checkpoint_path: str, seeds: List[int], episodes: int, cpu: bool) -> Dict:
    if not cpu and torch.cuda.is_available():
        device = "cuda"
    elif not cpu and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    env = make_env(seeds[0])
    obs, _ = env.reset()
    state_dim = int(np.asarray(obs, dtype=np.float32).size)
    action_dim = int(env.action_space.n)
    env.close()

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    agent.load(checkpoint_path)

    def step_fn(obs):
        state = agent.flatten_obs(np.asarray(obs))
        return agent.select_action(state, epsilon=0.0)

    per_seed: List[Dict] = []
    for seed in seeds:
        env = make_env(seed)
        result = run_episodes(env, step_fn, episodes, seed)
        result["seed"] = seed
        per_seed.append(result)
        env.close()
        print(f"  [DQN-scratch] seed={seed} mean={result['mean_reward']:.3f} ± {result['std_reward']:.3f}  crash={result['crash_rate']:.2%}")

    overall_means = [r["mean_reward"] for r in per_seed]
    return {
        "model": "DQN-scratch",
        "per_seed": per_seed,
        "overall_mean": float(np.mean(overall_means)),
        "overall_std": float(np.std(overall_means)),
    }


def _load_sb3_autodetect(model_path: str):
    """Charge un modèle SB3 en détectant automatiquement l'algo depuis le zip."""
    import zipfile
    from stable_baselines3 import PPO, DQN

    with zipfile.ZipFile(model_path) as zf:
        raw = zf.read("data")

    if b"PPOPolicy" in raw or b"ActorCriticPolicy" in raw:
        return PPO.load(model_path), "SB3-PPO"
    else:
        return DQN.load(model_path), "SB3-DQN"


def evaluate_sb3(model_path: str, seeds: List[int], episodes: int, label: str = "") -> Dict:
    model, detected_label = _load_sb3_autodetect(model_path)
    label = label or detected_label

    per_seed: List[Dict] = []
    for seed in seeds:
        env = make_env(seed)

        def step_fn(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        result = run_episodes(env, step_fn, episodes, seed)
        result["seed"] = seed
        per_seed.append(result)
        env.close()
        print(f"  [{label}]     seed={seed} mean={result['mean_reward']:.3f} ± {result['std_reward']:.3f}  crash={result['crash_rate']:.2%}")

    overall_means = [r["mean_reward"] for r in per_seed]
    return {
        "model": label,
        "per_seed": per_seed,
        "overall_mean": float(np.mean(overall_means)),
        "overall_std": float(np.std(overall_means)),
    }


def print_comparison_table(results: List[Dict]) -> None:
    print("\n" + "=" * 65)
    print(f"{'Model':<15} {'Overall mean':>14} {'Overall std':>12}")
    print("-" * 65)
    for r in results:
        print(f"{r['model']:<15} {r['overall_mean']:>14.3f} {r['overall_std']:>12.3f}")
    print("=" * 65)

    seeds = [s["seed"] for s in results[0]["per_seed"]]
    header = f"{'Model':<15}" + "".join(f"  seed={s}" for s in seeds)
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        row = f"{r['model']:<15}"
        for s_data in r["per_seed"]:
            row += f"  {s_data['mean_reward']:>7.3f}±{s_data['std_reward']:.2f}"
        print(row)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn-checkpoint", type=str, default="results/dqn_scratch/checkpoints/best_model.pt")
    parser.add_argument("--sb3-model", type=str, default="results/sb3_dqn/model.zip")
    parser.add_argument("--ppo-model", type=str, default="", help="Chemin vers le modèle SB3 PPO (.zip)")
    parser.add_argument("--extra-models", type=str, nargs="*", default=[],
                        metavar="CHEMIN:LABEL",
                        help="Modèles SB3 supplémentaires au format chemin:label (ex: results/sb3_distance/model.zip:SB3-Distance)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 200, 300])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-json", type=str, default="results/comparison_multiseed.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = []

    print(f"\nEvaluating DQN-scratch ({args.episodes} eps × {len(args.seeds)} seeds)...")
    results.append(evaluate_dqn_scratch(args.dqn_checkpoint, args.seeds, args.episodes, args.cpu))

    print(f"\nEvaluating SB3-DQN ({args.episodes} eps × {len(args.seeds)} seeds)...")
    results.append(evaluate_sb3(args.sb3_model, args.seeds, args.episodes))

    if args.ppo_model:
        print(f"\nEvaluating SB3-PPO ({args.episodes} eps × {len(args.seeds)} seeds)...")
        results.append(evaluate_sb3(args.ppo_model, args.seeds, args.episodes))

    for entry in args.extra_models:
        if ":" not in entry:
            print(f"  [skip] format invalide '{entry}' — attendu chemin:label")
            continue
        path, label = entry.split(":", 1)
        print(f"\nEvaluating {label} ({args.episodes} eps × {len(args.seeds)} seeds)...")
        results.append(evaluate_sb3(path, args.seeds, args.episodes, label=label))

    print_comparison_table(results)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved to {args.output_json}")
