"""
Crash stress test : lance N simulations avec des seeds différentes
et compte le nombre de crashs.

Usage:
    # DQN scratch
    python crash_stress_test.py --checkpoint results/dqn_scratch/checkpoints/best_model.pt

    # SB3 DQN
    python crash_stress_test.py --sb3-model results/sb3_dqn/model.zip

    # SB3 PPO
    python crash_stress_test.py --ppo-model results/sb3_ppo/model.zip

    # Les trois en même temps
    python crash_stress_test.py \
        --checkpoint results/dqn_scratch/checkpoints/best_model.pt \
        --sb3-model results/sb3_dqn/model.zip \
        --ppo-model results/sb3_ppo/model.zip \
        --simulations 500
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import highway_env
import numpy as np
import torch

from dqn_agent import DQNAgent
from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID


def make_env():
    _ = highway_env.__name__
    env = gym.make(SHARED_CORE_ENV_ID)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    return env


def run_stress_test(step_fn, n_simulations: int, base_seed: int = 0, label: str = "model"):
    """
    Lance n_simulations épisodes, chaque fois avec seed = base_seed + i.
    Retourne les statistiques de crash.
    """
    env = make_env()

    n_crashes = 0
    rewards = []
    lengths = []
    crash_episodes = []

    print(f"\n[{label}] Lancement de {n_simulations} simulations...")
    print("-" * 55)

    for i in range(n_simulations):
        seed = base_seed + i
        obs, _ = env.reset(seed=seed)
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

        if ep_crashed:
            n_crashes += 1
            crash_episodes.append(seed)

        rewards.append(ep_reward)
        lengths.append(ep_len)

        if (i + 1) % 50 == 0 or (i + 1) == n_simulations:
            crash_rate_so_far = n_crashes / (i + 1)
            print(
                f"  {i+1:>4}/{n_simulations}  |  crashs: {n_crashes:>4}  |  "
                f"taux: {crash_rate_so_far:.1%}  |  "
                f"reward moy: {np.mean(rewards):.3f}"
            )

    env.close()

    results = {
        "model": label,
        "n_simulations": n_simulations,
        "n_crashes": n_crashes,
        "crash_rate": n_crashes / n_simulations,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "crash_seeds": crash_episodes,
    }

    print(f"\n{'='*55}")
    print(f"  [{label}] RESULTATS FINAUX")
    print(f"{'='*55}")
    print(f"  Simulations     : {n_simulations}")
    print(f"  Crashs          : {n_crashes}")
    print(f"  Taux de crash   : {n_crashes / n_simulations:.2%}")
    print(f"  Reward moyenne  : {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"  Longueur moy.   : {results['mean_episode_length']:.1f} steps")
    print(f"{'='*55}\n")

    return results


def load_dqn_scratch(checkpoint_path: str, cpu: bool):
    if not cpu and torch.cuda.is_available():
        device = "cuda"
    elif not cpu and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    env = make_env()
    obs, _ = env.reset(seed=0)
    state_dim = int(np.asarray(obs, dtype=np.float32).size)
    action_dim = int(env.action_space.n)
    env.close()

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    agent.load(checkpoint_path)

    def step_fn(obs):
        state = agent.flatten_obs(np.asarray(obs))
        return agent.select_action(state, epsilon=0.0)

    return step_fn


def load_sb3(model_path: str):
    """Charge un modèle SB3 en auto-détectant l'algo depuis le zip."""
    import zipfile
    from stable_baselines3 import PPO, DQN

    with zipfile.ZipFile(model_path) as zf:
        raw = zf.read("data")

    if b"PPOPolicy" in raw or b"ActorCriticPolicy" in raw:
        model = PPO.load(model_path)
    else:
        model = DQN.load(model_path)

    def step_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return step_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Stress test : N simulations et comptage des crashs.")
    parser.add_argument("--checkpoint", type=str, default="", help="Chemin vers le checkpoint DQN scratch (.pt)")
    parser.add_argument("--sb3-model", type=str, default="", help="Chemin vers le modèle SB3 DQN (.zip)")
    parser.add_argument("--ppo-model", type=str, default="", help="Chemin vers le modèle SB3 PPO (.zip)")
    parser.add_argument("--extra-models", type=str, nargs="*", default=[],
                        metavar="CHEMIN:LABEL",
                        help="Modèles SB3 supplémentaires au format chemin:label (ex: results/sb3_distance/model.zip:SB3-Distance)")
    parser.add_argument("--simulations", type=int, default=500, help="Nombre de simulations (défaut: 500)")
    parser.add_argument("--base-seed", type=int, default=0, help="Seed de départ (chaque sim = base_seed + i)")
    parser.add_argument("--cpu", action="store_true", help="Forcer l'utilisation du CPU")
    parser.add_argument("--output-json", type=str, default="results/crash_stress_test.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.checkpoint and not args.sb3_model and not args.ppo_model:
        print("Erreur : spécifie au moins --checkpoint, --sb3-model ou --ppo-model.")
        raise SystemExit(1)

    all_results = []

    if args.checkpoint:
        step_fn = load_dqn_scratch(args.checkpoint, args.cpu)
        result = run_stress_test(
            step_fn,
            n_simulations=args.simulations,
            base_seed=args.base_seed,
            label="DQN-scratch",
        )
        all_results.append(result)

    if args.sb3_model:
        step_fn = load_sb3(args.sb3_model)
        result = run_stress_test(
            step_fn,
            n_simulations=args.simulations,
            base_seed=args.base_seed,
            label="SB3-DQN",
        )
        all_results.append(result)

    if args.ppo_model:
        step_fn = load_sb3(args.ppo_model)
        result = run_stress_test(
            step_fn,
            n_simulations=args.simulations,
            base_seed=args.base_seed,
            label="SB3-PPO",
        )
        all_results.append(result)

    for entry in args.extra_models:
        if ":" not in entry:
            print(f"  [skip] format invalide '{entry}' — attendu chemin:label")
            continue
        path, label = entry.split(":", 1)
        step_fn = load_sb3(path)
        result = run_stress_test(
            step_fn,
            n_simulations=args.simulations,
            base_seed=args.base_seed,
            label=label,
        )
        all_results.append(result)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"Resultats sauvegardés dans {args.output_json}")
