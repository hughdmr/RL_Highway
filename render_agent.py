"""
Visualise a trained agent (DQN-scratch, SB3 DQN, ou SB3 PPO) on highway-v0.

Modes
-----
  --mode human    : opens a window and plays in real-time (default)
  --mode video    : renders to an MP4 file (no window required)
  --mode gif      : same as video but exports an animated GIF

Examples
--------
  # DQN scratch en direct :
  python render_agent.py --checkpoint results/dqn_scratch/checkpoints/best_model.pt

  # SB3 DQN en vidéo :
  python render_agent.py --sb3 --checkpoint results/sb3_dqn/model.zip \
                          --mode video --episodes 5 --output results/videos/sb3_dqn.mp4

  # SB3 PPO en direct (l'algo est auto-détecté) :
  python render_agent.py --sb3 --checkpoint results/sb3_ppo/model.zip

  # SB3 PPO en GIF :
  python render_agent.py --sb3 --checkpoint results/sb3_ppo/model.zip \
                          --mode gif --output results/videos/ppo_demo.gif

  # Scénarios aléatoires à chaque lancement :
  python render_agent.py --sb3 --checkpoint results/sb3_ppo/model.zip --random-seed

  # Fixer un scénario précis (reproductible) :
  python render_agent.py --sb3 --checkpoint results/sb3_ppo/model.zip --seed 42
"""

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import highway_env 
import numpy as np

from eval_shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID


def make_env(render_mode: str):
    """Create and configure the highway environment."""
    _ = highway_env.__name__
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    return env


def load_dqn(checkpoint_path: str, state_dim: int, action_dim: int):
    """Load a from-scratch DQN checkpoint."""
    import torch
    from dqn_agent import DQNAgent

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    agent.load(checkpoint_path)
    return agent


def load_sb3(checkpoint_path: str):
    """Load a Stable-Baselines3 model (DQN ou PPO, auto-détecté)."""
    import zipfile
    from stable_baselines3 import PPO, DQN

    with zipfile.ZipFile(checkpoint_path) as zf:
        raw = zf.read("data")

    if b"PPOPolicy" in raw or b"ActorCriticPolicy" in raw:
        print("  → algo détecté : PPO")
        return PPO.load(checkpoint_path)
    else:
        print("  → algo détecté : DQN")
        return DQN.load(checkpoint_path)


def run_episodes_human(agent, is_sb3: bool, episodes: int, seed: int, delay: float = 0.0):
    """Play episodes in a live window."""
    env = make_env("human")
    rewards = _run(env, agent, is_sb3, episodes, seed, delay)
    env.close()
    return rewards


def run_episodes_rgb(agent, is_sb3: bool, episodes: int, seed: int):
    """Collect RGB frames from all episodes."""
    env = make_env("rgb_array")
    all_frames: list[list[np.ndarray]] = []

    obs, _ = env.reset(seed=seed)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_frames: list[np.ndarray] = []
        ep_reward = 0.0

        while not done:
            frame = env.render()
            if frame is not None:
                ep_frames.append(frame)

            if is_sb3:
                action, _ = agent.predict(obs, deterministic=True)
            else:
                state = agent.flatten_obs(obs)
                action = agent.select_action(state, epsilon=0.0)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)

        print(f"  Episode {ep + 1}/{episodes}  reward={ep_reward:.3f}  frames={len(ep_frames)}")
        all_frames.append(ep_frames)

    env.close()
    return all_frames


def _run(env, agent, is_sb3: bool, episodes: int, seed: int, delay: float = 0.0):
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        while not done:
            if is_sb3:
                action, _ = agent.predict(obs, deterministic=True)
            else:
                state = agent.flatten_obs(obs)
                action = agent.select_action(state, epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)
            if delay > 0:
                time.sleep(delay)
        print(f"  Episode {ep + 1}/{episodes}  reward={ep_reward:.3f}")
        rewards.append(ep_reward)
    return rewards


def save_video(all_frames: list[list[np.ndarray]], output: Path, fps: int = 15):
    """Save all episode frames to an MP4 file using imageio."""
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "imageio is required for video export.\n"
            "Install it with:  pip install imageio[ffmpeg]"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    flat_frames = [f for ep in all_frames for f in ep]
    writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8)
    for frame in flat_frames:
        writer.append_data(frame)
    writer.close()
    print(f"Video saved → {output}")


def save_gif(all_frames: list[list[np.ndarray]], output: Path, fps: int = 15):
    """Save all episode frames to an animated GIF using imageio."""
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "imageio is required for GIF export.\n"
            "Install it with:  pip install imageio"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    flat_frames = [f for ep in all_frames for f in ep]
    duration = 1.0 / fps
    imageio.mimsave(str(output), flat_frames, duration=duration, loop=0)
    print(f"GIF saved → {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a trained DQN agent on highway-v0.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint (.pt for DQN-scratch, .zip for SB3).")
    parser.add_argument("--sb3", action="store_true",
                        help="Load checkpoint as a Stable-Baselines3 model.")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to render.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-seed", action="store_true",
                        help="Tire un seed aléatoire à chaque lancement (ignore --seed).")
    parser.add_argument("--mode", type=str, default="human",
                        choices=["human", "video", "gif"],
                        help="Rendering mode: live window, MP4 video, or animated GIF.")
    parser.add_argument("--output", type=str, default="",
                        help="Output file path (only for video/gif modes). "
                             "Defaults to results/videos/render.<ext>.")
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames per second for video/gif export.")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Pause in seconds between each step in human mode (e.g. 0.05 to slow down).")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.random_seed:
        import random
        args.seed = random.randint(0, 100_000)
        print(f"Seed aléatoire : {args.seed}")

    if args.mode in ("video", "gif") and not args.output:
        ext = "mp4" if args.mode == "video" else "gif"
        ckpt_stem = Path(args.checkpoint).stem
        args.output = f"results/videos/{ckpt_stem}.{ext}"

    print(f"Loading {'SB3' if args.sb3 else 'DQN-scratch'} checkpoint: {args.checkpoint}")
    if args.sb3:
        agent = load_sb3(args.checkpoint)
    else:
        _tmp_env = make_env("rgb_array")
        _tmp_obs, _ = _tmp_env.reset()
        state_dim = int(np.asarray(_tmp_obs, dtype=np.float32).size)
        action_dim = int(_tmp_env.action_space.n)
        _tmp_env.close()
        agent = load_dqn(args.checkpoint, state_dim, action_dim)

    print(f"Running {args.episodes} episode(s) in mode='{args.mode}' ...")

    if args.mode == "human":
        run_episodes_human(agent, args.sb3, args.episodes, args.seed, args.delay)

    elif args.mode == "video":
        all_frames = run_episodes_rgb(agent, args.sb3, args.episodes, args.seed)
        save_video(all_frames, Path(args.output), fps=args.fps)

    elif args.mode == "gif":
        all_frames = run_episodes_rgb(agent, args.sb3, args.episodes, args.seed)
        save_gif(all_frames, Path(args.output), fps=args.fps)


if __name__ == "__main__":
    main()
