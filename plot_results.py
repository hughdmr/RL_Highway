"""
Generate training curves and comparison plots.

Produces:
  results/figures/training_curves.png  — smoothed episode reward vs timestep
  results/figures/comparison_bar.png   — mean ± std bar chart (from comparison_multiseed.json)

Usage:
    python plot_results.py
    python plot_results.py --dqn-csv results/dqn_scratch/metrics.csv \
                           --sb3-csv results/sb3_dqn/metrics.csv \
                           --ppo-csv results/sb3_ppo/metrics.csv \
                           --comparison-json results/comparison_multiseed.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def smooth(values, window: int = 20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def format_steps_in_k(steps: np.ndarray) -> np.ndarray:
    return steps / 1000.0


COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


def plot_training_curves(paths: list, output: Path) -> None:
    fig, (ax_first, ax_rest) = plt.subplots(2, 1, figsize=(10, 8), sharey=True)
    first_window = 50_000
    max_step = 200_000

    for (csv_path, label), color in zip(paths, COLORS):
        p = Path(csv_path)
        if not p.exists():
            print(f"  [skip] {csv_path} not found")
            continue
        df = pd.read_csv(p)
        steps = df["global_step"].values
        rewards = df["episode_reward"].values
        smoothed = smooth(rewards)
        # align x after smoothing
        trim = len(rewards) - len(smoothed)
        x = steps[trim:]
        y = smoothed
        y_std = rewards[trim:].std()

        first_mask = x <= first_window
        rest_mask = x <= max_step

        if np.any(first_mask):
            ax_first.plot(x[first_mask], y[first_mask], label=label, color=color, linewidth=1.8)
            ax_first.fill_between(
                x[first_mask],
                y[first_mask] - y_std,
                y[first_mask] + y_std,
                alpha=0.15,
                color=color,
            )

        if np.any(rest_mask):
            ax_rest.plot(format_steps_in_k(x[rest_mask]), y[rest_mask], label=label, color=color, linewidth=1.8)
            ax_rest.fill_between(
                format_steps_in_k(x[rest_mask]),
                y[rest_mask] - y_std,
                y[rest_mask] + y_std,
                alpha=0.15,
                color=color,
            )

    ax_first.set_ylabel("Episode reward (smoothed)")
    ax_first.set_title("Training curves — first 50k steps")
    ax_first.set_xlim(0, first_window)
    ax_first.grid(True, alpha=0.3)
    ax_first.legend()

    ax_rest.set_xlabel("Environment steps (k)")
    ax_rest.set_ylabel("Episode reward (smoothed)")
    ax_rest.set_title("Training curves — 0 to 200k steps")
    ax_rest.set_xlim(0, max_step / 1000.0)
    ax_rest.grid(True, alpha=0.3)
    ax_rest.legend()

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved: {output}")


def plot_comparison_bar(comparison_json: Path, output: Path) -> None:
    if not comparison_json.exists():
        print(f"  [skip] {comparison_json} not found")
        return

    with comparison_json.open() as f:
        data = json.load(f)

    models = [d["model"] for d in data]
    means = [d["overall_mean"] for d in data]
    stds = [d["overall_std"] for d in data]

    seeds = [s["seed"] for s in data[0]["per_seed"]]
    seed_means = {d["model"]: [s["mean_reward"] for s in d["per_seed"]] for d in data}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = COLORS[: len(models)]

    bars = axes[0].bar(models, means, yerr=stds, capsize=8, color=colors, alpha=0.8, edgecolor="black")
    axes[0].set_ylabel("Mean reward (50 eps × 3 seeds)")
    axes[0].set_title("Overall comparison")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, m, s in zip(bars, means, stds):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            m + s + 0.2,
            f"{m:.2f}±{s:.2f}",
            ha="center", va="bottom", fontsize=9,
        )

    x = np.arange(len(seeds))
    n_models = len(models)
    width = 0.8 / n_models
    for i, (model, color) in enumerate(zip(models, colors)):
        offset = (i - n_models / 2 + 0.5) * width
        axes[1].bar(x + offset, seed_means[model], width, label=model, color=color, alpha=0.8, edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"seed={s}" for s in seeds])
    axes[1].set_ylabel("Mean reward (50 eps)")
    axes[1].set_title("Per-seed breakdown")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(" vs ".join(models) + " — highway-v0", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn-csv", type=str, default="results/dqn_scratch/metrics.csv")
    parser.add_argument("--sb3-csv", type=str, default="results/sb3_dqn/metrics.csv")
    parser.add_argument("--ppo-csv", type=str, default="", help="CSV de training SB3 PPO (optionnel)")
    parser.add_argument("--extra-csvs", type=str, nargs="*", default=[],
                        metavar="CHEMIN:LABEL",
                        help="CSVs supplémentaires au format chemin:label (ex: results/sb3_distance/metrics.csv:SB3-Distance)")
    parser.add_argument("--comparison-json", type=str, default="results/comparison_multiseed.json")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.output_dir)

    curves = [
        (args.dqn_csv, "DQN-scratch"),
        (args.sb3_csv, "SB3-DQN"),
    ]
    if args.ppo_csv:
        curves.append((args.ppo_csv, "SB3-PPO"))
    for entry in args.extra_csvs:
        if ":" not in entry:
            print(f"  [skip] format invalide '{entry}' — attendu chemin:label")
            continue
        path, label = entry.split(":", 1)
        curves.append((path, label))

    print("Generating training curves...")
    plot_training_curves(curves, out_dir / "training_curves.png")

    print("Generating comparison bar chart...")
    plot_comparison_bar(Path(args.comparison_json), out_dir / "comparison_bar.png")
