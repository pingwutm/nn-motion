"""Synthetic dataset generation for NN Motion demo.

This script creates random driving scenes with static obstacles.  Obstacles
whose longitudinal distance exceeds ``speed * 7`` are assigned zero cost
weight, allowing the outer network to ignore far-away objects.

The scenes are generated in parallel using the ``multiprocessing`` module and
both JSON data and corresponding matplotlib visualizations are written to the
``data/`` directory.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, asdict
from multiprocessing import Pool
from typing import List

import matplotlib.pyplot as plt


@dataclass
class Obstacle:
    distance: float
    lateral: float
    weight: float


@dataclass
class Scene:
    speed: float
    obstacles: List[Obstacle]


def generate_scene(idx: int) -> None:
    random.seed()
    speed = random.uniform(5.0, 20.0)
    num_obs = random.randint(1, 5)
    obstacles: List[Obstacle] = []
    for _ in range(num_obs):
        dist = random.uniform(0.0, 100.0)
        lat = random.uniform(-3.0, 3.0)
        weight = 1.0 if dist <= speed * 7.0 else 0.0
        obstacles.append(Obstacle(distance=dist, lateral=lat, weight=weight))

    scene = Scene(speed=speed, obstacles=obstacles)

    os.makedirs("data", exist_ok=True)
    with open(f"data/sample_{idx:03d}.json", "w", encoding="utf-8") as f:
        json.dump(asdict(scene), f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.plot(0.0, 0.0, "bo", label="ego")
    for obs in obstacles:
        color = "red" if obs.weight > 0 else "gray"
        ax.scatter(obs.distance, obs.lateral, c=color, s=80)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("Longitudinal [m]")
    ax.set_ylabel("Lateral [m]")
    ax.set_title(f"Scene {idx} speed={speed:.1f} m/s")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"data/vis_{idx:03d}.png")
    plt.close(fig)


def main(num_samples: int = 100, workers: int = 4) -> None:
    with Pool(processes=workers) as pool:
        pool.map(generate_scene, range(num_samples))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic scenes")
    parser.add_argument("--num", type=int, default=100, help="number of scenes")
    parser.add_argument("--workers", type=int, default=4, help="parallel workers")
    args = parser.parse_args()

    main(num_samples=args.num, workers=args.workers)
