"""Synthetic dataset generation for NN Motion demo.

This script creates random driving scenes with static obstacles and basic
roadside context.  Each obstacle is assigned a category (e.g. vehicle,
pedestrian) and a corresponding cost weight.  Pedestrians close to a generated
sidewalk receive larger weights so that the learning algorithm can emphasise
their avoidance.  Obstacles whose longitudinal distance exceeds ``speed * 7``
are given zero weight, allowing the outer network to ignore far-away objects.

The scenes are generated in parallel using the ``multiprocessing`` module and
both JSON data and corresponding matplotlib visualisations are written to the
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
    category: str
    distance: float
    lateral: float
    weight: float


@dataclass
class Scene:
    speed: float
    has_sidewalk: bool
    obstacles: List[Obstacle]


def generate_scene(idx: int) -> None:
    random.seed()

    speed = random.uniform(5.0, 20.0)
    has_sidewalk = random.choice([True, False])

    # Base weights per obstacle category
    cat_weights = {"vehicle": 1.0, "pedestrian": 1.2, "cone": 0.5}
    categories = list(cat_weights.keys())

    num_obs = random.randint(1, 5)
    obstacles: List[Obstacle] = []
    for _ in range(num_obs):
        dist = random.uniform(0.0, 100.0)
        lat = random.uniform(-3.0, 3.0)
        cat = random.choice(categories)

        if dist <= speed * 7.0:
            weight = cat_weights[cat]
            if (
                cat == "pedestrian"
                and has_sidewalk
                and abs(lat) > 2.0  # near sidewalk
            ):
                weight *= 2.0
        else:
            weight = 0.0

        obstacles.append(
            Obstacle(category=cat, distance=dist, lateral=lat, weight=weight)
        )

    scene = Scene(speed=speed, has_sidewalk=has_sidewalk, obstacles=obstacles)

    os.makedirs("data", exist_ok=True)
    with open(f"data/sample_{idx:03d}.json", "w", encoding="utf-8") as f:
        json.dump(asdict(scene), f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axhline(0.0, color="black", linewidth=1, label="lane center")
    if has_sidewalk:
        ax.axhline(3.5, color="green", linestyle="--", linewidth=1, label="sidewalk")
        ax.axhline(-3.5, color="green", linestyle="--", linewidth=1)
    ax.plot(0.0, 0.0, "bo", label="ego")

    color_map = {"vehicle": "red", "pedestrian": "orange", "cone": "purple"}
    for obs in obstacles:
        color = color_map[obs.category] if obs.weight > 0 else "gray"
        ax.scatter(obs.distance, obs.lateral, c=color, s=80)

    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("Longitudinal [m]")
    ax.set_ylabel("Lateral [m]")
    title = f"Scene {idx} speed={speed:.1f} m/s"
    if has_sidewalk:
        title += " with sidewalk"
    ax.set_title(title)
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
