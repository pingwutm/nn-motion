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
from typing import Dict, List

import torch
import theseus as th

import matplotlib.pyplot as plt


@dataclass
class Obstacle:
    category: str
    distance: float
    lateral: float
    weight: float
    cost: float = 0.0


@dataclass
class TrajectoryPoint:
    x: float
    y: float


@dataclass
class Scene:
    speed: float
    has_sidewalk: bool
    obstacles: List[Obstacle]
    trajectory: List[TrajectoryPoint]
    total_cost: float
    cost_breakdown: Dict[str, float]


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

    def optimise_trajectory() -> tuple[list[TrajectoryPoint], float, Dict[str, float]]:
        """Solve a simple lane-keeping/obstacle-avoidance problem and record cost terms."""

        horizon = 20
        dt = 1.0
        xs = torch.linspace(0.0, speed * dt * (horizon - 1), horizon)

        # optimisation variable: lateral position at each step
        y = th.Vector(tensor=torch.zeros(1, horizon))
        objective = th.Objective()

        lane_scale = math.sqrt(2.0 * 0.1)

        def lane_err(optim_vars, aux_vars):
            return optim_vars[0].tensor

        lane_cf = th.AutoDiffCostFunction(
            [y],
            lane_err,
            horizon,
            cost_weight=th.ScaleCostWeight(torch.tensor([[lane_scale]])),
        )
        objective.add(lane_cf)

        smooth_scale = math.sqrt(2.0 * 5.0)

        def smooth_err(optim_vars, aux_vars):
            vals = optim_vars[0].tensor
            return vals[:, 1:] - vals[:, :-1]

        smooth_cf = th.AutoDiffCostFunction(
            [y],
            smooth_err,
            horizon - 1,
            cost_weight=th.ScaleCostWeight(torch.tensor([[smooth_scale]])),
        )
        objective.add(smooth_cf)

        obs_cfs: list[tuple[Obstacle, th.CostFunction]] = []
        for obs in obstacles:
            t = int(obs.distance / (speed * dt))
            if 0 <= t < horizon and obs.weight > 0:
                scale = math.sqrt(2.0 * obs.weight)

                def make_err(t_index: int, lat: float):
                    def err(optim_vars, aux_vars):
                        vals = optim_vars[0].tensor
                        diff = vals[:, t_index] - lat
                        return torch.relu(1.0 - torch.abs(diff)).view(1, 1)

                    return err

                cf = th.AutoDiffCostFunction(
                    [y],
                    make_err(t, obs.lateral),
                    1,
                    cost_weight=th.ScaleCostWeight(torch.tensor([[scale]])),
                )
                objective.add(cf)
                obs_cfs.append((obs, cf))

        objective.update()
        solver = th.optimizer.nonlinear.GaussNewton(objective, max_iterations=25)
        solver.optimize()

        ys = y.tensor[0].clone()
        lane_cost = 0.5 * torch.sum(lane_cf.weighted_error() ** 2)
        smooth_cost = 0.5 * torch.sum(smooth_cf.weighted_error() ** 2)
        obs_cost_total = 0.0
        for obs, cf in obs_cfs:
            c = 0.5 * torch.sum(cf.weighted_error() ** 2)
            obs.cost = float(c)
            obs_cost_total += float(c)

        traj = [TrajectoryPoint(float(xs[i]), float(ys[i])) for i in range(horizon)]
        total_cost = float(lane_cost + smooth_cost + obs_cost_total)
        breakdown = {
            "lane": float(lane_cost),
            "smoothness": float(smooth_cost),
            "obstacles": float(obs_cost_total),
        }
        return traj, total_cost, breakdown

    trajectory, total_cost, breakdown = optimise_trajectory()

    scene = Scene(
        speed=speed,
        has_sidewalk=has_sidewalk,
        obstacles=obstacles,
        trajectory=trajectory,
        total_cost=total_cost,
        cost_breakdown=breakdown,
    )

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

    ax.plot([p.x for p in trajectory], [p.y for p in trajectory], "b-", label="solution")

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
