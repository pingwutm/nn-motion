"""Simple imitation learning demo for NN Motion."""

import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running the script without installing the package
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nn_motion.optimizer import TheseusMotionOptimizer
from nn_motion.problem import build_problem


@dataclass
class Obstacle:
    category: str
    distance: float
    lateral: float


@dataclass
class Scene:
    speed: float
    has_sidewalk: bool
    obstacles: List[Obstacle]
    trajectory: List[float]  # lateral positions only


class WeightNet(nn.Module):
    """Predict obstacle weights from scene and obstacle features."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(7, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus(),  # ensure positive weights
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features).squeeze(-1)


def load_scene(path: str) -> Scene:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    obstacles = [
        Obstacle(o["category"], o["distance"], o["lateral"]) for o in data["obstacles"]
    ]
    traj = [p["y"] for p in data["trajectory"]]
    return Scene(
        speed=data["speed"],
        has_sidewalk=data["has_sidewalk"],
        obstacles=obstacles,
        trajectory=traj,
    )


def obstacle_features(scene: Scene, obs: Obstacle) -> torch.Tensor:
    cat_map = {"vehicle": [1, 0, 0], "pedestrian": [0, 1, 0], "cone": [0, 0, 1]}
    has_sw = 1.0 if scene.has_sidewalk else 0.0
    speed_n = scene.speed / 20.0
    dist_n = obs.distance / 100.0
    lat_n = obs.lateral / 3.0
    return torch.tensor([has_sw, speed_n, dist_n, lat_n] + cat_map[obs.category], dtype=torch.float32)


def main(data_glob: str = "data/*.json", epochs: int = 5) -> None:
    scenes = [load_scene(p) for p in sorted(glob.glob(data_glob))]
    if not scenes:
        raise RuntimeError("no scene data found; run generate_dataset.py first")

    net = WeightNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    horizon = len(scenes[0].trajectory)

    for epoch in range(epochs):
        total = 0.0
        for scene in scenes:
            layer_builder = lambda s=scene: build_problem(s.speed, [o.__dict__ for o in s.obstacles])
            optimizer = TheseusMotionOptimizer(layer_builder)
            inputs = {"y": torch.zeros(1, horizon)}
            for i, obs in enumerate(scene.obstacles):
                feats = obstacle_features(scene, obs)
                w = net(feats)
                scale = torch.sqrt(2.0 * w + 1e-6).view(1, 1)
                inputs[f"w_obs_{i}"] = scale

            result = optimizer.solve(inputs)
            y_sol = result.controls["y"]
            ref_y = torch.tensor(scene.trajectory).view(1, horizon)
            loss = F.mse_loss(y_sol, ref_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss)
        print(f"epoch {epoch}: loss={total / len(scenes):.4f}")

    # print learned weights for first scene as a sanity check
    test_scene = scenes[0]
    for i, obs in enumerate(test_scene.obstacles):
        feats = obstacle_features(test_scene, obs)
        w = net(feats).item()
        print(f"obstacle {i} category={obs.category} weight={w:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Imitation learning demo")
    parser.add_argument("--data_glob", type=str, default="data/*.json")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    main(data_glob=args.data_glob, epochs=args.epochs)
