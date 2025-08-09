import glob
import json
from dataclasses import dataclass
from typing import List

from torch.utils.data import Dataset


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


class SceneDataset(Dataset):
    """Dataset of synthetic driving scenes stored as JSON files."""

    def __init__(self, data_glob: str):
        self.paths = sorted(glob.glob(data_glob))
        if not self.paths:
            raise RuntimeError(
                "no scene data found; run generate_dataset.py first"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Scene:
        path = self.paths[idx]
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
