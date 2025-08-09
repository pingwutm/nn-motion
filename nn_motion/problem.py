import math
from typing import List, Dict

import torch
import theseus as th


def build_problem(speed: float, obstacles: List[Dict[str, float]], horizon: int = 20, dt: float = 1.0) -> th.TheseusLayer:
    """Construct a Theseus layer for the lane/obstacle problem.

    Parameters
    ----------
    speed: float
        Ego vehicle speed used to compute obstacle time indices.
    obstacles: list of dict
        Each dictionary should contain ``distance`` and ``lateral`` keys.
    horizon: int, optional
        Number of discrete time steps of the trajectory.
    dt: float, optional
        Time step in seconds.

    Returns
    -------
    th.TheseusLayer
        Differentiable solver for the trajectory problem with named weight
        variables ``w_obs_i`` for each obstacle.
    """
    y = th.Vector(tensor=torch.zeros(1, horizon), name="y")
    objective = th.Objective()

    # Lane keeping with fixed weight 1.0 (scale sqrt(2 * 1.0))
    lane_scale = math.sqrt(2.0 * 1.0)

    def lane_err(optim_vars, aux_vars):
        return optim_vars[0].tensor

    lane_cf = th.AutoDiffCostFunction(
        [y], lane_err, horizon, cost_weight=th.ScaleCostWeight(lane_scale)
    )
    objective.add(lane_cf)

    # Smoothness with fixed relative weight 50 (since original was 5 vs lane 0.1)
    smooth_scale = math.sqrt(2.0 * 50.0)

    def smooth_err(optim_vars, aux_vars):
        vals = optim_vars[0].tensor
        return vals[:, 1:] - vals[:, :-1]

    smooth_cf = th.AutoDiffCostFunction(
        [y], smooth_err, horizon - 1, cost_weight=th.ScaleCostWeight(smooth_scale)
    )
    objective.add(smooth_cf)

    for i, obs in enumerate(obstacles):
        t = int(obs["distance"] / (speed * dt))
        if 0 <= t < horizon:
            def make_err(t_index: int, lat: float):
                def err(optim_vars, aux_vars):
                    vals = optim_vars[0].tensor
                    diff = vals[:, t_index] - lat
                    return torch.relu(1.0 - torch.abs(diff)).view(1, 1)

                return err

            w_var = th.Variable(torch.ones(1, 1), name=f"w_obs_{i}")
            cf = th.AutoDiffCostFunction(
                [y],
                make_err(t, obs["lateral"]),
                1,
                cost_weight=th.ScaleCostWeight(w_var),
            )
            objective.add(cf)

    objective.update()
    optimizer = th.GaussNewton(objective, max_iterations=25)
    layer = th.TheseusLayer(optimizer)
    return layer


__all__ = ["build_problem"]
