"""Motion optimization interfaces using Theseus.

This module defines a thin wrapper around Theseus so that gradients of the
inner optimization problem can be propagated to outer neural network weights.
The design is intentionally minimal to allow future replacement with a C++
implementation that exposes the same Python API.
"""

from dataclasses import dataclass
from typing import Any, Dict

try:
    import torch
    import theseus as th
except Exception as exc:  # pragma: no cover - handled at runtime
    torch = None
    th = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class OptimizationResult:
    """Container for optimization outputs."""
    controls: Any
    solver_info: Dict[str, Any]


class TheseusMotionOptimizer:
    """Minimal differentiable optimizer using Theseus.

    Parameters
    ----------
    problem_builder: callable
        A function that returns a ``th.TheseusLayer`` given no arguments.
        This allows users to customize the underlying optimization problem
        while keeping the wrapper interface stable.  A similar C++ backed
        optimizer can later expose the same interface.
    """

    def __init__(self, problem_builder):
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "theseus and torch are required but failed to import"
            ) from _IMPORT_ERROR
        self.layer = problem_builder()

    def solve(self, inputs: Dict[str, torch.Tensor]) -> OptimizationResult:
        """Solve the inner problem with the provided inputs.

        Parameters
        ----------
        inputs: Dict[str, torch.Tensor]
            Mapping from input names to tensors, including cost weights.

        Returns
        -------
        OptimizationResult
            Result containing optimized controls and solver information.
        """
        solution = self.layer.forward(inputs, {})
        controls = {k: v.detach() for k, v in solution.items()}
        info = {"status": "success"}
        return OptimizationResult(controls=controls, solver_info=info)


__all__ = ["TheseusMotionOptimizer", "OptimizationResult"]
