"""Adam solver (stub)."""
from __future__ import annotations

import torch

from .problem import Problem
from .base import Solver
from .registry import SOLVERS


@SOLVERS.register("adam")
class AdamSolver(Solver):
    """Gradient descent solver using torch.optim.Adam."""

    def solve(
        self,
        problem: Problem,
        max_iter: int = 100,
        **kwargs: object,
    ) -> torch.Tensor:
        raise NotImplementedError
