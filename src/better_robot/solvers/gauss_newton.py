"""Gauss-Newton solver (stub)."""
from __future__ import annotations

import torch

from .problem import Problem
from .base import Solver
from .registry import SOLVERS


@SOLVERS.register("gn")
class GaussNewton(Solver):
    """GN solver. Faster than LM on well-conditioned problems (no damping)."""

    def solve(
        self,
        problem: Problem,
        max_iter: int = 100,
        **kwargs: object,
    ) -> torch.Tensor:
        raise NotImplementedError
