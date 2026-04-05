"""L-BFGS solver (stub)."""
from __future__ import annotations

import torch

from .problem import Problem
from .base import Solver
from .registry import SOLVERS


@SOLVERS.register("lbfgs")
class LBFGSSolver(Solver):
    """Quasi-Newton solver using torch.optim.LBFGS."""

    def solve(
        self,
        problem: Problem,
        max_iter: int = 100,
        **kwargs: object,
    ) -> torch.Tensor:
        raise NotImplementedError
