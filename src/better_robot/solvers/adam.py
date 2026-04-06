"""Adam gradient-descent solver."""
from __future__ import annotations

import torch

from .problem import Problem
from .base import Solver
from .registry import SOLVERS


@SOLVERS.register("adam")
class AdamSolver(Solver):
    """Gradient descent solver using torch.optim.Adam."""

    def __init__(self, lr: float = 1e-2) -> None:
        self.lr = lr

    def solve(self, problem: Problem, max_iter: int = 100, **kwargs) -> torch.Tensor:
        x = problem.variables.clone().float().requires_grad_(True)
        lo = problem.lower_bounds
        hi = problem.upper_bounds
        optimizer = torch.optim.Adam([x], lr=self.lr)

        for _ in range(max_iter):
            optimizer.zero_grad()
            r = problem.total_residual(x)
            loss = 0.5 * r.dot(r)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if lo is not None and hi is not None:
                    x.data.clamp_(
                        lo.to(dtype=x.dtype, device=x.device),
                        hi.to(dtype=x.dtype, device=x.device),
                    )
        return x.detach()
