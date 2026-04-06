"""L-BFGS solver."""
from __future__ import annotations

import torch

from .problem import Problem
from .base import Solver
from .registry import SOLVERS


@SOLVERS.register("lbfgs")
class LBFGSSolver(Solver):
    """Quasi-Newton solver using torch.optim.LBFGS."""

    def __init__(self, lr: float = 1.0) -> None:
        self.lr = lr

    def solve(self, problem: Problem, max_iter: int = 100, **kwargs) -> torch.Tensor:
        x = problem.variables.clone().float().requires_grad_(True)
        lo = problem.lower_bounds
        hi = problem.upper_bounds
        optimizer = torch.optim.LBFGS([x], lr=self.lr, max_iter=20, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            r = problem.total_residual(x)
            loss = 0.5 * r.dot(r)
            loss.backward()
            return loss

        for _ in range(max(1, max_iter // 20)):
            optimizer.step(closure)
            with torch.no_grad():
                if lo is not None and hi is not None:
                    x.data.clamp_(
                        lo.to(dtype=x.dtype, device=x.device),
                        hi.to(dtype=x.dtype, device=x.device),
                    )
        return x.detach()
