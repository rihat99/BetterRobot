"""Gauss-Newton solver."""
from __future__ import annotations

import torch

from .problem import Problem
from .base import Solver
from .registry import SOLVERS


@SOLVERS.register("gn")
class GaussNewton(Solver):
    """GN solver. Faster than LM on well-conditioned problems (no damping)."""

    def solve(self, problem: Problem, max_iter: int = 100, **kwargs) -> torch.Tensor:
        x = problem.variables.clone().float()
        lo = problem.lower_bounds
        hi = problem.upper_bounds

        for _ in range(max_iter):
            r = problem.total_residual(x)
            J = self._jacobian(problem, x)
            JtJ = J.T @ J
            Jtr = J.T @ r
            # Add tiny regularization to handle rank deficiency
            n_vars = x.shape[0]
            A = JtJ + 1e-8 * torch.eye(n_vars, dtype=x.dtype, device=x.device)
            delta = torch.linalg.solve(A, -Jtr)
            x_new = x + delta
            if lo is not None and hi is not None:
                x_new = x_new.clamp(
                    lo.to(dtype=x.dtype, device=x.device),
                    hi.to(dtype=x.dtype, device=x.device),
                )
            x = x_new
        return x

    def _jacobian(self, problem: Problem, x: torch.Tensor) -> torch.Tensor:
        if problem.jacobian_fn is not None:
            return problem.jacobian_fn(x)
        return torch.func.jacrev(problem.total_residual)(x)
