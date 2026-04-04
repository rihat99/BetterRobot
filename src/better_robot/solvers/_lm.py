"""Custom Levenberg-Marquardt solver.

Supports two Jacobian modes (set via problem.jacobian_fn):
  - autodiff:  J = torch.func.jacrev(problem.total_residual)(x)
  - analytic:  J = problem.jacobian_fn(x)   [user-provided]

Parameters mirror PyPose's LevenbergMarquardt:
  damping  — initial lambda (same as Adaptive(damping=...))
  factor   — multiplicative scale for adaptive damping
  reject   — max retries per step before accepting increased lambda
"""

from __future__ import annotations

import torch

from ._base import Problem, Solver


class LevenbergMarquardt(Solver):
    """LM solver. Uses torch.func.jacrev when problem.jacobian_fn is None."""

    def __init__(
        self,
        damping: float = 1e-4,
        factor: float = 2.0,
        reject: int = 16,
    ) -> None:
        self.damping = damping
        self.factor = factor
        self.reject = reject

    def solve(self, problem: Problem, max_iter: int = 100) -> torch.Tensor:
        x = problem.variables.clone().float()
        lam = float(self.damping)
        lo = problem.lower_bounds
        hi = problem.upper_bounds

        for _ in range(max_iter):
            r = problem.total_residual(x)                    # (m,)
            J = self._jacobian(problem, x)                   # (m, n)
            JtJ = J.T @ J                                    # (n, n)
            Jtr = J.T @ r                                    # (n,)
            n_vars = x.shape[0]

            for _ in range(self.reject):
                A = JtJ + lam * torch.eye(n_vars, dtype=x.dtype, device=x.device)
                delta = torch.linalg.solve(A, -Jtr)
                x_new = x + delta
                if lo is not None and hi is not None:
                    x_new = x_new.clamp(
                        lo.to(dtype=x.dtype, device=x.device),
                        hi.to(dtype=x.dtype, device=x.device),
                    )
                if problem.total_residual(x_new).norm() <= r.norm():
                    x = x_new
                    lam = max(lam / self.factor, 1e-7)
                    break
                lam = min(lam * self.factor, 1e7)

        return x

    def _jacobian(self, problem: Problem, x: torch.Tensor) -> torch.Tensor:
        if problem.jacobian_fn is not None:
            return problem.jacobian_fn(x)
        return torch.func.jacrev(problem.total_residual)(x)
