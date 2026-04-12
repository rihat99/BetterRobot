"""Gauss-Newton optimiser.

Undamped Gauss-Newton with a tiny Tikhonov regularisation for rank
deficiency.  Shares the same plug-in linear-solver interface as LM so
the outer shell of ``minimize`` is almost identical — the only
difference is the fixed (near-zero) damping.

See ``docs/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

import torch

from ..problem import LeastSquaresProblem
from .base import OptimizationResult, Optimizer


class GaussNewton(Optimizer):
    """Undamped Gauss-Newton with tiny regularization for rank deficiency."""

    def __init__(self, *, tol: float = 1e-6, eps: float = 1e-8) -> None:
        self.tol = tol
        self.eps = eps

    def minimize(
        self,
        problem: LeastSquaresProblem,
        *,
        max_iter: int = 50,
        linear_solver=None,
        kernel=None,
        strategy=None,
        scheduler=None,
    ) -> OptimizationResult:
        """Run Gauss-Newton until convergence or ``max_iter`` is reached.

        See docs/07_RESIDUALS_COSTS_SOLVERS.md §5.
        """
        from ..solvers.cholesky import Cholesky

        solver = linear_solver if linear_solver is not None else Cholesky()

        x = problem.x0.clone().detach()
        nv = problem._nv

        r = problem.residual(x)
        history: list[dict] = []
        converged = False
        it = 0

        for it in range(max_iter):
            J = problem.jacobian(x)                                    # (dim, nv)
            JtJ = J.mT @ J                                             # (nv, nv)
            Jtr = J.mT @ r                                             # (nv,)
            H = JtJ + self.eps * torch.eye(nv, dtype=J.dtype, device=J.device)
            delta_v = solver.solve(H, -Jtr)                            # (nv,)

            x_new = problem.step(x, delta_v)
            if problem.lower is not None:
                x_new = x_new.clamp(
                    min=problem.lower.to(x_new.device, x_new.dtype),
                    max=problem.upper.to(x_new.device, x_new.dtype),
                )

            x = x_new
            r = problem.residual(x)
            cost = float(0.5 * (r @ r).sum())
            history.append({"iter": it, "cost": cost})

            if float(Jtr.norm()) < self.tol:
                converged = True
                break

        return OptimizationResult(
            x=x, residual=r, iters=it + 1, converged=converged, history=history
        )
