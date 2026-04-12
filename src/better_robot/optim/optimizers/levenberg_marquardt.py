"""Levenberg-Marquardt optimiser.

Pluggable kernel / linear solver / damping strategy. The minimize loop is
agnostic to the residual shape — it calls ``problem.residual(x)`` and
``problem.jacobian(x)`` and uses ``problem.step(x, delta_v)`` for the
manifold-aware update.

See ``docs/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

import torch

from ..problem import LeastSquaresProblem
from .base import OptimizationResult, Optimizer


class LevenbergMarquardt(Optimizer):
    """Damped Gauss-Newton with adaptive lambda and pluggable components."""

    def __init__(
        self,
        *,
        lam0: float = 1e-4,
        factor: float = 2.0,
        tol: float = 1e-6,
    ) -> None:
        self.lam0 = lam0
        self.factor = factor
        self.tol = tol

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
        """Run LM until convergence or ``max_iter`` is reached.

        See docs/07_RESIDUALS_COSTS_SOLVERS.md §5.
        """
        from ..solvers.cholesky import Cholesky
        from ..strategies.adaptive import Adaptive

        solver = linear_solver if linear_solver is not None else Cholesky()
        strat = strategy if strategy is not None else Adaptive(
            lam0=self.lam0, factor=self.factor
        )

        x = problem.x0.clone().detach()
        lam = strat.init(problem)
        nv = problem._nv

        # Initial residual
        r = problem.residual(x)
        cost = float(0.5 * (r @ r).sum())
        converged = False
        history: list[dict] = []

        for it in range(max_iter):
            J = problem.jacobian(x)  # (dim, nv)

            # Normal equations: (J^T J + lam * I) delta_v = -J^T r
            JtJ = J.mT @ J                                    # (nv, nv)
            Jtr = J.mT @ r                                    # (nv,)
            H = JtJ + lam * torch.eye(nv, dtype=J.dtype, device=J.device)
            delta_v = solver.solve(H, -Jtr)                   # (nv,)

            # Manifold-aware update
            x_new = problem.step(x, delta_v)

            # Clamp to box bounds (no-op for ±inf)
            if problem.lower is not None:
                x_new = x_new.clamp(
                    min=problem.lower.to(x_new.device, x_new.dtype),
                    max=problem.upper.to(x_new.device, x_new.dtype),
                )

            r_new = problem.residual(x_new)
            cost_new = float(0.5 * (r_new @ r_new).sum())

            history.append({"iter": it, "cost": cost, "lam": lam})

            if cost_new < cost:
                x = x_new
                r = r_new
                cost = cost_new
                lam = strat.accept(lam)

                # Convergence check on gradient norm
                if float(Jtr.norm()) < self.tol:
                    converged = True
                    break
            else:
                lam = strat.reject(lam)

        return OptimizationResult(
            x=x, residual=r, iters=it + 1, converged=converged, history=history
        )
