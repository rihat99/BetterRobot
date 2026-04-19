"""Levenberg-Marquardt optimiser.

Pluggable kernel / linear solver / damping strategy. The minimize loop is
agnostic to the residual shape — it calls ``problem.residual(x)`` and
``problem.jacobian(x)`` and uses ``problem.step(x, delta_v)`` for the
manifold-aware update.

Returns a :class:`~better_robot.optim.state.SolverState` whose
``status`` is ``"converged"`` when the gradient norm drops below ``tol``
and ``"maxiter"`` otherwise.

See ``docs/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

import torch

from ..problem import LeastSquaresProblem
from ..state import SolverState


class LevenbergMarquardt:
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
    ) -> SolverState:
        """Run LM until convergence or ``max_iter`` is reached.

        See docs/07_RESIDUALS_COSTS_SOLVERS.md §5.
        """
        from ..solvers.cholesky import Cholesky
        from ..strategies.adaptive import Adaptive

        solver = linear_solver if linear_solver is not None else Cholesky()
        strat = strategy if strategy is not None else Adaptive(
            lam0=self.lam0, factor=self.factor
        )

        state = SolverState.from_problem(problem)
        state.damping = strat.init(problem)
        nv = problem._nv
        cost = float(state.residual_norm)

        it = -1
        for it in range(max_iter):
            J = problem.jacobian(state.x)  # (dim, nv)

            # Normal equations: (J^T J + lam * I) delta_v = -J^T r
            JtJ = J.mT @ J                                           # (nv, nv)
            Jtr = J.mT @ state.residual                              # (nv,)
            H = JtJ + state.damping * torch.eye(nv, dtype=J.dtype, device=J.device)
            delta_v = solver.solve(H, -Jtr)                          # (nv,)

            # Manifold-aware update
            x_new = problem.step(state.x, delta_v)

            # Clamp to box bounds (no-op for ±inf)
            if problem.lower is not None:
                x_new = x_new.clamp(
                    min=problem.lower.to(x_new.device, x_new.dtype),
                    max=problem.upper.to(x_new.device, x_new.dtype),
                )

            r_new = problem.residual(x_new)
            cost_new = float(0.5 * (r_new @ r_new).sum())

            state.history.append({"iter": it, "cost": cost, "lam": state.damping})

            if cost_new < cost:
                # Gain ratio = (actual decrease) / (predicted decrease).
                predicted = float(-delta_v @ Jtr - 0.5 * (delta_v @ (JtJ @ delta_v)))
                state.gain_ratio = (cost - cost_new) / predicted if predicted > 0 else None

                state.x = x_new
                state.residual = r_new
                state.residual_norm = torch.as_tensor(cost_new)
                cost = cost_new
                state.damping = strat.accept(state.damping)

                # Convergence check on gradient norm
                if float(Jtr.norm()) < self.tol:
                    state.status = "converged"
                    state.iters = it + 1
                    return state
            else:
                state.damping = strat.reject(state.damping)

        state.iters = it + 1
        state.status = "maxiter"
        return state
