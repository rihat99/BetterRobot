"""Levenberg-Marquardt optimiser.

Pluggable kernel / linear solver / damping strategy. The minimize loop is
agnostic to the residual shape — it calls ``problem.residual(x)`` and
``problem.jacobian(x)`` and uses ``problem.step(x, delta_v)`` for the
manifold-aware update.

If a non-trivial ``kernel`` is supplied, the residual and Jacobian are
re-weighted IRLS-style each iteration: every row ``i`` is scaled by
``sqrt(kernel.weight(r_i²))`` before forming the normal equations.
Trial steps are accepted using the matching robust objective
``Σ kernel.rho(r_i²)`` (or ``0.5·‖r‖²`` without a kernel).

Box bounds are enforced by projecting every trial point onto
``[lower, upper]`` before evaluating its residual. This is not a bounded-LM
algorithm: acceptance is a bare objective comparison, with no active set,
projected-gradient test, or KKT termination. A run limited by active bounds
can therefore stall with residual error and terminate as ``"maxiter"``.
Active-set LM or a reflective trust region is planned for M2b.

Returns a :class:`~better_robot.optim.state.SolverState` whose
``status`` is ``"converged"`` when the gradient norm drops below ``tol``
and ``"maxiter"`` when the iteration budget is exhausted. The LM path does
not currently emit ``"stalled"``.

See ``docs/concepts/solver_stack.md §5``.
"""

from __future__ import annotations

import torch

from ..problem import LeastSquaresProblem
from ..state import SolverState


def _apply_kernel(
    r: torch.Tensor,
    J: torch.Tensor,
    kernel,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(sqrt(w)·r, sqrt(w)·J)`` for IRLS reweighting.

    ``kernel.weight(r_i²)`` produces a per-row weight. Built-in kernels use
    the normalized convention ``w_i = 2·rho'(r_i²)``, and scaling each
    residual row and matching Jacobian row by ``sqrt(w_i)`` gives the IRLS
    normal equations (Triggs et al. 2000, §4).
    """
    if kernel is None:
        return r, J
    sq = r * r
    w = kernel.weight(sq)
    sw = torch.sqrt(w.clamp(min=0.0))
    return r * sw, J * sw.unsqueeze(-1)


def _robust_cost(r: torch.Tensor, kernel) -> torch.Tensor:
    """Return the scalar objective used for LM trial-step acceptance."""
    squared = r * r
    if kernel is None:
        return 0.5 * squared.sum()
    return kernel.rho(squared).sum()


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

        See docs/concepts/solver_stack.md §5.
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
        cost = float(_robust_cost(state.residual, kernel))

        it = -1
        for it in range(max_iter):
            J = problem.jacobian(state.x)                                 # (dim, nv)
            r_w, J_w = _apply_kernel(state.residual, J, kernel)

            # Normal equations: (J^T J + lam * I) delta_v = -J^T r
            JtJ = J_w.mT @ J_w                                            # (nv, nv)
            Jtr = J_w.mT @ r_w                                            # (nv,)
            H = JtJ + state.damping * torch.eye(nv, dtype=J.dtype, device=J.device)
            delta_v = solver.solve(H, -Jtr)                               # (nv,)

            # Manifold-aware update
            x_new = problem.step(state.x, delta_v)

            # Clamp to box bounds (no-op for ±inf)
            if problem.lower is not None:
                x_new = x_new.clamp(
                    min=problem.lower.to(x_new.device, x_new.dtype),
                    max=problem.upper.to(x_new.device, x_new.dtype),
                )

            r_new = problem.residual(x_new)
            cost_new = float(_robust_cost(r_new, kernel))

            state.history.append({"iter": it, "cost": cost, "lam": state.damping})

            if cost_new < cost:
                # Gain ratio = (actual decrease) / (predicted decrease).
                predicted = float(-delta_v @ Jtr - 0.5 * (delta_v @ (JtJ @ delta_v)))
                state.gain_ratio = (cost - cost_new) / predicted if predicted > 0 else None

                state.x = x_new
                state.residual = r_new
                # ``residual_norm`` remains the raw 0.5·‖r‖² diagnostic even
                # when robust rho drives acceptance.
                state.residual_norm = 0.5 * (r_new * r_new).sum()
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
