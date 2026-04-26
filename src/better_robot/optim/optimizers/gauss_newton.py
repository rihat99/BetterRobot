"""Gauss-Newton optimiser.

Undamped Gauss-Newton with a tiny Tikhonov regularisation for rank
deficiency. Shares the same plug-in linear-solver interface as LM so the
outer shell of ``minimize`` is almost identical — the only difference is
the fixed (near-zero) damping. Honours ``kernel`` via IRLS reweighting.

See ``docs/concepts/solver_stack.md §5``.
"""

from __future__ import annotations

import torch

from ..problem import LeastSquaresProblem
from ..state import SolverState
from .levenberg_marquardt import _apply_kernel


class GaussNewton:
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
    ) -> SolverState:
        """Run Gauss-Newton until convergence or ``max_iter`` is reached.

        See docs/concepts/solver_stack.md §5.
        """
        from ..solvers.cholesky import Cholesky

        solver = linear_solver if linear_solver is not None else Cholesky()
        state = SolverState.from_problem(problem)
        nv = problem._nv

        it = -1
        for it in range(max_iter):
            J = problem.jacobian(state.x)                                # (dim, nv)
            r_w, J_w = _apply_kernel(state.residual, J, kernel)
            JtJ = J_w.mT @ J_w                                           # (nv, nv)
            Jtr = J_w.mT @ r_w                                           # (nv,)
            H = JtJ + self.eps * torch.eye(nv, dtype=J.dtype, device=J.device)
            delta_v = solver.solve(H, -Jtr)                              # (nv,)

            x_new = problem.step(state.x, delta_v)
            if problem.lower is not None:
                x_new = x_new.clamp(
                    min=problem.lower.to(x_new.device, x_new.dtype),
                    max=problem.upper.to(x_new.device, x_new.dtype),
                )

            state.x = x_new
            state.residual = problem.residual(x_new)
            cost = float(0.5 * (state.residual @ state.residual).sum())
            state.residual_norm = torch.as_tensor(cost)
            state.history.append({"iter": it, "cost": cost})

            if float(Jtr.norm()) < self.tol:
                state.status = "converged"
                state.iters = it + 1
                return state

        state.iters = it + 1
        state.status = "maxiter"
        return state
