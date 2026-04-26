"""L-BFGS optimiser in tangent space.

Two-loop recursion L-BFGS that stores its history in the ``nv`` tangent
space and applies updates through ``problem.step`` (the model's
manifold retraction).  Driven by ``problem.jacobian`` so it works for
both fixed-base and free-flyer robots, same as the Adam implementation
in ``optim/optimizers/adam.py``.

A simple Armijo backtracking line search guards against divergence on
nonconvex objectives.

Returns a :class:`~better_robot.optim.state.SolverState`; ``status`` is
``"converged"`` when the gradient norm drops below ``tol``,
``"stalled"`` when the line search cannot find a descent direction, and
``"maxiter"`` otherwise.

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

import torch

from ..problem import LeastSquaresProblem
from ..state import SolverState


class LBFGS:
    """L-BFGS with Armijo backtracking, evaluated in tangent space."""

    # Read by ``MultiStageOptimizer`` to suppress top-level kwarg forwarding
    # — warning is intended for direct user calls, not internal wrappers.
    _ignores_normal_eqn_kwargs: bool = True

    def __init__(
        self,
        *,
        history_size: int = 10,
        lr: float = 1.0,
        tol: float = 1e-6,
        c1: float = 1e-4,
        max_ls: int = 20,
    ) -> None:
        self.history_size = history_size
        self.lr = lr
        self.tol = tol
        self.c1 = c1
        self.max_ls = max_ls

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
        """Run L-BFGS until convergence or ``max_iter`` is reached.

        See docs/design/07_RESIDUALS_COSTS_SOLVERS.md §5.
        """
        import warnings
        for _name, _val in (("linear_solver", linear_solver),
                            ("kernel", kernel),
                            ("strategy", strategy)):
            if _val is not None:
                warnings.warn(
                    f"LBFGS ignores {_name}={_val!r} — quasi-Newton method has "
                    f"no normal-equation step. See docs/design/07_RESIDUALS_COSTS_SOLVERS.md §5.",
                    UserWarning,
                    stacklevel=2,
                )
        state = SolverState.from_problem(problem)

        s_hist: list[torch.Tensor] = []
        y_hist: list[torch.Tensor] = []
        rho_hist: list[torch.Tensor] = []

        J = problem.jacobian(state.x)
        g = J.mT @ state.residual                  # current gradient (nv,)
        cost = float(state.residual_norm)

        it = -1
        for it in range(max_iter):
            if float(g.norm()) < self.tol:
                state.status = "converged"
                state.iters = it
                return state

            # ── two-loop recursion: d = -H_k * g ─────────────────────────────
            q = g.clone()
            alpha = [torch.zeros((), dtype=g.dtype, device=g.device)] * len(s_hist)  # bench-ok: scalar list, not a hot-path tensor alloc
            for i in range(len(s_hist) - 1, -1, -1):
                alpha[i] = rho_hist[i] * (s_hist[i] @ q)
                q = q - alpha[i] * y_hist[i]

            if s_hist:
                s_last, y_last = s_hist[-1], y_hist[-1]
                gamma = (s_last @ y_last) / (y_last @ y_last).clamp(min=1e-12)
                z = gamma * q
            else:
                z = q

            for i in range(len(s_hist)):
                beta = rho_hist[i] * (y_hist[i] @ z)
                z = z + (alpha[i] - beta) * s_hist[i]

            direction = -z                               # (nv,)

            # ── Armijo backtracking line search ─────────────────────────────
            step_size = self.lr
            dir_dot_grad = float(direction @ g)
            if dir_dot_grad >= 0.0:
                # Not a descent direction — reset to steepest descent.
                direction = -g
                dir_dot_grad = -float(g @ g)
                s_hist.clear(); y_hist.clear(); rho_hist.clear()

            x_new = state.x
            r_new = state.residual
            cost_new = cost
            accepted = False
            for _ls in range(self.max_ls):
                delta_v = step_size * direction
                x_try = problem.step(state.x, delta_v)
                if problem.lower is not None:
                    x_try = x_try.clamp(
                        min=problem.lower.to(x_try.device, x_try.dtype),
                        max=problem.upper.to(x_try.device, x_try.dtype),
                    )
                r_try = problem.residual(x_try)
                cost_try = float(0.5 * (r_try @ r_try).sum())
                if cost_try <= cost + self.c1 * step_size * dir_dot_grad:
                    x_new, r_new, cost_new = x_try, r_try, cost_try
                    accepted = True
                    break
                step_size *= 0.5

            if not accepted:
                # Line search failed — bail out with "stalled".
                state.iters = it + 1
                state.status = "stalled"
                return state

            # ── update curvature pair ───────────────────────────────────────
            J_new = problem.jacobian(x_new)
            g_new = J_new.mT @ r_new
            s_k = step_size * direction
            y_k = g_new - g
            sy = float(s_k @ y_k)
            if sy > 1e-12:
                s_hist.append(s_k)
                y_hist.append(y_k)
                rho_hist.append(torch.tensor(1.0 / sy, dtype=g.dtype, device=g.device))
                if len(s_hist) > self.history_size:
                    s_hist.pop(0)
                    y_hist.pop(0)
                    rho_hist.pop(0)

            state.x = x_new
            state.residual = r_new
            state.residual_norm = torch.as_tensor(cost_new)
            g = g_new
            cost = cost_new
            state.history.append({"iter": it, "cost": cost})

        state.iters = it + 1
        state.status = "maxiter"
        return state
