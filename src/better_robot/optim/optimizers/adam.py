"""Adam first-order optimiser in tangent space.

Runs Adam over ``0.5 * ||r(x)||²``. The gradient is pulled from
``problem.jacobian(x)`` — specifically ``Jᵀ r`` — instead of routing
autograd through the residual. This sidesteps PyPose's known
factor-of-2 bug in the quaternion ``Log().backward()`` path (see
``docs/05_KINEMATICS.md §3``) and makes Adam work identically on
fixed-base and free-flyer robots: every moment is tracked in ``nv``
space and applied through ``problem.step`` (the manifold retraction).

See ``docs/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

import torch

from ..problem import LeastSquaresProblem
from .base import OptimizationResult, Optimizer


class Adam(Optimizer):
    """Adam on ``0.5 * ||r(x)||^2`` driven by ``problem.jacobian``."""

    def __init__(
        self,
        *,
        lr: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        tol: float = 1e-6,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.tol = tol

    def minimize(
        self,
        problem: LeastSquaresProblem,
        *,
        max_iter: int = 200,
        linear_solver=None,
        kernel=None,
        strategy=None,
        scheduler=None,
    ) -> OptimizationResult:
        """Run Adam until convergence or ``max_iter`` is reached.

        See docs/07_RESIDUALS_COSTS_SOLVERS.md §5.
        """
        x = problem.x0.clone().detach()
        nv = problem._nv
        device, dtype = x.device, x.dtype

        m = torch.zeros(nv, dtype=dtype, device=device)
        v = torch.zeros(nv, dtype=dtype, device=device)

        history: list[dict] = []
        converged = False
        it = 0
        r = problem.residual(x)

        for it in range(1, max_iter + 1):
            J = problem.jacobian(x)          # (dim, nv)
            grad = J.mT @ r                  # (nv,)  — ∇ (½‖r‖²)

            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)
            m_hat = m / (1.0 - self.beta1 ** it)
            v_hat = v / (1.0 - self.beta2 ** it)
            delta_v = -self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

            x = problem.step(x, delta_v)
            if problem.lower is not None:
                x = x.clamp(
                    min=problem.lower.to(x.device, x.dtype),
                    max=problem.upper.to(x.device, x.dtype),
                )

            r = problem.residual(x)
            cost = float(0.5 * (r @ r).sum())
            history.append({"iter": it, "cost": cost})

            if float(grad.norm()) < self.tol:
                converged = True
                break

        return OptimizationResult(
            x=x, residual=r, iters=it, converged=converged, history=history
        )
