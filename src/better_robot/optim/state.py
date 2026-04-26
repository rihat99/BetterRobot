"""``SolverState`` — shared iteration record passed through every optimiser.

Docs: ``docs/concepts/solver_stack.md §5``. The same struct serves as:

1. the **iteration state** mutated by :meth:`Optimizer.minimize` each step;
2. the **terminal return value** of :meth:`Optimizer.minimize`;
3. the **contract** observed by damping strategies, linear solvers, and
   downstream consumers such as ``tasks.ik.solve_ik``.

Having one struct avoids the mjwarp "pass tensors everywhere and keep
them in sync" trap and makes it easy for a consumer to assert
``isinstance(result, SolverState)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from .problem import LeastSquaresProblem


SolverStatus = Literal["running", "converged", "stalled", "maxiter"]


@dataclass
class SolverState:
    """Iteration-level solver state — also the terminal result.

    Parameters
    ----------
    x : Tensor
        Current iterate (``(nx,)`` or batched ``(B..., nx)``).
    residual : Tensor
        Residual at ``x``. Shape ``(dim,)`` or ``(B..., dim)``.
    residual_norm : Tensor
        ``0.5 * ‖r‖²``. Scalar or leading-batch tensor.
    iters : int
        Completed iteration count. ``0`` before the loop starts.
    damping : float
        Current LM ``λ``. ``0.0`` for solvers that don't use damping
        (``GaussNewton``, ``Adam``, ``LBFGS``).
    gain_ratio : float | None
        Ratio of actual to predicted decrease on the last *accepted*
        step. ``None`` before the first accepted step.
    status : SolverStatus
        ``"running"`` while iterating; terminal value tells the caller
        why the loop ended (``"converged"``, ``"stalled"``, ``"maxiter"``).
    history : list[dict]
        Optional trajectory log. Solvers append one dict per iteration
        with whatever they want to expose (``cost``, ``lam``, …).

    Notes
    -----
    ``.converged`` is a ``@property`` so existing callers that were
    written against the old ``OptimizationResult`` interface keep
    working during the deprecation window.
    """

    x: torch.Tensor
    residual: torch.Tensor
    residual_norm: torch.Tensor
    iters: int = 0
    damping: float = 0.0
    gain_ratio: float | None = None
    status: SolverStatus = "running"
    history: list[dict] = field(default_factory=list)

    # ──────────────────────────── helpers ────────────────────────────

    @property
    def converged(self) -> bool:
        """True iff the loop terminated with ``status="converged"``.

        Kept as a property so callers (``IKResult``, contract tests, user
        scripts) that branch on ``result.converged`` keep working.
        """
        return self.status == "converged"

    @classmethod
    def from_problem(cls, problem: "LeastSquaresProblem") -> "SolverState":
        """Allocate an initial :class:`SolverState` for ``problem``.

        Evaluates the residual once at ``problem.x0`` so the loop can skip
        the 0-iteration special case.
        """
        x0 = problem.x0.clone().detach()
        r0 = problem.residual(x0)
        return cls(
            x=x0,
            residual=r0,
            residual_norm=0.5 * (r0 @ r0),
            iters=0,
            damping=0.0,
            gain_ratio=None,
            status="running",
        )
