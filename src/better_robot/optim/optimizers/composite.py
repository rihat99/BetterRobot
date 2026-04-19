"""Composite two-stage solver — ``LM`` for coarse, ``LBFGS`` for refinement.

Pattern adopted from cuRobo's high-DoF humanoid IK: an LM stage gets into
the basin of attraction quickly, then a quasi-Newton stage finishes the
residual to tight tolerance without the cost of repeatedly solving a
full ``JᵀJ + λI`` system.

``stage2_disabled_items`` optionally deactivates named entries of
``problem.cost_stack`` during the LBFGS refinement — the typical use is
to drop the collision residual once LM has the configuration safely
inside free space (``docs/08_TASKS.md``).

See ``docs/08_TASKS.md`` and ``docs/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..problem import LeastSquaresProblem
from ..state import SolverState
from .lbfgs import LBFGS
from .levenberg_marquardt import LevenbergMarquardt


class LMThenLBFGS:
    """Run Levenberg-Marquardt then L-BFGS on the same problem."""

    def __init__(
        self,
        *,
        stage1_max_iter: int = 50,
        stage2_max_iter: int = 50,
        stage2_disabled_items: Iterable[str] = (),
        tol: float = 1e-6,
        lm: LevenbergMarquardt | None = None,
        lbfgs: LBFGS | None = None,
    ) -> None:
        self.stage1_max_iter = stage1_max_iter
        self.stage2_max_iter = stage2_max_iter
        self.stage2_disabled_items = tuple(stage2_disabled_items)
        self.tol = tol
        self.lm = lm if lm is not None else LevenbergMarquardt(tol=tol)
        self.lbfgs = lbfgs if lbfgs is not None else LBFGS(tol=tol)

    def minimize(
        self,
        problem: LeastSquaresProblem,
        *,
        max_iter: int | None = None,
        linear_solver=None,
        kernel=None,
        strategy=None,
        scheduler=None,
    ) -> SolverState:
        """Run LM for ``stage1_max_iter`` iterations then LBFGS for the rest.

        If ``max_iter`` is provided, it replaces the sum of the two stage
        budgets (split 50/50). Otherwise the constructor budgets are used.
        """
        if max_iter is not None:
            half = max(1, max_iter // 2)
            stage1 = half
            stage2 = max(1, max_iter - half)
        else:
            stage1 = self.stage1_max_iter
            stage2 = self.stage2_max_iter

        # ── Stage 1: Levenberg-Marquardt ────────────────────────────────────
        state1 = self.lm.minimize(
            problem,
            max_iter=stage1,
            linear_solver=linear_solver,
            kernel=kernel,
            strategy=strategy,
            scheduler=scheduler,
        )
        if state1.status == "converged":
            return state1

        # ── Stage 2: L-BFGS from the LM solution ────────────────────────────
        # Temporarily disable named cost items; reactivate in `finally`.
        disabled_previous: dict[str, bool] = {}
        for name in self.stage2_disabled_items:
            if name in problem.cost_stack.items:
                disabled_previous[name] = problem.cost_stack.items[name].active
                problem.cost_stack.set_active(name, False)

        try:
            import dataclasses
            problem_stage2 = dataclasses.replace(problem, x0=state1.x.detach())
            state2 = self.lbfgs.minimize(
                problem_stage2,
                max_iter=stage2,
                linear_solver=linear_solver,
                kernel=kernel,
                strategy=strategy,
                scheduler=scheduler,
            )
        finally:
            for name, was_active in disabled_previous.items():
                problem.cost_stack.set_active(name, was_active)

        # Merge: keep stage-2's terminal state but extend the history.
        state2.history = state1.history + state2.history
        state2.iters = state1.iters + state2.iters
        return state2
