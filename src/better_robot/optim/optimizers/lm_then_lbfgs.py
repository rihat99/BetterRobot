"""``LMThenLBFGS`` — thin wrapper around :class:`MultiStageOptimizer`.

Two-stage chain (LM coarse, L-BFGS fine), with optional cost-stack items
disabled in the L-BFGS stage. Kept as a named class because it is the
common case and the pre-multi-stage call sites reach for it by name.

See ``docs/concepts/solver_stack.md §5``.
"""

from __future__ import annotations

from collections.abc import Iterable

from .lbfgs import LBFGS
from .levenberg_marquardt import LevenbergMarquardt
from .multi_stage import MultiStageOptimizer, OptimizerStage


class LMThenLBFGS:
    """LM (coarse) → L-BFGS (refine), with optional stage-2 disabled items."""

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
        self._inner = MultiStageOptimizer(
            stages=(
                OptimizerStage(optimizer=self.lm, max_iter=stage1_max_iter),
                OptimizerStage(
                    optimizer=self.lbfgs,
                    max_iter=stage2_max_iter,
                    disabled_items=self.stage2_disabled_items,
                ),
            )
        )

    def minimize(
        self,
        problem,
        *,
        max_iter=None,
        linear_solver=None,
        kernel=None,
        strategy=None,
        scheduler=None,
    ):
        return self._inner.minimize(
            problem,
            max_iter=max_iter,
            linear_solver=linear_solver,
            kernel=kernel,
            strategy=strategy,
            scheduler=scheduler,
        )
